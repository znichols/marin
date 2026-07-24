# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM local inference backend."""

import contextlib
import hashlib
import logging
import os
import socket
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from fsspec.core import url_to_fs

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.inference.backend import OPENAI_API_SUFFIX, ModelSpec
from marin.inference.config import DEFAULT_CUDA_VLLM_VERSION, VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.tpu_vllm_pins import tpu_inference_fork_ref, vllm_fork_ref
from marin.inference.vllm_server import (
    IsolatedCudaVllm,
    IsolatedTpuVllm,
    VllmEnvironment,
    VllmLauncher,
    VllmType,
    WorkspaceVllm,
)

logger = logging.getLogger(__name__)
_LOCAL_MODEL_STAGING_ROOT_ENV = "MARIN_LOCAL_MODEL_STAGING_ROOT"


def stage_object_store_model_locally(model_path: str) -> str:
    """Materialize an fsspec object-store model directory on this worker's disk.

    CUDA eval pods on CoreWeave reserve enough ephemeral disk for a complete
    checkpoint.  Local staging is deliberately performed before vLLM starts so
    its normal local-file loader is used instead of the RunAI streaming plugin.
    The deterministic path also makes a retry within the same pod resumable.
    """
    if not model_path.startswith(("s3://", "gs://")):
        return model_path

    logger.info("Enumerating object-store model files for local staging: %s", model_path)
    filesystem, remote_root = url_to_fs(model_path)
    remote_root = remote_root.rstrip("/")
    if not remote_root:
        raise ValueError(f"Object-store model path must include a checkpoint prefix: {model_path}")
    remote_files = sorted(filesystem.find(remote_root))
    if not remote_files:
        raise FileNotFoundError(f"No files found under object-store model path: {model_path}")

    cache_root = Path(os.environ.get(_LOCAL_MODEL_STAGING_ROOT_ENV, tempfile.gettempdir())) / "marin-models"
    destination = cache_root / hashlib.sha256(model_path.encode()).hexdigest()[:16]
    expected_config = f"{remote_root}/config.json"
    if expected_config not in remote_files:
        raise FileNotFoundError(f"Object-store model path has no config.json: {model_path}")

    logger.info("Staging %s object-store model files locally at %s", len(remote_files), destination)
    copied = 0
    for file_index, remote_file in enumerate(remote_files, start=1):
        info = filesystem.info(remote_file)
        if info.get("type") == "directory":
            continue
        relative = os.path.relpath(remote_file, remote_root)
        if relative == os.pardir or relative.startswith(f"{os.pardir}{os.sep}"):
            raise ValueError(f"Object-store listing escaped model prefix: {remote_file}")
        local_file = destination / relative
        expected_size = info.get("size")
        if expected_size is not None and local_file.is_file() and local_file.stat().st_size == expected_size:
            continue
        local_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = local_file.with_suffix(f"{local_file.suffix}.partial")
        filesystem.get_file(remote_file, str(temporary_file))
        os.replace(temporary_file, local_file)
        copied += 1
        if copied % 25 == 0:
            logger.info("Local model staging progress: %s/%s files copied", file_index, len(remote_files))

    logger.info("Staged %s object-store model files locally at %s (%s copied)", len(remote_files), destination, copied)
    return str(destination)


def vllm_launcher(config: VllmEngineConfig) -> VllmLauncher:
    if config.launcher is VllmLauncherType.WORKSPACE:
        return WorkspaceVllm()
    if config.launcher is VllmLauncherType.TPU:
        return IsolatedTpuVllm(vllm_ref=vllm_fork_ref(), tpu_inference_ref=tpu_inference_fork_ref())
    source = VllmType.MARIN_FORK if config.source is VllmSource.MARIN_FORK else VllmType.UPSTREAM
    version = config.version if source is VllmType.UPSTREAM else None
    if source is VllmType.UPSTREAM and version is None:
        version = DEFAULT_CUDA_VLLM_VERSION
    return IsolatedCudaVllm(source=source, version=version)


def _reserve_localhost_port(host: str) -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _served_model_name_args(model: str, extra_args: tuple[str, ...]) -> tuple[str, ...]:
    """Supply the default served name only when a caller has not explicitly selected one."""
    has_explicit_name = any(arg == "--served-model-name" or arg.startswith("--served-model-name=") for arg in extra_args)
    return () if has_explicit_name else ("--served-model-name", model)


@contextlib.contextmanager
def _chat_template_argument(content: str | None) -> Iterator[tuple[str, ...]]:
    if content is None:
        yield ()
        return
    with tempfile.NamedTemporaryFile("w", suffix=".jinja", prefix="marin_serve_chat_", delete=False) as handle:
        handle.write(content)
        path = handle.name
    try:
        yield ("--chat-template", path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


@dataclass(frozen=True)
class VllmServedModel:
    base_url: str
    model_id: str
    environment: VllmEnvironment

    def check_alive(self) -> None:
        server = self.environment.vllm_server
        if server is not None and server.process.poll() is not None:
            raise RuntimeError(f"vLLM server exited unexpectedly with code {server.process.returncode}")


@dataclass(frozen=True)
class VllmBackend:
    config: VllmEngineConfig
    host: str = "127.0.0.1"
    port: int | None = None
    name: str = "vllm"

    @contextlib.contextmanager
    def serve(self, spec: ModelSpec) -> Iterator[VllmServedModel]:
        resolved_port = _reserve_localhost_port(self.host) if self.port is None else self.port
        engine_kwargs: dict[str, object] = {"dtype": spec.dtype}
        if spec.max_model_len is not None:
            engine_kwargs["max_model_len"] = spec.max_model_len
        if self.config.max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = self.config.max_num_batched_tokens
        model_path = spec.model_path
        if model_path is not None and self.config.stage_object_store_model_locally:
            model_path = stage_object_store_model_locally(model_path)
        model = ModelConfig(name=spec.model, path=model_path, engine_kwargs=engine_kwargs)
        with _chat_template_argument(spec.chat_template_content) as chat_template_args:
            with VllmEnvironment(
                model=model,
                host=self.host,
                port=resolved_port,
                timeout_seconds=self.config.startup_timeout_seconds,
                extra_args=[
                    *(
                        ("--tensor-parallel-size", str(spec.tensor_parallel_size))
                        if spec.tensor_parallel_size is not None
                        else ()
                    ),
                    *_served_model_name_args(spec.model, self.config.extra_args),
                    *chat_template_args,
                    *self.config.extra_args,
                ],
                launcher=vllm_launcher(self.config),
            ) as environment:
                if environment.model_id is None:
                    raise RuntimeError("vLLM server did not report a model id")
                yield VllmServedModel(
                    base_url=environment.server_url.removesuffix(OPENAI_API_SUFFIX),
                    model_id=environment.model_id,
                    environment=environment,
                )
