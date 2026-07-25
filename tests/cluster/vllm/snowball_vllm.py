# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture Snowball observations through the standard vLLM serving path."""

import dataclasses
import hashlib
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from marin.inference import config as inference_config_module
from marin.inference import vllm_backend as vllm_backend_module
from marin.inference import vllm_cache as vllm_cache_module
from marin.inference import vllm_server as vllm_server_module
from marin.inference.backend import OPENAI_API_SUFFIX, ModelSpec
from marin.inference.config import VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.tpu_vllm_pins import fork_source_revision, tpu_inference_fork_ref, vllm_fork_ref
from marin.inference.vllm_backend import VllmBackend
from rigging.filesystem import StoragePath

from tests.cluster.vllm import backend_parity as backend_parity_module
from tests.cluster.vllm import snowball as snowball_module
from tests.cluster.vllm.backend_parity import (
    NextTokenObservation,
    ObservationReport,
    RunProvenance,
    request_next_token_observation,
    source_digest,
)
from tests.cluster.vllm.snowball import (
    GPU_RANK_SENTINEL_CASE_ID,
    PROMPT_FIXTURE_SHA256,
    SNOWBALL,
    VLLM_HTTP_TIMEOUT,
    VLLM_MAX_MODEL_LEN,
    VLLM_MAX_NUM_BATCHED_TOKENS,
    RepresentativeGolden,
    VllmCell,
    prepare_capture_selection,
)

CAPTURE_SOURCE_DIGEST = source_digest(
    backend_parity_module.__file__,
    inference_config_module.__file__,
    vllm_backend_module.__file__,
    vllm_cache_module.__file__,
    vllm_server_module.__file__,
    snowball_module.__file__,
    __file__,
)


def _detect_device_kind(platform: str, *, expected_count: int) -> str:
    """Read accelerator identity from the hardware after the server releases it."""
    if platform == "tpu":
        import jax  # noqa: PLC0415 -- importing before vLLM exits would contend for libtpu

        device_kinds = {device.device_kind for device in jax.devices()}
        device_count = jax.device_count()
    elif platform == "gpu":
        output = subprocess.check_output(
            ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"),
            text=True,
        )
        names = tuple(line.strip() for line in output.splitlines() if line.strip())
        device_kinds = set(names)
        device_count = len(names)
    else:
        raise ValueError(f"Unsupported platform {platform!r}")
    if device_count != expected_count or len(device_kinds) != 1:
        raise RuntimeError(
            f"Expected {expected_count} identical {platform} devices, found count={device_count} kinds={device_kinds}"
        )
    return next(iter(device_kinds))


def capture_vllm(
    cell: VllmCell,
    *,
    same_process_repeats: int = 1,
    batch_indices: tuple[int, ...] | None = None,
    case_ids: tuple[str, ...] | None = None,
    goldens: tuple[RepresentativeGolden, ...] | None = None,
    additional_goldens: tuple[tuple[RepresentativeGolden, ...], ...] = (),
    golden_digest: str,
    candidate_revision: str = "development",
) -> ObservationReport:
    """Capture GPU or TPU vLLM through one OpenAI request/response adapter."""
    if same_process_repeats <= 0:
        raise ValueError("same_process_repeats must be positive")
    if cell.location.export_uri is None:
        raise ValueError("vLLM requires the verified platform-local export")

    prompt_fixture, selected_ids_by_case, selected_indices = prepare_capture_selection(
        cell.location,
        goldens=goldens,
        additional_goldens=additional_goldens,
        batch_indices=batch_indices,
    )
    selected_case_ids = None if case_ids is None else set(case_ids)
    if selected_case_ids is not None:
        if len(selected_case_ids) != len(case_ids):
            raise ValueError(f"Duplicate case ids {case_ids}")
        known_case_ids = {case.id for case in prompt_fixture.cases}
        unknown_case_ids = selected_case_ids - known_case_ids
        if unknown_case_ids:
            raise ValueError(f"Unknown case ids {sorted(unknown_case_ids)}")

    config_bytes = (StoragePath(cell.location.export_uri) / "config.json").read_bytes()
    num_chips = cell.tensor_parallel_size * cell.data_parallel_size
    spec = ModelSpec(
        weights=cell.location.export_uri,
        api_model=SNOWBALL.model_name,
        num_chips=num_chips,
        tensor_parallel_size=cell.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=VLLM_MAX_MODEL_LEN,
        chat_template_content=None,
    )
    is_tpu = cell.location.name == "tpu"
    vllm_requirement = vllm_fork_ref()
    tpu_inference_requirement = tpu_inference_fork_ref() if is_tpu else None
    extra_args = [
        "--max-num-seqs",
        "1",
        "--max-logprobs",
        str(cell.returned_logprobs),
    ]
    if is_tpu:
        extra_args.append("--no-enable-prefix-caching")
    else:
        if cell.attention_backend is None:
            raise ValueError("GPU vLLM capture requires an attention backend")
        extra_args.extend(
            [
                "--data-parallel-size",
                str(cell.data_parallel_size),
                "--enable-expert-parallel",
                "--model-loader-extra-config",
                '{"distributed":true}',
                "--attention-backend",
                cell.attention_backend,
            ]
        )
    backend = VllmBackend(
        VllmEngineConfig(
            launcher=VllmLauncherType.TPU if is_tpu else VllmLauncherType.CUDA,
            source=VllmSource.UPSTREAM if is_tpu else VllmSource.MARIN_FORK,
            max_num_batched_tokens=VLLM_MAX_NUM_BATCHED_TOKENS,
            extra_args=tuple(extra_args),
        ),
    )

    observations = []
    failures = []
    diagnostics = []
    diagnostic_failures = []
    runtime_fingerprint = None
    with backend.serve(spec) as served:
        runtime_fingerprint = served.environment.runtime_fingerprint()
        completions_url = f"{served.base_url}{OPENAI_API_SUFFIX}/completions"

        def request_case(case, bucket_max_tokens, repeat_index, backend_index):
            headers = None
            if cell.data_parallel_size > 1:
                request_id = f"repeat-{repeat_index}-{case.id}-rank-{backend_index}"
                headers = {
                    "X-data-parallel-rank": str(backend_index),
                    "X-Request-Id": request_id,
                }
            return request_next_token_observation(
                completions_url,
                served.model_id,
                case_id=case.id,
                prompt_token_ids=case.prompt_token_ids,
                selected_token_ids=selected_ids_by_case[case.id],
                bucket_max_tokens=bucket_max_tokens,
                repeat_index=repeat_index,
                backend_index=backend_index,
                returned_logprobs=cell.returned_logprobs,
                headers=headers,
                timeout=VLLM_HTTP_TIMEOUT,
            )

        for repeat_index in range(same_process_repeats):
            for batch_index in selected_indices:
                batch = prompt_fixture.batches[batch_index]
                cases = tuple(case for case in batch.cases if selected_case_ids is None or case.id in selected_case_ids)
                if cell.data_parallel_size > 1:
                    with ThreadPoolExecutor(max_workers=cell.data_parallel_size) as executor:
                        futures = [
                            executor.submit(request_case, case, batch.max_tokens, repeat_index, rank)
                            for rank, case in enumerate(cases)
                        ]
                        results = [future.result() for future in as_completed(futures)]
                else:
                    results = [request_case(case, batch.max_tokens, repeat_index, 0) for case in cases]
                for result in results:
                    if isinstance(result, NextTokenObservation):
                        observations.append(result)
                    else:
                        failures.append(result)

        if cell.data_parallel_size > 1:
            sentinel = next(case for case in prompt_fixture.cases if case.id == GPU_RANK_SENTINEL_CASE_ID)
            if len(sentinel.prompt_token_ids) <= 2048:
                raise ValueError("GPU rank sentinel must remain a long-context prompt")
            sentinel_bucket = next(batch.max_tokens for batch in prompt_fixture.batches if sentinel in batch.cases)
            for repeat_index in range(same_process_repeats):
                with ThreadPoolExecutor(max_workers=cell.data_parallel_size) as executor:
                    futures = [
                        executor.submit(request_case, sentinel, sentinel_bucket, repeat_index, rank)
                        for rank in range(cell.data_parallel_size)
                    ]
                    results = [future.result() for future in as_completed(futures)]
                for result in results:
                    if isinstance(result, NextTokenObservation):
                        diagnostics.append(result)
                    else:
                        diagnostic_failures.append(result.model_copy(update={"stage": "rank-sentinel"}))

    assert runtime_fingerprint is not None
    device_kind = _detect_device_kind(cell.location.name, expected_count=num_chips)
    runtime_payload = dataclasses.asdict(runtime_fingerprint)
    fork_revisions = [("vllm", fork_source_revision(vllm_requirement, package="vllm"))]
    if tpu_inference_requirement is not None:
        fork_revisions.append(
            (
                "tpu-inference",
                fork_source_revision(tpu_inference_requirement, package="tpu-inference"),
            )
        )
    return ObservationReport(
        provenance=RunProvenance(
            backend="vllm",
            platform=cell.location.name,
            process_id=uuid.uuid4().hex,
            code_digest=CAPTURE_SOURCE_DIGEST,
            candidate_revision=candidate_revision,
            parameter_digest=SNOWBALL.export_sha256,
            model_config_digest=hashlib.sha256(config_bytes).hexdigest(),
            prompt_fixture_digest=PROMPT_FIXTURE_SHA256,
            mesh_shape=(
                ("tensor", cell.tensor_parallel_size),
                ("data", cell.data_parallel_size),
                ("expert", 1),
            ),
            device_kind=device_kind,
            golden_digest=golden_digest,
            fork_source_revisions=tuple(fork_revisions),
            runtime_fingerprint=runtime_payload,
            runtime_fingerprint_digest=runtime_fingerprint.digest(),
        ),
        observations=tuple(observations),
        failures=tuple(failures),
        diagnostics=tuple(diagnostics),
        diagnostic_failures=tuple(diagnostic_failures),
    )
