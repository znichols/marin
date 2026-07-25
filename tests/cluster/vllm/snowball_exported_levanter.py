# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture Snowball next-token observations after loading the canonical HF export.

Snowball implements Levanter's full-forward model interface but not the paged
``initial_cache``/``decode`` interface. ``LevanterBackend.serve`` therefore
rejects it before loading weights. This capture uses ``LevanterBackend.load_model``
and a teacher-forced full forward so the exact fixed-batch next-token scores can
still be compared without pretending an endpoint path exists for this model.
"""

import hashlib
import uuid

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from jax.sharding import PartitionSpec as P
from levanter.models import snowball as levanter_snowball_module
from levanter.models.snowball import SnowballLMHeadModel
from marin.inference import config as inference_config_module
from marin.inference import levanter_backend as levanter_backend_module
from marin.inference import model_preparation as model_preparation_module
from marin.inference.backend import ModelSpec
from marin.inference.config import LevanterEngineConfig
from marin.inference.levanter_backend import LevanterBackend
from marin.inference.model_preparation import read_attention_heads, select_tensor_parallel_size
from rigging.filesystem import StoragePath

from tests.cluster.vllm import backend_parity as backend_parity_module
from tests.cluster.vllm import snowball as snowball_module
from tests.cluster.vllm import snowball_levanter as snowball_levanter_module
from tests.cluster.vllm.backend_parity import (
    ObservationReport,
    RunProvenance,
    numerical_runtime_fingerprint,
    source_digest,
)
from tests.cluster.vllm.snowball import (
    BATCH_SIZE,
    PROMPT_FIXTURE_SHA256,
    REPRESENTATIVE_GOLDEN_SHA256,
    SNOWBALL,
    VLLM_MAX_MODEL_LEN,
    ExportedLevanterCell,
    RepresentativeGolden,
    prepare_capture_selection,
)
from tests.cluster.vllm.snowball_levanter import (
    build_batch_observations,
    rectangular_selected_token_ids,
    score_logprob_diagnostics,
)

CAPTURE_SOURCE_DIGEST = source_digest(
    backend_parity_module.__file__,
    inference_config_module.__file__,
    levanter_backend_module.__file__,
    levanter_snowball_module.__file__,
    model_preparation_module.__file__,
    snowball_levanter_module.__file__,
    snowball_module.__file__,
    __file__,
)


def capture_exported_levanter(
    cell: ExportedLevanterCell,
    *,
    same_process_repeats: int = 1,
    batch_indices: tuple[int, ...] | None = None,
    goldens: tuple[RepresentativeGolden, ...] | None = None,
    additional_goldens: tuple[tuple[RepresentativeGolden, ...], ...] = (),
    golden_digest: str = REPRESENTATIVE_GOLDEN_SHA256,
    candidate_revision: str = "development",
) -> ObservationReport:
    """Load the canonical export and capture selected prompt batches."""
    if same_process_repeats <= 0:
        raise ValueError("same_process_repeats must be positive")
    if jax.default_backend() != cell.location.name:
        raise RuntimeError(f"Expected {cell.location.name} backend, found {jax.default_backend()}")
    if cell.location.export_uri is None:
        raise ValueError(f"No verified export is configured for {cell.location.name}")

    prompt_fixture, selected_ids_by_case, selected_indices = prepare_capture_selection(
        cell.location,
        goldens=goldens,
        additional_goldens=additional_goldens,
        batch_indices=batch_indices,
    )
    export_uri = cell.location.export_uri
    config_bytes = (StoragePath(export_uri) / "config.json").read_bytes()
    num_heads, num_kv_heads = read_attention_heads(export_uri)
    num_chips = jax.device_count()
    tensor_parallel_size = select_tensor_parallel_size(num_heads, num_chips, num_kv_heads)
    spec = ModelSpec(
        weights=export_uri,
        api_model=SNOWBALL.model_name,
        num_chips=num_chips,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=VLLM_MAX_MODEL_LEN,
        chat_template_content=None,
    )
    config_overrides = {
        "moe_implementation": cell.moe_implementation,
        "attention_implementation": cell.attention_implementation,
    }
    with LevanterBackend(LevanterEngineConfig()).load_model(spec, config_overrides=config_overrides) as loaded:
        model = loaded.model
        if not isinstance(model, SnowballLMHeadModel):
            raise TypeError(f"Expected SnowballLMHeadModel, found {type(model).__name__}")
        parameter_dtypes = {leaf.dtype for leaf in jax.tree.leaves(model) if eqx.is_inexact_array(leaf)}
        if parameter_dtypes != {jnp.dtype(jnp.bfloat16)}:
            raise ValueError(f"Expected only BF16 parameters, got {parameter_dtypes}")
        if loaded.trainer.data_axis_size != BATCH_SIZE:
            raise ValueError(f"Expected data axis {BATCH_SIZE}, got {loaded.trainer.data_axis_size}")
        if loaded.tokenizer.eos_token_id is None:
            raise ValueError("Snowball tokenizer must define eos_token_id")

        @hax.named_jit(axis_resources=loaded.trainer.compute_axis_mapping)
        def score_next_token(model, input_ids, last_positions, selected_token_ids):
            hidden = model.activations(input_ids).rearrange(("batch", "position", "embed")).array
            last_hidden = hidden.at[jnp.arange(hidden.shape[0]), last_positions.array].get(out_sharding=P("data"))
            logits = jnp.einsum("bh,hv->bv", last_hidden, model.transformer.output_proj, out_sharding=P("data"))
            if logits.dtype != jnp.bfloat16:
                raise ValueError(f"Expected BF16 logits, got {logits.dtype}")
            logprobs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            return score_logprob_diagnostics(logprobs, selected_token_ids)

        observations = []
        Batch = Axis("batch", BATCH_SIZE)
        for repeat_index in range(same_process_repeats):
            for batch_index in selected_indices:
                batch = prompt_fixture.batches[batch_index]
                token_ids, last_token_indices = snowball_module.pad_prompt_batch(batch, loaded.tokenizer.eos_token_id)
                selected_ids = rectangular_selected_token_ids(
                    tuple(case.id for case in batch.cases),
                    selected_ids_by_case,
                    vocab_size=model.Vocab.size,
                )
                Pos = Axis("position", batch.max_tokens)
                outputs = score_next_token(
                    model,
                    hax.named(jnp.asarray(token_ids), (Batch, Pos)),
                    hax.named(jnp.asarray(last_token_indices), (Batch,)),
                    jnp.asarray(selected_ids),
                )
                device_outputs = jax.tree.map(lambda output: np.asarray(jax.device_get(output)), outputs)
                observations.extend(
                    build_batch_observations(
                        batch,
                        repeat_index=repeat_index,
                        top_logprobs=device_outputs.top_logprobs,
                        top_token_ids=device_outputs.top_token_ids,
                        selected_token_ids_by_row=selected_ids,
                        selected_logprobs=device_outputs.selected_logprobs,
                        selected_ranks=device_outputs.selected_ranks,
                        capacity_overflow=np.empty((0,), dtype=np.float32),
                        has_nonfinite=device_outputs.has_nonfinite,
                    )
                )

        mesh_shape = tuple((name, int(size)) for name, size in loaded.trainer.device_mesh.shape.items())

    runtime_fingerprint, runtime_fingerprint_digest = numerical_runtime_fingerprint(("jax", "jaxlib", "libtpu"))
    provenance = RunProvenance(
        backend="levanter-exported",
        platform=cell.location.name,
        process_id=uuid.uuid4().hex,
        code_digest=CAPTURE_SOURCE_DIGEST,
        candidate_revision=candidate_revision,
        # Exact export integrity is established separately; this binds the
        # loaded cell to those already verified, content-addressed bytes.
        parameter_digest=SNOWBALL.export_sha256,
        model_config_digest=hashlib.sha256(config_bytes).hexdigest(),
        prompt_fixture_digest=PROMPT_FIXTURE_SHA256,
        mesh_shape=mesh_shape,
        device_kind=",".join(sorted({device.device_kind for device in jax.devices()})),
        golden_digest=golden_digest,
        runtime_fingerprint=runtime_fingerprint,
        runtime_fingerprint_digest=runtime_fingerprint_digest,
    )
    return ObservationReport(provenance=provenance, observations=tuple(observations))
