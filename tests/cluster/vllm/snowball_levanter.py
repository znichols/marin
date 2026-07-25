# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture Snowball native-Levanter next-token observations on GPU or TPU.

This module is both the non-pytest golden/measurement entry point and the shared
implementation used by the standing-cluster gate. It intentionally knows nothing
about Iris submission so persistent dev accelerators can run it directly.
"""

import dataclasses
import uuid
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax.partitioning import set_mesh
from huggingface_hub import snapshot_download
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import compact_grug_mesh
from levanter.models import snowball as levanter_snowball_module
from levanter.tokenizers import load_tokenizer
from marin.inference import config as inference_config_module

from tests.cluster.vllm import backend_parity as backend_parity_module
from tests.cluster.vllm import snowball as snowball_module
from tests.cluster.vllm import snowball_checkpoint as snowball_checkpoint_module
from tests.cluster.vllm.backend_parity import (
    NextTokenObservation,
    ObservationReport,
    RunProvenance,
    SelectedTokenScore,
    TokenScore,
    canonical_json_digest,
    numerical_runtime_fingerprint,
    source_digest,
)
from tests.cluster.vllm.snowball import (
    PROMPT_FIXTURE_SHA256,
    REPRESENTATIVE_GOLDEN_SHA256,
    TOKENIZER_FILE_PATTERNS,
    NativeLevanterCell,
    PromptBatch,
    RepresentativeGolden,
    prepare_capture_selection,
)
from tests.cluster.vllm.snowball_checkpoint import (
    VendoredTransformer,
    apply_pending_qb_betas,
    decode_vendored_config,
    load_checkpoint,
    logical_array_digest,
    prepare_bf16_parameters,
    read_executor_info,
)

OBSERVATION_TOP_K = 50


class ScoredTokenBatch(NamedTuple):
    top_logprobs: jax.Array
    top_token_ids: jax.Array
    selected_logprobs: jax.Array
    selected_ranks: jax.Array
    capacity_overflow: jax.Array
    has_nonfinite: jax.Array


class CanonicalTop25Observations(NamedTuple):
    top_logprobs: np.ndarray
    top_token_ids: np.ndarray
    selected_logprobs: np.ndarray
    selected_ranks: np.ndarray
    greedy_token_ids: np.ndarray


class ScoredLogprobDiagnostics(NamedTuple):
    top_logprobs: jax.Array
    top_token_ids: jax.Array
    selected_logprobs: jax.Array
    selected_ranks: jax.Array
    has_nonfinite: jax.Array


def rectangular_selected_token_ids(
    case_ids: tuple[str, ...],
    selected_ids_by_case: dict[str, tuple[int, ...]],
    *,
    vocab_size: int,
) -> np.ndarray:
    """Pad ragged reference unions with deterministic, valid, non-reference IDs."""
    width = max(len(selected_ids_by_case[case_id]) for case_id in case_ids)
    rows = []
    for case_id in case_ids:
        selected = list(selected_ids_by_case[case_id])
        if len(selected) != len(set(selected)):
            raise ValueError(f"Duplicate selected token IDs for {case_id}")
        if any(token_id < 0 or token_id >= vocab_size for token_id in selected):
            raise ValueError(f"Selected token ID outside vocabulary for {case_id}")
        selected_set = set(selected)
        filler = (token_id for token_id in range(vocab_size) if token_id not in selected_set)
        selected.extend(next(filler) for _ in range(width - len(selected)))
        rows.append(selected)
    return np.asarray(rows, dtype=np.int32)


def _last_token_logprobs(
    model: VendoredTransformer,
    hidden: jax.Array,
    last_token_indices: jax.Array,
) -> jax.Array:
    last_hidden = hidden.at[jnp.arange(hidden.shape[0]), last_token_indices].get(
        out_sharding=P(("replica_dcn", "data", "expert"))
    )
    logits = jnp.einsum(
        "bh,hv->bv",
        last_hidden,
        model.output_proj,
        out_sharding=P(("replica_dcn", "data", "expert")),
    )
    assert logits.dtype == jnp.bfloat16
    return jax.nn.log_softmax(logits.astype(jnp.float32))


def score_logprob_diagnostics(
    logprobs: jax.Array,
    selected_token_ids: jax.Array,
) -> ScoredLogprobDiagnostics:
    """Compute the shared top-token and selected-token diagnostics."""
    top_logprobs, top_token_ids = jax.lax.top_k(logprobs, OBSERVATION_TOP_K)
    selected_logprobs = jnp.take_along_axis(logprobs, selected_token_ids, axis=-1)
    selected_ranks = jnp.sum(
        logprobs[:, None, :] > selected_logprobs[:, :, None],
        axis=-1,
        dtype=jnp.int32,
    )
    has_nonfinite = jnp.any(~jnp.isfinite(logprobs), axis=-1)
    return ScoredLogprobDiagnostics(
        top_logprobs=top_logprobs,
        top_token_ids=top_token_ids,
        selected_logprobs=selected_logprobs,
        selected_ranks=selected_ranks,
        has_nonfinite=has_nonfinite,
    )


@eqx.filter_jit
def score_next_token(
    model: VendoredTransformer,
    token_ids: jax.Array,
    last_token_indices: jax.Array,
    selected_token_ids: jax.Array,
) -> ScoredTokenBatch:
    """Score only each row's final real token while retaining parity diagnostics."""
    hidden, router_metrics = model(token_ids)
    logprobs = _last_token_logprobs(model, hidden, last_token_indices)
    diagnostics = score_logprob_diagnostics(logprobs, selected_token_ids)
    return ScoredTokenBatch(
        top_logprobs=diagnostics.top_logprobs,
        top_token_ids=diagnostics.top_token_ids,
        selected_logprobs=diagnostics.selected_logprobs,
        selected_ranks=diagnostics.selected_ranks,
        capacity_overflow=router_metrics["capacity_overflow_per_layer"],
        has_nonfinite=diagnostics.has_nonfinite,
    )


@eqx.filter_jit
def score_gpu_canonical_top25(
    model: VendoredTransformer,
    pending_qb_betas: jax.Array,
    token_ids: jax.Array,
    last_token_indices: jax.Array,
    policy: jmp.Policy,
) -> tuple[jax.Array, jax.Array]:
    """Preserve the exact canonical GPU scoring graph for its observations."""
    model = apply_pending_qb_betas(model, pending_qb_betas)
    model = policy.cast_to_compute(model)
    hidden, _ = model(token_ids)
    logprobs = _last_token_logprobs(model, hidden, last_token_indices)
    return jax.lax.top_k(logprobs, snowball_module.TOP_K)


def apply_canonical_top25(
    selected_token_ids: np.ndarray,
    diagnostic_logprobs: np.ndarray,
    diagnostic_ranks: np.ndarray,
    canonical_logprobs: np.ndarray,
    canonical_token_ids: np.ndarray,
) -> CanonicalTop25Observations:
    """Use baseline-compatible GPU scores for canonical tokens, retaining top-50 diagnostics."""
    canonical_selected_logprobs = diagnostic_logprobs.copy()
    canonical_selected_ranks = diagnostic_ranks.copy()
    for row in range(selected_token_ids.shape[0]):
        canonical_by_id = {
            int(token_id): (float(logprob), rank)
            for rank, (logprob, token_id) in enumerate(
                zip(canonical_logprobs[row], canonical_token_ids[row], strict=True)
            )
        }
        for column, token_id in enumerate(selected_token_ids[row]):
            canonical = canonical_by_id.get(int(token_id))
            if canonical is not None:
                canonical_selected_logprobs[row, column] = canonical[0]
                canonical_selected_ranks[row, column] = canonical[1]
    return CanonicalTop25Observations(
        top_logprobs=canonical_logprobs,
        top_token_ids=canonical_token_ids,
        selected_logprobs=canonical_selected_logprobs,
        selected_ranks=canonical_selected_ranks,
        greedy_token_ids=canonical_token_ids[:, 0],
    )


def build_batch_observations(
    batch: PromptBatch,
    *,
    repeat_index: int,
    top_logprobs: np.ndarray,
    top_token_ids: np.ndarray,
    selected_token_ids_by_row: np.ndarray,
    selected_logprobs: np.ndarray,
    selected_ranks: np.ndarray,
    capacity_overflow: np.ndarray,
    has_nonfinite: np.ndarray,
    greedy_token_ids: np.ndarray | None = None,
) -> tuple[NextTokenObservation, ...]:
    """Convert device arrays into the versioned backend-neutral schema."""
    rows = len(batch.cases)
    expected_row_shapes = {
        "top_logprobs": top_logprobs.shape[0],
        "top_token_ids": top_token_ids.shape[0],
        "selected_logprobs": selected_logprobs.shape[0],
        "selected_ranks": selected_ranks.shape[0],
        "has_nonfinite": has_nonfinite.shape[0],
    }
    if any(actual != rows for actual in expected_row_shapes.values()):
        raise ValueError(f"Batch rows do not match {rows} cases: {expected_row_shapes}")

    overflow = tuple(float(value) for value in capacity_overflow)
    observations = []
    for row, case in enumerate(batch.cases):
        observations.append(
            NextTokenObservation(
                case_id=case.id,
                bucket_max_tokens=batch.max_tokens,
                repeat_index=repeat_index,
                backend_index=row,
                greedy_token_id=int(top_token_ids[row, 0] if greedy_token_ids is None else greedy_token_ids[row]),
                backend_top_scores=tuple(
                    TokenScore(token_id=int(token_id), logprob=float(logprob))
                    for logprob, token_id in zip(top_logprobs[row], top_token_ids[row], strict=True)
                ),
                selected_token_scores=tuple(
                    SelectedTokenScore(
                        token_id=int(token_id),
                        logprob=float(logprob),
                        rank=int(rank),
                    )
                    for token_id, logprob, rank in zip(
                        selected_token_ids_by_row[row],
                        selected_logprobs[row],
                        selected_ranks[row],
                        strict=True,
                    )
                ),
                capacity_overflow=overflow,
                has_nonfinite=bool(has_nonfinite[row]),
            )
        )
    return tuple(observations)


# By-value remote serialization preserves this value without reopening controller
# worktree paths on the worker. A directly imported CLI computes it on that host.
CAPTURE_SOURCE_DIGEST = source_digest(
    backend_parity_module.__file__,
    inference_config_module.__file__,
    levanter_snowball_module.__file__,
    snowball_checkpoint_module.__file__,
    __file__,
)


def _parameter_dtype(cell: NativeLevanterCell):
    if cell.parameter_dtype is None:
        return None
    if cell.parameter_dtype == "bfloat16":
        return jnp.bfloat16
    raise ValueError(f"Unsupported parameter dtype {cell.parameter_dtype!r}")


def capture_native_levanter(
    cell: NativeLevanterCell,
    *,
    same_process_repeats: int = 1,
    batch_indices: tuple[int, ...] | None = None,
    goldens: tuple[RepresentativeGolden, ...] | None = None,
    additional_goldens: tuple[tuple[RepresentativeGolden, ...], ...] = (),
    golden_digest: str = REPRESENTATIVE_GOLDEN_SHA256,
    candidate_revision: str = "development",
) -> ObservationReport:
    """Load one regional checkpoint and capture selected batches repeatedly."""
    if same_process_repeats <= 0:
        raise ValueError("same_process_repeats must be positive")
    if jax.default_backend() != cell.location.name:
        raise RuntimeError(f"Expected {cell.location.name} backend, found {jax.default_backend()}")

    prompt_fixture, selected_ids_by_case, selected_indices = prepare_capture_selection(
        cell.location,
        goldens=goldens,
        additional_goldens=additional_goldens,
        batch_indices=batch_indices,
    )
    executor_info = read_executor_info(cell.location)
    if executor_info["config"]["data"]["tokenizer"] != prompt_fixture.tokenizer:
        raise ValueError("Checkpoint and prompt fixture use different tokenizers")
    inference_model_config = dataclasses.replace(
        decode_vendored_config(executor_info),
        moe_implementation=cell.moe_implementation,
        attention_implementation=cell.attention_implementation,
    )
    gpu_policy = jmp.get_policy(executor_info["config"]["mp"]) if cell.location.name == "gpu" else None
    tokenizer = load_tokenizer(
        snapshot_download(
            prompt_fixture.tokenizer,
            revision=prompt_fixture.tokenizer_revision,
            allow_patterns=list(TOKENIZER_FILE_PATTERNS),
        )
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Snowball tokenizer must define eos_token_id")

    mesh = compact_grug_mesh()
    if mesh.shape.get("expert", 1) != 1:
        raise ValueError(f"Native parity expects EP1, got mesh {mesh.shape}")
    with set_mesh(mesh):
        params, pending_qb_betas = load_checkpoint(
            inference_model_config,
            mesh,
            location=cell.location,
            parameter_dtype=_parameter_dtype(cell),
        )
        canonical_gpu_outputs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        if gpu_policy is not None:
            for repeat_index in range(same_process_repeats):
                for batch_index in selected_indices:
                    batch = prompt_fixture.batches[batch_index]
                    token_ids, last_token_indices = snowball_module.pad_prompt_batch(batch, tokenizer.eos_token_id)
                    canonical_logprobs, canonical_token_ids = score_gpu_canonical_top25(
                        params,
                        pending_qb_betas,
                        jnp.asarray(token_ids),
                        jnp.asarray(last_token_indices),
                        gpu_policy,
                    )
                    canonical_gpu_outputs[repeat_index, batch_index] = (
                        np.asarray(jax.device_get(canonical_logprobs)),
                        np.asarray(jax.device_get(canonical_token_ids)),
                    )

        model = prepare_bf16_parameters(params, pending_qb_betas)
        del params, pending_qb_betas
        parameter_dtypes = {leaf.dtype for leaf in jax.tree.leaves(model) if eqx.is_inexact_array(leaf)}
        if parameter_dtypes != {jnp.dtype(jnp.bfloat16)}:
            raise ValueError(f"Expected only BF16 parameters, got {parameter_dtypes}")
        parameter_digest = logical_array_digest(model)

        observations = []
        diagnostics = []
        for repeat_index in range(same_process_repeats):
            for batch_index in selected_indices:
                batch = prompt_fixture.batches[batch_index]
                token_ids, last_token_indices = snowball_module.pad_prompt_batch(batch, tokenizer.eos_token_id)
                selected_ids = rectangular_selected_token_ids(
                    tuple(case.id for case in batch.cases),
                    selected_ids_by_case,
                    vocab_size=model.output_proj.shape[-1],
                )
                outputs = score_next_token(
                    model,
                    jnp.asarray(token_ids),
                    jnp.asarray(last_token_indices),
                    jnp.asarray(selected_ids),
                )
                device_outputs = jax.tree.map(lambda output: np.asarray(jax.device_get(output)), outputs)
                selected_logprobs = device_outputs.selected_logprobs
                selected_ranks = device_outputs.selected_ranks
                top_logprobs = device_outputs.top_logprobs
                top_token_ids = device_outputs.top_token_ids
                greedy_token_ids = None
                if gpu_policy is not None:
                    canonical_logprobs, canonical_token_ids = canonical_gpu_outputs[repeat_index, batch_index]
                    diagnostics.extend(
                        build_batch_observations(
                            batch,
                            repeat_index=repeat_index,
                            top_logprobs=device_outputs.top_logprobs,
                            top_token_ids=device_outputs.top_token_ids,
                            selected_token_ids_by_row=selected_ids,
                            selected_logprobs=device_outputs.selected_logprobs,
                            selected_ranks=device_outputs.selected_ranks,
                            capacity_overflow=device_outputs.capacity_overflow,
                            has_nonfinite=device_outputs.has_nonfinite,
                        )
                    )
                    canonical = apply_canonical_top25(
                        selected_ids,
                        selected_logprobs,
                        selected_ranks,
                        canonical_logprobs,
                        canonical_token_ids,
                    )
                    top_logprobs = canonical.top_logprobs
                    top_token_ids = canonical.top_token_ids
                    selected_logprobs = canonical.selected_logprobs
                    selected_ranks = canonical.selected_ranks
                    greedy_token_ids = canonical.greedy_token_ids
                observations.extend(
                    build_batch_observations(
                        batch,
                        repeat_index=repeat_index,
                        top_logprobs=top_logprobs,
                        top_token_ids=top_token_ids,
                        selected_token_ids_by_row=selected_ids,
                        selected_logprobs=selected_logprobs,
                        selected_ranks=selected_ranks,
                        capacity_overflow=device_outputs.capacity_overflow,
                        has_nonfinite=device_outputs.has_nonfinite,
                        greedy_token_ids=greedy_token_ids,
                    )
                )

    device_kind = ",".join(sorted({device.device_kind for device in jax.devices()}))
    runtime_fingerprint, runtime_fingerprint_digest = numerical_runtime_fingerprint(("jax", "jaxlib", "libtpu"))
    provenance = RunProvenance(
        backend="levanter-native",
        platform=cell.location.name,
        process_id=uuid.uuid4().hex,
        code_digest=CAPTURE_SOURCE_DIGEST,
        candidate_revision=candidate_revision,
        parameter_digest=parameter_digest,
        model_config_digest=canonical_json_digest(executor_info["config"]["model"]),
        prompt_fixture_digest=PROMPT_FIXTURE_SHA256,
        mesh_shape=tuple((name, int(size)) for name, size in mesh.shape.items()),
        device_kind=device_kind,
        golden_digest=golden_digest,
        runtime_fingerprint=runtime_fingerprint,
        runtime_fingerprint_digest=runtime_fingerprint_digest,
    )
    return ObservationReport(
        provenance=provenance,
        observations=tuple(observations),
        diagnostics=tuple(diagnostics),
    )
