# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball model identity and representative inference goldens."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict
from rigging.filesystem import StoragePath

from tests.cluster.vllm.backend_parity import (
    ObservationReport,
    ParityContract,
    ReportExpectation,
    TokenScore,
    assert_report_matches_exact_golden,
    assert_report_matches_golden,
    assert_report_matches_provenance,
    capture_selection,
)
from tests.cluster.vllm.snowball_contracts import (
    EXPORTED_LEVANTER_GPU_CONTRACT_SHA256,
    EXPORTED_LEVANTER_TPU_CONTRACT_SHA256,
    VLLM_TPU_CONTRACT_SHA256,
)

BATCH_SIZE = 8
TOP_K = 25
PROMPT_BUCKET_MAX_TOKENS = (256, 1024, 4096, 16384, 32768)
VLLM_MAX_MODEL_LEN = 32768
VLLM_MAX_NUM_BATCHED_TOKENS = 512
VLLM_HTTP_TIMEOUT = (30.0, 5 * 60.0)
TOKENIZER_FILE_PATTERNS = ("tokenizer*", "special_tokens*", "added_tokens*", "chat_template*")
GPU_VLLM_MAX_PROBABILITY_ERROR_BY_BUCKET = tuple((bucket, 0.075) for bucket in PROMPT_BUCKET_MAX_TOKENS)
GPU_RANK_SENTINEL_CASE_ID = "knowledge-longbench-02"

_RESOURCES = Path(__file__).parent / "resources"
# Frozen artifacts retain the identifiers of the June 67B training lineage.
_REPRESENTATIVE_GOLDEN_PATH = _RESOURCES / "june_tpu_67b_a2b_step_42150_representative_eval_golden.json"
_NATIVE_TPU_GOLDEN_PATH = _RESOURCES / "snowball_levanter_tpu_golden.json"
_EXPORTED_LEVANTER_GPU_CONTRACT_PATH = _RESOURCES / "snowball_levanter_gpu_contract.json"
_EXPORTED_LEVANTER_TPU_CONTRACT_PATH = _RESOURCES / "snowball_levanter_tpu_contract.json"
_VLLM_TPU_CONTRACT_PATH = _RESOURCES / "snowball_vllm_tpu_contract.json"
REPRESENTATIVE_GOLDEN_SHA256 = "d695624cc411d7bf79a6c2e28f34538437985a9e07612d2a07a5095128ff4b2d"
NATIVE_TPU_GOLDEN_SHA256 = "84fbb4229cab10a98a4010f84f39d28894d59b8b5c95ce0d39d031f0fa69fbbf"
PROMPT_FIXTURE_SHA256 = "47863868cbfe336739c8097535f113f4d2dae4954f772eb91511c911433596e8"
PROMPT_FIXTURE_URL = (
    "https://storage.googleapis.com/marin-public/test-data/vllm/e2e/representative-eval-prompts/"
    f"{PROMPT_FIXTURE_SHA256}.json"
)
TPU_PROMPT_FIXTURE_URI = (
    f"gs://marin-us-east5/test-data/vllm/e2e/representative-eval-prompts/{PROMPT_FIXTURE_SHA256}.json"
)


@dataclass(frozen=True)
class ModelLineage:
    """Immutable model identity shared by every platform-local copy."""

    checkpoint_step: int
    export_sha256: str

    @property
    def model_name(self) -> str:
        return f"snowball-step-{self.checkpoint_step}-bf16"


@dataclass(frozen=True)
class ModelLocation:
    """Platform-local inputs and outputs for one model lineage."""

    name: str
    lineage: ModelLineage
    run_root: str
    prompt_fixture_uri: str
    compilation_cache_dir: str
    export_uri: str | None

    @property
    def executor_info_path(self) -> str:
        return str(StoragePath(self.run_root) / ".executor_info")

    @property
    def checkpoint_path(self) -> str:
        return str(StoragePath(self.run_root) / "checkpoints" / f"step-{self.lineage.checkpoint_step}")


@dataclass(frozen=True)
class NativeLevanterCell:
    """One native Levanter platform configuration in the parity matrix."""

    name: str
    backend: str
    platform: str
    reference: str
    location: ModelLocation
    parameter_dtype: str | None
    attention_implementation: str
    moe_implementation: str
    artifact_root: str
    additional_reference_platforms: tuple[str, ...] = ()

    @property
    def data_parallel_size(self) -> int:
        return BATCH_SIZE

    def capture(
        self,
        *,
        goldens: tuple[RepresentativeGolden, ...],
        additional_goldens: tuple[tuple[RepresentativeGolden, ...], ...],
        golden_digest: str,
        candidate_revision: str,
    ) -> ObservationReport:
        from tests.cluster.vllm.snowball_levanter import capture_native_levanter  # noqa: PLC0415

        return capture_native_levanter(
            self,
            goldens=goldens,
            additional_goldens=additional_goldens,
            golden_digest=golden_digest,
            candidate_revision=candidate_revision,
        )

    def comparison_checks(
        self,
        report: ObservationReport,
        goldens: tuple[RepresentativeGolden, ...],
        expectation: ReportExpectation,
    ) -> tuple[ComparisonCheck, ...]:
        expected_by_id = {golden.id: golden.top_logprobs for golden in goldens}
        return (
            (
                f"{self.name} exact {self.reference} golden",
                lambda: assert_report_matches_exact_golden(report, expected_by_id, expectation),
            ),
        )


@dataclass(frozen=True)
class ExportedLevanterCell:
    """One HF-export-loaded Levanter platform configuration."""

    name: str
    backend: str
    platform: str
    reference: str
    location: ModelLocation
    attention_implementation: str
    moe_implementation: str
    contract_bytes: bytes
    contract_sha256: str
    artifact_root: str
    additional_reference_platforms: tuple[str, ...] = ()

    @property
    def data_parallel_size(self) -> int:
        return BATCH_SIZE

    def capture(
        self,
        *,
        goldens: tuple[RepresentativeGolden, ...],
        additional_goldens: tuple[tuple[RepresentativeGolden, ...], ...],
        golden_digest: str,
        candidate_revision: str,
    ) -> ObservationReport:
        from tests.cluster.vllm.snowball_exported_levanter import capture_exported_levanter  # noqa: PLC0415

        return capture_exported_levanter(
            self,
            goldens=goldens,
            additional_goldens=additional_goldens,
            golden_digest=golden_digest,
            candidate_revision=candidate_revision,
        )

    def comparison_checks(
        self,
        report: ObservationReport,
        goldens: tuple[RepresentativeGolden, ...],
        expectation: ReportExpectation,
    ) -> tuple[ComparisonCheck, ...]:
        return frozen_contract_checks(self, report, goldens, expectation)


@dataclass(frozen=True)
class VllmCell:
    """One exported-checkpoint vLLM serving configuration."""

    name: str
    backend: str
    platform: str
    reference: str
    location: ModelLocation
    tensor_parallel_size: int
    data_parallel_size: int
    attention_backend: str | None
    returned_logprobs: int
    contract_bytes: bytes | None
    contract_sha256: str
    artifact_root: str
    additional_reference_platforms: tuple[str, ...] = ()

    def capture(
        self,
        *,
        goldens: tuple[RepresentativeGolden, ...],
        additional_goldens: tuple[tuple[RepresentativeGolden, ...], ...],
        golden_digest: str,
        candidate_revision: str,
    ) -> ObservationReport:
        from tests.cluster.vllm.snowball_vllm import capture_vllm  # noqa: PLC0415

        return capture_vllm(
            self,
            goldens=goldens,
            additional_goldens=additional_goldens,
            golden_digest=golden_digest,
            candidate_revision=candidate_revision,
        )

    def comparison_checks(
        self,
        report: ObservationReport,
        goldens: tuple[RepresentativeGolden, ...],
        expectation: ReportExpectation,
    ) -> tuple[ComparisonCheck, ...]:
        if self.contract_bytes is not None:
            return frozen_contract_checks(self, report, goldens, expectation)
        # Replayed data-parallel-rank observations remain persisted diagnostics;
        # the shared 64-case matrix is the parity contract for this cell.
        expected_by_id = {golden.id: golden.top_logprobs for golden in goldens}
        return (
            (
                f"{self.name} representative prompts against {self.reference} golden",
                lambda: assert_report_matches_golden(
                    report,
                    expected_by_id,
                    expectation,
                    max_probability_error_by_bucket=GPU_VLLM_MAX_PROBABILITY_ERROR_BY_BUCKET,
                ),
            ),
        )


ComparisonCheck = tuple[str, Callable[[], object]]


class ParityCell(Protocol):
    """Configuration and behavior required by the shared parity runner."""

    name: str
    backend: str
    platform: str
    reference: str
    location: ModelLocation
    artifact_root: str
    data_parallel_size: int
    additional_reference_platforms: tuple[str, ...]

    def capture(
        self,
        *,
        goldens: tuple[RepresentativeGolden, ...],
        additional_goldens: tuple[tuple[RepresentativeGolden, ...], ...],
        golden_digest: str,
        candidate_revision: str,
    ) -> ObservationReport: ...

    def comparison_checks(
        self,
        report: ObservationReport,
        goldens: tuple[RepresentativeGolden, ...],
        expectation: ReportExpectation,
    ) -> tuple[ComparisonCheck, ...]: ...


SNOWBALL = ModelLineage(
    checkpoint_step=42150,
    export_sha256="d819cbc63780bd866a942e47f9283cbd7932bbb237b52df527edd750c65be8f0",
)

SNOWBALL_GPU = ModelLocation(
    name="gpu",
    lineage=SNOWBALL,
    run_root=(
        "s3://marin-us-east-02a/marin/grug/"
        "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k-79ebf3"
    ),
    prompt_fixture_uri=PROMPT_FIXTURE_URL,
    compilation_cache_dir=(
        "s3://marin-us-east-02a/tmp/ttl=30d/compilation-cache/june-tpu-67b-a2b-step-42150-sonic-fa4-representative-v2"
    ),
    export_uri="s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/d819cbc63780bd86/",
)

SNOWBALL_TPU = ModelLocation(
    name="tpu",
    lineage=SNOWBALL,
    run_root=(
        "gs://marin-us-east5/grug/moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k-79ebf3"
    ),
    prompt_fixture_uri=TPU_PROMPT_FIXTURE_URI,
    compilation_cache_dir=("gs://marin-us-east5/tmp/ttl=30d/compilation-cache/snowball-step-42150-levanter-native-v1"),
    export_uri=(
        "gs://marin-us-east5/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/"
        "d819cbc63780bd866a942e47f9283cbd7932bbb237b52df527edd750c65be8f0"
    ),
)

SNOWBALL_NATIVE_GPU = NativeLevanterCell(
    name="levanter-native-gpu",
    backend="levanter-native",
    platform="gpu",
    reference="gpu",
    location=SNOWBALL_GPU,
    parameter_dtype=None,
    attention_implementation="gpu_fa4_cute",
    moe_implementation="sonic",
    artifact_root="s3://marin-us-east-02a/tmp/ttl=30d/snowball-parity/reports",
)

SNOWBALL_NATIVE_TPU = NativeLevanterCell(
    name="levanter-native-tpu",
    backend="levanter-native",
    platform="tpu",
    reference="tpu",
    location=SNOWBALL_TPU,
    parameter_dtype="bfloat16",
    attention_implementation="tpu_splash",
    moe_implementation="ring",
    artifact_root="gs://marin-us-east5/tmp/ttl=30d/snowball-parity/reports",
    additional_reference_platforms=("gpu",),
)

SNOWBALL_EXPORTED_GPU = ExportedLevanterCell(
    name="levanter-exported-gpu",
    backend="levanter-exported",
    platform="gpu",
    reference="gpu",
    location=SNOWBALL_GPU,
    attention_implementation="gpu_fa4_cute",
    moe_implementation="sonic",
    contract_bytes=_EXPORTED_LEVANTER_GPU_CONTRACT_PATH.read_bytes(),
    contract_sha256=EXPORTED_LEVANTER_GPU_CONTRACT_SHA256,
    artifact_root="s3://marin-us-east-02a/tmp/ttl=30d/snowball-parity/reports",
)

SNOWBALL_EXPORTED_TPU = ExportedLevanterCell(
    name="levanter-exported-tpu",
    backend="levanter-exported",
    platform="tpu",
    reference="tpu",
    location=SNOWBALL_TPU,
    attention_implementation="tpu_splash",
    moe_implementation="ring",
    contract_bytes=_EXPORTED_LEVANTER_TPU_CONTRACT_PATH.read_bytes(),
    contract_sha256=EXPORTED_LEVANTER_TPU_CONTRACT_SHA256,
    artifact_root="gs://marin-us-east5/tmp/ttl=30d/snowball-parity/reports",
)

SNOWBALL_VLLM_GPU = VllmCell(
    name="vllm-gpu",
    backend="vllm",
    platform="gpu",
    reference="gpu",
    location=SNOWBALL_GPU,
    tensor_parallel_size=1,
    data_parallel_size=8,
    attention_backend="FLASH_ATTN",
    returned_logprobs=50,
    contract_bytes=None,
    contract_sha256="",
    artifact_root="s3://marin-us-east-02a/tmp/ttl=30d/snowball-parity/reports",
)

SNOWBALL_VLLM_TPU = VllmCell(
    name="vllm-tpu",
    backend="vllm",
    platform="tpu",
    reference="tpu",
    location=SNOWBALL_TPU,
    tensor_parallel_size=8,
    data_parallel_size=1,
    attention_backend=None,
    returned_logprobs=4096,
    contract_bytes=_VLLM_TPU_CONTRACT_PATH.read_bytes(),
    contract_sha256=VLLM_TPU_CONTRACT_SHA256,
    artifact_root="gs://marin-us-east5/tmp/ttl=30d/snowball-parity/reports",
)


PARITY_CELLS = (
    SNOWBALL_NATIVE_GPU,
    SNOWBALL_NATIVE_TPU,
    SNOWBALL_EXPORTED_GPU,
    SNOWBALL_EXPORTED_TPU,
    SNOWBALL_VLLM_GPU,
    SNOWBALL_VLLM_TPU,
)


class SnowballRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class RepresentativeGolden(SnowballRecord):
    id: str
    top_logprobs: tuple[TokenScore, ...]


class GoldenDocument(SnowballRecord):
    cases: tuple[RepresentativeGolden, ...]


@dataclass(frozen=True)
class RepresentativeCase:
    id: str
    prompt_token_ids: tuple[int, ...]
    top_logprobs: tuple[TokenScore, ...]


@dataclass(frozen=True)
class PromptBatch:
    max_tokens: int
    cases: tuple[RepresentativeCase, ...]


@dataclass(frozen=True)
class RepresentativePromptFixture:
    tokenizer: str
    tokenizer_revision: str
    cases: tuple[RepresentativeCase, ...]
    batches: tuple[PromptBatch, ...]


def read_representative_goldens() -> tuple[RepresentativeGolden, ...]:
    return _read_goldens(
        _REPRESENTATIVE_GOLDEN_PATH,
        expected_sha256=REPRESENTATIVE_GOLDEN_SHA256,
        label="Representative golden",
    )


def read_native_tpu_goldens() -> tuple[RepresentativeGolden, ...]:
    """Read the exact TPU snapshot created only after cross-slice stability."""
    return _read_goldens(
        _NATIVE_TPU_GOLDEN_PATH,
        expected_sha256=NATIVE_TPU_GOLDEN_SHA256,
        label="Native TPU golden",
    )


def read_goldens_for_platform(platform: str) -> tuple[RepresentativeGolden, ...]:
    if platform == "gpu":
        return read_representative_goldens()
    if platform == "tpu":
        return read_native_tpu_goldens()
    raise ValueError(f"Unsupported platform {platform!r}")


def golden_digest_for_platform(platform: str) -> str:
    if platform == "gpu":
        return REPRESENTATIVE_GOLDEN_SHA256
    if platform == "tpu":
        return NATIVE_TPU_GOLDEN_SHA256
    raise ValueError(f"Unsupported platform {platform!r}")


def _read_goldens(path: Path, *, expected_sha256: str, label: str) -> tuple[RepresentativeGolden, ...]:
    golden_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(golden_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    goldens = GoldenDocument.model_validate_json(golden_bytes).cases
    if len(goldens) != 64:
        raise ValueError(f"{label} must contain 64 cases, got {len(goldens)}")
    if len({golden.id for golden in goldens}) != len(goldens):
        raise ValueError(f"{label} contains duplicate case IDs")
    for golden in goldens:
        if len(golden.top_logprobs) != TOP_K:
            raise ValueError(f"{label} case {golden.id} has {len(golden.top_logprobs)} scores, expected {TOP_K}")
        if len({score.token_id for score in golden.top_logprobs}) != TOP_K:
            raise ValueError(f"{label} case {golden.id} contains duplicate token IDs")
        if any(not np.isfinite(score.logprob) for score in golden.top_logprobs):
            raise ValueError(f"{label} case {golden.id} contains a nonfinite logprob")
    return goldens


def canonical_golden_bytes(goldens: tuple[RepresentativeGolden, ...]) -> bytes:
    """Serialize both platform goldens with the same compact deterministic layout."""
    lines = ["{", '  "cases": [']
    for case_index, golden in enumerate(goldens):
        lines.extend(
            [
                "    {",
                f'      "id": {json.dumps(golden.id)},',
                '      "top_logprobs": [',
            ]
        )
        for score_index, score in enumerate(golden.top_logprobs):
            comma = "," if score_index + 1 < len(golden.top_logprobs) else ""
            lines.append(
                "        "
                + json.dumps(
                    {"logprob": score.logprob, "token_id": score.token_id},
                    sort_keys=True,
                    separators=(", ", ": "),
                )
                + comma
            )
        lines.extend(["      ]", f"    }}{',' if case_index + 1 < len(goldens) else ''}"])
    lines.extend(["  ]", "}"])
    return ("\n".join(lines) + "\n").encode()


def goldens_from_native_report(
    report: ObservationReport,
    *,
    expected_platform: str,
    case_order: tuple[str, ...],
) -> tuple[RepresentativeGolden, ...]:
    """Validate and promote one complete healthy native report to top-25 values."""
    if report.provenance.backend != "levanter-native" or report.provenance.platform != expected_platform:
        raise ValueError("Only a native-Levanter report from the requested platform can produce a golden")
    if report.failures:
        raise ValueError(f"Cannot promote a partial report with {len(report.failures)} capture failures")
    by_case = {}
    for observation in report.observations:
        if observation.repeat_index != 0 or observation.case_id in by_case:
            raise ValueError("Golden promotion requires exactly one observation per case in repeat 0")
        if observation.has_nonfinite or any(value != 0.0 for value in observation.capacity_overflow):
            raise ValueError(f"Unhealthy observation for {observation.case_id}")
        if len(observation.backend_top_scores) < TOP_K:
            raise ValueError(f"Observation {observation.case_id} has fewer than {TOP_K} top scores")
        by_case[observation.case_id] = observation
    if set(by_case) != set(case_order):
        raise ValueError("Native report case coverage differs from the canonical prompt fixture")
    return tuple(
        RepresentativeGolden(id=case_id, top_logprobs=by_case[case_id].backend_top_scores[:TOP_K])
        for case_id in case_order
    )


def _read_parity_contract(contract_bytes: bytes, *, expected_sha256: str, label: str) -> ParityContract:
    actual_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    return ParityContract.model_validate_json(contract_bytes)


def read_parity_contract(cell: ExportedLevanterCell | VllmCell) -> ParityContract:
    if cell.contract_bytes is None:
        raise ValueError(f"{cell.name} has no frozen contract")
    return _read_parity_contract(
        cell.contract_bytes,
        expected_sha256=cell.contract_sha256,
        label=f"{cell.name} contract",
    )


def frozen_contract_checks(
    cell: ExportedLevanterCell | VllmCell,
    report: ObservationReport,
    goldens: tuple[RepresentativeGolden, ...],
    expectation: ReportExpectation,
) -> tuple[ComparisonCheck, ...]:
    """Keep runtime identity and numerical comparison as independent checks."""
    contract = read_parity_contract(cell)
    expected_by_id = {golden.id: golden.top_logprobs for golden in goldens}
    return (
        (
            f"{cell.name} frozen provenance",
            lambda: assert_report_matches_provenance(report, contract),
        ),
        (
            f"{cell.name} against {cell.reference} golden",
            lambda: assert_report_matches_golden(
                report,
                expected_by_id,
                expectation,
                max_probability_error_by_bucket=contract.max_probability_error_by_bucket,
                exact_top_scores=contract.exact_top_scores,
            ),
        ),
    )


def report_expectation(
    fixture: RepresentativePromptFixture,
    *,
    data_parallel_size: int,
    repeat_count: int = 1,
) -> ReportExpectation:
    case_buckets = []
    backend_indices = []
    for batch in fixture.batches:
        for row, case in enumerate(batch.cases):
            case_buckets.append((case.id, batch.max_tokens))
            backend_indices.append((case.id, row if data_parallel_size > 1 else 0))
    return ReportExpectation(
        case_buckets=tuple(case_buckets),
        repeat_indices=tuple(range(repeat_count)),
        backend_indices_by_case=tuple(backend_indices),
    )


def read_prompt_fixture(
    expected_cases: tuple[RepresentativeGolden, ...],
    *,
    fixture_uri: str = PROMPT_FIXTURE_URL,
    expected_sha256: str = PROMPT_FIXTURE_SHA256,
) -> RepresentativePromptFixture:
    fixture_bytes = StoragePath(fixture_uri).read_bytes()
    actual_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Prompt fixture SHA-256 mismatch for {fixture_uri}: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = json.loads(fixture_bytes)
    expected_by_id = {case.id: case for case in expected_cases}
    assert {case["id"] for case in payload["cases"]} == expected_by_id.keys()
    cases = tuple(
        RepresentativeCase(
            id=raw_case["id"],
            prompt_token_ids=tuple(raw_case["prompt_token_ids"]),
            top_logprobs=expected_by_id[raw_case["id"]].top_logprobs,
        )
        for raw_case in payload["cases"]
    )
    return RepresentativePromptFixture(
        tokenizer=payload["tokenizer"],
        tokenizer_revision=payload["tokenizer_revision"],
        cases=cases,
        batches=_prompt_batches(cases),
    )


def prepare_capture_selection(
    location: ModelLocation,
    *,
    goldens: tuple[RepresentativeGolden, ...] | None,
    additional_goldens: tuple[tuple[RepresentativeGolden, ...], ...],
    batch_indices: tuple[int, ...] | None,
) -> tuple[RepresentativePromptFixture, dict[str, tuple[int, ...]], tuple[int, ...]]:
    """Load one cell's prompt fixture and select the shared reference-token union."""
    primary_goldens = read_goldens_for_platform(location.name) if goldens is None else goldens
    prompt_fixture = read_prompt_fixture(primary_goldens, fixture_uri=location.prompt_fixture_uri)
    references = (
        {golden.id: golden.top_logprobs for golden in primary_goldens},
        *({golden.id: golden.top_logprobs for golden in reference} for reference in additional_goldens),
    )
    selected_ids_by_case, selected_indices = capture_selection(
        references,
        batch_count=len(prompt_fixture.batches),
        batch_indices=batch_indices,
    )
    return prompt_fixture, selected_ids_by_case, selected_indices


def _prompt_batches(cases: tuple[RepresentativeCase, ...]) -> tuple[PromptBatch, ...]:
    batches = []
    remaining_cases = cases
    for max_tokens in PROMPT_BUCKET_MAX_TOKENS:
        bucket = tuple(
            sorted(
                (case for case in remaining_cases if len(case.prompt_token_ids) <= max_tokens), key=lambda case: case.id
            )
        )
        remaining_cases = tuple(case for case in remaining_cases if len(case.prompt_token_ids) > max_tokens)
        if len(bucket) % BATCH_SIZE:
            raise ValueError(
                f"Prompt bucket <= {max_tokens} has {len(bucket)} cases; expected full batches of {BATCH_SIZE}"
            )
        batches.extend(
            PromptBatch(max_tokens=max_tokens, cases=bucket[start : start + BATCH_SIZE])
            for start in range(0, len(bucket), BATCH_SIZE)
        )

    if remaining_cases:
        raise ValueError(f"Prompts exceed {PROMPT_BUCKET_MAX_TOKENS[-1]} tokens")
    return tuple(batches)


def pad_prompt_batch(batch: PromptBatch, eos_token_id: int) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.full((BATCH_SIZE, batch.max_tokens), eos_token_id, dtype=np.int32)
    last_token_indices = np.empty(BATCH_SIZE, dtype=np.int32)
    for row, case in enumerate(batch.cases):
        token_ids[row, : len(case.prompt_token_ids)] = case.prompt_token_ids
        last_token_indices[row] = len(case.prompt_token_ids) - 1
    return token_ids, last_token_indices
