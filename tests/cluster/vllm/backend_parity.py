# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral capture, persistence, and comparison for next-token parity."""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from marin.inference.config import NUMERICAL_RUNTIME_ENV_NAMES
from pydantic import BaseModel, ConfigDict
from rigging.filesystem import StoragePath

OBSERVATION_SCHEMA_VERSION = 2


def source_digest(*module_paths: str | Path) -> str:
    """Hash numerical source files without retaining controller-local paths."""
    digest = hashlib.sha256()
    for path in sorted(Path(path) for path in module_paths):
        relative_name = path.name.encode()
        digest.update(len(relative_name).to_bytes(8, "little"))
        digest.update(relative_name)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def canonical_json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def numerical_runtime_fingerprint(package_names: Iterable[str]) -> tuple[dict[str, Any], str]:
    """Capture normalized package/compiler inputs, excluding diagnostic host facts."""
    packages = {}
    for name in package_names:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        packages[name] = {
            "version": distribution.version,
            "direct_url": distribution.read_text("direct_url.json") or "",
        }
    environment = {
        key: os.environ[key] for key in sorted(NUMERICAL_RUNTIME_ENV_NAMES & os.environ.keys()) if os.environ[key]
    }
    payload = {
        "packages": packages,
        "environment": environment,
        "python_version": platform.python_version(),
    }
    return payload, canonical_json_digest(payload)


class FrozenRecord(BaseModel):
    """Strict immutable wire record shared by capture, promotion, and comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class TokenScore(FrozenRecord):
    logprob: float
    token_id: int


class SelectedTokenScore(FrozenRecord):
    """One explicitly requested token as observed by a backend."""

    token_id: int
    logprob: float | None
    rank: int | None


class NextTokenObservation(FrozenRecord):
    """Reference-neutral next-token output for one case."""

    case_id: str
    bucket_max_tokens: int
    repeat_index: int
    backend_index: int | None
    greedy_token_id: int
    backend_top_scores: tuple[TokenScore, ...]
    selected_token_scores: tuple[SelectedTokenScore, ...]
    capacity_overflow: tuple[float, ...] = ()
    has_nonfinite: bool = False


class CaptureFailure(FrozenRecord):
    """A transport or normalization failure that did not stop later captures."""

    stage: str
    message: str
    case_id: str | None = None
    repeat_index: int | None = None
    backend_index: int | None = None


class RunProvenance(FrozenRecord):
    """Measured and content-pinned numerical inputs for one process."""

    backend: str
    platform: str
    process_id: str
    code_digest: str
    candidate_revision: str
    parameter_digest: str
    model_config_digest: str
    prompt_fixture_digest: str
    mesh_shape: tuple[tuple[str, int], ...]
    device_kind: str
    golden_digest: str
    fork_source_revisions: tuple[tuple[str, str], ...] = ()
    runtime_fingerprint: dict[str, Any] | None = None
    runtime_fingerprint_digest: str = ""


class ParityContract(FrozenRecord):
    """Frozen numerical and measured-or-pinned provenance for one matrix cell."""

    schema_version: Literal[2]
    name: str
    backend: str
    platform: str
    parameter_digest: str
    model_config_digest: str
    prompt_fixture_digest: str
    canonical_golden_digest: str
    code_digest: str
    candidate_revision: str
    device_kind: str
    mesh_shape: tuple[tuple[str, int], ...]
    max_probability_error_by_bucket: tuple[tuple[int, float], ...]
    fork_source_revisions: tuple[tuple[str, str], ...] = ()
    runtime_fingerprint_digest: str = ""
    exact_top_scores: bool = False


@dataclass(frozen=True)
class ReportExpectation:
    """Coverage and scheduling identity expected from one capture."""

    case_buckets: tuple[tuple[str, int], ...]
    repeat_indices: tuple[int, ...]
    backend_indices_by_case: tuple[tuple[str, int | None], ...]


class ObservationReport(FrozenRecord):
    provenance: RunProvenance
    observations: tuple[NextTokenObservation, ...]
    failures: tuple[CaptureFailure, ...] = ()
    diagnostics: tuple[NextTokenObservation, ...] = ()
    diagnostic_failures: tuple[CaptureFailure, ...] = ()
    schema_version: Literal[2] = OBSERVATION_SCHEMA_VERSION


@dataclass(frozen=True)
class NextTokenParity:
    case_id: str
    backend_index: int | None
    greedy_token_id: int
    golden_top_token_ids: tuple[int, ...]
    golden_probability_gap_to_greedy: float
    max_probability_error: float
    top_probability_l1_error: float


class ParityFailureCode(StrEnum):
    CAPTURE = "capture"
    COVERAGE = "coverage"
    EXACT_TOP_SCORES = "exact_top_scores"
    GOLDEN = "golden"
    HEALTH = "health"
    NUMERICAL_BOUND = "numerical_bound"
    PROVENANCE = "provenance"


@dataclass(frozen=True)
class ParityViolation:
    code: ParityFailureCode
    message: str
    case_id: str | None = None
    repeat_index: int | None = None
    backend_index: int | None = None
    field: str | None = None


@dataclass(frozen=True)
class BucketMaximum:
    bucket: int
    max_probability_error: float
    bound: float
    case_id: str


class ParityValidationError(AssertionError):
    """Structured aggregate raised after evaluating every reachable observation."""

    def __init__(
        self,
        heading: str,
        violations: Iterable[ParityViolation],
        *,
        bucket_maxima: Iterable[BucketMaximum] = (),
    ):
        self.heading = heading
        self.violations = tuple(violations)
        self.bucket_maxima = tuple(bucket_maxima)
        maxima = "\n".join(
            (
                f"bucket={maximum.bucket}: max_probability_error={maximum.max_probability_error:.17g} "
                f"bound={maximum.bound:.17g} case={maximum.case_id}"
            )
            for maximum in self.bucket_maxima
        )
        details = "\n".join(violation.message for violation in self.violations)
        sections = [heading]
        if maxima:
            sections.append(f"Bucket maxima:\n{maxima}")
        if details:
            sections.append(f"Violations:\n{details}")
        super().__init__("\n".join(sections))

    @property
    def codes(self) -> frozenset[ParityFailureCode]:
        return frozenset(violation.code for violation in self.violations)


class PersistedParityError(AssertionError):
    """Validation failures raised only after the complete report is durable."""

    def __init__(self, report_uri: str, failures: Iterable[tuple[str, str]]):
        self.report_uri = report_uri
        self.failures = tuple(failures)
        details = "\n\n".join(f"{label}:\n{message}" for label, message in self.failures)
        super().__init__(f"Report persisted before validation: {report_uri}\n\n{details}")


def _float32_bits(value: float | None) -> int | None:
    if value is None:
        return None
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def observation_from_completion_response(
    payload: dict,
    *,
    case_id: str,
    prompt_token_ids: tuple[int, ...],
    selected_token_ids: tuple[int, ...],
    bucket_max_tokens: int,
    repeat_index: int,
    backend_index: int | None,
) -> NextTokenObservation:
    """Normalize one OpenAI completions response without asserting parity."""
    try:
        (choice,) = payload["choices"]
        if choice["prompt_token_ids"] != list(prompt_token_ids):
            raise ValueError("response prompt token IDs differ from request")
        (greedy_token_id,) = choice["token_ids"]
        (raw_top_logprobs,) = choice["logprobs"]["top_logprobs"]
        scores_by_id = {
            int(token.removeprefix("token_id:")): float(logprob) for token, logprob in raw_top_logprobs.items()
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{case_id}: malformed completions response: {error}") from error

    backend_top_scores = tuple(
        TokenScore(token_id=token_id, logprob=logprob)
        for token_id, logprob in sorted(scores_by_id.items(), key=lambda item: (-item[1], item[0]))
    )
    rank_by_id = {score.token_id: rank for rank, score in enumerate(backend_top_scores)}
    return NextTokenObservation(
        case_id=case_id,
        bucket_max_tokens=bucket_max_tokens,
        repeat_index=repeat_index,
        backend_index=backend_index,
        greedy_token_id=int(greedy_token_id),
        backend_top_scores=backend_top_scores,
        selected_token_scores=tuple(
            SelectedTokenScore(
                token_id=token_id,
                logprob=scores_by_id.get(token_id),
                rank=rank_by_id.get(token_id),
            )
            for token_id in selected_token_ids
        ),
        has_nonfinite=any(not math.isfinite(score.logprob) for score in backend_top_scores),
    )


def request_next_token_observation(
    completions_url: str,
    model_id: str,
    *,
    case_id: str,
    prompt_token_ids: tuple[int, ...],
    selected_token_ids: tuple[int, ...],
    bucket_max_tokens: int,
    repeat_index: int,
    backend_index: int | None,
    returned_logprobs: int,
    headers: dict[str, str] | None = None,
    timeout: tuple[float, float] = (30.0, 300.0),
) -> NextTokenObservation | CaptureFailure:
    """Capture one request, returning a structured failure instead of aborting a wave."""
    import requests  # noqa: PLC0415

    try:
        response = requests.post(
            completions_url,
            headers=headers,
            json={
                "model": model_id,
                "prompt": list(prompt_token_ids),
                "add_special_tokens": False,
                "temperature": 0.0,
                "max_tokens": 1,
                "logprobs": returned_logprobs,
                "return_tokens_as_token_ids": True,
                "return_token_ids": True,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return observation_from_completion_response(
            response.json(),
            case_id=case_id,
            prompt_token_ids=prompt_token_ids,
            selected_token_ids=selected_token_ids,
            bucket_max_tokens=bucket_max_tokens,
            repeat_index=repeat_index,
            backend_index=backend_index,
        )
    except Exception as error:
        return CaptureFailure(
            stage="request",
            message=f"{type(error).__name__}: {error}",
            case_id=case_id,
            repeat_index=repeat_index,
            backend_index=backend_index,
        )


def selected_token_ids_by_case(*references: Mapping[str, tuple[TokenScore, ...]]) -> dict[str, tuple[int, ...]]:
    """Return a deterministic per-case union across every declared reference."""
    case_ids = set().union(*(reference.keys() for reference in references))
    return {
        case_id: tuple(dict.fromkeys(score.token_id for reference in references for score in reference.get(case_id, ())))
        for case_id in sorted(case_ids)
    }


def capture_selection(
    references: tuple[Mapping[str, tuple[TokenScore, ...]], ...],
    *,
    batch_count: int,
    batch_indices: tuple[int, ...] | None,
) -> tuple[dict[str, tuple[int, ...]], tuple[int, ...]]:
    """Build the selected-token union and validate one shared batch slice."""
    selected_ids_by_case = selected_token_ids_by_case(*references)
    selected_indices = tuple(range(batch_count)) if batch_indices is None else batch_indices
    if len(set(selected_indices)) != len(selected_indices) or any(
        index < 0 or index >= batch_count for index in selected_indices
    ):
        raise ValueError(f"Invalid batch indices {selected_indices}")
    return selected_ids_by_case, selected_indices


def _parity_from_observation(
    observation: NextTokenObservation,
    expected_top_logprobs: tuple[TokenScore, ...],
) -> NextTokenParity:
    actual = {}
    for score in observation.selected_token_scores:
        if score.token_id in actual:
            raise AssertionError("duplicate selected token")
        if score.logprob is not None:
            actual[score.token_id] = score.logprob
    expected = {score.token_id: score.logprob for score in expected_top_logprobs}
    missing = expected.keys() - actual.keys()
    if missing:
        raise AssertionError(f"selected tokens missing from backend scores: {sorted(missing)}")
    probability_errors = tuple(
        abs(math.exp(actual[token_id]) - math.exp(logprob)) for token_id, logprob in expected.items()
    )
    maximum_expected = max(expected.values())
    selected_expected = expected.get(observation.greedy_token_id, -math.inf)
    return NextTokenParity(
        case_id=observation.case_id,
        backend_index=observation.backend_index,
        greedy_token_id=observation.greedy_token_id,
        golden_top_token_ids=tuple(expected),
        golden_probability_gap_to_greedy=math.exp(maximum_expected) - math.exp(selected_expected),
        max_probability_error=max(probability_errors),
        top_probability_l1_error=sum(probability_errors),
    )


def _coverage_failures(report: ObservationReport, expectation: ReportExpectation) -> list[ParityViolation]:
    expected_buckets = dict(expectation.case_buckets)
    expected_indices = dict(expectation.backend_indices_by_case)
    expected_keys = {
        (repeat_index, case_id, expected_indices[case_id])
        for repeat_index in expectation.repeat_indices
        for case_id in expected_buckets
    }
    actual_keys = [
        (observation.repeat_index, observation.case_id, observation.backend_index) for observation in report.observations
    ]
    failures = []
    if len(actual_keys) != len(set(actual_keys)):
        failures.append(
            ParityViolation(
                ParityFailureCode.COVERAGE,
                "duplicate (repeat, case, backend_index) observation identity",
            )
        )
    missing = expected_keys - set(actual_keys)
    extra = set(actual_keys) - expected_keys
    if missing:
        failures.append(ParityViolation(ParityFailureCode.COVERAGE, f"missing observations: {sorted(missing)}"))
    if extra:
        failures.append(ParityViolation(ParityFailureCode.COVERAGE, f"unexpected observations: {sorted(extra)}"))
    for observation in report.observations:
        expected_bucket = expected_buckets.get(observation.case_id)
        if expected_bucket is not None and observation.bucket_max_tokens != expected_bucket:
            failures.append(
                ParityViolation(
                    ParityFailureCode.COVERAGE,
                    f"case={observation.case_id}: bucket {observation.bucket_max_tokens} != {expected_bucket}",
                )
            )
    failures.extend(
        ParityViolation(
            ParityFailureCode.CAPTURE,
            f"capture stage={failure.stage} case={failure.case_id} repeat={failure.repeat_index}: {failure.message}",
        )
        for failure in report.failures
    )
    return failures


def _provenance_failures(report: ObservationReport, contract: ParityContract) -> list[ParityViolation]:
    provenance = report.provenance
    expected = {
        "backend": contract.backend,
        "platform": contract.platform,
        "parameter_digest": contract.parameter_digest,
        "model_config_digest": contract.model_config_digest,
        "prompt_fixture_digest": contract.prompt_fixture_digest,
        "golden_digest": contract.canonical_golden_digest,
        "code_digest": contract.code_digest,
        "candidate_revision": contract.candidate_revision,
        "device_kind": contract.device_kind,
        "mesh_shape": contract.mesh_shape,
        "fork_source_revisions": contract.fork_source_revisions,
        "runtime_fingerprint_digest": contract.runtime_fingerprint_digest,
    }
    return [
        ParityViolation(
            ParityFailureCode.PROVENANCE,
            f"Unexpected {field}: expected {wanted!r}, got {getattr(provenance, field)!r}",
            field=field,
        )
        for field, wanted in expected.items()
        if getattr(provenance, field) != wanted
    ]


def assert_report_matches_provenance(
    report: ObservationReport,
    contract: ParityContract,
) -> None:
    """Validate only measured and content-pinned runtime identity."""
    failures = _provenance_failures(report, contract)
    if failures:
        raise ParityValidationError("Report provenance does not match the frozen contract.", failures)


def _observation_health_failures(observation: NextTokenObservation) -> list[ParityViolation]:
    context = f"case={observation.case_id} repeat={observation.repeat_index}"
    reasons = []
    if observation.has_nonfinite:
        reasons.append("nonfinite scores")
    if any(value != 0.0 for value in observation.capacity_overflow):
        reasons.append("router capacity overflow")
    if not observation.backend_top_scores:
        reasons.append("no backend top scores")
    elif observation.greedy_token_id != observation.backend_top_scores[0].token_id:
        reasons.append("greedy token is not the returned top token")
    if len({score.token_id for score in observation.backend_top_scores}) != len(observation.backend_top_scores):
        reasons.append("duplicate backend top token")
    return [
        ParityViolation(
            ParityFailureCode.HEALTH,
            f"{context}: {reason}",
            case_id=observation.case_id,
            repeat_index=observation.repeat_index,
            backend_index=observation.backend_index,
        )
        for reason in reasons
    ]


def _report_integrity_failures(
    report: ObservationReport,
    expectation: ReportExpectation,
) -> list[ParityViolation]:
    failures = _coverage_failures(report, expectation)
    for observation in report.observations:
        failures.extend(_observation_health_failures(observation))
    return failures


def assert_report_matches_golden(
    report: ObservationReport,
    expected_by_id: Mapping[str, tuple[TokenScore, ...]],
    expectation: ReportExpectation,
    *,
    max_probability_error_by_bucket: tuple[tuple[int, float], ...],
    exact_top_scores: bool = False,
) -> tuple[NextTokenParity, ...]:
    """Evaluate coverage, health, and numerical parity against one golden."""
    failures = _report_integrity_failures(report, expectation)
    bucket_bounds = dict(max_probability_error_by_bucket)
    expected_buckets = dict(expectation.case_buckets)
    unknown_buckets = set(expected_buckets.values()) - bucket_bounds.keys()
    if unknown_buckets:
        failures.append(
            ParityViolation(
                ParityFailureCode.NUMERICAL_BOUND,
                f"contract has no bound for buckets {sorted(unknown_buckets)}",
            )
        )

    parities = []
    bucket_maxima: dict[int, tuple[float, str]] = {}
    for observation in report.observations:
        context = f"case={observation.case_id} repeat={observation.repeat_index}"
        expected_scores = expected_by_id.get(observation.case_id)
        if expected_scores is None:
            failures.append(
                ParityViolation(
                    ParityFailureCode.GOLDEN,
                    f"{context}: case absent from reference",
                    case_id=observation.case_id,
                    repeat_index=observation.repeat_index,
                    backend_index=observation.backend_index,
                )
            )
        else:
            try:
                parity = _parity_from_observation(observation, expected_scores)
            except AssertionError as error:
                failures.append(
                    ParityViolation(
                        ParityFailureCode.GOLDEN,
                        f"{context}: {error}",
                        case_id=observation.case_id,
                        repeat_index=observation.repeat_index,
                        backend_index=observation.backend_index,
                    )
                )
            else:
                parities.append(parity)
                bucket = observation.bucket_max_tokens
                if bucket in bucket_bounds:
                    maximum = bucket_maxima.get(bucket)
                    if maximum is None or parity.max_probability_error > maximum[0]:
                        bucket_maxima[bucket] = (parity.max_probability_error, observation.case_id)
                    if parity.greedy_token_id not in parity.golden_top_token_ids:
                        failures.append(
                            ParityViolation(
                                ParityFailureCode.GOLDEN,
                                f"{context}: greedy token is outside reference top tokens",
                                case_id=observation.case_id,
                                repeat_index=observation.repeat_index,
                                backend_index=observation.backend_index,
                            )
                        )
                    if parity.golden_probability_gap_to_greedy > 2 * parity.max_probability_error:
                        failures.append(
                            ParityViolation(
                                ParityFailureCode.GOLDEN,
                                f"{context}: greedy-token change is not explained by measured error",
                                case_id=observation.case_id,
                                repeat_index=observation.repeat_index,
                                backend_index=observation.backend_index,
                            )
                        )
                    if parity.max_probability_error > bucket_bounds[bucket]:
                        failures.append(
                            ParityViolation(
                                ParityFailureCode.NUMERICAL_BOUND,
                                (
                                    f"{context}: maximum probability error {parity.max_probability_error:.17g} "
                                    f"exceeds bound {bucket_bounds[bucket]:.17g}"
                                ),
                                case_id=observation.case_id,
                                repeat_index=observation.repeat_index,
                                backend_index=observation.backend_index,
                            )
                        )

    if exact_top_scores:
        failures.extend(_exact_top_score_failures(report, expected_by_id))
    if failures:
        maxima = tuple(
            BucketMaximum(
                bucket=bucket,
                max_probability_error=maximum,
                bound=bucket_bounds[bucket],
                case_id=case_id,
            )
            for bucket, (maximum, case_id) in sorted(bucket_maxima.items())
        )
        raise ParityValidationError(
            "Golden comparison failed after evaluating all reachable observations.",
            failures,
            bucket_maxima=maxima,
        )
    return tuple(parities)


def _exact_top_score_failures(
    report: ObservationReport,
    expected_by_id: Mapping[str, tuple[TokenScore, ...]],
) -> list[ParityViolation]:
    failures = []
    for observation in report.observations:
        expected = expected_by_id.get(observation.case_id)
        if expected is None:
            continue
        actual = observation.backend_top_scores[: len(expected)]
        expected_signature = tuple((score.token_id, _float32_bits(score.logprob)) for score in expected)
        actual_signature = tuple((score.token_id, _float32_bits(score.logprob)) for score in actual)
        if actual_signature != expected_signature:
            failures.append(
                ParityViolation(
                    ParityFailureCode.EXACT_TOP_SCORES,
                    f"case={observation.case_id} repeat={observation.repeat_index}: exact top scores differ",
                    case_id=observation.case_id,
                    repeat_index=observation.repeat_index,
                    backend_index=observation.backend_index,
                )
            )
    return failures


def assert_report_matches_exact_golden(
    report: ObservationReport,
    expected_by_id: Mapping[str, tuple[TokenScore, ...]],
    expectation: ReportExpectation,
) -> None:
    failures = _report_integrity_failures(report, expectation)
    failures.extend(_exact_top_score_failures(report, expected_by_id))
    if failures:
        raise ParityValidationError("Exact-golden comparison failed.", failures)


def persist_report_before_assert(
    report: ObservationReport,
    report_uri: str,
    checks: Iterable[tuple[str, Callable[[], object]]],
) -> None:
    """Persist first, then run every independent comparison and raise once."""
    StoragePath(report_uri).write_bytes((report.model_dump_json() + "\n").encode())
    failures = []
    for label, check in checks:
        try:
            check()
        except AssertionError as error:
            failures.append((label, str(error)))
    if failures:
        raise PersistedParityError(report_uri, failures)
