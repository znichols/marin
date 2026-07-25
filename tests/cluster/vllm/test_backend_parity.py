# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import requests
from pydantic import ValidationError

from tests.cluster.vllm.backend_parity import (
    CaptureFailure,
    NextTokenObservation,
    ObservationReport,
    ParityContract,
    ParityFailureCode,
    ParityValidationError,
    PersistedParityError,
    ReportExpectation,
    RunProvenance,
    SelectedTokenScore,
    TokenScore,
    assert_report_matches_exact_golden,
    assert_report_matches_golden,
    assert_report_matches_provenance,
    numerical_runtime_fingerprint,
    persist_report_before_assert,
    request_next_token_observation,
)


def provenance(**changes) -> RunProvenance:
    values = {
        "backend": "backend",
        "platform": "tpu",
        "process_id": "process",
        "code_digest": "source",
        "candidate_revision": "a" * 40,
        "parameter_digest": "parameters",
        "model_config_digest": "config",
        "prompt_fixture_digest": "prompt",
        "mesh_shape": (("data", 1),),
        "device_kind": "TPU v6 lite",
        "golden_digest": "golden",
        "runtime_fingerprint": {"jax": "0.10.1"},
        "runtime_fingerprint_digest": "runtime",
    }
    return RunProvenance(**(values | changes))


def observation(
    case_id: str = "case",
    *,
    logprob: float = -0.25,
    bucket: int = 256,
    repeat: int = 0,
) -> NextTokenObservation:
    return NextTokenObservation(
        case_id=case_id,
        bucket_max_tokens=bucket,
        repeat_index=repeat,
        backend_index=0,
        greedy_token_id=2,
        backend_top_scores=(TokenScore(token_id=2, logprob=logprob),),
        selected_token_scores=(SelectedTokenScore(token_id=2, logprob=logprob, rank=0),),
    )


def contract(**changes) -> ParityContract:
    values = {
        "schema_version": 2,
        "name": "cell",
        "backend": "backend",
        "platform": "tpu",
        "parameter_digest": "parameters",
        "model_config_digest": "config",
        "prompt_fixture_digest": "prompt",
        "canonical_golden_digest": "golden",
        "code_digest": "source",
        "candidate_revision": "a" * 40,
        "device_kind": "TPU v6 lite",
        "mesh_shape": (("data", 1),),
        "max_probability_error_by_bucket": ((256, 0.1), (4096, 0.1)),
        "runtime_fingerprint_digest": "runtime",
    }
    return ParityContract(**(values | changes))


def expectation(
    *,
    cases: tuple[tuple[str, int], ...] = (("case", 256),),
    repeats: tuple[int, ...] = (0,),
) -> ReportExpectation:
    return ReportExpectation(
        case_buckets=cases,
        repeat_indices=repeats,
        backend_indices_by_case=tuple((case_id, 0) for case_id, _ in cases),
    )


def assert_golden(report: ObservationReport, expected: dict[str, tuple[TokenScore, ...]], wanted: ReportExpectation):
    return assert_report_matches_golden(
        report,
        expected,
        wanted,
        max_probability_error_by_bucket=((256, 0.1), (4096, 0.1)),
    )


def test_request_failure_does_not_abort_later_capture(monkeypatch) -> None:
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    payloads = iter(
        [
            {"choices": []},
            {
                "choices": [
                    {
                        "prompt_token_ids": [1],
                        "token_ids": [2],
                        "logprobs": {"top_logprobs": [{"token_id:2": -0.1}]},
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Response(next(payloads)))
    kwargs = {
        "completions_url": "http://example/v1/completions",
        "model_id": "model",
        "case_id": "case",
        "prompt_token_ids": (1,),
        "selected_token_ids": (2,),
        "bucket_max_tokens": 256,
        "repeat_index": 0,
        "backend_index": 0,
        "returned_logprobs": 50,
    }
    assert isinstance(request_next_token_observation(**kwargs), CaptureFailure)
    assert isinstance(request_next_token_observation(**kwargs), NextTokenObservation)


@pytest.mark.parametrize(
    ("report", "wanted"),
    [
        (
            ObservationReport(provenance=provenance(), observations=(observation(),)),
            expectation(repeats=(0, 1)),
        ),
        (
            ObservationReport(provenance=provenance(), observations=(observation(bucket=4096),)),
            expectation(),
        ),
        (
            ObservationReport(provenance=provenance(), observations=(observation(), observation())),
            expectation(),
        ),
    ],
)
def test_coverage_failures_are_structured(report: ObservationReport, wanted: ReportExpectation) -> None:
    with pytest.raises(ParityValidationError) as error:
        assert_golden(report, {"case": (TokenScore(logprob=-0.25, token_id=2),)}, wanted)
    assert ParityFailureCode.COVERAGE in error.value.codes


def test_bucket_failures_aggregate_and_unknown_bucket_fails_closed() -> None:
    report = ObservationReport(
        provenance=provenance(),
        observations=(
            observation("short", logprob=-1.0, bucket=256),
            observation("long", logprob=-1.0, bucket=4096),
        ),
    )
    expected = {
        "short": (TokenScore(logprob=0.0, token_id=2),),
        "long": (TokenScore(logprob=0.0, token_id=2),),
    }
    wanted = expectation(cases=(("short", 256), ("long", 4096)))
    with pytest.raises(ParityValidationError) as error:
        assert_golden(report, expected, wanted)
    bounded_cases = {
        violation.case_id for violation in error.value.violations if violation.code == ParityFailureCode.NUMERICAL_BOUND
    }
    assert bounded_cases == {"short", "long"}

    with pytest.raises(ParityValidationError) as error:
        assert_report_matches_golden(
            report,
            expected,
            wanted,
            max_probability_error_by_bucket=((256, 0.1),),
        )
    assert ParityFailureCode.NUMERICAL_BOUND in error.value.codes
    assert {maximum.bucket for maximum in error.value.bucket_maxima} == {256}
    assert any(
        violation.code == ParityFailureCode.NUMERICAL_BOUND and violation.case_id is None
        for violation in error.value.violations
    )


@pytest.mark.parametrize("field", ["code_digest", "candidate_revision", "device_kind"])
def test_provenance_mismatch_reports_the_field(field: str) -> None:
    report = ObservationReport(
        provenance=provenance(**{field: "wrong"}),
        observations=(observation(),),
    )
    with pytest.raises(ParityValidationError) as error:
        assert_report_matches_provenance(report, contract())
    assert {violation.field for violation in error.value.violations} == {field}


def test_exact_golden_uses_float32_bits() -> None:
    report = ObservationReport(provenance=provenance(), observations=(observation(logprob=0.0),))
    assert_report_matches_exact_golden(
        report,
        {"case": (TokenScore(logprob=0.0, token_id=2),)},
        expectation(),
    )
    with pytest.raises(ParityValidationError) as error:
        assert_report_matches_exact_golden(
            report,
            {"case": (TokenScore(logprob=-0.0, token_id=2),)},
            expectation(),
        )
    assert error.value.codes == {ParityFailureCode.EXACT_TOP_SCORES}


def test_report_is_persisted_before_all_checks(tmp_path) -> None:
    report = ObservationReport(
        provenance=provenance(),
        observations=(observation(),),
        failures=(CaptureFailure(stage="request", message="transport", case_id="other"),),
    )
    path = tmp_path / "partial.json"
    calls = []

    def fail(label):
        calls.append(label)
        raise AssertionError(label)

    with pytest.raises(PersistedParityError) as error:
        persist_report_before_assert(
            report,
            str(path),
            (("first", lambda: fail("one")), ("second", lambda: fail("two"))),
        )
    assert ObservationReport.model_validate_json(path.read_bytes()) == report
    assert calls == ["one", "two"]
    assert error.value.report_uri == str(path)
    assert tuple(label for label, _ in error.value.failures) == ("first", "second")


def test_runtime_fingerprint_ignores_empty_noop_controls(monkeypatch) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "")
    payload, empty_digest = numerical_runtime_fingerprint(())
    assert "JAX_PLATFORMS" not in payload["environment"]

    monkeypatch.setenv("JAX_PLATFORMS", "gpu")
    payload, configured_digest = numerical_runtime_fingerprint(())
    assert payload["environment"]["JAX_PLATFORMS"] == "gpu"
    assert configured_digest != empty_digest


def test_persisted_records_reject_nonfinite_values() -> None:
    with pytest.raises(ValidationError):
        TokenScore(logprob=float("nan"), token_id=7)
