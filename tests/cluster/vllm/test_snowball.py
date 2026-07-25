# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.cluster.vllm.backend_parity import (
    NextTokenObservation,
    ObservationReport,
    RunProvenance,
)
from tests.cluster.vllm.snowball import (
    RepresentativeGolden,
    canonical_golden_bytes,
    goldens_from_native_report,
    read_native_tpu_goldens,
    read_prompt_fixture,
    read_representative_goldens,
)
from tests.cluster.vllm.snowball_levanter import apply_canonical_top25
from tests.cluster.vllm.snowball_promote_golden import GoldenPromotionProvenance, promote_native_report


def test_canonical_gpu_and_tpu_goldens_use_the_same_byte_stable_schema() -> None:
    gpu = read_representative_goldens()
    tpu = read_native_tpu_goldens()
    assert len(gpu) == len(tpu) == 64
    assert {case.id for case in gpu} == {case.id for case in tpu}
    assert (
        canonical_golden_bytes(gpu)
        == (
            Path(__file__).parent / "resources" / "june_tpu_67b_a2b_step_42150_representative_eval_golden.json"
        ).read_bytes()
    )
    assert (
        canonical_golden_bytes(tpu)
        == (Path(__file__).parent / "resources" / "snowball_levanter_tpu_golden.json").read_bytes()
    )


def test_canonical_gpu_scores_remain_primary_when_diagnostics_differ() -> None:
    canonical = apply_canonical_top25(
        selected_token_ids=np.asarray([[7, 9, 11]], dtype=np.int32),
        diagnostic_logprobs=np.asarray([[-0.7, -0.9, -1.1]], dtype=np.float32),
        diagnostic_ranks=np.asarray([[3, 4, 5]], dtype=np.int32),
        canonical_logprobs=np.asarray([[-0.1, -0.2]], dtype=np.float32),
        canonical_token_ids=np.asarray([[9, 7]], dtype=np.int32),
    )

    np.testing.assert_array_equal(canonical.top_logprobs, np.asarray([[-0.1, -0.2]], dtype=np.float32))
    np.testing.assert_array_equal(canonical.top_token_ids, np.asarray([[9, 7]], dtype=np.int32))
    np.testing.assert_array_equal(canonical.selected_logprobs, np.asarray([[-0.2, -0.1, -1.1]], dtype=np.float32))
    np.testing.assert_array_equal(canonical.greedy_token_ids, np.asarray([9], dtype=np.int32))


def test_native_golden_promotion_requires_complete_reviewed_capture(tmp_path) -> None:
    gpu = read_representative_goldens()
    case_order = tuple(golden.id for golden in gpu)
    observations = tuple(
        NextTokenObservation(
            case_id=golden.id,
            bucket_max_tokens=256,
            repeat_index=0,
            backend_index=0,
            greedy_token_id=golden.top_logprobs[0].token_id,
            backend_top_scores=golden.top_logprobs,
            selected_token_scores=(),
        )
        for golden in gpu
    )
    provenance = RunProvenance(
        backend="levanter-native",
        platform="gpu",
        process_id="process",
        code_digest="source",
        candidate_revision="a" * 40,
        parameter_digest="parameters",
        model_config_digest="config",
        prompt_fixture_digest="prompt",
        mesh_shape=(("data", 8),),
        device_kind="H100",
        golden_digest="golden",
    )
    report = ObservationReport(provenance=provenance, observations=observations)
    assert goldens_from_native_report(report, expected_platform="gpu", case_order=case_order) == gpu
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "golden.json"
    manifest_path = tmp_path / "manifest.json"
    report_path.write_text(report.model_dump_json() + "\n")
    promote_native_report(
        report_uri=str(report_path),
        output_path=output_path,
        manifest_path=manifest_path,
        platform="gpu",
        expected_provenance=GoldenPromotionProvenance(
            candidate_revision="a" * 40,
            code_digest="source",
            runtime_fingerprint_digest="",
            parameter_digest="parameters",
            model_config_digest="config",
            prompt_fixture_digest="prompt",
            device_kind="H100",
            mesh_shape=(("data", 8),),
        ),
    )
    assert output_path.read_bytes() == canonical_golden_bytes(gpu)
    assert json.loads(manifest_path.read_text())["requires_human_review"] is True

    with pytest.raises(ValueError):
        goldens_from_native_report(
            ObservationReport(provenance=provenance, observations=observations[:-1]),
            expected_platform="gpu",
            case_order=case_order,
        )


def test_prompt_fixture_digest_mismatch_fails_closed(tmp_path) -> None:
    goldens = tuple(RepresentativeGolden(id=f"case-{index}", top_logprobs=()) for index in range(8))
    payload = {
        "tokenizer": "test-tokenizer",
        "tokenizer_revision": "revision",
        "cases": [{"id": golden.id, "prompt_token_ids": [index + 1]} for index, golden in enumerate(goldens)],
    }
    fixture_bytes = (json.dumps(payload, sort_keys=True) + "\n").encode()
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_bytes(fixture_bytes)
    digest = hashlib.sha256(fixture_bytes).hexdigest()
    fixture = read_prompt_fixture(goldens, fixture_uri=str(fixture_path), expected_sha256=digest)
    assert fixture.tokenizer == "test-tokenizer"
    with pytest.raises(ValueError):
        read_prompt_fixture(goldens, fixture_uri=str(fixture_path), expected_sha256="0" * 64)
