# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Promote a complete native-Levanter report to a reviewed golden candidate."""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from rigging.filesystem import StoragePath

from tests.cluster.vllm.backend_parity import ObservationReport
from tests.cluster.vllm.snowball import (
    canonical_golden_bytes,
    goldens_from_native_report,
    read_representative_goldens,
)


class GoldenPromotionProvenance(BaseModel):
    """Reviewed subset of measured fields required before golden promotion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_revision: str
    code_digest: str
    runtime_fingerprint_digest: str
    parameter_digest: str
    model_config_digest: str
    prompt_fixture_digest: str
    device_kind: str
    mesh_shape: tuple[tuple[str, int], ...]


def promote_native_report(
    *,
    report_uri: str,
    output_path: Path,
    manifest_path: Path,
    platform: str,
    expected_provenance: GoldenPromotionProvenance,
) -> None:
    report_bytes = StoragePath(report_uri).read_bytes()
    report = ObservationReport.model_validate_json(report_bytes)
    provenance = report.provenance

    mismatches = [
        f"{field}: expected {value!r}, got {getattr(provenance, field)!r}"
        for field, value in expected_provenance.model_dump().items()
        if getattr(provenance, field) != value
    ]
    if provenance.platform != platform:
        mismatches.append(f"platform: expected {platform!r}, got {provenance.platform!r}")
    if mismatches:
        raise ValueError("Golden promotion provenance mismatch:\n" + "\n".join(mismatches))

    canonical_order = tuple(golden.id for golden in read_representative_goldens())
    goldens = goldens_from_native_report(
        report,
        expected_platform=platform,
        case_order=canonical_order,
    )
    golden_bytes = canonical_golden_bytes(goldens)
    output_path.write_bytes(golden_bytes)
    manifest = {
        "schema_version": 1,
        "requires_human_review": True,
        "source_report_uri": report_uri,
        "source_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "candidate_golden_sha256": hashlib.sha256(golden_bytes).hexdigest(),
        "provenance": provenance.model_dump(mode="json"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
