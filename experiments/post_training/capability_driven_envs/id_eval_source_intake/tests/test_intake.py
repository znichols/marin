# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rights-gated metadata joins and immutable private acquisition guards."""

from pathlib import Path

import pytest

from experiments.post_training.capability_driven_envs.id_eval_source_intake.acquire import (
    reviewed_source,
    storage,
    write_pinned,
)
from experiments.post_training.capability_driven_envs.id_eval_source_intake.inventory import historical_id_rows


def test_historical_id_never_promotes_to_training_rights() -> None:
    source = {
        "records": [
            {"family_id": "biology", "variant_id": "v1", "name": "Example", "assignment": "ID", "sources": []},
            {"family_id": "biology", "variant_id": "v2", "name": "Other", "assignment": "OOD", "sources": []},
        ]
    }
    catalog = {
        "biology": {
            "selected_variant": "v2",
            "variants": [
                {"id": "v1", "assignment": "TBD", "sources": [{"label": "Benchmark", "url": "https://example.org"}]},
                {"id": "v2", "assignment": "OOD", "sources": []},
            ],
        }
    }
    rows = historical_id_rows(source, catalog)
    assert len(rows) == 1
    assert rows[0]["catalog_assignment"] == "TBD"
    assert rows[0]["selected_variant"] is False
    assert rows[0]["training_rights_status"] == "not_reviewed"
    assert rows[0]["eval_access_status"] == "not_reviewed"


def test_only_explicitly_reviewed_variant_can_be_acquired() -> None:
    record = reviewed_source("mmlu", "mmlu-pro-biology")
    assert record["review"]["training_from_eval_questions_or_instructions"] == "prohibited_by_marin_policy"
    with pytest.raises(ValueError, match="No unique reviewed source"):
        reviewed_source("gpqa", "gpqa-diamond-biology")


def test_output_must_be_outside_git_and_frozen_bytes_cannot_change(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inside the Git checkout"):
        storage(str(Path(__file__).resolve().parent), None)
    with pytest.raises(ValueError, match="designated private intake prefix"):
        storage("s3://unreviewed-bucket/benchmark-items", "https://cwobject.com")
    fs, root = storage(str(tmp_path), None)
    path = root + "/source/file.bin"
    write_pinned(fs, path, b"frozen")
    write_pinned(fs, path, b"frozen")
    assert Path(path).read_bytes() == b"frozen"
    with pytest.raises(ValueError, match="differs"):
        write_pinned(fs, path, b"changed")
