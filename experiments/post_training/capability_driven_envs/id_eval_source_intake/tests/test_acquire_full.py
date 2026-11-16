# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
from pathlib import Path

import pytest

from experiments.post_training.capability_driven_envs.id_eval_source_intake.acquire_full import (
    MAX_TRANSFER_BYTES,
    git_blob_sha1,
    plan_record,
    verify_stream,
)
from experiments.post_training.capability_driven_envs.id_eval_source_intake.inventory import CATALOG_COMMIT, sha256


def test_git_and_lfs_pinned_object_verification() -> None:
    data = b"some benchmark file\n"
    git = git_blob_sha1(len(data)).copy()
    git.update(data)
    assert (
        verify_stream(io.BytesIO(data), {"path": "README.md", "size": len(data), "oid": git.hexdigest(), "lfs": False})
        == hashlib.sha256(data).hexdigest()
    )
    assert (
        verify_stream(
            io.BytesIO(data), {"path": "a.zip", "size": len(data), "oid": hashlib.sha256(data).hexdigest(), "lfs": True}
        )
        == hashlib.sha256(data).hexdigest()
    )
    with pytest.raises(ValueError, match="Pinned source"):
        verify_stream(
            io.BytesIO(b"wrong"), {"path": "README.md", "size": len(data), "oid": git.hexdigest(), "lfs": False}
        )


def test_full_plan_rejects_over_10gb_without_size_specific_signoff(tmp_path: Path) -> None:
    plan = {
        "catalog_commit": CATALOG_COMMIT,
        "revision": "test-v1",
        "records": [
            {
                "family": "bio",
                "variant": "bio",
                "artifact": {
                    "files": [{"path": "README.md", "size": MAX_TRANSFER_BYTES + 1, "oid": "0" * 40, "lfs": False}]
                },
            }
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="explicit size-specific sign-off"):
        plan_record(path, sha256(path.read_bytes()), "bio", "bio")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        plan_record(path, "0" * 64, "bio", "bio")
