# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive every file in a pinned public HF dataset repository to private S3.

The signed plan contains the complete HF tree at an exact revision. Git blobs
are verified against their Git SHA-1 object IDs and LFS files against SHA-256;
each downloaded file is spooled and verified before it can become an S3 object.
No evaluation item is used as a TaskForge training input.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import requests

from experiments.post_training.capability_driven_envs.id_eval_source_intake.acquire import (
    HF_API,
    HF_RESOLVE,
    PRIVATE_S3_ROOT,
    storage,
    write_pinned,
)
from experiments.post_training.capability_driven_envs.id_eval_source_intake.inventory import CATALOG_COMMIT, sha256

MAX_TRANSFER_BYTES = 10_000_000_000  # Larger cross-region copies require separate, size-specific sign-off.
CHUNK_BYTES = 4 * 1024 * 1024


def git_blob_sha1(size: int) -> Any:
    """Initialize a hash for Git's length-prefixed blob object format."""
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode())
    return digest


def verify_stream(stream: Any, file: dict[str, Any]) -> str:
    """Verify pinned Git/LFS object ID and return the content's SHA-256."""
    pinned = hashlib.sha256() if file["lfs"] else git_blob_sha1(file["size"])
    content = hashlib.sha256()
    size = 0
    while chunk := stream.read(CHUNK_BYTES):
        size += len(chunk)
        if size > file["size"]:
            raise ValueError(f"Object exceeds pinned size: {file['path']}")
        pinned.update(chunk)
        content.update(chunk)
    if size != file["size"] or pinned.hexdigest() != file["oid"]:
        raise ValueError(f"Pinned source hash or size changed: {file['path']}")
    return content.hexdigest()


def plan_record(
    path: Path, expected_sha256: str, family: str, variant: str, transfer_approval: str | None = None
) -> tuple[dict[str, Any], str]:
    """Validate a private full-tree plan, including the cross-region transfer cap."""
    raw = path.read_bytes()
    if sha256(raw) != expected_sha256:
        raise ValueError("Full-tree acquisition plan SHA-256 mismatch")
    plan = json.loads(raw)
    if plan["catalog_commit"] != CATALOG_COMMIT:
        raise ValueError("Catalog revision mismatch")
    matches = [row for row in plan["records"] if (row["family"], row["variant"]) == (family, variant)]
    if len(matches) != 1:
        raise ValueError("No unique acquisition plan entry")
    record = matches[0]
    files = record["artifact"]["files"]
    paths = [row["path"] for row in files]
    if not files or len(paths) != len(set(paths)) or "README.md" not in paths:
        raise ValueError("Incomplete or duplicate source tree")
    for file in files:
        parts = Path(file["path"])
        if parts.is_absolute() or ".." in parts.parts or file["size"] < 0:
            raise ValueError("Unsafe source path or size")
        if len(file["oid"]) != (64 if file["lfs"] else 40):
            raise ValueError("Unpinned source object")
    if sum(row["size"] for row in files) > MAX_TRANSFER_BYTES and not transfer_approval:
        raise ValueError("Full-tree transfer exceeds 10 GB; request explicit size-specific sign-off")
    return record, plan["revision"]


def acquire_full(
    record: dict[str, Any],
    plan_revision: str,
    plan_sha256: str,
    output: str,
    endpoint: str,
    transfer_approval: str | None = None,
) -> dict[str, Any]:
    """Download and verify every pinned source blob, then seal the S3 snapshot."""
    if not output.startswith(PRIVATE_S3_ROOT):
        raise ValueError("Full eval source must be archived under the private S3 intake prefix")
    fs, prefix = storage(output, endpoint)
    artifact = record["artifact"]
    repo, revision = artifact["repo"], artifact["revision"]
    response = requests.get(HF_API + repo + "/revision/" + revision, timeout=30)
    response.raise_for_status()
    source = response.json()
    if source["sha"] != revision or source["gated"] is not False or source["private"] is not False:
        raise ValueError("Pinned repo changed or is no longer public")
    if source["cardData"].get("license") != artifact["declared_license"]:
        raise ValueError("Dataset-level license declaration changed")
    tree_response = requests.get(f"{HF_API}{repo}/tree/{revision}?recursive=true", timeout=60)
    tree_response.raise_for_status()
    observed = {
        row["path"]: {
            "path": row["path"],
            "size": row["size"],
            "oid": row["lfs"]["oid"] if row.get("lfs") else row["oid"],
            "lfs": bool(row.get("lfs")),
        }
        for row in tree_response.json()
        if row["type"] == "file"
    }
    if observed != {row["path"]: row for row in artifact["files"]}:
        raise ValueError("Pinned plan no longer matches the complete repository tree")

    content_sha256 = {}
    for index, file in enumerate(artifact["files"], start=1):
        destination = f"{prefix}/{file['path']}"
        if fs.exists(destination):
            with fs.open(destination, "rb") as stream:
                content_sha256[file["path"]] = verify_stream(stream, file)
        else:
            with tempfile.TemporaryFile() as spool:
                url = f"{HF_RESOLVE}{repo}/resolve/{revision}/{file['path']}"
                with requests.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        spool.write(chunk)
                        if spool.tell() > file["size"]:
                            raise ValueError(f"Downloaded object exceeded pinned size: {file['path']}")
                spool.seek(0)
                content_sha256[file["path"]] = verify_stream(spool, file)
                spool.seek(0)
                fs.makedirs(destination.rsplit("/", 1)[0], exist_ok=True)
                with fs.open(destination, "wb") as target:
                    while chunk := spool.read(CHUNK_BYTES):
                        target.write(chunk)
            if fs.size(destination) != file["size"]:
                raise ValueError(f"Remote object has wrong size: {file['path']}")
        print(
            json.dumps(
                {
                    "family": record["family"],
                    "verified_files": index,
                    "total_files": len(artifact["files"]),
                    "path": file["path"],
                }
            ),
            flush=True,
        )

    with fs.open(f"{prefix}/README.md", "rb") as stream:
        card = stream.read().decode("utf-8")
    markers = ("evaluation-only", "evaluation only", "not for training", "non-commercial", "all rights reserved")
    manifest = {
        "family": record["family"],
        "variant": record["variant"],
        "catalog_commit": CATALOG_COMMIT,
        "plan_revision": plan_revision,
        "plan_sha256": plan_sha256,
        "source_repo": repo,
        "source_revision": revision,
        "coverage": "complete_pinned_hf_repository_tree_not_linked_external_assets",
        "files": [{**file, "sha256": content_sha256[file["path"]]} for file in artifact["files"]],
        "rights_inspection": {
            "dataset_license_declared": artifact["declared_license"],
            "card_restriction_markers": [marker for marker in markers if marker in card.lower()],
            "underlying_source_terms_checked": False,
        },
        "status": "private_eval_source_upstream_rights_unresolved_not_training_input",
        **({"transfer_approval": transfer_approval} if transfer_approval else {}),
    }
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    write_pinned(fs, f"{prefix}/manifest.json", raw)
    return {
        "family": record["family"],
        "files": len(artifact["files"]),
        "bytes": sum(f["size"] for f in artifact["files"]),
        "manifest_sha256": sha256(raw),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--s3-endpoint-url", required=True)
    parser.add_argument("--transfer-approval", help="Record explicit human sign-off for copies over 10 GB")
    args = parser.parse_args()
    record, revision = plan_record(args.plan, args.plan_sha256, args.family, args.variant, args.transfer_approval)
    print(
        json.dumps(
            acquire_full(record, revision, args.plan_sha256, args.output, args.s3_endpoint_url, args.transfer_approval),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
