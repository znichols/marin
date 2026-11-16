# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive a reviewed public evaluation source privately, with byte-for-byte pins.

This is evaluation-data acquisition for provenance/overlap checks, not a
TaskForge training import. No gated dataset, unchecked license or unknown
upstream asset may bypass the explicit review record.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import fsspec
import requests
from fsspec.spec import AbstractFileSystem

from experiments.post_training.capability_driven_envs.id_eval_source_intake.inventory import CATALOG_COMMIT, sha256

MAX_TOTAL_BYTES = 10_000_000
PRIVATE_S3_ROOT = "s3://marin-us-east-02a/marin4sc/taskforge/source-corpora/id-evals/"
REPO_ROOT = Path(__file__).resolve().parents[4]
REVIEW_PATH = Path(__file__).with_name("reviewed_sources.json")
HF_API = "https://huggingface.co/api/datasets/"
HF_RESOLVE = "https://huggingface.co/datasets/"


def reviewed_source(family: str, variant: str) -> dict[str, Any]:
    """Require an explicit, pinned, private-eval-only acquisition decision."""
    review = json.loads(REVIEW_PATH.read_text())
    if review["catalog_commit"] != CATALOG_COMMIT:
        raise ValueError("Rights review refers to another catalog revision")
    matches = [row for row in review["records"] if (row["family"], row["variant"]) == (family, variant)]
    if len(matches) != 1:
        raise ValueError("No unique reviewed source for this variant")
    record = matches[0]
    if (
        record["review"]["evaluation_acquisition"] != "approved_private_eval_only"
        or record["review"]["training_from_eval_questions_or_instructions"] != "prohibited_by_marin_policy"
        or record["review"]["training_from_upstream_sources"] != "blocked_pending_per_source_rights_review"
    ):
        raise ValueError("Dataset must remain an eval-only acquisition with unresolved upstream training rights")
    paths = [row["path"] for row in record["artifact"]["files"]]
    if not paths or len(set(paths)) != len(paths) or any(Path(p).is_absolute() or ".." in Path(p).parts for p in paths):
        raise ValueError("Unsafe or missing artifact file paths")
    if sum(int(row["size"]) for row in record["artifact"]["files"]) > MAX_TOTAL_BYTES:
        raise ValueError("Acquisition exceeds the pilot's ten-megabyte ceiling")
    return record


def storage(uri: str, endpoint_url: str | None) -> tuple[AbstractFileSystem, str]:
    """Resolve local or credential-injected CoreWeave S3 output, without implicit public ACLs."""
    if uri.startswith("s3://"):
        if not uri.startswith(PRIVATE_S3_ROOT):
            raise ValueError("Evaluation artifacts must stay under the designated private intake prefix")
        if not endpoint_url or not os.environ.get("CW_KEY_ID") or not os.environ.get("CW_KEY_SECRET"):
            raise ValueError("S3 output requires explicit endpoint and private CW storage credentials")
        fs = fsspec.filesystem(
            "s3",
            key=os.environ["CW_KEY_ID"],
            secret=os.environ["CW_KEY_SECRET"],
            client_kwargs={"endpoint_url": endpoint_url},
            config_kwargs={"s3": {"addressing_style": "virtual"}},
        )
        return fs, uri.removeprefix("s3://").rstrip("/")
    if "://" in uri:
        raise ValueError("Only a local path or private s3:// root is supported")
    if Path(uri).resolve().is_relative_to(REPO_ROOT):
        raise ValueError("Evaluation data must not be stored inside the Git checkout")
    fs, path = fsspec.core.url_to_fs(uri)
    return fs, path.rstrip("/")


def write_pinned(fs: AbstractFileSystem, path: str, data: bytes) -> None:
    """Never overwrite a differently-hashed object, even after an interrupted attempt."""
    if fs.exists(path):
        with fs.open(path, "rb") as stream:
            existing = stream.read()
        if sha256(existing) != sha256(data):
            raise ValueError(f"Existing private artifact differs: {path}")
        return
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(path, "wb") as stream:
        stream.write(data)


def download(repo: str, revision: str, file: dict[str, Any]) -> bytes:
    """Stream a pinned HF object into a bounded buffer and verify its expected SHA-256."""
    url = f"{HF_RESOLVE}{repo}/resolve/{revision}/{file['path']}"
    content = bytearray()
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        for part in response.iter_content(chunk_size=1024 * 1024):
            content.extend(part)
            if len(content) > file["size"] or len(content) > MAX_TOTAL_BYTES:
                raise ValueError("Source object exceeded its pinned size ceiling")
    data = bytes(content)
    if len(data) != file["size"] or sha256(data) != file["sha256"]:
        raise ValueError("Downloaded source bytes do not match the pinned LFS/card content")
    return data


def acquire(record: dict[str, Any], fs: AbstractFileSystem, prefix: str) -> dict[str, Any]:
    """Verify public access and license, archive exact bytes, then seal a private manifest."""
    artifact = record["artifact"]
    if artifact["kind"] != "huggingface_dataset":
        raise ValueError("Unsupported acquisition source")
    repo, revision = artifact["repo"], artifact["revision"]
    metadata = requests.get(HF_API + repo + "/revision/" + revision, timeout=30)
    metadata.raise_for_status()
    source = metadata.json()
    if source["sha"] != revision or source["gated"] is not False or source["private"] is not False:
        raise ValueError("Source changed, became gated, or is no longer public")
    if source["cardData"]["license"] != "mit":
        raise ValueError("Pinned review assumed a dataset-level MIT declaration")
    for file in artifact["files"]:
        destination = f"{prefix}/{file['path']}"
        if fs.exists(destination):
            with fs.open(destination, "rb") as stream:
                existing = stream.read()
            if len(existing) == file["size"] and sha256(existing) == file["sha256"]:
                continue
            raise ValueError(f"Existing source artifact differs: {destination}")
        write_pinned(fs, destination, download(repo, revision, file))
    manifest = {
        "review_revision": json.loads(REVIEW_PATH.read_text())["revision"],
        "catalog_commit": CATALOG_COMMIT,
        "family": record["family"],
        "variant": record["variant"],
        "source_repo": repo,
        "source_revision": revision,
        "files": artifact["files"],
        "status": "private_eval_only_not_taskforge_training_input",
        "underlying_source_rights_cleared": False,
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    write_pinned(fs, f"{prefix}/manifest.json", manifest_bytes)
    return {
        "files": len(artifact["files"]),
        "bytes": sum(f["size"] for f in artifact["files"]),
        "manifest_sha256": sha256(manifest_bytes),
        "status": manifest["status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True, help="Private, fresh local or s3:// directory; never a Git path")
    parser.add_argument("--s3-endpoint-url", help="Explicit CoreWeave virtual-hosted S3 endpoint for S3 output")
    args = parser.parse_args()
    record = reviewed_source(args.family, args.variant)
    fs, prefix = storage(args.output, args.s3_endpoint_url)
    print(json.dumps(acquire(record, fs, prefix), sort_keys=True))


if __name__ == "__main__":
    main()
