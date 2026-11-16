# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive a public eval source and inspect its rights evidence and provenance.

This is evaluation-data acquisition for provenance/overlap checks, not a
TaskForge training import. Gated and unlicensed sources stay out; underlying
assets are recorded as unresolved until checked against their own terms.
"""

import argparse
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import fsspec
import pyarrow.parquet as pq
import requests
from fsspec.spec import AbstractFileSystem

from experiments.post_training.capability_driven_envs.id_eval_source_intake.inventory import CATALOG_COMMIT, sha256

MAX_TOTAL_BYTES = 10_000_000
PRIVATE_S3_ROOT = "s3://marin-us-east-02a/marin4sc/taskforge/source-corpora/id-evals/"
REPO_ROOT = Path(__file__).resolve().parents[4]
CANDIDATE_PATH = Path(__file__).with_name("source_candidates.json")
HF_API = "https://huggingface.co/api/datasets/"
HF_RESOLVE = "https://huggingface.co/datasets/"


def source_candidate(
    family: str, variant: str, candidate_path: Path = CANDIDATE_PATH, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Resolve a pinned public-source candidate without granting training permission."""
    raw = candidate_path.read_bytes()
    if candidate_path != CANDIDATE_PATH and (not expected_sha256 or sha256(raw) != expected_sha256):
        raise ValueError("External candidate inventory requires its exact SHA-256")
    candidates = json.loads(raw)
    if candidates["catalog_commit"] != CATALOG_COMMIT:
        raise ValueError("Source candidates refer to another catalog revision")
    matches = [row for row in candidates["records"] if (row["family"], row["variant"]) == (family, variant)]
    if len(matches) != 1:
        raise ValueError("No unique source candidate for this variant")
    record = matches[0]
    paths = [row["path"] for row in record["artifact"]["files"]]
    if not paths or len(set(paths)) != len(paths) or any(Path(p).is_absolute() or ".." in Path(p).parts for p in paths):
        raise ValueError("Unsafe or missing artifact file paths")
    if sum(int(row["size"]) for row in record["artifact"]["files"]) > MAX_TOTAL_BYTES:
        raise ValueError("Acquisition exceeds the pilot's ten-megabyte ceiling")
    return {**record, "candidate_revision": candidates["revision"]}


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
    if source["cardData"].get("license") != artifact["declared_license"]:
        raise ValueError("Pinned dataset-level license declaration changed")
    for file in artifact["files"]:
        destination = f"{prefix}/{file['path']}"
        if fs.exists(destination):
            with fs.open(destination, "rb") as stream:
                existing = stream.read()
            if len(existing) == file["size"] and sha256(existing) == file["sha256"]:
                continue
            raise ValueError(f"Existing source artifact differs: {destination}")
        write_pinned(fs, destination, download(repo, revision, file))
    with fs.open(f"{prefix}/README.md", "rb") as card_stream:
        card = card_stream.read().decode("utf-8")
    restriction_markers = (
        "evaluation-only",
        "evaluation only",
        "not for training",
        "non-commercial",
        "all rights reserved",
    )
    provenance_counts: Counter[str] = Counter()
    parquet_row_counts: dict[str, int] = {}
    for file in artifact["files"]:
        if not file["path"].endswith(".parquet"):
            continue
        with fs.open(f"{prefix}/{file['path']}", "rb") as stream:
            contents = io.BytesIO(stream.read())
        if artifact["provenance_column"] is None:
            parquet_row_counts[file["path"]] = pq.ParquetFile(contents).metadata.num_rows
        else:
            table = pq.read_table(contents, columns=[artifact["provenance_column"]])
            provenance_counts.update(str(value) for value in table.column(0).to_pylist())
    rights_inspection: dict[str, Any] = {
        "dataset_license_declared": artifact["declared_license"],
        "pinned_card_sha256": next(file["sha256"] for file in artifact["files"] if file["path"] == "README.md"),
        "card_restriction_markers": [marker for marker in restriction_markers if marker in card.lower()],
        "underlying_source_labels": dict(sorted(provenance_counts.items())),
        "underlying_source_terms_checked": False,
    }
    if parquet_row_counts:
        rights_inspection["parquet_row_counts"] = parquet_row_counts
    manifest = {
        "candidate_revision": record["candidate_revision"],
        "catalog_commit": CATALOG_COMMIT,
        "family": record["family"],
        "variant": record["variant"],
        "source_repo": repo,
        "source_revision": revision,
        "files": artifact["files"],
        "rights_inspection": rights_inspection,
        "status": "private_eval_source_upstream_rights_unresolved_not_training_input",
        **({"artifact_coverage": artifact["coverage"]} if "coverage" in artifact else {}),
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
    parser.add_argument("--candidates", type=Path, default=CANDIDATE_PATH, help="Private pinned acquisition plan")
    parser.add_argument("--candidate-sha256", help="Required SHA-256 when --candidates is outside this package")
    args = parser.parse_args()
    record = source_candidate(args.family, args.variant, args.candidates, args.candidate_sha256)
    fs, prefix = storage(args.output, args.s3_endpoint_url)
    print(json.dumps(acquire(record, fs, prefix), sort_keys=True))


if __name__ == "__main__":
    main()
