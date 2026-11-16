"""Freeze screened dual-arm GLM outputs without treating agreement as validation."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import fsspec

from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.run_glm import (
    REVISION,
    SOURCE_MANIFEST_SHA256,
    read_rows,
)

ARMS = ("qfirst", "afirst")
STAGES = ("qfirst", "anchor", "afirst", "qfirst-review", "afirst-review", "qfirst-fit", "afirst-fit")


def read_bytes(uri: str) -> bytes:
    """Read one required immutable response object."""
    with fsspec.open(uri, "rb") as stream:
        return stream.read()


def read_json(uri: str) -> tuple[dict[str, Any], str]:
    """Parse and hash a private JSON record."""
    raw = read_bytes(uri)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write_new(uri: str, raw: bytes) -> None:
    """Refuse to replace a frozen artifact with different bytes."""
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        if read_bytes(uri) != raw:
            raise ValueError(f"Frozen output collision: {uri}")
        return
    with fs.open(path, "wb") as stream:
        stream.write(raw)


def freeze(input_uri: str, results: str, old_smoke: str, new_smoke: str, output: str) -> dict[str, object]:
    """Verify every assigned pair and publish a private immutable completion snapshot."""
    rows = read_rows(input_uri)
    counts: Counter[str] = Counter()
    families: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    yield_per_passage: Counter[int] = Counter()
    key_positions: Counter[int] = Counter()
    summaries: list[dict[str, str]] = []
    candidates: list[dict[str, object]] = []
    stage_counts: Counter[str] = Counter()
    for row in rows:
        identity = str(row["id"])
        uri = results.rstrip("/") + "/" + identity + "/summary.json"
        record, digest = read_json(uri)
        if record["id"] != identity or record["revision"] != REVISION or record["source_url"] != row["source_url"]:
            raise ValueError(f"Unexpected summary identity: {uri}")
        outcomes = record["outcomes"]
        if set(outcomes) != set(ARMS):
            raise ValueError(f"Missing generation arm: {uri}")
        summaries.append({"id": identity, "sha256": digest})
        n = 0
        for arm in ARMS:
            outcome = outcomes[arm]
            status = outcome["status"]
            if status not in ("screened_candidate", "quarantined"):
                raise ValueError(f"Unknown screening status: {status}")
            counts[f"{arm}:{status}"] += 1
            families[f"{row['source_family']}:{arm}:{status}"] += 1
            issues.update(outcome["issues"])
            if status != "screened_candidate":
                continue
            n += 1
            stage = arm + ".json"
            stage_uri = results.rstrip("/") + "/" + identity + "/" + stage
            fs, path = fsspec.core.url_to_fs(stage_uri)
            if not fs.exists(path):
                stage_uri = old_smoke.rstrip("/") + "/" + identity + "/" + stage
            proposal, _ = read_json(stage_uri)
            item = proposal["value"]["item"]
            if item is None:
                raise ValueError(f"Screened item is absent: {stage_uri}")
            key_positions[item["correct_index"]] += 1
            candidates.append(
                {
                    "id": identity + "-" + arm,
                    "arm": arm,
                    "source_url": row["source_url"],
                    "source_family": row["source_family"],
                    "source_page_sha256": row["source_page_sha256"],
                    "passage_sha256": row["passage_sha256"],
                    "item": item,
                    "status": "screened_training_candidate_not_validated",
                }
            )
        yield_per_passage[n] += 1
    if len(summaries) != 195 or len({entry["id"] for entry in summaries}) != 195:
        raise ValueError("The 195-passage snapshot is incomplete or duplicated")
    for label, prefix in (("full", results), ("old_smoke", old_smoke), ("new_smoke", new_smoke)):
        fs, path = fsspec.core.url_to_fs(prefix.rstrip("/") + "/")
        for name in fs.find(path):
            stage = Path(name).name.removesuffix(".json")
            if stage in STAGES:
                stage_counts[label + ":" + stage] += 1
    aggregate: dict[str, object] = {
        "status": "screening_only_not_scientifically_validated",
        "revision": REVISION,
        "source_passages_sha256": SOURCE_MANIFEST_SHA256,
        "passage_count": len(rows),
        "arm_outcomes": dict(sorted(counts.items())),
        "source_arm_outcomes": dict(sorted(families.items())),
        "screened_yield_per_passage": {str(k): v for k, v in sorted(yield_per_passage.items())},
        "quarantine_reasons": dict(sorted(issues.items())),
        "screened_answer_position_counts": {str(k): v for k, v in sorted(key_positions.items())},
        "stage_counts": dict(sorted(stage_counts.items())),
        "summary_sha256": summaries,
    }
    aggregate_data = (json.dumps(aggregate, sort_keys=True, indent=2) + "\n").encode()
    candidates_data = ("".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates)).encode()
    base = output.rstrip("/") + "/"
    write_new(base + "aggregate.json", aggregate_data)
    write_new(base + "screened-candidates.jsonl", candidates_data)
    manifest: dict[str, object] = {
        "status": "private_screened_candidates_not_training_ready",
        "revision": REVISION,
        "source_passages_sha256": SOURCE_MANIFEST_SHA256,
        "full_result_root": results,
        "old_smoke_root": old_smoke,
        "new_smoke_root": new_smoke,
        "aggregate_sha256": hashlib.sha256(aggregate_data).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidates_data).hexdigest(),
        "screened_candidates": len(candidates),
    }
    write_new(base + "manifest.json", (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--old-smoke", required=True)
    parser.add_argument("--new-smoke", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.input, args.results, args.old_smoke, args.new_smoke, args.output), indent=2))


if __name__ == "__main__":
    main()
