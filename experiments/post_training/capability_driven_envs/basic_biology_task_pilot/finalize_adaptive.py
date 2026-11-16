# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze a private, auditable snapshot of the ten-passage adaptive GLM pilot."""

import argparse
import hashlib
import json
from collections import Counter
from typing import Any

from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.adaptive import (
    ARMS,
    COLLISIONS_TO_STOP,
    MAX_ATTEMPTS,
    REVISION,
    PassageProgress,
    ScreenedItem,
    apply_decision,
)
from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.finalize import read_json, write_new
from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.run_glm import (
    MCQ,
    SOURCE_MANIFEST_SHA256,
    read_rows,
)

PILOT_INDICES = (86, 63, 46, 51, 120, 150, 169, 136, 189, 131)


def snapshot(
    input_uri: str, result_root: str, output_root: str, *, all_passages: bool = False, pilot_root: str | None = None
) -> dict[str, object]:
    """Validate saved stage decisions and freeze summary hashes and screened items."""
    rows = read_rows(input_uri)
    if all_passages != (pilot_root is not None):
        raise ValueError("Full-corpus snapshots must supply the completed pilot root")
    indices = tuple(range(len(rows))) if all_passages else PILOT_INDICES
    root = result_root.rstrip("/") + "/"
    results: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    marginal: Counter[int] = Counter()
    by_source: Counter[str] = Counter()
    stops: Counter[str] = Counter()
    for index in indices:
        row = rows[index]
        passage_id = str(row["id"])
        prefix = (pilot_root.rstrip("/") + "/" if pilot_root and index in PILOT_INDICES else root) + passage_id + "/"
        final, final_sha = read_json(prefix + "summary.json")
        if (
            final["id"] != passage_id
            or final["revision"] != REVISION
            or final["source_url"] != row["source_url"]
            or final["max_attempts"] != MAX_ATTEMPTS
            or final["collision_threshold"] != COLLISIONS_TO_STOP
        ):
            raise ValueError(f"Unexpected completed passage: {passage_id}")
        progress = PassageProgress()
        attempts: list[dict[str, str]] = []
        for number in range(MAX_ATTEMPTS // len(ARMS)):
            for arm in ARMS:
                if number >= final["arms"][arm]["attempts"]:
                    continue
                if progress.arms[arm].stopped:
                    raise ValueError(f"Attempt after collision stop: {passage_id}/{arm}/{number}")
                uri = prefix + arm + f"/attempt-{number:02d}/summary.json"
                attempt, digest = read_json(uri)
                identity = f"{passage_id}-{arm}-{number:02d}"
                if (
                    attempt["id"] != identity
                    or attempt["revision"] != REVISION
                    or attempt["arm"] != arm
                    or attempt["attempt"] != number
                    or attempt["prior_novel_ids"] != [item.id for item in progress.novel]
                ):
                    raise ValueError(f"Inconsistent attempt: {identity}")
                status = attempt["status"]
                item: MCQ | None = None
                if status != "quarantined":
                    if not attempt["proposal_uri"] or attempt["issues"]:
                        raise ValueError(f"Screened attempt has missing proposal or quality issues: {identity}")
                    proposal, _ = read_json(attempt["proposal_uri"])
                    item = MCQ.model_validate(proposal["value"]["item"])
                    if status == "screened_novel":
                        candidates.append(
                            {
                                "id": identity,
                                "arm": arm,
                                "source_url": row["source_url"],
                                "source_family": row["source_family"],
                                "source_page_sha256": row["source_page_sha256"],
                                "passage_sha256": row["passage_sha256"],
                                "item": item.model_dump(mode="json"),
                                "status": "glm_screened_novel_not_validated",
                            }
                        )
                        marginal[number + 1] += 1
                    elif status == "screened_collision":
                        if attempt["matched_id"] not in {old.id for old in progress.novel}:
                            raise ValueError(f"Unknown collision target: {identity}")
                        review, _ = read_json(prefix + arm + f"/attempt-{number:02d}/collision.json")
                        decision = review["value"]
                        if not decision["same_learning_objective"] or decision["matched_id"] != attempt["matched_id"]:
                            raise ValueError(f"Inconsistent collision call: {identity}")
                    else:
                        raise ValueError(f"Unknown status: {identity}: {status}")
                elif attempt["matched_id"] is not None:
                    raise ValueError(f"Rejected item has collision match: {identity}")
                status_counts[f"{arm}:{status}"] += 1
                reason_counts.update(attempt["issues"])
                by_source[f"{row['source_family']}:{status}"] += 1
                attempts.append({"id": identity, "sha256": digest})
                apply_decision(
                    progress,
                    arm,
                    status,
                    ScreenedItem(identity, item) if status == "screened_novel" and item is not None else None,
                    COLLISIONS_TO_STOP,
                )
        if [item.id for item in progress.novel] != final["screened_distinct_ids"]:
            raise ValueError(f"Distinct set differs: {passage_id}")
        for arm in ARMS:
            state = progress.arms[arm]
            reported = final["arms"][arm]
            if (
                state.attempts != reported["attempts"]
                or state.collision_streak != reported["collision_streak"]
                or reported["stop_reason"] != ("collision_threshold" if state.stopped else "attempt_cap")
            ):
                raise ValueError(f"Stopping decision differs: {passage_id}/{arm}")
            stops[f"{arm}:{reported['stop_reason']}"] += 1
        results.append(
            {
                "offset": index,
                "id": passage_id,
                "family": row["source_family"],
                "summary_sha256": final_sha,
                "attempts": attempts,
                "novel_screened": len(progress.novel),
                "stops": {arm: final["arms"][arm]["stop_reason"] for arm in ARMS},
            }
        )
    aggregate: dict[str, Any] = {
        "revision": REVISION,
        "source_passages_sha256": SOURCE_MANIFEST_SHA256,
        "status": "glm_screening_and_novelty_only_not_scientifically_validated",
        ("passage_indices" if all_passages else "pilot_indices"): indices,
        "max_total_attempts_per_passage": MAX_ATTEMPTS,
        "collision_threshold_per_arm": COLLISIONS_TO_STOP,
        "status_counts": dict(sorted(status_counts.items())),
        "quarantine_reasons": dict(sorted(reason_counts.items())),
        "by_source": dict(sorted(by_source.items())),
        "marginal_distinct_by_attempt_number": dict(sorted(marginal.items())),
        "stop_reasons": dict(sorted(stops.items())),
        "passages": results,
    }
    aggregate_bytes = (json.dumps(aggregate, sort_keys=True, indent=2) + "\n").encode()
    candidate_bytes = ("".join(json.dumps(item, sort_keys=True) + "\n" for item in candidates)).encode()
    base = output_root.rstrip("/") + "/"
    write_new(base + "aggregate.json", aggregate_bytes)
    write_new(base + "screened-novel.jsonl", candidate_bytes)
    manifest: dict[str, object] = {
        "status": (
            "private_glm_screened_full_wave_not_training_ready"
            if all_passages
            else "private_glm_screened_pilot_not_training_ready"
        ),
        "revision": REVISION,
        "result_root": result_root,
        "source_sha256": SOURCE_MANIFEST_SHA256,
        "aggregate_sha256": hashlib.sha256(aggregate_bytes).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "distinct_screened_items": len(candidates),
        "completed_attempts": sum(status_counts.values()),
    }
    if pilot_root is not None:
        manifest["completed_pilot_root"] = pilot_root
    write_new(base + "manifest.json", (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-passages", action="store_true")
    parser.add_argument("--pilot-root", help="Read the ten completed v3 pilot passages instead of repeating them")
    args = parser.parse_args()
    print(
        json.dumps(
            snapshot(args.input, args.results, args.output, all_passages=args.all_passages, pilot_root=args.pilot_root),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
