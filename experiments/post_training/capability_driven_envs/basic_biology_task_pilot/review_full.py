# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Review every frozen MCQ with an independent, blinded GLM science/quality pass.

The reviewer is independent of the saved generation/review calls, not an
independent biologist. Every response is persisted before any disposition is
computed; this does not clear source rights or authorize training.
"""

import argparse
import asyncio
import hashlib
import json
import os
from typing import Any

import fsspec
from pydantic import BaseModel, ConfigDict, Field
from taskforge.llm.policy import LLMPolicy, ReasoningEffort
from taskforge_adapters.campaign import OpenAIModelPort, endpoint_source, provider_config
from taskforge_adapters.glm_client import GlmClient

from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.run_glm import (
    MCQ,
    SOURCE_MANIFEST_SHA256,
    _quote_matches,
    call_or_load,
    read_rows,
    write_once,
)
from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL

CANDIDATE_SHA256 = "f748948a6bfb715e3caaf83164f84e2a3318de093c85350482982b12549c5501"
REVISION = "gov-bio-full-science-review-glm-v1"
POLICY = LLMPolicy(max_tokens=1800, reasoning_effort=ReasoningEffort.LOW, temperature=0, top_p=1)


class ScienceReview(BaseModel):
    """Blinded candidate assessment, not a biological expert attestation."""

    model_config = ConfigDict(extra="forbid")

    best_index: int | None = Field(ge=0, le=3)
    other_defensible_indices: list[int]
    correct_biology: bool
    supported_by_passage: bool
    fits_intro_biology: bool
    distractors_plausible: bool
    time_sensitive_or_outdated: bool
    evidence_quote: str
    biological_issue: str
    curricular_issue: str
    distractor_issue: str
    explanation: str


def load_candidates(uri: str) -> list[dict[str, Any]]:
    """Read the exact frozen snapshot, not a mutable generation prefix."""
    with fsspec.open(uri, "rb") as file:
        raw = file.read()
    if hashlib.sha256(raw).hexdigest() != CANDIDATE_SHA256:
        raise ValueError("Frozen candidate hash mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 840 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Expected 840 distinct frozen candidates")
    return rows


def review_prompt(passage: str, item: MCQ) -> str:
    """Do not show the generated key, quote, original GLM decision, or earlier reviews."""
    return (
        "Review this multiple-choice question independently against the supplied source and basic biology. "
        "Choose the best 0-based option index, or null if no option is defensible. List every OTHER defensible "
        "index. Explain any biological errors, unproved premises, incorrect units or quantities (especially "
        "chromosomes versus chromosome pairs), missing assumptions, ambiguity, or dated claims. Even when the "
        "source says something, flag it if it is biologically wrong or time-sensitive. Check that the objective "
        "belongs in a general Bio 101 curriculum (cells, heredity, evolution, core molecular biology, etc.), "
        "not specialist diagnostics, policy, researcher trivia, or source-table memorization. Are the incorrect "
        "options plausible enough to assess understanding without themselves becoming defensible? Give a short "
        "verbatim supporting quote; if there is none use an empty string. Return each issue field as an empty "
        "string only if there is no issue. Be skeptical: approving a wrong key is worse than quarantining a "
        "good item. This is screening, not scientific validation.\n\n"
        f"SOURCE PASSAGE:\n{passage}\n\nQUESTION:\n{item.question}\nOPTIONS:\n"
        + "\n".join(f"{i}: {option}" for i, option in enumerate(item.options))
    )


def disposition(item: MCQ, review: ScienceReview, passage: str) -> tuple[str, list[str]]:
    """Reject on any model objection; never turn a missing review into acceptance."""
    issues = []
    if review.best_index != item.correct_index:
        issues.append("reviewer_disagrees_with_key")
    if review.other_defensible_indices:
        issues.append("multiple_defensible_choices")
    if not review.evidence_quote or not _quote_matches(passage, review.evidence_quote):
        issues.append("review_quote_not_in_passage")
    for field in (
        "correct_biology",
        "supported_by_passage",
        "fits_intro_biology",
        "distractors_plausible",
    ):
        if not getattr(review, field):
            issues.append(f"failed_{field}")
    if review.time_sensitive_or_outdated:
        issues.append("time_sensitive_or_outdated")
    for field in ("biological_issue", "curricular_issue", "distractor_issue"):
        if getattr(review, field).strip():
            issues.append(field)
    return ("model_screened_for_taskforge_probe" if not issues else "quarantined_for_review", issues)


async def review_one(
    model: GlmClient,
    record: dict[str, Any],
    passage: dict[str, Any],
    output_root: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Persist one paid call and a separate deterministic decision for each item."""
    item = MCQ.model_validate(record["item"])
    item_id = record["id"]
    if (
        record["passage_sha256"] != passage["passage_sha256"]
        or record["source_page_sha256"] != passage["source_page_sha256"]
    ):
        raise ValueError(f"Source mismatch: {item_id}")
    base = output_root.rstrip("/") + "/" + item_id + "/"
    prompt = review_prompt(passage["text"], item)
    async with semaphore:
        saved = await call_or_load(model, prompt, ScienceReview, POLICY, base + "review.json")
    review = ScienceReview.model_validate(saved["value"])
    status, issues = disposition(item, review, passage["text"])
    result: dict[str, Any] = {
        "id": item_id,
        "revision": REVISION,
        "source_sha256": SOURCE_MANIFEST_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "status": status,
        "issues": issues,
        "rights_cleared": False,
        "expert_validated": False,
        "review_uri": base + "review.json",
    }
    write_once(base + "decision.json", result)
    return result


async def run(args: argparse.Namespace) -> None:
    candidates = load_candidates(args.candidates)
    passages = read_rows(args.passages)
    if len(passages) != 195:
        raise ValueError("Expected 195 pinned passages")
    by_passage = {row["id"]: row for row in passages}
    selected = [row for index, row in enumerate(candidates) if index % args.shard_count == args.shard_index]
    if args.ids is not None:
        requested = set(args.ids.split(","))
        selected = [row for row in selected if row["id"] in requested]
        if len(selected) != len(requested):
            raise ValueError("Pilot IDs must each belong to this shard's frozen candidates")
    if args.limit is not None:
        selected = selected[: args.limit]
    port = OpenAIModelPort(
        GLM_MODEL,
        GLM_BULK_TOKEN_ENV,
        base_url=args.base_url,
        relay_job=None if args.base_url else args.relay_job,
        relay_service=None if args.base_url else args.relay_service,
    )
    async with GlmClient(endpoint_source(port, os.environ[port.api_key_env]), provider_config(port)) as model:
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            review_one(model, record, by_passage[record["id"].rsplit("-", 2)[0]], args.output, semaphore)
            for record in selected
        ]
        for future in asyncio.as_completed(tasks):
            result = await future
            print(json.dumps({"id": result["id"], "status": result["status"], "issues": result["issues"]}), flush=True)
    print(json.dumps({"reviewed": len(selected), "shard": args.shard_index}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--passages", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Pilot only; not part of full finalization")
    parser.add_argument("--ids", help="Pilot only: comma-separated frozen candidate IDs")
    parser.add_argument("--base-url")
    parser.add_argument("--relay-job", default="/muchanem/glm53-relay-tok-02a")
    parser.add_argument("--relay-service", default="glm-5.3-tok")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count or args.concurrency < 1:
        raise ValueError("Invalid shard settings")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
