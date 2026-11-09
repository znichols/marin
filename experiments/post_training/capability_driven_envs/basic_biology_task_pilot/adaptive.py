# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pilot adaptive, paired MCQ generation with a bounded semantic-collision stop.

All per-attempt GLM calls and decisions are private, immutable, and resumable.
This is screening for *training* candidates, not scientific validation.
"""

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from taskforge_adapters.campaign import OpenAIModelPort, endpoint_source, provider_config
from taskforge_adapters.glm_client import GlmClient

from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.run_glm import (
    GENERATION,
    MCQ,
    REVIEW,
    AnswerAnchor,
    BlindReview,
    IntroFit,
    Proposal,
    call_or_load,
    candidate_issues,
    load,
    prompt_for_anchor,
    prompt_for_answer_first,
    prompt_for_intro_fit,
    prompt_for_question_first,
    prompt_for_review,
    read_rows,
    review_issues,
    write_once,
)
from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL

REVISION = "gov-bio-adaptive-mcq-glm-v1"
MAX_ATTEMPTS = 20  # Total across both arms; never twenty per arm.
COLLISIONS_TO_STOP = 3
ARMS = ("qfirst", "afirst")


class CollisionReview(BaseModel):
    """GLM screening decision, not proof of semantic equivalence."""

    model_config = ConfigDict(extra="forbid")

    same_learning_objective: bool
    matched_id: str | None
    rationale: str = Field(min_length=8)


@dataclass(frozen=True)
class ScreenedItem:
    id: str
    item: MCQ


@dataclass
class ArmProgress:
    attempts: int = 0
    collision_streak: int = 0
    stopped: bool = False


@dataclass
class PassageProgress:
    arms: dict[str, ArmProgress] = field(default_factory=lambda: {arm: ArmProgress() for arm in ARMS})
    novel: list[ScreenedItem] = field(default_factory=list)


def generation_prompt(passage: str, arm: str, prior: list[ScreenedItem]) -> str:
    """Avoid already-covered learning objectives; preserve v2's first prompt."""
    base = prompt_for_question_first(passage) if arm == "qfirst" else prompt_for_anchor(passage)
    if not prior:
        return base
    covered = [{"question": old.item.question, "answer": old.item.options[old.item.correct_index]} for old in prior]
    return (
        base + "\n\nADDITIONAL REQUIREMENT: Earlier questions already test the following learning objectives. "
        "Choose a DIFFERENT passage-supported biological fact or reasoning step, not a paraphrase, "
        "substitution of names, or a more obscure piece of trivia. Return no_task if there is no distinct "
        "introductory-biology question left.\nALREADY COVERED: " + json.dumps(covered, ensure_ascii=False)
    )


def collision_prompt(candidate: MCQ, prior: list[ScreenedItem]) -> str:
    """Judge equivalence of the learning objective, not surface-word overlap."""
    examples = [
        {
            "id": old.id,
            "question": old.item.question,
            "correct_answer": old.item.options[old.item.correct_index],
            "evidence": old.item.evidence_quote,
        }
        for old in prior
    ]
    candidate_data = {
        "question": candidate.question,
        "correct_answer": candidate.options[candidate.correct_index],
        "evidence": candidate.evidence_quote,
    }
    return (
        "Do the candidate and ANY earlier question assess essentially the SAME biology fact or reasoning step? "
        "Paraphrases, changing names/numbers for the same fact, or reversing question and answer count as a collision. "
        "Sharing a broad topic or vocabulary alone does NOT; distinguish e.g. DNA replication from DNA transcription. "
        "If any collide, give ONE earlier id in matched_id and same_learning_objective=true; "
        "otherwise matched_id=null and same_learning_objective=false. Explain the comparison. "
        "This is a novelty screen, NOT a scientific correctness assessment.\n"
        f"CANDIDATE: {json.dumps(candidate_data, ensure_ascii=False)}\n"
        f"EARLIER DISTINCT QUESTIONS: {json.dumps(examples, ensure_ascii=False)}"
    )


def collision_review_issue(review: CollisionReview, prior: list[ScreenedItem]) -> str | None:
    """Quarantine inconsistent or unknown model matches without repeating a completed call."""
    if review.same_learning_objective != (review.matched_id is not None):
        return "invalid_collision_review"
    if review.matched_id is not None and review.matched_id not in {item.id for item in prior}:
        return "invalid_collision_review"
    return None


def apply_decision(progress: PassageProgress, arm: str, status: str, item: ScreenedItem | None, threshold: int) -> None:
    """Advance one arm; rejected attempts do not reset or increment collision streaks."""
    state = progress.arms[arm]
    state.attempts += 1
    if status == "screened_novel":
        if item is None:
            raise ValueError("Novel item missing")
        progress.novel.append(item)
        state.collision_streak = 0
    elif status == "screened_collision":
        state.collision_streak += 1
        state.stopped = state.collision_streak >= threshold
    elif status != "quarantined":
        raise ValueError(f"Unexpected attempt status: {status}")


def first_stage_uri(passage_id: str, stage: str, prior_root: str, smoke_root: str, fit_smoke_root: str) -> str:
    """Reuse completed first-attempt v2 calls, including the four older smoke items."""
    roots = (prior_root, fit_smoke_root) if stage.endswith("-fit") else (prior_root, smoke_root)
    for root in roots:
        uri = root.rstrip("/") + "/" + passage_id + "/" + stage + ".json"
        if load(uri) is not None:
            return uri
    raise ValueError(f"Missing frozen v2 first-attempt stage: {passage_id}/{stage}")


async def one_attempt(
    model: GlmClient,
    passage: str,
    passage_id: str,
    arm: Literal["qfirst", "afirst"],
    number: int,
    progress: PassageProgress,
    root: str,
    prior_root: str,
    smoke_root: str,
    fit_smoke_root: str,
    threshold: int,
) -> dict[str, object]:
    """Save generation, review, fit and novelty evidence before recording a decision."""
    attempt_root = root + arm + f"/attempt-{number:02d}/"

    def stage_uri(name: str) -> str:
        if number == 0:
            return first_stage_uri(passage_id, name, prior_root, smoke_root, fit_smoke_root)
        return attempt_root + name + ".json"

    generated = generation_prompt(passage, arm, [] if number == 0 else progress.novel)
    issues: list[str] = []
    anchor: AnswerAnchor | None = None
    if arm == "afirst":
        anchor = AnswerAnchor.model_validate(
            (await call_or_load(model, generated, AnswerAnchor, GENERATION, stage_uri("anchor")))["value"]
        )
        if anchor.disposition != "candidate":
            issues.append("no_anchor")
        elif (
            not anchor.answer
            or not anchor.evidence_quote
            or anchor.answer not in passage
            or anchor.evidence_quote not in passage
        ):
            issues.append("unproven_anchor")
        elif any(
            anchor.answer.casefold() == old.item.options[old.item.correct_index].casefold() for old in progress.novel
        ):
            issues.append("reused_anchor")
    proposal: Proposal | None = None
    if not issues:
        if arm == "qfirst":
            question_prompt = generated
        else:
            assert anchor is not None and anchor.answer is not None and anchor.evidence_quote is not None
            question_prompt = prompt_for_answer_first(passage, anchor.answer, anchor.evidence_quote)
        proposal = Proposal.model_validate(
            (await call_or_load(model, question_prompt, Proposal, GENERATION, stage_uri(arm)))["value"]
        )
        issues = candidate_issues(passage, proposal, anchor=anchor.answer if anchor else None)
    if not issues and proposal is not None and proposal.item is not None:
        item = proposal.item
        review = BlindReview.model_validate(
            (
                await call_or_load(
                    model, prompt_for_review(passage, item), BlindReview, REVIEW, stage_uri(arm + "-review")
                )
            )["value"]
        )
        issues = review_issues(passage, proposal, review)
    if not issues and proposal is not None and proposal.item is not None:
        fit = IntroFit.model_validate(
            (await call_or_load(model, prompt_for_intro_fit(proposal.item), IntroFit, REVIEW, stage_uri(arm + "-fit")))[
                "value"
            ]
        )
        if not fit.fits_intro_biology:
            issues.append("outside_intro_biology")
    match: str | None = None
    status = "quarantined"
    if not issues and proposal is not None and proposal.item is not None:
        status = "screened_novel"
        if progress.novel:
            judge = CollisionReview.model_validate(
                (
                    await call_or_load(
                        model,
                        collision_prompt(proposal.item, progress.novel),
                        CollisionReview,
                        REVIEW,
                        attempt_root + "collision.json",
                    )
                )["value"]
            )
            if collision_review_issue(judge, progress.novel):
                issues.append("invalid_collision_review")
                status = "quarantined"
            elif judge.matched_id is not None:
                match = judge.matched_id
                status = "screened_collision"
    identity = f"{passage_id}-{arm}-{number:02d}"
    result: dict[str, object] = {
        "id": identity,
        "revision": REVISION,
        "arm": arm,
        "attempt": number,
        "status": status,
        "issues": issues,
        "matched_id": match,
        "proposal_uri": stage_uri(arm) if proposal is not None else None,
        "prior_novel_ids": [old.id for old in progress.novel],
    }
    write_once(attempt_root + "summary.json", result)
    apply_decision(
        progress, arm, status, ScreenedItem(identity, proposal.item) if status == "screened_novel" else None, threshold
    )
    return result


async def run_passage(
    row: dict[str, object],
    port: OpenAIModelPort,
    destination: str,
    prior_root: str,
    smoke_root: str,
    fit_smoke_root: str,
    *,
    completed_pilot_root: str | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    collision_threshold: int = COLLISIONS_TO_STOP,
) -> dict[str, object]:
    """Try paired arms until both saturate or their total budget is exhausted."""
    if max_attempts < 2 or max_attempts > MAX_ATTEMPTS or max_attempts % 2 or collision_threshold < 1:
        raise ValueError("Use a positive collision threshold and an even total attempt cap of 2..20")
    passage = str(row["text"])
    if hashlib.sha256(passage.encode()).hexdigest() != row["passage_sha256"]:
        raise ValueError("Passage digest mismatch")
    passage_id = str(row["id"])
    root = destination.rstrip("/") + "/" + passage_id + "/"
    completed = load(root + "summary.json")
    if completed is None and completed_pilot_root is not None:
        completed = load(completed_pilot_root.rstrip("/") + "/" + passage_id + "/summary.json")
    if completed is not None:
        if (
            completed["revision"] != REVISION
            or completed["id"] != passage_id
            or completed["source_url"] != row["source_url"]
            or completed["source_family"] != row["source_family"]
            or completed["model"] != port.model
            or completed["max_attempts"] != max_attempts
            or completed["collision_threshold"] != collision_threshold
        ):
            raise ValueError(f"Incompatible completed adaptive passage: {passage_id}")
        return completed
    # Reuse is legitimate only for a completed frozen paired run on this passage.
    original = load(prior_root.rstrip("/") + "/" + passage_id + "/summary.json") or load(
        smoke_root.rstrip("/") + "/" + passage_id + "/summary.json"
    )
    if original is None or original["id"] != passage_id or original["source_url"] != row["source_url"]:
        raise ValueError(f"No matching frozen v2 paired run: {passage_id}")
    progress = PassageProgress()
    token = os.environ[port.api_key_env]
    async with GlmClient(endpoint_source(port, token), provider_config(port)) as model:
        for number in range(max_attempts // 2):
            for arm in ARMS:
                if progress.arms[arm].stopped:
                    continue
                result = await one_attempt(
                    model,
                    passage,
                    passage_id,
                    arm,
                    number,
                    progress,
                    root,
                    prior_root,
                    smoke_root,
                    fit_smoke_root,
                    collision_threshold,
                )
                print(f"{passage_id} {arm} {number + 1} {result['status']} match={result['matched_id']}", flush=True)
            if all(progress.arms[arm].stopped for arm in ARMS):
                break
    outcome: dict[str, object] = {
        "id": passage_id,
        "revision": REVISION,
        "source_url": row["source_url"],
        "source_family": row["source_family"],
        "model": port.model,
        "max_attempts": max_attempts,
        "collision_threshold": collision_threshold,
        "arms": {
            arm: {
                "attempts": progress.arms[arm].attempts,
                "collision_streak": progress.arms[arm].collision_streak,
                "stop_reason": "collision_threshold" if progress.arms[arm].stopped else "attempt_cap",
            }
            for arm in ARMS
        },
        "screened_distinct_ids": [item.id for item in progress.novel],
        "status": "screened_training_candidates_not_validated",
    }
    write_once(root + "summary.json", outcome)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prior-prefix", required=True)
    parser.add_argument("--old-smoke-prefix", required=True)
    parser.add_argument("--fit-smoke-prefix", required=True)
    parser.add_argument("--indices", help="Distinct frozen passage offsets; pin these for a pilot")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--completed-pilot-prefix", help="Reuse complete v3 pilot passages in the full run")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--collision-threshold", type=int, default=COLLISIONS_TO_STOP)
    parser.add_argument("--base-url")
    parser.add_argument("--relay-job", default="/muchanem/glm53-relay-tok-02a")
    parser.add_argument("--relay-service", default="glm-5.3-tok")
    args = parser.parse_args()
    port = OpenAIModelPort(
        GLM_MODEL,
        GLM_BULK_TOKEN_ENV,
        base_url=args.base_url,
        relay_job=None if args.base_url else args.relay_job,
        relay_service=None if args.base_url else args.relay_service,
    )
    rows = read_rows(args.input)
    if (args.indices is None) == (args.shard_count is None):
        raise ValueError("Specify either --indices or --shard-index/--shard-count")
    if args.shard_count is not None:
        if args.shard_count < 1 or args.shard_index is None or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("Invalid shard index/count")
        indices = [i for i in range(len(rows)) if i % args.shard_count == args.shard_index]
    else:
        indices = [int(index) for index in args.indices.split(",")]
    if len(indices) != len(set(indices)) or any(i < 0 or i >= len(rows) for i in indices):
        raise ValueError("Passage offsets must be distinct and in range")
    for index in indices:
        outcome = asyncio.run(
            run_passage(
                rows[index],
                port,
                args.output,
                args.prior_prefix,
                args.old_smoke_prefix,
                args.fit_smoke_prefix,
                completed_pilot_root=args.completed_pilot_prefix,
                max_attempts=args.max_attempts,
                collision_threshold=args.collision_threshold,
            )
        )
        print(json.dumps(outcome), flush=True)


if __name__ == "__main__":
    main()
