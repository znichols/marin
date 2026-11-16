# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project model-screened private biology MCQs into provisional TaskForge TaskSpecs.

This is a structural import, not scientific acceptance, source-rights clearance,
or training export. The full review must exist for all 840 candidates before the
projection writes anything. Only use a fresh, private output prefix.
"""

import argparse
import asyncio
import hashlib
import json
import random
from collections import Counter
from typing import Any

import fsspec
from shellbox.machine import CommandSemantics
from taskcompendium.grader import verifyit_package
from taskcompendium.grading import grade_answer
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    JsonValueAnswer,
    ResourceGroups,
    Source,
    TaskSpec,
    TextMessage,
)
from taskforge.content_hash import digest
from taskforge.stage10_proposal.model import (
    BuildItem,
    Environment,
    Grounding,
    ProposalHeader,
    ResearchNeed,
    SolverMachine,
    SourceKind,
    SourceRef,
    TaskProposal,
    Verification,
    parse,
    render,
)
from taskforge.stage20_triage.preflight import preflight
from verifyit.spec import StructuredExactSpec

from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.finalize import read_json, write_new
from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.review_full import (
    CANDIDATE_SHA256,
    ScienceReview,
    disposition,
    load_candidates,
    review_prompt,
)
from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.review_full import (
    REVISION as REVIEW_REVISION,
)
from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.run_glm import (
    MCQ,
    SOURCE_MANIFEST_SHA256,
    read_rows,
)

REVISION = "gov-bio-mcq-taskforge-provisional-v2"
DESK_REVIEW_SHA256 = "2afab1c70a07bdcb710045a2738e648fecb17aa3e7b887de23ec154153cd6f87"
DUPLICATE_REVIEW_SHA256 = "32d507952ee78ba2a03d56537ee1bddacff5af11252b5078877094fa73abf04b"


def shuffled_prompt(item_id: str, item: MCQ) -> tuple[str, str, list[int]]:
    """Return a deterministic option order, remapped letter, and explicit mapping."""
    order = list(range(4))
    seed = int(hashlib.sha256((REVISION + ":" + item_id).encode()).hexdigest(), 16)
    random.Random(seed).shuffle(order)
    key = chr(65 + order.index(item.correct_index))
    prompt = (
        item.question
        + "\n\n"
        + "\n".join(f"{chr(65 + index)}. {item.options[original]}" for index, original in enumerate(order))
        + '\n\nReturn only a JSON object with a single key "choice" containing A, B, C, or D.'
    )
    return prompt, key, order


def provisional_task(record: dict[str, Any]) -> tuple[TaskSpec, dict[str, Any]]:
    """Build an explicitly unapproved TaskSpec and preserve option-key provenance."""
    item = MCQ.model_validate(record["item"])
    prompt, key, order = shuffled_prompt(record["id"], item)
    package = verifyit_package(StructuredExactSpec(expected={"choice": key}))
    task = TaskSpec(
        id="private-gov-bio-" + hashlib.sha256((REVISION + ":" + record["id"]).encode()).hexdigest()[:24],
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell",), command_semantics=CommandSemantics.LINUX_PROCESS, working_directory="/workspace"
        ),
        answer_type=AnswerType.JSON,
        answer_format=JsonValueAnswer(),
        grader=package.grader,
        source=Source(
            dataset="us-gov-biology-adaptive-glm-v3-private-training-leads",
            revision=record["passage_sha256"],
            row=record["id"],
            importer_revision=REVISION,
        ),
        resources=ResourceGroups(verifier=package.resources),
        tags=("biology", "intro-biology", "private-model-screened", "rights-unreviewed", "not-training-ready"),
    )
    mapping = {
        "candidate_id": record["id"],
        "task_id": task.id,
        "task_digest": digest(task),
        "original_correct_index": item.correct_index,
        "shuffled_to_original_indices": order,
        "shuffled_correct_letter": key,
        "source_page_sha256": record["source_page_sha256"],
        "passage_sha256": record["passage_sha256"],
        "rights_cleared": False,
        "expert_validated": False,
    }
    return task, mapping


def proposal_for(task: TaskSpec, record: dict[str, Any]) -> TaskProposal:
    """Build a full TaskForge proposal for deterministic preflight (not review acceptance)."""
    sections = {
        "Task": (
            "The private candidate proposes an introductory biology question with four lettered answers. "
            "The solver selects a single answer and returns its letter in a JSON object. Frozen provenance "
            "links the question to a specific archived government page and passage. This proposal tests "
            "only that the provisional MCQ can be represented and checked by TaskForge; neither the "
            "LLM review nor a passing structural check makes it scientifically validated or training-ready."
        ),
        "Realism and workflow": (
            "The solver sees a self-contained, source-grounded multiple-choice question, four distinct choices, "
            "and a simple answer format. This is a prompt-only basic-biology reasoning workflow, not a "
            "laboratory experiment or an open-ended agentic investigation. The task source page, extracted "
            "passage, proposal, and screening records have immutable identifiers, but only the question and "
            "options are visible to a solver. The private reference must remain hidden."
        ),
        "Research plan": (
            "No additional model generation, browsing, or retrieval is needed to build this task. The source "
            "passage was archived and content-hashed before the questions were written. A separate domain expert "
            "must assess biological truth, curriculum, ambiguity, and distractor quality. A rights reviewer must "
            "check the particular source page for third-party material before training. Neither human decision "
            "is present or inferred by this structural projection."
        ),
        "Build plan": (
            "Project a single item at its frozen identity. Shuffle the four options with a versioned hash seed, "
            "remap the correct letter, and record the entire old-to-new mapping and the source digest in a "
            "private companion artifact. Build a TaskSpec with a deterministic exact-match grader; retain the "
            "original candidate and reviewer records untouched. Reject any missing decision, inconsistent source, "
            "or changed bytes. Do not publish or use this result for training."
        ),
        "Grader design and controls": (
            "The executable VerifyIt StructuredExactSpec parses a terminal JSON object and matches its "
            "choice field to the remapped reference letter. The positive reference answer must grade one and "
            "a different letter must grade zero. These controls test the grader contract alone; an incorrectly "
            "keyed biology question would also pass them. The reviewer must assess that the reference is "
            "scientifically correct before a separate approval can occur."
        ),
        "Risks and null conditions": (
            "The additional GLM reviewer can miss subtle biology errors and specialist trivia, even when it "
            "agrees with the generator. Source-level duplication, third-party page rights, stale claims, and "
            "unhelpful distractors may still be present. A near-duplicate or wrong reference must be withheld. "
            "This provisional import never grants permission to train, evaluate, publish, or declare a task "
            "scientifically validated merely because structural preflight passes."
        ),
    }
    body = "\n\n".join(f"## {title}\n{text}" for title, text in sections.items())
    header = ProposalHeader(
        id=task.id,
        source=SourceRef(kind=SourceKind.REPO, ref=record["source_url"], hash=record["source_page_sha256"]),
        environment=Environment.REASONING,
        solver_machine=SolverMachine.NONE,
        verification=Verification.CODE,
        composite=(),
        grounding=Grounding.GROUNDED,
        research_need=ResearchNeed.NOT_NEEDED,
        research_reason="The frozen source passage is already captured; no new retrieval is needed.",
        research=(),
        build=(BuildItem(name="task", description="Provisional, private MCQ TaskSpec"),),
        resources=(),
        null_reason=None,
    )
    return parse(render(TaskProposal(header=header, body=body)))


def grade_controls(task: TaskSpec, key: str) -> tuple[float | None, float | None]:
    """Verify both reference and wrong choice against the actual TaskSpec grader."""

    def reward(choice: str) -> float | None:
        attempt = GradingAttempt(
            conversation=ConversationTrace(
                events=(*task.context.events, TextMessage(role="assistant", content=json.dumps({"choice": choice})))
            )
        )
        return grade_answer(task, attempt).reward

    return reward(key), reward(next(letter for letter in "ABCD" if letter != key))


async def build(args: argparse.Namespace) -> dict[str, Any]:
    candidates = load_candidates(args.candidates)
    passages = {row["id"]: row for row in read_rows(args.passages)}
    decisions: list[dict[str, Any]] = []
    # Fail closed before writing a single provisional task if even one review is missing.
    for record in candidates:
        uri = args.reviews.rstrip("/") + "/" + record["id"] + "/"
        decision, _ = read_json(uri + "decision.json")
        review, _ = read_json(uri + "review.json")
        item = MCQ.model_validate(record["item"])
        parsed = ScienceReview.model_validate(review["value"])
        passage = passages[record["id"].rsplit("-", 2)[0]]
        prompt_hash = hashlib.sha256(review_prompt(passage["text"], item).encode()).hexdigest()
        status, issues = disposition(item, parsed, passage["text"])
        if (
            decision["id"] != record["id"]
            or decision["revision"] != REVIEW_REVISION
            or decision["candidate_sha256"] != CANDIDATE_SHA256
            or decision["source_sha256"] != SOURCE_MANIFEST_SHA256
            or decision["status"] != status
            or decision["issues"] != issues
            or decision["review_uri"] != uri + "review.json"
            or review["prompt_sha256"] != prompt_hash
            or not review["completions"]
            or decision["rights_cleared"] is not False
            or decision["expert_validated"] is not False
            or record["passage_sha256"] != passage["passage_sha256"]
        ):
            raise ValueError(f"Changed or inconsistent review: {record['id']}")
        decisions.append(decision)
    if len(decisions) != 840:
        raise ValueError("Expected complete 840-item review ledger")

    with fsspec.open(args.desk_review, "rb") as file:
        desk_bytes = file.read()
    if hashlib.sha256(desk_bytes).hexdigest() != DESK_REVIEW_SHA256:
        raise ValueError("Desk review changed or missing")
    held = {row["id"]: row["disposition"] for row in map(json.loads, desk_bytes.splitlines())}
    if len(held) != 17 or not set(held).issubset({row["id"] for row in candidates}):
        raise ValueError("Unexpected desk-review identities")
    with fsspec.open(args.duplicate_review, "rb") as file:
        duplicate_bytes = file.read()
    if hashlib.sha256(duplicate_bytes).hexdigest() != DUPLICATE_REVIEW_SHA256:
        raise ValueError("Cross-passage duplicate review changed or missing")
    duplicate_pairs = [json.loads(row) for row in duplicate_bytes.splitlines()]
    if len(duplicate_pairs) != 3 or any(len(row["candidate_ids"]) != 2 for row in duplicate_pairs):
        raise ValueError("Unexpected cross-passage duplicate review")
    duplicate_holds = {item_id for row in duplicate_pairs for item_id in row["candidate_ids"]}
    if len(duplicate_holds) != 6 or not duplicate_holds.issubset({row["id"] for row in candidates}):
        raise ValueError("Unknown cross-passage duplicate candidate")
    statuses = Counter(decision["status"] for decision in decisions)
    selected = []
    projections = []
    for record, decision in zip(candidates, decisions, strict=True):
        if (
            decision["status"] != "model_screened_for_taskforge_probe"
            or record["id"] in held
            or record["id"] in duplicate_holds
        ):
            continue
        task, mapping = provisional_task(record)
        verdict = await preflight(proposal_for(task, record))
        positive, negative = grade_controls(task, mapping["shuffled_correct_letter"])
        if verdict.decision.value != "accept" or positive != 1.0 or negative != 0.0:
            raise ValueError(f"Structural TaskForge probe failed: {record['id']}: {verdict.reasons}")
        selected.append(task.model_dump_json() + "\n")
        projections.append({**mapping, "preflight": verdict.decision.value})

    root = args.output.rstrip("/") + "/"
    task_bytes = "".join(selected).encode()
    mapping_bytes = ("".join(json.dumps(row, sort_keys=True) + "\n" for row in projections)).encode()
    decision_bytes = ("".join(json.dumps(row, sort_keys=True) + "\n" for row in decisions)).encode()
    manifest = {
        "revision": REVISION,
        "source_candidate_sha256": CANDIDATE_SHA256,
        "source_passages_sha256": SOURCE_MANIFEST_SHA256,
        "status": "private_structurally_taskforge_compatible_not_validated_or_training_ready",
        "model_review_status_counts": dict(sorted(statuses.items())),
        "model_screened_provisional_tasks": len(selected),
        "desk_review_hold_ids": len(held),
        "desk_review_sha256": DESK_REVIEW_SHA256,
        "duplicate_review_sha256": DUPLICATE_REVIEW_SHA256,
        "cross_passage_duplicate_hold_ids": len(duplicate_holds),
        "task_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "mapping_sha256": hashlib.sha256(mapping_bytes).hexdigest(),
        "decision_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "scientific_validation": False,
        "page_level_rights_clearance": False,
    }
    write_new(root + "taskspecs.jsonl", task_bytes)
    write_new(root + "option-mappings.jsonl", mapping_bytes)
    write_new(root + "decisions.jsonl", decision_bytes)
    write_new(root + "manifest.json", (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--passages", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--desk-review", required=True, help="Frozen private desk-review overrides")
    parser.add_argument("--duplicate-review", required=True, help="Frozen cross-passage duplicate hold pairs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(build(args)), indent=2))


if __name__ == "__main__":
    main()
