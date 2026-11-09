"""Run paired, source-grounded GLM MCQ generation and blinded GLM screening.

Each stage is persisted independently under a private prefix. Resuming skips
completed calls instead of repeating them; no result is automatically accepted
as biologically validated training data.
"""

import argparse
import asyncio
import hashlib
import json
import os
import unicodedata
from dataclasses import asdict
from typing import Literal

import fsspec
from pydantic import BaseModel, ConfigDict, Field
from taskforge.llm.policy import LLMPolicy, ReasoningEffort
from taskforge_adapters.campaign import OpenAIModelPort, endpoint_source, provider_config
from taskforge_adapters.glm_client import GlmClient

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL

SOURCE_MANIFEST_SHA256 = "b492e8e93a913e383245c8d290ba90c5abaaa7f4931522e2c9e07fae7ebc7d5d"
REVISION = "gov-bio-dual-mcq-glm-v2"
GENERATION = LLMPolicy(max_tokens=2400, reasoning_effort=ReasoningEffort.LOW, temperature=0.3, top_p=0.95)
REVIEW = LLMPolicy(max_tokens=1800, reasoning_effort=ReasoningEffort.LOW, temperature=0, top_p=1)


class MCQ(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=20)
    options: tuple[str, str, str, str]
    correct_index: int = Field(ge=0, le=3)
    evidence_quote: str = Field(min_length=8)


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disposition: Literal["candidate", "no_task"]
    item: MCQ | None
    reason: str


class AnswerAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disposition: Literal["candidate", "no_task"]
    answer: str | None
    evidence_quote: str | None
    reason: str


class BlindReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_index: int | None
    supported_by_passage: bool
    other_defensible_indices: list[int]
    unsupported_assumptions: list[str]
    evidence_quote: str
    explanation: str


class IntroFit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fits_intro_biology: bool
    rationale: str


def prompt_for_question_first(passage: str) -> str:
    """Generate exactly one passage-grounded question, or an explicit no-task."""
    return (
        "Create ONE simple but nontrivial four-option introductory biology MCQ based only on the source passage. "
        "The question must be self-contained: include any needed premise in the stem, "
        "but do not copy an entire source paragraph. "
        "One answer must be unambiguously correct and all three distractors clearly false under the stated conditions. "
        "If the passage cannot support such a question, set disposition=no_task, item=null, and explain why. "
        "For a candidate give 0-based correct_index and an evidence_quote copied verbatim from the passage; "
        "avoid questions about extraneous citations, publication dates, or medical advice.\n\n"
        f"SOURCE PASSAGE:\n{passage}"
    )


def prompt_for_anchor(passage: str) -> str:
    """Select the answer before writing any question, following answer-first QA."""
    return (
        "Select one short, unambiguous biology term or phrase (2-80 characters) that appears verbatim in the "
        "source passage and would be a good answer to a basic biology question. Include an evidence_quote "
        "copied verbatim from the passage that explains the answer. Do not write the question yet. "
        "If no suitable anchor exists, set disposition=no_task and answer/evidence_quote=null.\n\n"
        f"SOURCE PASSAGE:\n{passage}"
    )


def prompt_for_answer_first(passage: str, answer: str, quote: str) -> str:
    """Write a question around a predetermined passage-supported answer."""
    return (
        "Create ONE self-contained introductory biology four-option MCQ from the passage. "
        f"The predetermined correct answer is EXACTLY {json.dumps(answer)}. Put this exact string in ONE option, "
        "and put its 0-based index in correct_index. Invent three clearly wrong, yet plausible, alternatives. "
        "Do not reveal the answer in the question stem. Keep the scientific conditions explicit. "
        "For a candidate give an evidence_quote copied verbatim from the passage. "
        f"The original anchor was supported by: {json.dumps(quote)}. "
        "If this cannot be done unambiguously set disposition=no_task, item=null and explain why.\n\n"
        f"SOURCE PASSAGE:\n{passage}"
    )


def prompt_for_review(passage: str, item: MCQ) -> str:
    """Do not reveal the candidate's answer key or source quote to the solver."""
    return (
        "You are an adversarial science-question reviewer. Independently answer the question using only the passage. "
        "Return the best 0-based selected_index (null if unanswerable). Check whether the passage actually supports "
        "the choice, list EVERY other defensible option index, and list any missing biological assumptions. "
        "Give one VERBATIM evidence_quote from the passage. Be skeptical of ambiguity and misleading distractors.\n\n"
        f"SOURCE PASSAGE:\n{passage}\n\nQUESTION:\n{item.question}\n\n"
        + "\n".join(f"{i}: {option}" for i, option in enumerate(item.options))
    )


def _quote_matches(passage: str, quote: str) -> bool:
    """Match verbatim evidence except typographic punctuation and whitespace."""
    punctuation = str.maketrans(
        {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"}
    )

    def normalize(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).translate(punctuation).split())

    return normalize(quote) in normalize(passage)


def prompt_for_intro_fit(item: MCQ) -> str:
    """Filter specialist biomedical and technical items without a hand-built taxonomy."""
    return (
        "Is this question appropriate for a general introductory biology (Bio 101) course? "
        "Accept core ideas in cells, molecular biology, heredity, evolution, organisms, and ecology. "
        "Reject niche clinical diagnostics, specialized laboratory technologies, research trivia, "
        "and source-specific names. "
        "Judge only curricular fit, not whether the proposed answer is correct. Give a brief rationale.\n\n"
        f"QUESTION: {item.question}\nOPTIONS:\n" + "\n".join(f"{i}: {option}" for i, option in enumerate(item.options))
    )


def candidate_issues(passage: str, proposal: Proposal, *, anchor: str | None = None) -> list[str]:
    """Check machine-verifiable shape and direct provenance only."""
    if proposal.disposition == "no_task":
        return ["no_task"] if proposal.item is None else ["invalid_no_task_with_item"]
    if proposal.item is None:
        return ["missing_candidate"]
    item = proposal.item
    issues: list[str] = []
    if len({s.strip().casefold() for s in item.options}) != 4:
        issues.append("duplicate_options")
    if not _quote_matches(passage, item.evidence_quote):
        issues.append("quote_not_in_passage")
    if anchor is not None and item.options[item.correct_index] != anchor:
        issues.append("wrong_anchor")
    if anchor is not None and anchor.casefold() in item.question.casefold():
        issues.append("answer_leaked_in_stem")
    return issues


def review_issues(passage: str, proposal: Proposal, review: BlindReview) -> list[str]:
    """Treat model agreement as screening evidence, not scientific validation."""
    item = proposal.item
    if item is None:
        raise ValueError("Cannot review absent candidate")
    issues = candidate_issues(passage, proposal)
    if review.selected_index != item.correct_index:
        issues.append("blind_solver_disagrees")
    if not review.supported_by_passage or not _quote_matches(passage, review.evidence_quote):
        issues.append("review_evidence_missing")
    if review.other_defensible_indices:
        issues.append("multiple_defensible_answers")
    if review.unsupported_assumptions:
        issues.append("unstated_assumptions")
    return issues


def write_once(uri: str, payload: dict[str, object]) -> None:
    """Never overwrite a completed call; its response is an immutable record."""
    data = (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode()
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        with fs.open(path, "rb") as file:
            if file.read() != data:
                raise ValueError(f"Existing stage differs: {uri}")
        return
    with fs.open(path, "wb") as file:
        file.write(data)


def load(uri: str) -> dict[str, object] | None:
    fs, path = fsspec.core.url_to_fs(uri)
    if not fs.exists(path):
        return None
    with fs.open(path, "rt") as file:
        return json.load(file)


def _stage_uri(root: str, prior_root: str | None, name: str) -> str:
    """Reuse a previously completed call without reissuing it or altering its bytes."""
    if prior_root is not None and load(prior_root + name + ".json") is not None:
        return prior_root + name + ".json"
    return root + name + ".json"


async def call_or_load(
    model: GlmClient, prompt: str, schema: type[BaseModel], policy: LLMPolicy, uri: str
) -> dict[str, object]:
    """Persist each individual provider call before moving to the next stage."""
    cached = load(uri)
    if cached is not None:
        if cached["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError(f"Prompt changed for frozen call: {uri}")
        return cached
    answer = await model.complete_structured([{"role": "user", "content": prompt}], schema, policy)
    record: dict[str, object] = {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "policy": asdict(policy),
        "value": answer.value.model_dump(mode="json"),
        "completions": [asdict(part) for part in answer.completions],
    }
    write_once(uri, record)
    return record


async def run_one(
    row: dict[str, object], port: OpenAIModelPort, destination: str, *, prior_destination: str | None = None
) -> dict[str, object]:
    """Run both paired arms, saving all generation and review calls independently."""
    text = str(row["text"])
    if hashlib.sha256(text.encode()).hexdigest() != row["passage_sha256"]:
        raise ValueError("Passage digest mismatch")
    root = destination.rstrip("/") + "/" + str(row["id"]) + "/"
    prior_root = prior_destination.rstrip("/") + "/" + str(row["id"]) + "/" if prior_destination else None
    token = os.environ[port.api_key_env]
    async with GlmClient(endpoint_source(port, token), provider_config(port)) as model:
        q = Proposal.model_validate(
            (
                await call_or_load(
                    model, prompt_for_question_first(text), Proposal, GENERATION, _stage_uri(root, prior_root, "qfirst")
                )
            )["value"]
        )
        a = AnswerAnchor.model_validate(
            (
                await call_or_load(
                    model, prompt_for_anchor(text), AnswerAnchor, GENERATION, _stage_uri(root, prior_root, "anchor")
                )
            )["value"]
        )
        anchor_issues: list[str] = []
        if a.disposition == "candidate":
            if not a.answer or not a.evidence_quote or a.answer not in text or a.evidence_quote not in text:
                anchor_issues.append("unproven_anchor")
        else:
            anchor_issues.append("no_anchor")
        second: Proposal | None = None
        if not anchor_issues:
            assert a.answer is not None and a.evidence_quote is not None
            second = Proposal.model_validate(
                (
                    await call_or_load(
                        model,
                        prompt_for_answer_first(text, a.answer, a.evidence_quote),
                        Proposal,
                        GENERATION,
                        _stage_uri(root, prior_root, "afirst"),
                    )
                )["value"]
            )
        outcomes: dict[str, object] = {}
        for name, proposal in (("qfirst", q), ("afirst", second)):
            issues = (
                candidate_issues(text, proposal, anchor=a.answer if name == "afirst" else None)
                if proposal
                else anchor_issues
            )
            if proposal is not None and not issues and proposal.item is not None:
                review = BlindReview.model_validate(
                    (
                        await call_or_load(
                            model,
                            prompt_for_review(text, proposal.item),
                            BlindReview,
                            REVIEW,
                            _stage_uri(root, prior_root, name + "-review"),
                        )
                    )["value"]
                )
                issues = review_issues(text, proposal, review)
            if proposal is not None and not issues and proposal.item is not None:
                fit = IntroFit.model_validate(
                    (
                        await call_or_load(
                            model, prompt_for_intro_fit(proposal.item), IntroFit, REVIEW, root + name + "-fit.json"
                        )
                    )["value"]
                )
                if not fit.fits_intro_biology:
                    issues.append("outside_intro_biology")
            outcomes[name] = {"status": "screened_candidate" if not issues else "quarantined", "issues": issues}
        result: dict[str, object] = {
            "id": row["id"],
            "source_url": row["source_url"],
            "revision": REVISION,
            "model": port.model,
            "outcomes": outcomes,
        }
        write_once(root + "summary.json", result)
        return result


def read_rows(uri: str) -> list[dict[str, object]]:
    """Require the exact frozen source-text input before making provider calls."""
    with fsspec.open(uri, "rb") as source:
        data = source.read()
    if hashlib.sha256(data).hexdigest() != SOURCE_MANIFEST_SHA256:
        raise ValueError("Wrong source-passage snapshot")
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--indices", help="Comma-separated frozen passage offsets for a stratified smoke run")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--reuse-prefix", help="Read completed v1 calls in place without repeating them")
    parser.add_argument("--base-url")
    parser.add_argument("--relay-job", default="/muchanem/glm53-relay-tok-02a")
    parser.add_argument("--relay-service", default="glm-5.3-tok")
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    port = OpenAIModelPort(
        GLM_MODEL,
        GLM_BULK_TOKEN_ENV,
        base_url=args.base_url,
        relay_job=None if args.base_url else args.relay_job,
        relay_service=None if args.base_url else args.relay_service,
    )
    full_rows = read_rows(args.input)
    if args.indices and args.shard_count is not None:
        raise ValueError("Specify either --indices or shard placement")
    if args.shard_count is not None:
        if args.shard_count < 1 or args.shard_index is None or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("Invalid shard index/count")
        indices = [i for i in range(len(full_rows)) if i % args.shard_count == args.shard_index]
    else:
        indices = [int(value) for value in args.indices.split(",")] if args.indices else list(range(args.limit))
    if len(indices) != len(set(indices)) or any(index < 0 or index >= len(full_rows) for index in indices):
        raise ValueError("Passage offsets must be distinct and in range")
    rows = [full_rows[index] for index in indices]
    for index, row in enumerate(rows, 1):
        result = asyncio.run(run_one(row, port, args.output, prior_destination=args.reuse_prefix))
        print(f"{index}/{len(rows)} {result['id']} {result['outcomes']}", flush=True)


if __name__ == "__main__":
    main()
