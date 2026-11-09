"""Independent MCQ screening and provenance guards."""

from pathlib import Path

import pytest

from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.adaptive import (
    PassageProgress,
    ScreenedItem,
    apply_decision,
    first_stage_uri,
)
from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.finalize import read_json, write_new
from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.prepare import extract_body
from experiments.post_training.capability_driven_envs.basic_biology_task_pilot.run_glm import (
    MCQ,
    BlindReview,
    Proposal,
    _stage_uri,
    candidate_issues,
    prompt_for_review,
    review_issues,
    write_once,
)


def test_extract_stops_before_external_references_and_omits_images() -> None:
    html = b"""<article><h1>What is DNA?</h1><p>A base pairs with T.</p>
    <figure><figcaption>Third-party copyrighted image</figcaption></figure>
    <h2>Scientific journal articles for further reading</h2>
    <p>External copyrighted paper.</p></article>"""
    assert extract_body(html, source="medlineplus") == ["What is DNA?", "A base pairs with T."]


def test_blinded_review_hides_reference_key_and_quote() -> None:
    item = MCQ(
        question="Which base pairs with adenine in DNA under standard base pairing?",
        options=("guanine", "cytosine", "thymine", "uracil"),
        correct_index=2,
        evidence_quote="A with T",
    )
    prompt = prompt_for_review("A with T", item)
    assert "correct_index" not in prompt
    assert "A with T" in prompt  # Source visible, generated evidence label not revealed.
    assert "best 0-based selected_index" in prompt


def test_final_snapshot_refuses_mutating_existing_artifacts(tmp_path: Path) -> None:
    uri = str(tmp_path / "aggregate.json")
    write_new(uri, b'{"ok":true}\n')
    write_new(uri, b'{"ok":true}\n')
    assert read_json(uri)[0] == {"ok": True}
    with pytest.raises(ValueError, match="Frozen output collision"):
        write_new(uri, b'{"ok":false}\n')


def test_version_two_reuses_completed_calls_without_mutating_old_records(tmp_path: Path) -> None:
    (tmp_path / "smoke").mkdir()
    prior = str(tmp_path / "smoke") + "/"
    current = str(tmp_path / "full") + "/"
    write_once(prior + "qfirst.json", {"value": "frozen"})
    assert _stage_uri(current, prior, "qfirst") == prior + "qfirst.json"
    assert _stage_uri(current, prior, "afirst") == current + "afirst.json"


def test_semantic_collision_stop_ignores_rejections_and_is_per_arm() -> None:
    item = MCQ(
        question="In DNA, which base pairs with adenine under standard base pairing?",
        options=("thymine", "uracil", "guanine", "cytosine"),
        correct_index=0,
        evidence_quote="A pairs with T",
    )
    progress = PassageProgress()
    apply_decision(progress, "qfirst", "screened_novel", ScreenedItem("initial", item), 3)
    for status in ("screened_collision", "quarantined", "screened_collision"):
        apply_decision(progress, "qfirst", status, None, 3)
    assert not progress.arms["qfirst"].stopped
    assert progress.arms["qfirst"].collision_streak == 2
    apply_decision(progress, "qfirst", "screened_collision", None, 3)
    assert progress.arms["qfirst"].stopped
    assert not progress.arms["afirst"].stopped
    assert [candidate.id for candidate in progress.novel] == ["initial"]
    apply_decision(progress, "afirst", "screened_novel", ScreenedItem("later", item), 3)
    assert progress.arms["afirst"].collision_streak == 0
    assert len(progress.novel) == 2


def test_first_attempt_prefers_frozen_sources_over_new_calls(tmp_path: Path) -> None:
    prior = str(tmp_path / "full")
    smoke = str(tmp_path / "smoke")
    fit = str(tmp_path / "fit")
    (tmp_path / "smoke" / "p").mkdir(parents=True)
    (tmp_path / "fit" / "p").mkdir(parents=True)
    write_once(smoke + "/p/qfirst.json", {"value": "unchanged"})
    write_once(fit + "/p/qfirst-fit.json", {"value": "unchanged"})
    assert first_stage_uri("p", "qfirst", prior, smoke, fit) == smoke + "/p/qfirst.json"
    assert first_stage_uri("p", "qfirst-fit", prior, smoke, fit) == fit + "/p/qfirst-fit.json"


def test_candidate_and_review_disagree_or_detect_ambiguity() -> None:
    text = "Adenine (A) pairs with thymine (T)."
    proposal = Proposal(
        disposition="candidate",
        item=MCQ(
            question="Under standard DNA base pairing, which base pairs with adenine?",
            options=("cytosine", "guanine", "thymine", "uracil"),
            correct_index=2,
            evidence_quote="pairs with thymine",
        ),
        reason="source-backed",
    )
    assert candidate_issues(text, proposal) == []
    review = BlindReview(
        selected_index=2,
        supported_by_passage=True,
        other_defensible_indices=[1],
        unsupported_assumptions=[],
        evidence_quote="pairs with thymine",
        explanation="An alternative was claimed defensible.",
    )
    assert review_issues(text, proposal, review) == ["multiple_defensible_answers"]
    punctuation = "The cell\u2019s nucleus contains DNA."
    assert (
        candidate_issues(
            punctuation,
            proposal.model_copy(update={"item": proposal.item.model_copy(update={"evidence_quote": "cell's nucleus"})}),
        )
        == []
    )
