# Basic biology: paired GLM question-first/answer-first pilot

**Status: experimental training candidates, not a validated exam or eval set.**
Frozen source: private government HTML archive `raw-html-v1-20261108` from
[the source inventory PR](https://github.com/marin-community/marin/pull/11106).
`prepare.py` checks its manifest SHA-256
`9f880299f679ec033b6dd68b422266b7f04a9ecc9ca38ef748b8e6f855b784cd`,
extracts text-only article bodies from MedlinePlus Genetics and NHGRI, omits
media, navigation, references, and images, then splits at paragraph boundaries.
It rejects extracted page-level copyright markers; NHGRI's public-domain-
unless-marked policy still requires spot checks before training. The resulting
**195 passages from 126 articles** have JSONL SHA-256
`b492e8e93a913e383245c8d290ba90c5abaaa7f4931522e2c9e07fae7ebc7d5d`.
Neither the raw HTML nor generated output goes in git.

Two generation arms use the **same passages** and GLM-5.3 bulk endpoint:

1. **Question-first:** ask for one self-contained, source-grounded, four-option
   MCQ or an explicit no-task decision. Bonito's text-conditional task
   generation motivates the shape, but this is a *prompted GLM adaptation*,
   **not** a Bonito reproduction.
2. **Answer-first:** choose a verbatim short answer and supporting quotation
   from the passage, then condition a second call on that answer to write a
   four-option MCQ. This adapts the answer-first/round-trip synthetic-QA
   approach of Alberti et al. (2019); it is not a reproduction of their model.

A separate, blinded GLM call sees passage, question, and options but **not the
proposed key**; it chooses its own answer and checks source support, multiple
defensible choices, and missing assumptions. Mechanical checks enforce exact
source quotes, four distinct options, and the answer-first anchor. All provider
outputs are private, retained per call, and resumable. Screening agreement is
**not independent biological validation**, especially with the same model in
both roles. OpenRouter is a later, separately frozen validation wave, not a
silent replacement for missing GLM responses.

Input/artifact locations (CoreWeave S3, co-located with GLM relay):

```text
s3://marin-us-east-02a/marin4sc/taskforge/source-corpora/us-gov-biology/raw-html-v1-20261108/
s3://marin-us-east-02a/marin4sc/taskforge/source-corpora/us-gov-biology/passages-v1/passages.jsonl
s3://marin-us-east-02a/marin4sc/taskforge/task-library/campaigns/basic-biology/dual-mcq-glm-v1/smoke-1/
```

The smoke used `--indices 0,1,122,123` to cover both source families.
Iris job `/zack/basic-biology-dual-glm-smoke-v1a` succeeded on 2026-11-09
in 4m44s (20 GLM calls for four passages). All eight generation proposals
were structurally complete; the blind GLM review screened six and quarantined
two (one reviewer used straight rather than typographic apostrophes in its
source quote; another identified two unstated assumptions). Two of the eight
items concern DNA sequencing technology, which is likely too specialized for
a core Bio 101 bank even though its sources are official. **None is certified
biologically correct.** The first attempted federated root job was rejected
before execution because the launcher omitted CoreWeave object-storage keys;
job `v1a` supplied them and ran successfully. Raw proposals, reviewers,
completion metadata, and per-call stage records remain private in `smoke-1/`.

Version 2 adds *only* a typography/whitespace-normalized exact evidence-quote
check and a separate GLM introductory-biology-fit review. The frozen v1
responses and statuses remain untouched. `--reuse-prefix .../smoke-1` reads
all completed generation/review calls from v1 and issues only the new fit calls
on the four smoke passages. For the full wave `--shard-index N --shard-count 8`
assigns each passage to exactly one root job; `--output .../full-v2` is a
separate immutable result root. Every passage is attempted **once per arm**,
so 195 passages can yield at most 390 candidate MCQs before screening and
no-task decisions. The full wave needs at most ~1,365 calls including the new
fit review (fewer for rejected/no-task items), minus reused smoke calls; no
completed call is repeated on a resume. Full-wave aggregates must be checked
for source-family and arm yield, concept duplicates, and answer-key imbalance.

## First full scale-up (GLM v2, 2026-11-09)

Eight separate Iris CoreWeave root jobs `/zack/basic-biology-dual-glm-full-v2-s0`
through `-s7` finished successfully. Each interleaved shard owns one eighth
of the 195 frozen passages; the four v1 smoke passages reused all 20 original
generation/review calls, and the v2 smoke fit calls were also reused. Across
all three disjoint result roots, **1,240 distinct GLM calls** were recorded;
there were no repeated completed calls. Finalized private snapshot:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/campaigns/basic-biology/dual-mcq-glm-v2/snapshots/complete-1/
```

`aggregate.json` SHA-256:
`cc5b9007e6045d672037cac2cedd52db48c0616688e705114f64bdedc66379ef`.
Private screened-candidate JSONL SHA-256:
`f093d21c0c4d62af8f001ae9bbb786664ec5deca2e2aeda04743e210b482f2bd`.
The finalizer freezes the manifest last and verifies a summary for each of
195 distinct passages, both assigned arms, and every screened proposal.

| Arm | Screened | Quarantined |
| --- | ---: | ---: |
| Question-first | 81 | 114 |
| Answer-first | 103 | 92 |
| **Total** | **184** | **206** |

Per-passage *screened* yield: **75 with zero, 56 with one, 64 with two**.
The main quarantine reason was outside-intro-biology (128 item-arm records),
followed by unstated assumptions (39) and source-quote mismatch (15). These
categories can co-occur. MedlinePlus contributed 118 of 244 screened arm
attempts (50 question-first, 68 answer-first); NHGRI 66 of 146 (31 and 35).
Question-first/answer-first differences are descriptive, not causal: they
also differ in generation-call count and anchoring constraints.

**Critical pre-training findings:** Generated correct-option positions are
strongly biased: among the 184 screened items positions 0/1/2/3 occur
**84/76/22/2** times. Any future training projection must shuffle options
and remap the key using a frozen deterministic rule. A spot-check also found
some *accepted* questions about cloning-history trivia, medical genomics,
intelligence tests and sequencing technology: the GLM Bio-101-fit screen is
not a reliable final content gate. These 184 are **only provisional screened
candidates**, not verified scientific references, accepted RL tasks, or a
benchmark. No OpenRouter calls or held-out eval comparisons have been made.
Missing/infrastructure/error states are not scored as zero or imputed. For an
Iris root job, use an authenticated CoreWeave launch and supply `GLM_BULK_TOKEN`
via Iris env/secret injection, not source files. All generated items are for
*training*, segregated from BioTasks and other held-out evaluation sets.

## Adaptive novelty pilot (v3; in progress)

`adaptive.py` tests whether passage-level yield can improve without setting a
fixed number of accepted tasks. For each passage, interleave the two arms for
**at most 20 total attempts** (up to 10 per arm), stopping an arm early after
three consecutive *screened* semantic collisions with previously distinct
screened questions from **either** arm. A failed generation, screening
quarantine, or repeated answer anchor neither increments nor resets the
collision streak. A fresh screened learning objective resets it. A GLM judge
compares question, answer, and evidence against all previous distinct screened
objectives from the same passage, including reversed/paraphrased questions;
this is **a model-based novelty heuristic, not a validated semantic metric or
scientific certification**. A hard cap limits false-negative collisions and
obscure trivia; low collision frequency does not prove that the source is
exhausted. Across passages, duplicate concepts still require a separate audit.

The first pair of attempts reuses *existing* v2 generation/review/fit calls
(the corresponding v1 and v2 smoke calls where applicable), never mutating
those records. Later prompts explicitly list previously covered objectives.
Every stage and collision decision is persisted independently, under a
**new v3 prefix**. If provider calls fail, the passage remains incomplete;
relaunching reads the completed stage records and rejects changed prompts.
No model-screened item is automatically promoted into training or held-out
evaluation, and the v2 snapshot remains frozen.

Frozen ten-passage pilot offsets:
`86,63,46,51,120,150,169,136,189,131`. They cover five passages from each
source family and, within each family, two passages with zero, one with one,
and two with two v2 screened candidates. Passage lengths span roughly
1,100–3,100 characters. A pilot can have at most 200 attempts (plus reviews);
**do not launch all 195 passages before inspecting novelty judgments, question
quality, source rights and per-attempt yield**. Option-key bias from v2 still
requires a deterministic shuffle/key remap before any training export.
