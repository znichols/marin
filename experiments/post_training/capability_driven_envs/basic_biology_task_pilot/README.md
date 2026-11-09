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

The smoke should use `--indices 0,1,122,123` to cover both source families.
Do not run the full 195-passage wave until the smoke output and projected
provider volume are examined. A completed full wave needs up to ~975 calls
(two generation arms, an extra answer-anchor call, two blinded reviews per
passage); no call is repeated on a resume. Missing/infrastructure/error states
are not scored as zero or imputed. For an Iris root job, use an authenticated
CoreWeave launch and supply `GLM_BULK_TOKEN` via Iris env/secret injection, not
source files. All generated items are for *training*, segregated from BioTasks
and other held-out evaluation sets.
