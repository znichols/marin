# ID evaluation source intake: exclude obvious blockers, inspect while acquiring

This is **not** a way to import benchmark questions into TaskForge training.
[Marin's evaluation policy](https://github.com/marin-community/marin/issues/10197)
allows independently sourced synthetic tasks *modeled on* ID eval skills but
forbids using ID eval instructions directly as RL data or training on teacher
rollouts from eval questions. Acquisition here is **eval-only**: retain a
private, hash-pinned copy for overlap checks, provenance, and upstream-source
discovery. Any upstream material used for TaskForge generation needs its own
source-specific rights check during acquisition.
No evaluation item, answer, CoT, or source snapshot is committed to Git.

## Stage 1: inventory and obvious exclusions (no question reads)

The pinned [biology eval catalog](https://github.com/Open-Athena/bio-evals/tree/c4f5c922a11acc68c0b70b46da8367a2a37a92cd)
contains **46 historically ID-labeled variant rows in 34 families**. Joining
its frozen imported spreadsheet projection (SHA-256
`1c4c7c768c08e507cf82aef017c89e7bd52bf55463d9257454f543c45844b2a3`)
to the current records yields **3 confirmed ID variants and 43 TBD**. TBD
means policy scope has not been confirmed for that particular protocol; it
says nothing about source rights. Only two of the confirmed ID variants
are the catalog's *selected* versions. Do not automatically promote historical
spreadsheet labels; BixBench3, for example, needs a separate OOD/rights review.

```bash
# Clone exactly this public catalog commit into a clean directory (not a Marin worktree).
uv run --no-sync python -m experiments.post_training.capability_driven_envs.id_eval_source_intake.inventory \
  --catalog-root /tmp/bio-evals-gamma \
  --output /tmp/id-eval-inventory-gamma-20261116.json
uv run --no-sync python -m experiments.post_training.capability_driven_envs.id_eval_source_intake.probe_rights \
  --inventory /tmp/id-eval-inventory-gamma-20261116.json \
  --output /tmp/id-eval-hf-exclusions-gamma-20261116-v2.json
```

The first pass found **14 linked Hugging Face dataset repos**. Four have
unreviewed gated access terms (GPQA, DrugDiscoveryBench, BiomniBench-DA,
BixBench-Verified-50); SciGym has no declared dataset license. The pipeline
**sets those five aside**. The other nine are leads to inspect while getting
pinned data; it says nothing about embedded third-party rights. A card's
license does not automatically cover papers, protocols, websites, figures,
notebooks, or third-party
datasets. GitHub-hosted and other non-HF sources have not had this exclusion
pass yet. An ID assignment or an implemented evaluator says nothing about
permission to make training data.

Private metadata-only snapshots (outside Git):

- `s3://marin-us-east-02a/marin4sc/taskforge/source-corpora/id-evals/inventory/catalog-c4f5c922-v1/historical-id-inventory.json`
  — SHA-256 `bca9ce5f2e4e06fb966b7ab85cb72f5b12a6adebf242c63d191c910f96023ea8`.
- The original metadata probe remains at `.../inventory/catalog-c4f5c922-v1/hf-rights-signals.json`.
  The updated exclusion pass is at
  `.../inventory/catalog-c4f5c922-v2/hf-exclusion-signals.json`
  — SHA-256 `0d8afa8fef1d6250f5b95ea55b660a46ae99654effa7ca30f48aac9c49c1a8ea`.

## Stage 2: inspect the pinned data and rights evidence together

`source_candidates.json` lists two small, public sources to inspect: MMLU-Pro
biology and Biomni Eval1. This is an acquisition list, **not an approval
registry**. The script checks access, declared license, exact revision and
file hashes while pulling the card and data. It reads only provenance columns
(`src` or `task_name`) to map upstream families, and records card restriction
markers and unresolved underlying terms in each private manifest. It refuses
gated sources, changed license declarations, missing pins, a download over
10 MB, output inside Git, and changed existing bytes. No question text or
answer trace is used in task generation.

The pinned [MMLU-Pro card](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/b189ec765aa7ed75c8acfea42df31fdae71f97be/README.md)
declares MIT; its questions also come from original MMLU, STEM websites,
TheoremQA, and SciBench. Biomni Eval1 declares Apache-2.0, but its task names
include Lab-Bench, GWAS Catalog, OpenTargets, and other separately sourced
material. Neither repository's top-level license resolves the rights of those
underlying assets.

```bash
# Local private snapshot:
uv run --no-sync python -m experiments.post_training.capability_driven_envs.id_eval_source_intake.acquire \
  --family mmlu --variant mmlu-pro-biology \
  --output /tmp/id-eval-intake-private-20261116/mmlu-pro-biology-inspect-v2
# Durable private S3 snapshot: inject CW_KEY_ID/CW_KEY_SECRET securely, never print them.
uv run --no-sync python -m experiments.post_training.capability_driven_envs.id_eval_source_intake.acquire \
  --family mmlu --variant mmlu-pro-biology \
  --output s3://marin-us-east-02a/marin4sc/taskforge/source-corpora/id-evals/mmlu-pro-biology/eval-inspect-v2-b189ec7 \
  --s3-endpoint-url https://cwobject.com
uv run --no-sync python -m experiments.post_training.capability_driven_envs.id_eval_source_intake.acquire \
  --family biomni-eval1 --variant biomni-eval1-tool-enabled \
  --output s3://marin-us-east-02a/marin4sc/taskforge/source-corpora/id-evals/biomni-eval1/eval-inspect-v2-51f97f \
  --s3-endpoint-url https://cwobject.com
```

The MMLU-Pro v2 remote snapshot contains the pinned card and **both** parquet
splits (4,198,765 bytes); `manifest.json` SHA-256 is
`ebcd03b163be9e17b5f5cc8ef4a4216f0e7b38ed8f07d808b2f3ed58578509d0`.
The separate Biomni Eval1 snapshot is under
`.../id-evals/biomni-eval1/eval-inspect-v2-51f97f/` (113,600 bytes;
manifest SHA-256 `a40e2f15cbcbb26a0b7664e7b49c5edd24bcefd6270a86bfe8ba1e1672d4743d`).
Both record the observed dataset license and upstream source labels, while
marking underlying-source terms unchecked. The older MMLU-Pro v1 snapshot
remains unchanged.
A private *metadata-only* scan of `category` and `src` found 717 biology rows
in the 12,032-item test split: **219** sourced from original MMLU biology and
**498** tagged STEM-website biology/genetics; the 70-item validation split has
five biology rows. These counts are provenance leads, not a cleared training
pool. The original questions, options, answers, and CoT were not printed,
committed, or given to a task generator.

## Continue with upstream sources

For each lead in the private manifests, inspect the actual upstream source and
its license, access terms, third-party credits, and usable content **when
fetching it**. Skip sources with explicit evaluation-only/no-training terms or
missing rights; record unresolved cases as unresolved. Pin eligible source
bytes and use those, plus high-level skill descriptions, to generate new
TaskForge workpieces. Eval copies stay private for overlap/decontamination.
TaskForge Stage 50/60 checks do not settle source rights or biology correctness.
