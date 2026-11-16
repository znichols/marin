# ID evaluation source intake: metadata → rights → private acquisition

This is **not** a way to import benchmark questions into TaskForge training.
[Marin's evaluation policy](https://github.com/marin-community/marin/issues/10197)
allows independently sourced synthetic tasks *modeled on* ID eval skills but
forbids using ID eval instructions directly as RL data or training on teacher
rollouts from eval questions. Acquisition here is **eval-only**: retain a
private, hash-pinned copy for overlap checks, provenance, and upstream-source
discovery. A separate rights-reviewed source must feed any TaskForge generator.
No evaluation item, answer, CoT, or source snapshot is committed to Git.

## Stage 1: inventory and rights evidence (no question reads)

The pinned [biology eval catalog](https://github.com/Open-Athena/bio-evals/tree/c4f5c922a11acc68c0b70b46da8367a2a37a92cd)
contains **46 historically ID-labeled variant rows in 34 families**. Joining
its frozen imported spreadsheet projection (SHA-256
`1c4c7c768c08e507cf82aef017c89e7bd52bf55463d9257454f543c45844b2a3`)
to the current records yields **3 confirmed ID variants and 43 TBD**. TBD
means policy scope has not been confirmed for that particular protocol—not
rejected, not approved for training. Only two of the confirmed ID variants
are the catalog's *selected* versions. Do not automatically promote historical
spreadsheet labels; BixBench3, for example, needs a separate OOD/rights review.

```bash
# Clone exactly this public catalog commit into a clean directory (not a Marin worktree).
uv run --no-sync python -m experiments.post_training.capability_driven_envs.id_eval_source_intake.inventory \
  --catalog-root /tmp/bio-evals-gamma \
  --output /tmp/id-eval-inventory-gamma-20261116.json
uv run --no-sync python -m experiments.post_training.capability_driven_envs.id_eval_source_intake.probe_rights \
  --inventory /tmp/id-eval-inventory-gamma-20261116.json \
  --output /tmp/id-eval-hf-rights-evidence-gamma-20261116.json
```

The first pass found **14 linked Hugging Face dataset repos**: ten public,
four gated; 13 declare a license in their card and one does not. These are
**repository metadata signals**, not item-level permissions or validation
that underlying papers, protocols, websites, figures, notebooks, or datasets
share the declared license. GPQA is gated with additional access terms; no
GPQA item was downloaded or used. GitHub-hosted and other non-HF sources still
need separate term and asset checks. No review may be inferred from `ID`,
`implemented`, `gated=False`, or a card's `license` field.

Private metadata-only snapshots (outside Git):

- `s3://marin-us-east-02a/marin4sc/taskforge/source-corpora/id-evals/inventory/catalog-c4f5c922-v1/historical-id-inventory.json`
  — SHA-256 `bca9ce5f2e4e06fb966b7ab85cb72f5b12a6adebf242c63d191c910f96023ea8`.
- `.../inventory/catalog-c4f5c922-v1/hf-rights-signals.json`
  — SHA-256 `860ce1b6f1e44a653ba520cd76c8186ad2ecfdae1283b180e05edd22b123fec8`.

## Stage 2 pilot: acquire one *eval-only* source with pinned bytes

`reviewed_sources.json` is an explicit, versioned **preliminary** review of
MMLU-Pro biology. The public [dataset card](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/b189ec765aa7ed75c8acfea42df31fdae71f97be/README.md)
declares MIT for this repository. That establishes a basis for **private eval
acquisition**, not rights to turn the evaluation questions into training
prompts. The card says questions derive from original MMLU, STEM websites,
TheoremQA, and SciBench; those underlying rights and training-source roles
remain unresolved. The script refuses an unreviewed variant, changed/gated
source, unexpected license, missing pin, unexpected file size/hash, a download
over 10 MB, output inside Git, and an attempt to overwrite changed bytes. It
records an eval-only manifest last and safely reuses completed files.

```bash
# Local private snapshot:
uv run --no-sync python -m experiments.post_training.capability_driven_envs.id_eval_source_intake.acquire \
  --family mmlu --variant mmlu-pro-biology \
  --output /tmp/id-eval-intake-private-20261116/mmlu-pro-biology
# Durable private S3 snapshot: inject CW_KEY_ID/CW_KEY_SECRET securely, never print them.
uv run --no-sync python -m experiments.post_training.capability_driven_envs.id_eval_source_intake.acquire \
  --family mmlu --variant mmlu-pro-biology \
  --output s3://marin-us-east-02a/marin4sc/taskforge/source-corpora/id-evals/mmlu-pro-biology/eval-only-b189ec7 \
  --s3-endpoint-url https://cwobject.com
```

The remote snapshot contains the pinned card and **both** parquet splits
(4,198,765 total bytes), with each file's SHA-256 checked on write and again
on read. The `manifest.json` SHA-256 is
`051e6dd0b24d9a1dab95d9afc669e19cbaac26fcf3adf371f51674059a35ac6a`.
A private *metadata-only* scan of `category` and `src` found 717 biology rows
in the 12,032-item test split: **219** sourced from original MMLU biology and
**498** tagged STEM-website biology/genetics; the 70-item validation split has
five biology rows. These counts are provenance leads, not a cleared training
pool. The original questions, options, answers, and CoT were not printed,
committed, or given to a task generator.

## Next gate before TaskForge generation

For each proposed training source, record a **separate** versioned decision
about license/access agreements, underlying third-party assets, whether the
specific source may be used for LLM task generation, and its eval-vs-training
role. Pin and archive independently cleared upstream material. Generate new
TaskForge workpieces from that material and high-level skill descriptions,
not from archived eval questions or answer traces. Keep eval source copies
private for overlap and decontamination audits. Stage 50/60 acceptance is
separate from rights and independent domain review; neither grants the other.
