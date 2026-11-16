# Biology eval source-to-training map (2026-11-16)

This is a **provenance and skill map**, not a license grant or a training dataset.
The first five rows describe complete pinned Hugging Face repository trees in
private S3; their paths, revisions, hashes and component-audit location are in
[README.md](README.md). Counts below come from schemas and categorical metadata
only. No held-out question, key, answer trace, or teacher rollout was copied to
Git or a task generator. The private metadata-only count/schema snapshot is
`s3://marin-us-east-02a/marin4sc/taskforge/source-corpora/id-evals/inventory/complete-hf-trees-5-v1/source-map-metadata-v1.json`
(SHA-256 `8c3ff3cf0206f74d392a6b00313f4ea1fe4e4992301d2f13af4e2ef1a4001a0c`).
A historical spreadsheet called these families ID leads; **the current bio-evals catalog labels only three variants ID and most
others TBD**. Do not turn the historical label into a rights or readiness claim.

## Five acquired source families

### BioKGBench — grounding, graph queries, claim verification

- **What is present:** KGCheck has 20 dev / 205 test items (node attributes,
  existence, one-hop and potential-edge checks); KGQA has 60 dev / 638 test
  items (one-hop 355, multi-hop 184, conjunction 99 in test); scientific claim
  verification (SCV) has 120 dev / 1,265 test items (test: 823 support, 419
  refute, 23 not-enough-information). The HF tree also contains 23 graph TSVs,
  12 ontology TSVs, 5,664-abstract SCV corpus and 51 full-paper PDFs. Graph
  source data is 10.66 GB; full papers are 97 MB.
- **Where it came from:** The [paper](https://arxiv.org/abs/2407.00466) and
  [pinned author README](https://github.com/westlake-autolab/Agent4S-BioKG/tree/9e166e2679122d2e064e7e6856ff294bd5763546)
  say KGQA was generated from expert-reviewed templates instantiated on a
  subset of the Clinical Knowledge Graph (CKG); KGCheck was expert annotated.
  CKG combines UniProt, TISSUES, STRING, SMPDB, SIGNOR, Reactome, IntAct,
  HGNC, DisGeNET, DISEASES and ontologies. SCV reconstructs SciFact and
  PubMedQA; its 5,664 abstracts are associated with those source datasets.
  The 51 open-access full papers were selected by experts using IntAct and CKG.
  The author README explicitly warns that the graph's component databases have
  **their own licenses**. An OA paper is not automatically licensed for training.
- **Possible ID RL route:** Use the **same pinned CKG-derived graph as the
  training and evaluation tool environment**, provided the licenses and terms
  of its constituent databases permit this use. Sharing source data is the
  point of an in-distribution evaluation; it is not permission to copy the
  benchmark's KGQA/KGCheck instructions, question templates, labels, or model
  rollouts. Independently author new graph-investigation objectives (for
  example, multi-relation evidence gathering or consistency checks), sample
  fresh entities and relationships, compute answers with an executable graph
  query, and independently validate biological meaning. Privately screen new
  tasks against eval targets at the question, entity-pair, edge, relation,
  answer and evidence levels; keep exact eval target facts out of training
  prompts/answers where they could turn the eval into memorization. This does
  **not** require making the whole graph or all entities disjoint. If a
  database component's terms prohibit training, remove that component and
  its derived edges before generating tasks, or use a separately licensed
  graph. The SCV corpus and KGCheck PDFs need their own rights assessment;
  do not train on their eval claims or question/answer traces.

### LAB-Bench — sequence operations, database lookup and literature/visual reasoning

- **What is present:** The public HF release has **1,967** MCQ rows in eight
  categories: SeqQA 600 (15 subtasks), DbQA 520 (10), FigQA 181, TableQA 244,
  LitQA2 199, SuppQA 82, ProtocolQA 108, CloningScenarios 33. The
  [author README](https://github.com/Future-House/lab-bench/tree/998a8e0a40cf116c80e1b0e7a805ebb5fb9fa838)
  says this public release is about 80% of the original set, with a **private
  20% contamination-monitoring subset and a canary**. Public does not mean
  suitable for training. This is the original LAB-Bench, not gated LABBench2.
- **Where it came from:** The [methods](https://arxiv.org/abs/2407.10362)
  describe programmatically generated sequence and database questions. DbQA
  combines resources including DisGeNET/OMIM, Ensembl, MSigDB, miRDB, Mouse
  Genome Informatics, GTRD, ClinVar/ProteinGym and P-HIPSter. Literature,
  supplements, figures and tables were assembled from scholarly papers by
  experts or contractors: FigQA's 181 public rows link to 41 DOIs; TableQA's
  244 to 134; SuppQA's 82 to DOI sources. LitQA2's 199 rows contain 205 DOI
  links, and 15 rows explicitly have `is_opensource=False`. ProtocolQA alters
  published protocols from protocols.io and STAR Protocols. No individual
  paper, figure, table, database or protocol is cleared by the HF CC-BY-SA tag.
- **Possible independent RL route:** First author genuinely new tasks on
  procedurally generated DNA/RNA/protein sequences (translation, reading
  frames, restriction-site logic, compositional statistics), with independent
  executable graders and biology review. Use **new instructions and seeds**,
  not eval stems/templates/distractors. Later, separately license and pin
  upstream database releases for *new* retrieval exercises. Literature,
  protocols and image/table work require source-by-source text and asset
  permissions; exclude the 15 flagged LitQA2 items outright from consideration.

### CORE-Bench — computational reproducibility

- **What is present:** The HF repository contains 45 train and 45 test paper
  records; the test JSON is GPG-encrypted but the
  [author README](https://github.com/siegelz/core-bench/tree/e32a2980e72fe6eb04ee04eb749458f570625663)
  publicly supplies the passphrase. A private, in-memory metadata-only
  decryption found **12 medical-sciences train and 13 medical-sciences test**
  capsules; plaintext was not saved. The [paper](https://arxiv.org/abs/2409.11363)
  applies three difficulty levels per paper (270 total task settings): extract
  reported results, run a supplied environment, or reconstruct dependencies
  and run the code. The HF JSON contains prompts and result specifications,
  **not** the code/data capsules.
- **Where it came from:** The authors selected 90 reproducible research
  [Code Ocean](https://codeocean.com/) papers and manually authored result
  checks. The harness obtains associated capsules from a separate endpoint;
  each paper's code, data and dependencies have their own provenance and terms.
- **Possible independent RL route:** Package original small Python/R
  biomedical analyses with **newly simulated or separately licensed data**,
  reproducible scripts, logged seeds, a container and independently computed
  result checks. Vary environment failure modes and output types. Never reuse
  benchmark capsule IDs, prompts, result questions or Code Ocean payloads as
  training material; review software licenses separately.

### DiscoveryBench — data-driven scientific discovery

- **What is present:** Real-track metadata has 25 train / 239 test tasks;
  the `domain` column identifies **10 biology train / 16 biology test** rows.
  The HF tree also has synthetic train/test data and answers across many
  domains. Its real-track folders include freshwater-fish evolution in train
  and non-native-plant introductions in test, alongside non-biology sources.
  The whole repository is **not** a biology-only benchmark. Its synthetic
  marine-biology folders occur in train, not a corresponding synthetic
  marine-biology test folder in this revision.
- **Where it came from:** The [paper](https://arxiv.org/abs/2407.01725) and
  [pinned code README](https://github.com/allenai/discoverybench/tree/c31fcf011e070f021a5f5b906896d0821f6880e8)
  describe DB-REAL as reproductions of published hypotheses using open
  datasets (e.g. GBIF, NLS, World Bank) or research-code repositories from
  Zenodo; DB-SYNTH was generated using an LLM-built hypothesis tree,
  executable column expressions and simulated/noisy tables. The dataset's
  ODC-BY declaration does **not** individually resolve paper, code, GBIF or
  survey-data terms. A metadata-only scan found no per-task license field in
  the repository JSON.
- **Possible independent RL route:** Build **new** simulated ecological or
  molecular measurements from an explicit data-generating process (signal,
  confounders, negative controls, missingness). Ask for an evidenced
  hypothesis, a reproducible analysis and calibrated uncertainty. Grade
  derivations, held-out simulations and concrete statistical results, not
  semantic resemblance to reference hypotheses. For real data, check each
  original archive's terms and do not reuse the benchmark's datasets/goals.

### BixBench — long-horizon computational biology

- **What is present:** 205 v1.5 questions in the index, with string, numeric-range
  and model-judged verifier modes (61 / 61 / 83 index rows).
  The [paper](https://arxiv.org/abs/2503.00096) describes 61 analytical
  scenarios; this HF revision has 64 capsule ZIPs. A central-directory-only
  audit counted 1,133 members, including 49 notebooks, 633 spreadsheets,
  24 FASTQs, nested archives, alignments and other biological files; it did
  **not** inspect item content. No member is named LICENSE, which neither
  proves restriction nor grants reuse.
- **Where it came from:** Experts assembled notebook-plus-input-data capsules,
  reviewed by other experts. An LLM proposed candidate questions and experts
  reviewed them. See the [author repo](https://github.com/Future-House/BixBench/tree/49311180bdacb324c596f2e07596c126f2004008).
  The top-level Apache declaration does not document every notebook's and
  input dataset's origin, permissions or patient-data status.
- **Possible independent RL route:** Create *different* reproducible analysis
  capsules using original simulated experiments or independently licensed
  studies: perturbation counts, sequence variants, imaging summaries, etc.
  Ask for a documented multi-step analysis and give executable or independently
  computed numerical/structural graders. Keep capsule IDs, data, scientific
  questions and answer traces disjoint from BixBench; adjudicate scientific
  interpretation and answer non-uniqueness before any training export.

## Existing ID leads outside the five-source batch

- **MMLU-Pro biology (current catalog ID):** The private pinned copy includes
  717 biology test rows and five biology validation rows. Test provenance:
  219 original-MMLU biology and 498 STEM-site biology/genetics. The wider
  MMLU-Pro corpus also mixes SciBench, TheoremQA and other categories and
  contains CoT fields; repository MIT does not establish rights in upstream
  exams/websites. Independent RL route: separately cleared instructional
  sources (e.g. particular pages of public biomedical resources), **new**
  concept questions, expert keys and alternative-answer review. The existing
  introductory-biology pilot remains provisional and must not be exported.
- **Biomni Eval1:** 433 validation items across ten task labels, including
  GWAS Catalog, OpenTargets, PharmaProjects, variant/gene prioritization,
  CRISPR delivery and **100 items labeled LAB-Bench DbQA/SeqQA**. This is a
  strong *overlap lead*, not proof of exact duplicate text. Resolve the
  individual data providers' terms (particularly proprietary or clinical
  data) before proposing independent GWAS/gene-prioritization training
  workpieces; do not import Biomni's prompts or labels.
- **GPQA Diamond biology (current catalog ID):** A pinned GPQA Diamond artifact
  already exists in Marin's task-curation cache, but the HF source is gated
  and its access terms have not been reviewed here. No GPQA item was copied
  into this intake or inspected for this map. Novel expert-authored research
  questions would require independently sourced papers, rights checks and
  expert review, **not** paraphrases of GPQA items or teacher eval rollouts.

## Practical first workpiece

Pilot a small, independently authored **synthetic sequence-analysis** family:
seeded source generator (no benchmark inputs), scientifically constrained
synthetic sequences, hidden deterministic reference program, independent
property-based checks, and TaskForge controls/adversarial trials. Generate
open-answer or tool-use tasks rather than imitating LAB-Bench's MCQ text.
Before expanding to database, literature or real-data tasks, separately
archive the exact upstream release and its terms/asset credits, then test
cross-source and eval overlap privately. Compare not just question text but
paper DOI, capsule/repository, dataset release, graph entity/edge, and source
passage identifiers; an exact-text-only check misses upstream contamination.
Source provenance, scientific correctness, grader reliability and Stage 50/60
evidence are separate gates.
