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

## Reusing an ID eval's source assets, not its tasks

The **same pinned graph, table, sequence resource, figure or capsule input** may
be a valid ID training environment if its component-specific terms permit
training use. "Independent task" refers to the *new objective and answer
construction*, not necessarily different source bytes. Keep these layers
separate:

1. **Source-only projection:** explicitly allowlist raw assets from the pinned
   snapshot, retain origin hashes and licenses, and give the generator access
   only to this projection and a human-authored **high-level skill brief**.
   Never mount benchmark question/goal/prompt, options, gold labels, reference
   analyses, CoT or teacher eval rollouts in the authoring job. For a graph,
   this could include graph/ontology TSVs but not `kgqa`, `kgcheck` or `scv`
   split files. For a capsule, separate raw inputs from solution notebooks.
2. **Component terms:** follow each asset to its original provider and record
   its access conditions, license and embedded credits. Skip obvious
   restrictions; label unresolved components unresolved rather than declaring
   the whole repository usable or forbidden. The repository-level license
   alone cannot resolve mixed-origin assets.
3. **New workpiece:** sample new objectives against the permitted source,
   compute independent references using a verifier or executable analysis,
   and review scientific interpretation and alternative answers. Use new
   tasks to teach the eval's *skill* and tool-use pattern, never rephrase a
   held-out instruction.
4. **Private decontamination:** compare candidate tasks, target facts,
   evidence, papers/capsules and answer results against the private eval.
   Reject exact or equivalent eval questions and answer-bearing tasks; this
   does **not** require holding out the whole source or every entity. Keep the
   overlap report private and separate from generation prompts.

This is an intake and generation design, not a finding that any particular
mixed-origin asset is cleared for RL training.

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
  subset of the Clinical Knowledge Graph (CKG). For KGCheck, authors used
  Cypher queries to sample nodes/edges, cross-checked against updated
  databases such as UniProt and IntAct and supporting literature, and had
  experts review evidence for the 225 support/refute labels.
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
  prompts/answers where they could turn the eval into memorization. **Graph
  queries alone can grade KGQA but cannot establish KGCheck correctness**:
  detecting erroneous or missing graph facts also requires rights-checked
  *external* evidence and independent expert adjudication. This does
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
- **Possible ID RL route:** Author new tasks on procedural DNA/RNA/protein
  sequences with independent graders. For DbQA, the **same underlying database
  releases** could support fresh retrieval objectives if their own terms
  permit this use; pin the release and keep eval stems, choices, keys and
  generation templates out of the authoring context. The original scientific
  figures/tables or protocols can likewise be considered as *source assets*
  only if their individual rights permit training use; ask genuinely different
  questions with independent reference checks. Never train on the public
  LAB-Bench questions just because 80% were released or on the 15
  `is_opensource=False` LitQA2 items.

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
- **Possible ID RL route:** If a particular Code Ocean capsule's code, data
  and dependencies allow this use, its *raw assets* could support new
  reproducibility tasks about different outputs or experiments. Do not give
  the task author the benchmark prompt, result questions, answer targets or
  successful eval traces; do not treat the capsule's own result notebooks as
  newly generated gold. Independently run the capsule, derive and check new
  targets, and audit answer/output overlap. Original simulated capsules are
  an easier first pilot; they are not a requirement for every eventual task.

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
- **Possible ID RL route:** If a real dataset's original archive permits
  training use, it can be the **same underlying table** supplied to a new
  analysis task; independently pose and verify a different discovery goal,
  without using the benchmark's goals, hypotheses, reference workflows or
  answer keys. Screen for equivalent findings even when wording differs.
  Original simulated ecological/molecular measurements with known signals,
  confounders and missingness are an easier first pilot. Grade concrete
  statistics and reproducible derivations, not semantic resemblance to an
  eval reference.

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
- **Possible ID RL route:** First separate a capsule's raw input files from
  its notebooks, hypotheses, result summaries and benchmark questions. If the
  inputs have usable source rights, the *same raw data* could support a
  genuinely new analysis objective, with an independently executed reference
  and numerical/structural grader. Do not feed the notebook's solved analysis
  or eval answer into a task author. Screen for identical or equivalent
  conclusions, and adjudicate ambiguity before export. New simulated or
  independently licensed capsules are useful alternatives, not an ID
  requirement.

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

## Practical first workpieces

**Priority 1: shared-graph BioKGBench pilot.** Map each of the 23 graph TSVs and
12 ontology TSVs to the upstream provider/version and applicable data terms.
If a usable subset exists, make a pinned *source-only* graph view (never mount
`kgqa`, `kgcheck`, `scv`, PDFs or eval-derived prompts into generation).
Author perhaps 20 new objectives across evidence-backed graph exploration,
relationship composition and consistency checking. Record each objective's
graph revision, generating seed, query program, referenced triples, computed
answer and independent verifier for KGQA. Treat KGCheck-style error detection
as a separate phase requiring updated external databases/literature and
expert-reviewed labels; do not grade it from the same graph alone. Privately
reject tasks equivalent to eval
queries or gold outcomes; retain the same graph and biological distribution.
This is a pilot design, **not a statement that the graph's rights are settled**.

**Priority 2: inexpensive procedural calibration.** Pilot a small,
independently authored **synthetic sequence-analysis** family:
seeded source generator (no benchmark inputs), scientifically constrained
synthetic sequences, hidden deterministic reference program, independent
property-based checks, and TaskForge controls/adversarial trials. Generate
open-answer or tool-use tasks rather than imitating LAB-Bench's MCQ text.
For any database, literature or real-data pilot, identify source-only assets
inside the pinned eval snapshot, check the terms and asset credits of their
original releases, then test task/answer overlap privately.
The underlying asset **may be shared** with the eval where its terms permit;
only benchmark-authored tasks, keys, and teacher rollouts are categorically
excluded from training. Compare not just question text but
paper DOI, capsule/repository, dataset release, graph entity/edge, and source
passage identifiers; an exact-text-only check misses upstream contamination.
Source provenance, scientific correctness, grader reliability and Stage 50/60
evidence are separate gates.
