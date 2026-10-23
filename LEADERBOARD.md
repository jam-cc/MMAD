# MMAD leaderboard

Last literature check: **2026-10-23**. Scope: the **full, seven-task MMAD multiple-choice benchmark**, not MMAD-BBox, binary-only anomaly detection, or a subset renamed MMAD-QA.

This is an initial, manually curated **paper-reported** leaderboard. M is the ICLR 2025 benchmark paper; C is the linked AD-Copilot arXiv v2 preprint. Other links identify checked paper versions; arXiv listing alone is not a claim of peer-reviewed acceptance. Inclusion is based on disclosed training and evaluation protocols, not a claim that the maintainers reran every model. A paper saying “training-free” or “1-shot” is not sufficient by itself. Results outside the main protocol remain visible below with their actual settings. Exclusion means a protocol difference, not an allegation of misconduct.

**Training on other industrial datasets is allowed**, provided all adapted components use sources disjoint from MMAD and its four constituent datasets. Industrial specialists and general-purpose backbones may share the main ranking when they satisfy the same inference protocol. See the training-provenance table below for the actual data and reasons for separate placement.

## Main-track rules

1. **No MMAD-specific adaptation anywhere in the pipeline.** Do not train, fine-tune, distill, learn prompts, calibrate, or select a configuration using MMAD questions/answers, images, annotations, domain-knowledge files, or derived descriptions. This applies to the MLLM, adapters, expert models and auxiliary components. Selecting hyperparameters on evaluation answers also fails this rule.
2. **No adaptation on the constituent image sources.** MVTec-AD/DS-MVTec, VisA, MVTec-LOCO and GoodsAD normal training images must not be used to train an expert, build a category-specific memory bank, or calibrate a detector. Ordinary one-reference in-context conditioning is allowed; extra normal images are not. Generic pretrained visual encoders are allowed when no prohibited task-specific adaptation is disclosed.
3. **Exactly one normal reference, using the standard MMAD template policy.** Use the benchmark-provided default reference. Query-driven retrieval across a gallery (`1-shot+`), extra abnormal demonstrations, multiple references, and hidden normal banks do not qualify simply because one image eventually reaches the MLLM. The repository's default VQA runner uses `random_templates[:1]`; `--similar_template` is a different setting. If the exact historical selection policy is not disclosed, record that limitation rather than invent it.
4. **No added domain text.** Retain the task question/options, normal-reference instruction and answer-format wrapper. Do not inject RAG documents, category descriptions, manufacturing knowledge, external expert-written hints, or MMAD domain knowledge at inference. A model's own generated reasoning is not an external knowledge prompt.
5. **Full coverage and official metrics.** Evaluate all four sources and seven question types. Do not mix a changed train/test split, filtered categories, five-task averages, binary accuracy, AUROC or BBox IoU into the overall ranking. Report the official anomaly-discrimination balanced accuracy and the unweighted mean of the seven task scores, retaining the official category/source aggregation. Missing/invalid answers must not disappear from the denominator.

Closed-model and generic pretraining corpora usually cannot be exhaustively audited. “No MMAD-specific adaptation disclosed” does **not** certify absence from all web-scale pretraining. New specialist submissions must identify their component training data; unknown expert/checkpoint provenance stays pending.

## Main leaderboard: reported standard 1-shot, no added knowledge

Scores are percentages, sorted by the **reported seven-task average**. **M** = original MMAD paper, Table 2; **C** = AD-Copilot paper, Table II. Both explicitly identify the standard/default one-shot setting; C §VII-A explicitly specifies no domain knowledge and §III lists disjoint industrial training sources. Historical template seeds, exact API snapshots and annotation hashes are not fully disclosed, so this is a historical reported-results track, not a certification of the current corrected dataset revision.

| Model | Scale | Anomaly disc. | Defect cls. | Defect loc. | Defect desc. | Defect analysis | Object cls. | Object analysis | Average | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AD-Copilot-Thinking | 7B | 73.95 | 74.29 | 76.40 | 84.92 | 86.93 | 91.86 | 87.67 | 82.29 | [C](https://arxiv.org/html/2603.13779v2#S6.T2) |
| AD-Copilot | 7B | 73.64 | 67.89 | 64.08 | 80.60 | 85.91 | 91.06 | 87.78 | 78.71 | [C](https://arxiv.org/html/2603.13779v2#S6.T2) |
| Qwen2.5-VL | 72B | 72.66 | 62.31 | 67.16 | 73.56 | 81.95 | 94.30 | 86.78 | 76.96 | [C](https://arxiv.org/html/2603.13779v2#S6.T2) |
| Seed1.5-VL | - | 65.30 | 64.32 | 63.35 | 73.80 | 84.38 | 91.65 | 83.67 | 75.21 | [C](https://arxiv.org/html/2603.13779v2#S6.T2) |
| GPT-4o | - | 68.63 | 65.80 | 55.62 | 73.21 | 83.41 | 94.98 | 82.80 | 74.92 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |
| Qwen3-VL | 8B | 68.71 | 63.27 | 61.01 | 68.18 | 79.99 | 92.27 | 83.96 | 73.91 | [C](https://arxiv.org/html/2603.13779v2#S6.T2) |
| Qwen3-VL-Thinking | 8B | 56.29 | 62.17 | 60.93 | 75.34 | 84.58 | 93.59 | 84.26 | 73.88 | [C](https://arxiv.org/html/2603.13779v2#S6.T2) |
| Seed1.5-VL-Thinking | - | 55.86 | 62.36 | 63.03 | 72.36 | 83.69 | 94.46 | 84.83 | 73.80 | [C](https://arxiv.org/html/2603.13779v2#S6.T2) |
| Gemini-1.5-pro | - | 68.63 | 60.12 | 58.56 | 70.38 | 82.46 | 89.20 | 82.25 | 73.09 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |
| Qwen2.5-VL | 7B | 71.10 | 56.02 | 60.69 | 64.13 | 78.26 | 91.49 | 83.67 | 72.19 | [C](https://arxiv.org/html/2603.13779v2#S6.T2) |
| InternVL2 | 76B | 68.25 | 54.22 | 56.66 | 66.30 | 80.47 | 86.40 | 82.92 | 70.75 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |
| Gemini-1.5-flash | - | 58.58 | 54.70 | 49.10 | 66.53 | 82.24 | 91.47 | 79.71 | 68.90 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |
| LLaVA-NeXT | 34B | 57.92 | 48.79 | 52.87 | 71.34 | 80.28 | 81.12 | 77.80 | 67.16 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |
| GPT-4o-mini | - | 64.33 | 48.58 | 38.75 | 63.68 | 80.40 | 88.56 | 79.74 | 66.29 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |
| MiniCPM-V2.6 | 8B | 57.31 | 49.22 | 43.28 | 65.86 | 75.24 | 92.02 | 80.80 | 66.25 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |
| InternVL3.5 | 8B | 64.93 | 48.33 | 53.83 | 58.65 | 78.00 | 71.97 | 81.24 | 65.28 | [C](https://arxiv.org/html/2603.13779v2#S6.T2) |
| LLaVA-OneVision | 7B | 51.77 | 46.13 | 41.85 | 62.19 | 69.73 | 90.31 | 80.93 | 63.27 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |
| InternVL2 | 8B | 59.97 | 43.85 | 47.91 | 57.60 | 78.10 | 74.18 | 80.37 | 63.14 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |
| InternLM-XComposer2-VL | 7B | 55.85 | 41.80 | 48.27 | 57.52 | 76.60 | 74.34 | 77.75 | 61.73 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |
| LLaVA-NEXT-Interleave | 7B | 57.64 | 33.79 | 47.72 | 51.84 | 67.93 | 81.39 | 74.91 | 59.32 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |
| LLaVA-1.5 | 13B | 49.96 | 38.78 | 46.17 | 58.17 | 73.09 | 73.62 | 70.98 | 58.68 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |
| LLaVA-1.5 | 7B | 51.33 | 37.04 | 36.62 | 50.60 | 69.79 | 68.29 | 69.53 | 54.74 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |
| Qwen-VL-Chat | 7B | 53.65 | 31.33 | 28.62 | 41.66 | 63.99 | 74.46 | 67.94 | 51.66 | [M](https://arxiv.org/html/2410.09453v3#S3.T2) |

Human references are not model ranks: expert **86.65**, ordinary **78.69** ([MMAD Table 2](https://arxiv.org/html/2410.09453v3#S3.T2)).

### Provenance and version limits

- **No best-of-paper score picking.** M is retained for checkpoints already evaluated there; C adds the listed newer checkpoints. Alternative reports are not silently substituted just because their numbers are higher. A genuinely different checkpoint or setting requires a separate identified row.
- **Current corrected annotations are a separate revision.** The 2026-10-22 release fixes GoodsAD similar-template product matching, the pill image reference and existing question corrections. Its `mmad.json` SHA-256 is `c219c6b0c8794c017519d398374b72522fe18649d0fbac1bc0c563501de08143` (8,366 queries, 39,670 questions). Historical scores above were not rerun on that revision. Future submissions must supply a revision/hash; the old 39,672-question count must not be described as the corrected release.
- **No disguised current-model claims.** API names above refer to the source paper's evaluations; they do not establish today's API performance. Exact checkpoint/date information absent from a paper is unavailable, not inferred.
- **Author overlap disclosed.** MMAD and AD-Copilot share authors. AD-Copilot has the same evidence standard as every other entry; its figures are author-reported, not an independent replication. Training uses industrial data from other sources; this is permitted because the rule prohibits MMAD/source adaptation, not all industrial training.

## Industrial training and provenance

This inventory distinguishes **disjoint industrial training**, **MMAD/source-data adaptation**, and **general-purpose backbones**. It is not a second ranking: published scores remain in their original protocol tables. “General-purpose” describes the checkpoint, not proof that its pretraining contained no industrial images. Sources below cover disclosed industrial adaptation; generic pretraining remains subject to the provenance limits above. Unknown component provenance is not assumed clean.

| Model | Model type | Disclosed industrial training sources | Main-track disposition | Primary evidence |
|---|---|---|---|---|
| General-purpose backbones | General-purpose | No MMAD-specific adaptation disclosed for these evaluated checkpoints; generic pretraining corpora are not exhaustively auditable. | Eligible under the reported standard 1-shot protocol; this is not a claim of industrial-data-free pretraining. | [MMAD Table 2](https://arxiv.org/html/2410.09453v3#S3.T2); [AD-Copilot Table II](https://arxiv.org/html/2603.13779v2#S6.T2) |
| AD-Copilot | Industrial-trained | Chat-AD: VIADUCT, Real-IAD, MANTA, 3CAD, MPDD, VISION; private Real-IAD-Variety and ReinAD. VISION appears in the training mixture table, in addition to the sources described in §III. | Main track: 78.71, standard 1-shot without added domain knowledge. Disjoint industrial training is permitted. Private training sources limit independent auditing; author overlap is disclosed above. | [Sources §III](https://arxiv.org/html/2603.13779v2#S3); [training mixture Table I](https://arxiv.org/html/2603.13779v2#S5.T1); [evaluation §VII-A](https://arxiv.org/html/2603.13779v2#S7.SS1) |
| AD-Copilot-Thinking | Industrial-trained | Chat-AD: VIADUCT, Real-IAD, MANTA, 3CAD, MPDD, VISION; private Real-IAD-Variety and ReinAD. Additional comparison-aware reasoning training is described in Table I. | Main track: 82.29, standard 1-shot without added domain knowledge. Self-generated reasoning does not count as external RAG. | [Training mixture Table I](https://arxiv.org/html/2603.13779v2#S5.T1); [results Table II](https://arxiv.org/html/2603.13779v2#S6.T2) |
| EIAD | Industrial-trained | EDDM: DDQA from VISION, PR-REAL, Real-IAD and BSData. The separate MDLM localization module also uses RSDD, DAGM2007 and Eyecandies mask supervision; review modules separately. | Separate setting: original MMAD result 69.4 is **0-shot**, not a verified standard 1-shot result. Disjoint industrial training itself is not the exclusion reason. | [DDQA §III-A](https://arxiv.org/html/2503.14162v1#S3.SS1); [mask training §III-C](https://arxiv.org/html/2503.14162v1#S3.SS3); [Table III](https://arxiv.org/html/2503.14162v1#S3.T3) |
| ADOPD | Industrial-trained | 6K training records constructed from Real-IAD; the paper states no overlap with MMAD. | Separate setting: 77.31 uses **0-shot**, query-only inference. Reference-privileged training is not a standard 1-shot inference result. | [Training §4.1](https://arxiv.org/html/2608.09789v1#S4.SS1); [dataset Appendix B](https://arxiv.org/html/2608.09789v1#A2); [Table 1](https://arxiv.org/html/2608.09789v1#S4.T1) |
| SAGE | Industrial-trained; source overlap | MVTec QA for first-stage SFT and VisA QA for second-stage preference optimization. | Fails the strict constituent-source training rule; no eligible full seven-task MMAD row established. Do not describe this as disjoint industrial training. | [Training and evaluation §5.1](https://arxiv.org/html/2507.07939v1#S5.SS1) |
| Triad | Industrial-trained; source overlap | Includes Real-IAD, VisA and MVTec-LOCO. | Source training plus filtered test categories; 70.92 is not a full untuned-MMAD result. | [Training §3.3 / filtered evaluation Appendix A.2](https://arxiv.org/html/2503.13184v2#S1.SS2) |
| AnomalyGPT | Industrial-trained; source overlap | Official training includes MVTec-AD. | Source adaptation; historical MMAD score 36.52 is separate. A disjointly trained checkpoint would require its own evidence. | [Official training instructions](https://github.com/CASIA-LMC-Lab/AnomalyGPT#33-training-anomalygpt) |
| MAU-GPT | Industrial-trained; source overlap | MAU-Set includes MVTec-AD. | Source adaptation; reported 61.41 also needs exact shot-setting confirmation. | [Training and dataset Table 1](https://arxiv.org/html/2602.07011v1) |

## Separate comparison: MMAD/source-data adaptation

These results do **not** enter the main leaderboard. The table is grouped by evaluation population; scores must not be ranked across the groups.

| Model / reported result | Reported average | Evaluation population / metric | Why separate; primary evidence |
|---|---:|---|---|
| AnomalyR1, 3B | 76.96 | Reported seven-task, 1-shot MMAD | §3.5 uses 600 images from the benchmark's constituent industrial datasets; method also consumes domain knowledge. [Original paper §3.5 / Table 2](https://arxiv.org/html/2504.11914v1#S3.SS5) |
| OmniAD, 7B, 1-shot | 79.9 | Reported seven-task MMAD | §3.3 explicitly adds one MMAD example per category to SFT and GRPO. [Original paper §3.3 / Table 2](https://arxiv.org/html/2505.22039v1#S3.SS3) |
| EMIT, 8B | 81.95 | Reported seven-task, 1-shot MMAD | Training descriptions are generated from five normal source-dataset images per object type, including MMAD constituents; derived text counts as adaptation. Do not label this identical to test-image memorization. [Data Collection / Table 1](https://arxiv.org/html/2507.21619v1#Sx3.SSx1.SSSx4) |
| **Changed split: InspectorGPT, 7B** | 82.38 | Seven tasks; 20% MMAD training, remaining 80% testing | Explicit stratified MMAD split in §Experiments. Not a full untuned-MMAD result. [Protocol / Table 1](https://arxiv.org/html/2608.29783v1#Sx4.SSx1) |
| **Changed split: Anomaly-LR, 3B / 7B** | 86.00 / 87.12 | Seven tasks; 32,059 questions from 6,766 held-out images | §3.1 trains on IAD-LR-22K-MMAD and uses an approximately 20%/80% split; query-only inference is also different from 1-shot. [Protocol / Table 1](https://arxiv.org/html/2609.29457v1#S3.SS1) |
| **Changed split: AgentIAD, 3B** | 82.88 | Seven tasks; 1,936 training images, 6,400 evaluation images | Training trajectories are constructed from MMAD; v2 also adds Web Searcher and Comparative Retriever memory access. Keep its seven-task score separate from binary-detection tables and from the full benchmark. [Original paper v2 §4.1 / Table 1](https://arxiv.org/html/2512.13671v2#S4.SS1) |
| **Different metric: AD-FM, 7B** | **83.56 (six tasks)** | Six analysis tasks, on an adapted split | §Experiments uses 20% source data for training. Table 3's 83.56 excludes anomaly discrimination; Table 1 reports 73.15 for binary discrimination. Neither number is a published full seven-task average. [Tables 1 and 3 / protocol](https://arxiv.org/html/2508.04175v2#Sx4.SSx1) |
| **Source-trained model: MAU-GPT, 4B** | 61.41 | Seven-task MMAD table; exact shot setting needs confirmation | MAU-Set includes MVTec-AD in Table 1; the model is trained on MAU-Set. This fails the strict source-data rule even if MMAD QA is not directly reused. [Training / Tables 1 and 5](https://arxiv.org/html/2602.07011v1) |
| **Source-trained specialist: AnomalyGPT, 7B** | 36.52 | Historical MMAD Table 2 | Official model training includes MVTec-AD. A future checkpoint demonstrably trained only on disjoint sources would be assessed separately. [Official training instructions](https://github.com/CASIA-LMC-Lab/AnomalyGPT#33-training-anomalygpt) |

## Separate comparison: retrieval, domain text, or extra normal banks

These are useful alternative protocols. They are not a common ranked track unless the text corpus, reference budget and evaluation population also match.

| Method / configuration | Reported score | Actual setting and eligibility evidence |
|---|---:|---|
| Reason-IAD, Qwen3-VL-8B | 79.43 | Seven-task 1-shot score, but retrieval-augmented knowledge is injected into reasoning. Appendix A.3 explicitly includes `{Domain_knowledge}` in the prompt. [Table 1 / Appendix A.3](https://arxiv.org/html/2602.09850v1#A1.SS3) |
| SiGMA | 85.2 | One retrieved reference per agent does not make the whole pipeline 1-shot: the defect expert has up to 256 normal images/category, text retrieval uses MMAD knowledge, and team/hyperparameters were selected using MMAD answers. Even K=1 does not remove its text/selection differences. [Appendices A–B / Table 1](https://arxiv.org/html/2610.12310v1#A2) |
| ADSeeker, Qwen2.5-VL-7B, Seek-Setting | 69.90 | Table 2 reports seven-task 1-shot results with Q2K RAG and visual expert evidence. The separate specialist comparison in Table 3 uses only part of MMAD; do not merge the two coverage claims. [Tables 2–3 / §5.2](https://arxiv.org/html/2508.03088v2#S5.SS2) |
| InternVL2-76B + RAG (original MMAD) | 76.18 | Original paper's added-domain-knowledge setting, distinct from its plain 70.75 result. [MMAD Table 4](https://arxiv.org/html/2410.09453v3#S4.T4) |
| InternVL2-40B + RAG (original MMAD) | 76.35 | Added domain knowledge; do not replace a plain 1-shot row with this result. [MMAD Table 4](https://arxiv.org/html/2410.09453v3#S4.T4) |
| LLaVA-NeXT-34B + RAG (original MMAD) | 72.97 | Added domain knowledge. [MMAD Table 4](https://arxiv.org/html/2410.09453v3#S4.T4) |

## Other settings and pending evidence

No unknown row is promoted to the main leaderboard by default. A method can fail multiple criteria; all known differences are retained here.

| Candidate / result | Disposition | Evidence or specific missing information |
|---|---|---|
| EIAD, 8B: 69.4 | **0-shot**, not main-track 1-shot | Original Table III labels its result zero-shot; §IV-A says all EIAD experiments use zero-shot. Some later comparison tables place it under a 1-shot heading; use the original source. DDQA uses VISION, PR-REAL, Real-IAD and BSData; MDLM additionally uses RSDD, DAGM2007 and Eyecandies masks. See the industrial-training inventory for module-level provenance. [Original Table III / §IV-A](https://arxiv.org/html/2503.14162v1#S3.T3) |
| ADOPD, 4B: 77.31 | **0-shot**, not main-track 1-shot | Training on disjoint industrial sources can be acceptable, but its main table explicitly evaluates without reference images. This is not the score of a 1-shot ADOPD checkpoint. [Table 1 / §4.1](https://arxiv.org/html/2608.09789v1#S4.T1) |
| Cambrian-1, 8B: 51.64; SPHINX, 7B: 59.96 | **0-shot fallback** | Original MMAD Table 2 marks these with `*`: no multi-image input, therefore 0-shot. A table-level “1-shot” caption does not override the row footnote. [Original Table 2](https://arxiv.org/html/2410.09453v3#S3.T2) |
| Triad: 70.92 | **Filtered categories + source training** | §3.3 includes VisA and LOCO in training; Appendix A.2 removes training categories before MMAD evaluation. Its 70.92 is a filtered-category result, not the full untuned benchmark, even without manufacturing-process text. [§3.3 / Appendix A.2](https://arxiv.org/html/2503.13184v2#S1.SS2) |
| GLLS, Qwen3-VL-8B: MVTec 91.2, VisA 84.7 | **Two domains / five tasks + atlas knowledge** | Table 1 covers MVTec and VisA only and uses a Part-Aware Visual-Logical Atlas. Do not call the five-task/domain average the seven-task full-MMAD average. [Table 1 / method](https://arxiv.org/html/2607.03817v1#S4.T1) |
| AnomalyClaw (GPT-5.5): 79.15; Direct GPT-5.5: 76.20 | **500-image subset; agent retrieves four references** | §4.4 specifies 500 images / 2,302 questions, with four retrieved normal references for the agent, despite a “standard 1-shot” table caption. Neither row establishes a full-benchmark score. [§4.4 / Table 2](https://arxiv.org/html/2605.10397v1#S4.SS4) |
| IAD-Unify: 72.5 | **Source adaptation; reference/oracle details pending** | Anomaly-56K uses MVTec-AD and VisA. Full-MMAD numbers alone do not prove a one-reference, no-oracle setting; isolate expert training and predicted-vs-GT region use. [§4.1 / §4.5](https://arxiv.org/html/2604.12440v1#S4.SS5) |
| Claude-3.5-Sonnet: 68.36 | **Pending published-table arithmetic** | Original Table 2 lists seven scores whose arithmetic mean is about 68.62, while the published overall score is 68.36. Keep the reported result visible but request the corrected task value or aggregation before ranking; do not silently change a number. [Original Table 2](https://arxiv.org/html/2410.09453v3#S3.T2) |
| Gemini-2.5-Pro: 81.19; Gemini-2.5-Flash: 78.65; GPT-5-mini: 77.65 (Reason-IAD baseline rows) | **Pending baseline prompt provenance** | Table 1 separates the base models from Reason-IAD, but Appendix A.3 describes a retrieved-domain-knowledge prompt without unambiguously mapping every baseline run to it. Later rounded copies do not resolve that ambiguity. Request baseline prompt/config and dataset revision before ranking. [Table 1 / Appendix A.3](https://arxiv.org/html/2602.09850v1#A1.SS3) |
| SiGMA's individual frozen baselines, including Qwen3.5-9B: 74.7 and Qwen3-VL-4B: 74.8 | **Pending exact reference policy** | Plain models are separated from the multimodal-searcher ablation, but the precise references for every individual run are not pinned; Appendix B describes a query-retrieved gallery. Confirm use of official default references, not retrieval over an extra bank. [Table 1 / Appendix B](https://arxiv.org/html/2610.12310v1#A2) |
| DCR²-AD | **Pending accessible primary protocol** | An OpenReview candidate reports separate with/without-domain-knowledge tables. Full text could not be retrieved during this check because of the site's browser challenge. Do not import the 83.36 domain-knowledge headline into the main track or infer training provenance from a secondary summary. [Primary submission](https://openreview.net/forum?id=qqlWHIjJ24) |
| EAGLE; SAGE; OPD-IAD; IAD-R1 | **No eligible full seven-task row established in this pass** | Their checked primary sources emphasize binary anomaly detection, different QA benchmarks, source adaptation and/or expert augmentation. Absence here means no verified eligible row, not a claim that the methods are ineffective. [EAGLE](https://arxiv.org/html/2602.17419v1), [SAGE](https://arxiv.org/html/2507.07939v1), [OPD-IAD](https://arxiv.org/html/2607.18850v1), [IAD-R1](https://ojs.aaai.org/index.php/AAAI/article/download/37588/41550) |

## Submit a result or correction

Open a GitHub issue or a pull request with:

- Paper version, checkpoint/model revision and exact API snapshot/date where applicable.
- All MLLM/expert/adapter training sources, any source-data calibration or memory construction, and a statement covering derived text and MMAD knowledge files.
- Dataset revision/hash, image/question counts, all seven task scores and the overall score; explain the official aggregation and handling of failed answers.
- The exact inference prompt, reference-selection code/seed, reference count for **every component**, text retrieval settings and tool calls. State whether evaluation labels were used for model/team/hyperparameter selection.
- A primary table/section link and preferably per-question predictions or a reproducible evaluation command. Do not invent absent per-task scores or recompute an overall score from incompatible splits.

Maintenance procedure: verify a new row against the primary paper plus available code; admit it only when the five rules are supported, otherwise retain it with the missing evidence. Update the literature-check date when papers are reviewed and preserve the previous row's provenance when corrected. This page is maintained through reviewed repository updates; no automatic polling or model reruns are implied.
