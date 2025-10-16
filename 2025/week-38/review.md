# EEG→Text: A Reproducible Pipeline for Recovering the Semantics of Inner Speech from EEG Signals (Review Version)

**Author(s):** *Verbatsev Eduard Igorevich*

**Affiliation:** *Moscow Aviation Institute (National Research University)*

**Contact:** *verbasik@gmail.com*

**Preprint Version:** v1 (Review)

---

## Abstract

We present a reproducible end-to-end "EEG → Text" pipeline for investigating inner speech in *read* and *imagine* paradigms. The pipeline standardizes preprocessing (artifact cleaning, referencing, ICA auto-labeling, adaptive wavelet filtering, normalization), introduces explicit data contracts, and supports multi-task training of an EEG encoder (CNN+Transformer) with vector quantization and auxiliary heads (coarse semantic categories, *read/imagine* domain, signal reconstruction). We formalize the task as retrieval over a fixed phrase dictionary and parallel classification of categories/domain, describe evaluation protocols (within-session, cross-session, cross-subject), anti-leakage measures, a set of metrics (Top-k, MRR, macro-F1/BA, confidence calibration), and a "repro-package" structure for result reproducibility. Preliminary experiments indicate the potential of VQ and multi-task optimization for stabilizing and interpreting representations; subsequent versions will include systematic comparison with strong baselines and cross-session/subj transfer.

---

## 1. Introduction and Motivation

Recovering the semantics of inner speech from non-invasive brain signals is a key challenge for BCI systems, cognitive neuroscience, and multimodal interfaces. Unlike invasive methods, EEG is safe and accessible but suffers from low signal-to-noise ratio and inter-subject variability. We propose a practical pipeline, from raw EDF files to model inference, focused on reproducibility, standardized formats, and transparent evaluation.

**Contributions (summary):**

1. Formalize the "EEG→Text retrieval" task over a fixed phrase dictionary with parallel classification of coarse categories and domain (*read/imagine*).
2. Describe a standardized preprocessing pipeline with anti-leakage practices.
3. Present an EEG encoder architecture (CNN+Transformer) with VQ and multi-task optimization (contrastive + classification + reconstruction).
4. Fix evaluation protocols (within / cross-session / cross-subject), metrics, and statistical tests.
5. Publish a "repro-package": code, configs, fixed versions, and checkpoints (structure and instructions).

---

## 2. Related Work (Overview)

Research on decoding "inner speech" from EEG can be broadly categorized into three directions. First, classical work on recognizing mentally articulated units: syllables, letters, words, and short commands. These studies demonstrate that even on budget, low-channel systems, stable patterns can be extracted; however, accuracy heavily depends on the number of classes, protocol, and inter-subject variability. For instance, recurrent models (LSTM-RNN) achieve high accuracy on four-class imagined command tasks with an 8-channel headset, while multi-class scenarios (~10–13 classes) using natural language inevitably degrade metrics, remaining statistically above chance. Parallelly, "letter-by-letter" decoding is advancing, where EEG patterns of handwritten letters are combined with language models: this modular "minimal units → language correction" approach shows viability for free input but currently relies heavily on strong LLM support in the second stage.

The second direction involves aligning brain and text representations and open-vocabulary approaches for EEG→Text. Here, the formulation often shifts from hard classification to retrieval/ranking or generation based on pre-trained language models. Several works use two-stage architectures: a specialized module for EEG feature extraction (convolutional-recurrent/transformer layers) and a language module (BART, etc.), augmented with a pre- or post-editing stage using a large model (GPT-like) to enhance fluency and semantic coherence. On sentence-reading corpora (e.g., ZuCo v1.0/v2.0), such systems report improvements in BLEU/ROUGE and BERTScore over earlier approaches, supporting the promise of open-vocabulary and semantic-oriented metrics in evaluation.

The third line concerns discretization of latent representations and vector quantization (VQ) in biosignals. Discrete codes stabilize the latent space, simplify semantic matching, and introduce interpretable "units" (codes) for analysis. Practice shows it is beneficial to control codebook "vitality" (perplexity, frequency distribution, preventing collapse) and combine VQ with tasks that anchor codes to meaningful goals (classification, reconstruction, contrastive alignment). In the context of EEG→Text, this opens a path to more robust and transferable features, especially under domain variability (*read/imagine*).

From the perspective of our work, our contribution is complementary to these directions. We build upon retrieval-style formulation and frozen text embeddings for semantic anchoring, extend the architecture with multi-task heads and domain-adversarial regularization, and use VQ for discretization and representation analysis (perplexity, code frequencies). This design unites the strengths of "classical" decoders, open-vocabulary approaches, and discrete latent spaces while preserving reproducibility and a strict anti-leakage protocol.

---

## 3. Problem Formulation and Scope

Let $x\in\mathbb{R}^{C\times T}$ be an EEG segment ($C$ channels, $T$ samples). The goal is to:

* (a) **Retrieval** over a phrase dictionary $\mathcal{Y}=\{y_1,\dots,y_N\}$: find top-$k$ candidates by score $s(f_\theta(x), g(y))$;
* (b) **Classify** coarse semantic category $c\in\{1,\dots,K\}$;
* (c) **Classify** domain $d\in\{\text{read},\text{imagine}\}$.

Here, $f_\theta$ is the EEG encoder. $g(\cdot)$ is a frozen text mapping of a phrase into a vector space (used **only** on the index/evaluation similarity side, see anti-leakage). We fix the **phrase corpus** and map $\texttt{phrase}\to\texttt{coarse}$ before training and evaluation.

**Evaluation scenarios:**

* **within-session:** training/evaluation within one subject session (different trials);
* **cross-session:** training in one session, evaluation in another session of the same subject;
* **cross-subject:** leave-one-subject-out (LOSO) and/or adaptation with ≤5% subject data.

**Constraints:** We do not consider OOD phrases outside the dictionary, audio/text synthesis, or online adaptation in v1.

---

## 4. Data and Preprocessing

Raw EEG recordings in EDF format with hardware event markers and epoch-to-phrase alignment are used as source material. For unified processing, we fix the sampling rate at 500 Hz, use a consistent montage, and stabilize the list of working channels after excluding artifact-prone and auxiliary (e.g., EOG) channels. Each file additionally preserves metadata: subject, session, and run identifiers; ordered list of channel names; montage hash; information on "bad" channels if identified during acquisition.

Preprocessing is structured as a sequential pipeline aiming to enhance signal-to-noise ratio and ensure reproducibility of subsequent learning steps. First, a bandpass filter (0.5–50 Hz) is applied, followed by notch filtering at 50 or 60 Hz and harmonics. To remove systematic offsets and anomalous electrodes, PREP is used: re-referencing is performed, noisy channels are detected (including RANSAC interpolations), and correction decisions are fixed. All PREP parameters are trained strictly on the training subset of the current fold and then applied unchanged to validation and test.

Event labels are extracted from the hardware trigger channel and matched against the phrase list, after which recordings are segmented into two non-overlapping windows: for *read* domain — 0 to 5 seconds; for *imagine* domain — 5 to 8.3 seconds. No baseline correction is applied, and use of post-event information outside the designated window is excluded. For outlier cleaning and corrupted trial recovery, AutoReject is applied: thresholds are evaluated only on training epochs, logs and threshold parameters are saved, and a pure transformation is applied to other splits without fitting.

The next step is Independent Component Analysis. We train ICA on the training subset (separately for *read* and *imagine* if needed), automatically label components using IClabel, and exclude non-brain sources (blinks, EMG, etc.). Preprocessing artifacts preserve mixing and inverse transformation matrices, the list of excluded components, and a full decision log. On validation and test, the exact same transformation is applied without retraining. To enhance signal quality, adaptive wavelet filtering (Daubechies family, soft thresholding) is applied on top; the choice of family and decomposition level is fixed based on the training set and documented for reproducibility.

Normalization is performed in one of two modes. In local mode, each epoch is scaled per channel using its own mean and standard deviation, a deterministic procedure without test-time learning. In global mode, channel-wise statistics are estimated on the training subset of the current fold, serialized, and then applied unchanged to validation and test data. Where needed, a robust version using interquartile range is used. At all steps, quality control reports are generated: fraction of interpolated channels, residual powerline noise, distributions of channel-specific means and variances, number of excluded ICA components, wavelet filter parameters. All artifacts are labeled with hashes and library versions.

Finally, the overarching anti-leakage rule: any operations with parameters estimated from data (PREP, AutoReject, ICA, global normalization, wavelet filter parameter selection) are trained exclusively on the train set within each fold and then applied as immutable transformations. Epoch segmentation is strictly limited to designated intervals, preventing future information leakage relative to event center.

---

## 5. Dataset Assembly and Data Contracts

Dataset assembly follows the principle of "data contracts," enabling unambiguous reconstruction and verification of each example's state. For each epoch, a machine-readable manifest is generated, recording subject, session, and run identifiers; epoch ordinal number; domain (*read* or *imagine*); original phrase and its categorical label; time window boundaries; sampling rate; ordered channel list and montage hash; and links to all preprocessing artifacts (PREP decisions, AutoReject parameters, ICA files, normalizer state, wavelet parameters). Additionally, the manifest contains a hash of the payload (e.g., SHA-256 of the feature array), code version, and creation timestamp, enabling experiment reproduction and data integrity verification.

We support two complementary storage formats. The FIF format serves as the canonical representation of epochs with full metadata; parallelly, compact containers in PKL, HDF5, or Parquet with feature tensors (float32) and artifact links are saved for training. The directory structure is uniform across all subjects and sessions: separate directories are reserved for canonical epochs, final training packages, preprocessing artifacts, derived features, and quality control logs.

Phrase-to-epoch alignment is strictly established before model training and relies on a static mapping "phrase → coarse category," equipped with its own hash. Two matching modes are allowed: exact match after string normalization and high-threshold fuzzy matching. Semantic search using embeddings as a "label refinement" tool is disabled by default, as it introduces model assumptions into labels. If this mode is used for exploratory analysis, each match is flagged with reduced confidence and either excluded from training or analyzed separately with assessment of noise label fraction and impact on final metrics.

For train/validation/test split scenarios, we provide explicit lists of epoch indices for within-session, cross-session, and cross-subject (LOSO), along with associated artifacts trained on the training subset. The structure of these descriptions can be stored in `split.yaml` files, where, besides indices, paths to serialized PREP, AutoReject, and ICA solutions, normalizer state, and wavelet parameters are specified. Dataset loading is accompanied by automatic compatibility verification: channel count and order, montage hash, label map hash, normalization mode, and time window boundaries are checked. On mismatch, a strict rejection policy or a pre-documented adaptation procedure is applied.

For convenience in training deep models, export to a PyTorch-compatible format is additionally supported: for each domain, a sequence of tensors (C, T), a list of label dictionaries (including phrase text, category ID, and confidence), and original lengths before padding are stored; in the metadata section, sampling rate, channel count, montage hash, channel mask, normalization mode, state hashes, and code version are fixed. All artifacts and manifests are equipped with checksums and versioning, and each dataset release includes a brief changelog and verification hashes for control subsets.

---

## 6. Model and Training (Overview)

The model architecture follows the principle of "local-frequency extraction → temporal aggregation → representation discretization → multi-task semantic supervision." The input EEG segment $x \in \mathbb{R}^{C \times T}$ first passes through a convolutional stem, extracting local-frequency and short-term temporal features and reducing time dimensionality. These maps are then fed into a transformer encoder, modeling long-range dependencies via self-attention. The aggregated representation is projected into a 768-dimensional semantic space and subjected to vector quantization: a codebook of $M=256$ vectors is used, updated via EMA scheme with soft discretization through Gumbel-Softmax. During the first $W=5$ epochs, warm-up is applied (discretization disabled), then temperature annealing is activated following $\tau_t = \max\{\tau_{\min}, \tau_0 \cdot \gamma^t\}$ (typically $\tau_0=0.05$, $\gamma=0.95$, $\tau_{\min}=0.01$). The quantized representation $z_q$ serves as the single fusion point for all target tasks.

Auxiliary heads are built on top of $z_q$. The coarse semantic category classifier is implemented as a multi-layer perceptron with dropout and optimized via cross-entropy. The domain classifier (*read/imagine*) is connected adversarially via gradient reversal: during forward pass, it predicts domain; during backward pass, it "penalizes" the encoder for domain-specific features, promoting invariance. The signal reconstructor restores the original $x$ from $z_q$ and acts as a regularizer preserving low-level information. Additionally, a semantic decoder based on Transformer directly approximates frozen text embeddings $v_{\text{text}} = g(y)$ from $z_q$. This eliminates the "codebook → semantics" gap and stabilizes space alignment. On intermediate levels (after CNN and after Transformer), a hierarchical semantic supervisor is included, generating auxiliary projections and losses, empirically improving learnability and structuredness of hidden representations.

Training is formulated as a multi-criteria optimization. The core component is the contrastive NT-Xent loss between $z_q$ and $v_{\text{text}}$:

$$
L_{\text{NTX}}(i) = -\log\left( \frac{\exp(\text{sim}(z_i, v_i)/\tau)}{\sum_{j \neq i} \exp(\text{sim}(z_i, v_j)/\tau)} \right),
$$

where $\text{sim}(u,v) = \cos(u,v)$.

Alongside the "hard" version, a soft variant is used, accounting for semantic similarities between phrases: positive weights are distributed among paraphrases and close formulations, reducing penalty for reasonable substitutions. Vector quantization is optimized with the standard commitment loss:

$$
L_{\text{VQ}} = \| \text{sg}[z] - e \|_2^2 + \beta \cdot \| z - \text{sg}[e] \|_2^2
$$

($e$ — closest codebook vector, $\text{sg}$ — stop gradient, $\beta \approx 0.25$).

To prevent codebook collapse, entropy regularization of code usage distribution is added (equivalent to encouraging high perplexity). Metric learning is enhanced by triplet loss with semi-hard negative mining and margin $m=0.3$, and semantic structuring is enforced by cross-entropy on coarse classes. Domain-invariant learning is implemented via adversarial loss with gradient reversal, and signal information preservation is ensured by MSE reconstruction. Finally, direct semantic forcing is performed via the decoder:

$$
L_{\text{sem}} = \alpha \cdot \| D(z_q) - v_{\text{text}} \|_2^2 + (1 - \alpha) \cdot (1 - \cos(D(z_q), v_{\text{text}})).
$$

The overall loss function is:

$$
L = \lambda_{\text{NTX}} \cdot L_{\text{NTX}} + \lambda_{\text{sem}} \cdot L_{\text{sem}} + \lambda_{\text{trip}} \cdot L_{\text{triplet}} + \lambda_{\text{coarse}} \cdot L_{\text{coarse}} + \lambda_{\text{dom}} \cdot L_{\text{domain}} + \lambda_{\text{rec}} \cdot \| \hat{x} - x \|_2^2 + \lambda_{\text{VQ}} \cdot L_{\text{VQ}} + \lambda_{\text{ent}} \cdot L_{\text{entropy}} + \lambda_{\text{aux}} \cdot L_{\text{aux}},
$$

where weights are tuned on validation. Optimization uses AdamW with gradient norm clipping.

A key practical detail is handling textual targets. The text encoder $g(\cdot)$ is frozen and used only to obtain reference phrase embeddings. To enhance robustness to formulations and synonyms, during training and especially evaluation, paraphrasing is applied: for each phrase, a set of close expressions is generated, their embeddings averaged, forming a "reference ensemble." Inference reduces to cosine similarity comparison between normalized $z_q$ and normalized reference vectors of the entire dictionary; top-k candidates are determined by ranking by similarity. This scheme, combined with domain-adversarial regularization and hierarchical supervision, ensures stable "brain → text" alignment and improves transferability between *read* and *imagine* domains.

---

## 7. Inference and Confidence Calibration

During inference, the latent vector is transformed into an L2-normalized embedding and compared with precomputed, also normalized, reference phrase vectors from the dictionary. As a similarity measure, cosine similarity is used (equivalent to dot product in normalized space). Reference vectors are stored in an approximate nearest neighbor index (e.g., FAISS): for exact searches, a flat index by dot product is used; for speed, hierarchical (IVF with adjustable nlist/nprobe) or graph-based (HNSW) structures are employed. All index versions are fixed and signed with hashes, ensuring reproducibility.

Output is formed as an ordered set of top-k candidates with their raw similarity scores. Post-processing includes deduplication of paraphrases (if the dictionary contains variants of the same phrase), tie-breaking rules (e.g., preference for more frequent or shorter formulations), and an optional domain priority: when the domain head (*read/imagine*) yields high confidence, ranking can be performed within the sub-index of the corresponding domain. For robustness to formulations, ensemble reference is applied: each phrase corresponds to a set of paraphrases, whose embeddings are averaged, reducing sensitivity to synonyms and punctuation.

Confidence calibration is performed separately for retrieval and classification (coarse and domain). For retrieval, similarity scores are treated as pre-logits and passed through temperature scaling: probabilities are obtained via softmax of logits divided by a positive temperature. Temperature parameter is tuned solely on the validation set, optimizing negative log-likelihood or Brier loss. An alternative is isotonic regression (monotonic, non-parametric calibration). For class-imbalanced classification, class-specific temperature scaling is permitted. Calibration quality is evaluated by Expected Calibration Error (ECE) with interval binning; additionally, Brier score and "reliability curves" (accuracy vs. confidence) before and after calibration are reported. Confidence intervals for ECE and Brier are estimated via bootstrap over episodes (95% CI).

For rejecting low-confidence responses and routing to human, standard uncertainty measures are computed: maximum predicted probability, entropy of distribution, difference between first and second candidate (margin), and support density among nearest neighbors (how many candidates exceed the similarity threshold). Based on these, an "abstain" strategy is implemented with thresholds tuned on risk-coverage curves. In extended mode, conformal predictive sets are supported: on validation, a non-inclusion scale is estimated; on test, the minimal set of phrases with guaranteed coverage (1−α) and controlled width is returned.

On the engineering side, inference is batched, supports mixed precision, caches embeddings for repeated windows, and features a "fast pass" with small k and aggressive index parameters for preliminary ranking followed by refined ranking in exact mode. Calibration parameters and index versions are strictly versioned; calibration fitting procedures never use test data, as documented in experiment logs, ensuring strict reproducibility.

---

## 8. Evaluation Protocol and Metrics

Evaluation is conducted in three scenarios: within-session, cross-session, and cross-subject (LOSO). In all cases, any trainable preprocessing elements (PREP, AutoReject, ICA, global normalization parameters, wavelet filter parameter selection) are tuned strictly on the training subset of each fold and then applied unchanged to validation and test. For each scenario, we fix lists of epoch indices and versioned sets of preprocessing artifacts. Reporting includes checksums, dependency versions, and reproducibility protocol.

### Retrieval

"EEG → Phrase" matching is formulated as ranking over the dictionary ($|Y|$ candidates). Base metrics include:

1. Top-k Accuracy — fraction of examples where the true phrase appears in top-k;
2. MRR (mean reciprocal rank) — average of inverse ranks;
3. Recall@k — fraction of true phrases retrieved among the first k candidates;
4. nDCG@k — normalized discounted cumulative gain with logarithmic decay by position (when relevant paraphrases exist).

For all ranking metrics, two-sided 95% confidence intervals are provided, estimated via bootstrap over episodes (≥10,000 resamplings) with percentile or BCa correction.

### Classification

For coarse category and domain (*read/imagine*) predictions, reporting includes accuracy, macro-F1 (class-average), and balanced accuracy (average of class sensitivities), as well as error matrices. For class imbalance, macro-precision/recall and class-specific reports are added.

### Calibration

Quality of confidence estimates is assessed via ECE/ACE and Brier score (before/after calibration by temperature or isotonic). Reliability curves are shown, and for retrieval, accuracy vs. confidence thresholds (risk–coverage) are plotted. Calibration parameters are tuned exclusively on validation.

### Comparative Analysis and Statistics

To test superiority over random ranking, a permutation test (permutations of answers within query) is used. Model and ablation comparisons are conducted via two-sided permutation test on metric differences with FDR control (Benjamini-Hochberg). Reports specify number of permutations, p-values, and corrected thresholds.

### Baselines

Considered:

- (a) Random (uniform ranking of dictionary);
- (b) Band features + linear model/logreg (energies of δ/θ/α/β/γ bands and ratios);
- (c) EEGNet and ShallowFBCSPNet;
- (d) Our stack without VQ (contrastive encoder without discretization);
- (e) Full variant (CNN→Transformer→VQ + multi-task heads).

All baselines are trained on the same splits and protocols; hyperparameters are tuned on validation with identical budgets.

### Ablations

Component contributions are evaluated: −VQ, −Recon, −AuxHeads, −Contrastive, optionally −AdvDomain and −SemDecoder. For each ablation, retrieval/classification metrics and calibration scores with CI are reported.

### Minimum Metric Set in Current Version (v1)

On the *imagine* domain, the model shows Top-5 Accuracy = 0.8288 and MRR = 0.6340; for coarse classification, Accuracy = 1.0000 is fixed on the current validation subset. Mean cosine similarities: $cos(z_q, v_{text})=0.5836$, $cos(\hat v_{text}, v_{text})=0.7279$, $cos(z_q, \hat v_{text})=0.7205$.

> Note: All other stated metrics (Top-1/10, Recall@k, nDCG@k, macro-F1/BA, ECE/ACE, Brier, permutation tests, bootstrap-CI across all splits) will be added in $v_{n+1}$ after full evaluation runs on fixed splits and baselines.

---

## 9. Results (Preliminary, Overview)

In this review version, we focus on qualitative illustrations and representative numerical examples, deferring full metric summaries to the final version. At the embedding level, space alignment is observed: the mean cosine similarity between quantized EEG representations and reference text vectors reaches ~0.584, while predicted semantic vectors from the decoder show even higher alignment with the reference (~0.728), indicating correct learning direction and benefit of auxiliary semantic decoding. On the *imagine* domain, the model achieves Top-5 Accuracy ≈ 0.829 and MRR ≈ 0.634 on the current subset, confirming its ability to rank relevant phrases among dictionary neighbors.

Discretization via codebook reveals moderate, but not extreme, code "vitality": ~16% of codebook entries are used (41/256), average code usage ~31.3, reflecting both frequency heterogeneity and presence of dominant patterns. Visualization of code frequency distributions and perplexity, and projections of $z_q$ and text vectors (t-SNE/UMAP), show partial clustering by coarse categories and domains, yet overlapping regions are observed where domain-adversarial regularization helps.

Auxiliary classifiers show high accuracy on coarse level and stable domain predictions, enabling their use as priors in ranking. Reliability curves after temperature calibration move closer to the diagonal, and Brier scores decrease (full values will be added in the final version). We also include examples of top-k outputs with commentary on "paraphrase proximity" errors, where the true phrase and top candidate are semantically equivalent despite different wording; such cases motivate use of nDCG@k and soft-contrastive learning.

Finally, we include illustrations:
1. Code frequency distribution and codebook perplexity;
2. 2D projections of embeddings $z_q$, $\hat v_{text}$, and $v_text$;
3. Example rankings with cosine similarities;
4. Error matrices for coarse/domain;
5. Reliability curves before/after calibration.

> Note: In the final version, full tables across all splits (within-/cross-session, LOSO) with 95% confidence intervals, comparisons with baselines, and ablation study results, as well as permutation p-values for key comparisons, will be published.

---

## 10. Reproducibility and Artifact Package

* **Environment:** `conda env.yml` or `Dockerfile`, version/seed fixation, `requirements.txt`.
* **Configurations:** Hydra configs (`configs/`), `Makefile`/`invoke` for scripts.
* **Data:** Loading/validation scripts, manifest and hashes of phrase dictionary/mapping.
* **Training/Evaluation:** `train.py`, `eval.py`, `ablation.py`, `baseline.py`.
* **Indexing:** Candidate index building script; fixed versions of text encoder.
* **Logs/Artifacts:** DVC (`dvc.yaml`) or equivalent, checkpoints, configs, reports.
* **MODEL_CARD.md:** Purpose, limitations, risks, applications.

---

## 11. Limitations

* Small number of subjects and variability in inter-subject transfer.
* Possible noise in phrase labeling, absence of OOD evaluation.
* No online adaptation or real-time experiments in v1.

---

## 12. Practical Value and Applications

The reproducible framework is useful for inner speech research, BCI interface prototyping, and as a component of multimodal "brain↔computer" systems. Unified data contracts and standardized processing steps simplify model comparison and result accumulation.

---

## 13. Conclusion and Plans

We present a systematic "EEG→Text" pipeline with emphasis on reproducibility and transparent evaluation. In upcoming versions:

1. Full result tables with statistics and CI;
2. Strong baselines and ablations;
3. Cross-session and cross-subject transfer;
4. Extended confidence calibration and VQ diagnostics.

---

## Acknowledgments

*Here — acknowledgments to colleagues, lab, and data contributors.*

## References

- [Lee et al., 2020] Neural Decoding of Imagined Speech and Visual Imagery as Intuitive Paradigms for BCI Communication. IEEE TNSRE 28(12):2647–2659.
- [Abdulghani et al., 2023] Imagined Speech Classification Using EEG and Deep Learning. MDPI Bioengineering 10(6):649.
- [Alharbi & Alotaibi, 2024] Decoding Imagined Speech from EEG Data: A Hybrid Deep Learning Approach to Capturing Spatial and Temporal Features. Life 14(11):1501.
- [Amrani et al., 2024] Deep Representation Learning for Open Vocabulary EEG‑to‑Text Decoding. IEEE JBHI / arXiv:2312.09430.
- [El Gedawy et al., 2025] Bridging Brain Signals and Language: A Deep Learning Approach to EEG‑to‑Text Decoding. arXiv:2502.17465.
- [Jiang et al., 2025] Neural Spelling: A Spell‑Based BCI System for Language Neural Decoding. arXiv:2501.17489.

---

### Appendix A. Anti-leakage Checklist

* Fit all preprocessors/scalers/ICA/wavelet thresholds — **only on train per-fold**.
* Hyperparameters tuned on **validation**, not test.
* Text encoder for index/labeling **not trained** on your data and distinct from model's; if same — remove semantic fallback.
* Index and phrase dictionary fixed before training (hashed).
* Time-leak check: strict inference time window, no access to post-event signals.

### Appendix B. VQ Diagnostics

* Perplexity and code "vitality", frequency distributions.
* Impact of codebook size and $\beta$ on quality and interpretability.
* Examples of code patterns linked to coarse/domain.

### Appendix C. "Repro-package" Structure (Example)

```
repo/
  ├─ env.yml | Dockerfile
  ├─ configs/
  ├─ data/
  │   └─ manifests/, hashes/
  ├─ src/
  │   ├─ preprocess/
  │   ├─ dataset/
  │   ├─ models/
  │   ├─ train.py  eval.py  ablation.py  baseline.py
  ├─ index/
  ├─ artifacts/  logs/
  ├─ MODEL_CARD.md  README.md  LICENSE
```

### Appendix D. Model Card (MODEL_CARD.md, Short Template)

**Purpose.** Retrieval over phrase dictionary and classification of coarse/domain from EEG.
**Data.** Source, acquisition protocol, consents.
**Training.** Architecture, losses, hyperparameters.
**Evaluation.** Splits, metrics, CI.
**Limitations.** Domains where model is unreliable.
**Ethics and Risk.** Privacy, permissible applications.
**Contact.** Support and connection.