# EEG→Text: a reproducible pipeline for restoring the semantics of inner speech from EEG signals (survey version)

**Author(s):** *Verbetsky Edward Igorevich*

**Affiliation:** *Moscow Aviation Institute (National Research University)*

**Contact:** *verbasik@gmail.com*

**Preprint version:** v1 (survey)

---

## Abstract

We present a reproducible end‑to‑end “EEG → Text” pipeline for studying inner speech in the *read* and *imagine* paradigms. The pipeline standardizes preprocessing (artifact cleaning, re‑referencing, ICA auto‑labeling, adaptive wavelet filtering, normalization), introduces explicit data contracts, and supports multi‑task training of an EEG encoder (CNN+Transformer) with vector quantization and auxiliary heads (coarse semantic categories, *read/imagine* domain, signal reconstruction). We formalize the task as retrieval over a fixed phrase vocabulary with parallel category/domain classification, describe evaluation protocols (within‑session, cross‑session, cross‑subject), leakage‑prevention measures, the metric set (Top‑k, MRR, macro‑F1/BA, confidence calibration), and the structure of a “repro‑package” for results reproducibility. Preliminary experiments indicate that VQ and multi‑task optimization can stabilize and improve the interpretability of representations; in subsequent versions we plan systematic comparisons with strong baselines and transfer across sessions and subjects.

---

## 1. Introduction and Motivation

Restoring the semantics of inner speech from non‑invasive brain signals is a key problem for BCI systems, cognitive neuroscience, and multimodal interfaces. Unlike invasive methods, EEG is safe and accessible, but suffers from low SNR and high inter‑subject variability. We propose a practical pipeline from raw EDF files to model inference, oriented toward reproducibility, standardized formats, and transparent evaluation.

**Contributions (summary):**

1. We formalize “EEG→Text retrieval” over a fixed phrase vocabulary with parallel coarse‑category and domain (*read/imagine*) classification.
2. We describe a standardized preprocessing pipeline with anti‑leakage practices.
3. We present an EEG encoder architecture (CNN+Transformer) with VQ and multi‑task optimization (contrastive + classification + reconstruction).
4. We fix evaluation protocols (within / cross‑session / cross‑subject), metrics, and statistical tests.
5. We publish a “repro‑package”: code, configs, pinned versions and checkpoints (structure and instructions).

---

## 2. Related Work (brief)

Research on decoding “inner speech” from EEG can be roughly divided into three directions. The first is the classical line of work on recognizing mentally articulated units: syllables, letters, words, and short commands. These studies demonstrate that even on budget, low‑channel systems it is possible to extract stable patterns; however, accuracy strongly depends on data quality, subject training, and vocabulary size. The second direction focuses on semantic decoding and alignment of EEG representations with language models’ embeddings, framing the problem as retrieval or classification in a joint latent space. The third direction explores modern deep architectures (CNNs, RNNs, Transformers), self‑supervision, contrastive and triplet objectives, and domain adaptation techniques to improve robustness across sessions and subjects. Our pipeline synthesizes these strands, emphasizing reproducibility, explicit data contracts, and evaluation discipline.

---

## 3. Data and Protocols

We consider inner‑speech experiments in two paradigms: (a) reading phrases (*read*) and (b) silent imagination (*imagine*). Each trial includes an EEG segment associated with a phrase from a fixed dictionary. We define clear splits and protocols:

- Within‑session: training/validation/test partitions inside a single session, with strict separation by trials.
- Cross‑session: training on one session, testing on another session of the same subject, with session‑wise normalization.
- Cross‑subject (LOSO): leave‑one‑subject‑out: training on all but one subject, testing on the held‑out subject; calibration and adapter procedures are allowed only on validation data of the held‑out subject.

We fix the phrase dictionary and mapping manifest with hashes to avoid any leakage. All preprocessing and encoding steps are applied within each split independently.

---

## 4. Preprocessing Pipeline (anti‑leakage)

The preprocessing chain standardizes EEG inputs and minimizes data leakage:

1. Band‑pass filtering and notch filtering (per protocol), removal of powerline noise.
2. Artifact handling: ICA with auto‑labeling of components (ocular/muscle), removal under pinned thresholds.
3. Adaptive wavelet denoising for residual artifacts; channel‑wise normalization.
4. Re‑referencing (e.g., average reference) and per‑session z‑scaling (parameters estimated on training part only).
5. Time‑windowing relative to cue onsets with fixed padding; masking of non‑stimulus segments if required.

All parameters are logged and versioned; seeds are fixed. We provide checks that train/val/test statistics are computed disjointly.

---

## 5. Model: EEG Encoder + Text Space

We use a hybrid encoder for EEG sequences: CNN blocks for local temporal patterns, followed by a Transformer for long‑range dependencies. The encoder outputs a latent vector that is then passed through vector quantization (codebook of size 256) producing a discrete representation. Auxiliary heads are attached:

- Coarse semantic categories (multi‑class classification).
- Domain (*read* vs *imagine*).
- Signal reconstruction (decoder from the quantized latent to the input space) for an auxiliary reconstruction loss.

The text space is defined by a frozen or lightly‑tuned language model encoder producing phrase embeddings. We align EEG and text vectors via a contrastive objective.

---

## 6. Training Objectives

The total loss is a weighted sum:

- Contrastive alignment (InfoNCE / supervised contrastive) between EEG and text embeddings.
- Cross‑entropy for coarse categories and for domain.
- Reconstruction loss (L1/L2) on the signal decoder.
- VQ commitment/codebook losses per standard VQ‑VAE practice.

Weights are selected on validation; we report ablations −VQ, −Recon, −AuxHeads, −Contrastive to quantify contributions.

---

## 7. Retrieval Formulation and Inference

We cast “EEG → phrase” as ranking over the fixed vocabulary. Given a query EEG embedding z (quantized) and text embeddings {v_y}, we compute cosine similarities and rank candidates.

Inference options:

- Direct nearest‑neighbor search in the text index (ANN or exact).
- Hybrid reranking with auxiliary priors (e.g., domain and coarse category scores).

We also explore a semantic decoder that predicts a text‑space vector from EEG, followed by nearest‑neighbor retrieval.

---

## 8. Evaluation

### Retrieval

We report:

1. Top‑k Accuracy — the fraction of examples for which the true phrase is within top‑k.
2. MRR (mean reciprocal rank).
3. Recall@k — the fraction of true phrases retrieved within the first k candidates.
4. nDCG@k — normalized discounted cumulative gain with logarithmic position discount (useful when relevant paraphrases exist).

Two‑sided 95% confidence intervals are provided via bootstrap over episodes (≥10,000 resamples) with percentile or BCa correction.

### Classification

For coarse categories and domain (*read/imagine*) we report accuracy, macro‑F1, and balanced accuracy, plus confusion matrices. For class imbalance we add macro‑precision/recall and per‑class breakdowns.

### Calibration

Confidence quality is evaluated via ECE/ACE and Brier score (before/after temperature or isotonic calibration). We include reliability curves and, for retrieval, risk–coverage plots (accuracy vs confidence threshold). Calibration parameters are tuned on validation only.

### Comparative Analysis and Statistics

To verify superiority over random ranking we use a permutation test (shuffling answers within a query). Model and ablation comparisons use two‑sided permutation tests on metric differences with FDR control (Benjamini–Hochberg). Reports include the number of permutations, p‑values, and adjusted thresholds.

### Baselines

We consider:

- (a) Random (uniform ranking over the vocabulary);
- (b) bandpower features + linear/logistic model (energies in δ/θ/α/β/γ bands and their ratios);
- (c) EEGNet and ShallowFBCSPNet;
- (d) our stack without VQ (contrastive encoder without discretization);
- (e) the full variant (CNN→Transformer→VQ + multi‑task heads).

All baselines follow the same splits and protocols; hyperparameters are tuned on validation with equal budgets.

### Ablations

We evaluate components: −VQ, −Recon, −AuxHeads, −Contrastive, and optionally −AdvDomain and −SemDecoder. For each ablation we report retrieval/classification metrics and calibration with CIs.

### Minimal metric set in the current version (v1)

On the imagine domain the model shows Top‑5 Accuracy = 0.8288 and MRR = 0.6340 on the current validated subset. Average cosine similarities: $cos(z_q, v_{text})=0.5836$, $cos(\hat v_{text}, v_{text})=0.7279$, $cos(z_q, \hat v_{text})=0.7205$.

> Note: the remaining declared metrics (Top‑1/10, Recall@k, nDCG@k, macro‑F1/BA, ECE/ACE, Brier, permutation tests and bootstrap CIs across all splits) will be added in $v_{n+1}$ after full evaluation on fixed splits and baselines.

---

## 9. Results (preliminary, overview)

In this survey version we focus on qualitative illustrations and representative numerical examples, deferring the full set of metrics to the final version. At the embedding level we observe alignment of spaces: the mean cosine similarity between quantized EEG representations and reference text vectors reaches ~0.584, while the semantic vectors predicted by the decoder show even higher alignment with the reference (~0.728). This indicates a proper training direction and the usefulness of auxiliary semantic decoding. In the imagine domain the model achieves Top‑5 Accuracy ≈ 0.829 and MRR ≈ 0.634 on the current subset, confirming the ability to rank relevant phrases among nearest neighbors of the vocabulary.

Discretization via the codebook reveals moderate, but not maximal, “liveliness” of codes: ~16% of codebook entries are used (41/256), the mean number of hits per code is ~31.3, reflecting both uneven frequencies and dominant patterns. Visualizations of code frequencies and perplexity, as well as projections of $z_q$ and text vectors (t‑SNE/UMAP), show partial clustering by coarse categories and domains; overlaps remain, where domain‑adversarial regularization helps.

Auxiliary classifiers demonstrate high accuracy at the coarse level and stable domain predictions, which can be used as a prior in reranking. Reliability curves after temperature calibration get closer to the diagonal, and Brier scores decrease (full values will be added in the final version). We also include examples of top‑k outputs with comments on “paraphrase proximity” errors, when the true phrase and the closest candidate are semantically equivalent but phrased differently; such cases motivate nDCG@k and soft‑contrastive training.

Finally, we include illustrations:
1. Code frequency and codebook perplexity distributions;
2. 2‑D projections of embeddings $z_q$, $\hat v_{text}$, and $v_{text}$;
3. Ranking examples with cosine similarities;
4. Confusion matrices for coarse/domain;
5. Reliability curves before/after calibration.

> Note: the final version will include full tables across all splits (within‑/cross‑session, LOSO) with 95% confidence intervals, comparisons with baselines and ablation results, as well as permutation p‑values for key comparisons.

---

## 10. Reproducibility and Artifact Package

* **Environment:** `conda env.yml` or `Dockerfile`, pinned versions/seeds, `requirements.txt`.
* **Configurations:** Hydra configs (`configs/`), `Makefile`/`invoke` for scenarios.
* **Data:** download/validation scripts, manifest and hashes of the phrase dictionary/mapping.
* **Training/Evaluation:** `train.py`, `eval.py`, `ablation.py`, `baseline.py`.
* **Indexing:** script to build the candidate index; pinned versions of the text encoder.
* **Logs/Artifacts:** DVC/MLflow, checkpoints, scalar summaries, and plots.
* **Docs:** README with step‑by‑step instructions; a “one‑command” reproduce script.

We provide clear instructions to reproduce all reported numbers, including random seeds and evaluation protocols.

---

## 11. Limitations and Ethical Aspects

This work focuses on non‑invasive EEG with limited vocabulary and controlled lab settings. Transfer across subjects remains challenging. Ethical aspects include data privacy, informed consent, and responsible communication of capabilities (avoid over‑claims about mind‑reading; the system retrieves phrases from a fixed dictionary under constrained conditions).

---

## 12. Conclusion

We introduced a reproducible “EEG → Text” pipeline for inner speech in *read* and *imagine* paradigms, with standardized preprocessing, a CNN+Transformer encoder with VQ and auxiliary heads, retrieval formulation, and a disciplined evaluation protocol. Preliminary results suggest that discretization and multi‑task optimization improve stability and interpretability. Future work will include systematic comparisons with strong baselines, robust calibration, and transfer across sessions and subjects.

---
