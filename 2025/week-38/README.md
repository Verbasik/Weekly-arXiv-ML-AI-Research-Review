# EEG→Text: A Reproducible Pipeline for Recovering the Semantics of Inner Speech from EEG Signals 🧠📝

## 📝 Description

✍️ I am preparing a preprint for arXiv — the foundation of my master's thesis: **EEG→Text** — a reproducible end-to-end pipeline for recovering the semantics of inner speech from EEG signals. The system standardizes signal preprocessing, enforces explicit data contracts, and employs multi-task training of an EEG encoder with vector quantization. The project formalizes the task as retrieval over a fixed phrase dictionary with parallel classification of semantic categories and domain (*read/imagine*).

- **Goal**: Learn to recover meaning from EEG signals — predict phrases, coarse semantic categories, and domain (*read/imagine*)
- **Format**: A practical, reproducible pipeline from raw EDF files to inference on a trained model

## 🔍 Key Features

- **Standardized preprocessing**: Noise and artifact suppression, referencing, automated cleaning and ICA, adaptive wavelet filtering, and normalization
- **Explicit data contracts**: Format validation, channel alignment, phrase-to-epoch binding, category mapping
- **Multi-task training**: Contrastive brain-text alignment, category classification, *read/imagine* domain classification, signal reconstruction
- **Vector quantization**: Discretization of representations to stabilize the latent space
- **Rigorous evaluation protocols**: within-session, cross-session, cross-subject with anti-leakage measures
- **Reproducibility**: Fixed versions, checkpoints, configs, and a "repro-package"

## 📈 Results and Impact

- **Retrieval**: Top-5 Accuracy = 0.8288 and MRR = 0.6340 on the *imagine* domain
- **Classification**: Accuracy = 1.0000 for coarse categories on the validation subset
- **Space alignment**: Cosine similarity between EEG and text vectors ~0.584
- **Discretization**: ~16% of codebook entries used (41/256), reflecting the presence of dominant patterns

## 🧠 Pipeline Overview (Briefly)

- **EEG preprocessing**: Noise filtering, referencing, ICA, automated artifact labeling, wavelet filtering
- **Dataset assembly**: Format validation, normalization, phrase-to-epoch binding, category mapping
- **Model architecture**: CNN+Transformer encoder with vector quantization and multi-task heads
- **Training**: Contrastive, classification, and reconstruction losses with domain-adversarial regularization
- **Inference**: Cosine similarity matching against reference vectors, top-k ranking, confidence calibration

## 🌟 Practical Applications

- **BCI interfaces**: Foundation for non-invasive "brain-computer" interfaces
- **Cognitive neuroscience**: Investigation of inner speech and semantic representations
- **Multimodal systems**: Component for "brain↔computer" systems
- **Standardization**: Unified contracts and formats for research in the field

## 📜 Citation

```bibtex
@misc{verbasik2025eegtotext,
  title        = {EEG→Text: A Reproducible Pipeline for Recovering the Semantics of Inner Speech from EEG Signals},
  author       = {Verbatsev, Eduard Igorevich},
  year         = {2025},
  howpublished = {Preprint},
  url          = {https://verbasik.github.io/Weekly-arXiv-ML-AI-Research-Review/#2025/week-38}
}
```

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>