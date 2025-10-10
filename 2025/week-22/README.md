[![arXiv](https://img.shields.io/badge/arXiv-2504.02495-b31b1b.svg)](https://arxiv.org/abs/2504.02495)

# Inference-Time Scaling for Generalist Reward Modeling

> Generative Reward Modeling (GRM) with Inference-Time Scaling (ITS) enables reward models to effectively compete with specialized models by scaling computation during query execution.

## 🚀 Key Achievements

* 🌐 **Generality and Flexibility** — A single generative evaluation model for diverse tasks, effectively adapting to various query types.
* 📈 **Inference-Time Scaling (ITS)** — Significant improvement in evaluation quality through increased computation (samples) during inference without model retraining.
* 🎯 **Meta Reward Model** — Enhanced voting via an auxiliary model that evaluates critique quality, substantially improving accuracy.
* 🔍 **Evaluation Transparency** — Generation of detailed textual principles and critical analyses, ensuring interpretable outcomes.
* ⚙️ **Self-Principled Principles (SPCT)** — A unique approach, Self-Principled Critique Tuning, that enhances the model’s ability to generate meaningful evaluation criteria.

## Why is GRM with ITS Important?

| Problems with Traditional Evaluation Models | Solution from GRM with ITS |
| ------------------------------------------- | -------------------------- |
| Narrow specialization and poor generalization | Universal model trained on a broad range of tasks |
| Lack of evaluation transparency | Generation of explicit principles and critical analyses |
| Limited use of computational resources during inference | Efficient scaling with compute during inference |
| Difficulty adapting to new tasks | Dynamic generation of principles and critique tailored to specific queries |
| Incomplete and noisy evaluations | Meta Reward Model filters out low-quality evaluations |

## ⚙️ Key Methodological Components

1. **Generative Reward Modeling (GRM):**

   * Generation of detailed textual critiques and evaluation principles instead of simple numerical values
   * Dynamic formulation of evaluation criteria per query

2. **Self-Principled Critique Tuning (SPCT):**

   * Rejective Fine-Tuning: Initial tuning of the model to generate correct critiques and principles
   * Rule-Based RL Fine-Tuning: Further training via reinforcement learning based on simple ranking rules

3. **Inference-Time Scaling (ITS):**

   * Parallel sampling and voting to enhance evaluation accuracy and robustness
   * Naive Voting (simple voting) and Meta Reward Model (enhanced weighted voting)

## 🔬 Results

* **Outperformance of Specialized RMs:** DeepSeek-GRM-27B with ITS surpassed 70B+ and even 340B parameter models (Nemotron-4) on multi-domain benchmarks
* **Scaling Efficiency:** Evaluation quality improves nearly linearly with increased sampling during inference
* **Reduced Systematic Bias:** Lower bias compared to traditional evaluation models
* **Approach Universality:** Consistently high performance across tasks of varying complexity, including PPE, Reward Bench, and RealMistake

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>