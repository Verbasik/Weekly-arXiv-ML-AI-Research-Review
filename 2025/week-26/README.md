[![arXiv](https://img.shields.io/badge/arXiv-2506.01928-b31b1b.svg)](https://arxiv.org/abs/2506.01928)
[![GitHub](https://img.shields.io/badge/GitHub-Eso--LMs-black?logo=github)](https://github.com/s-sahoo/Eso-LMs)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/sahoo-diffusion/eso-lms-6838e86cb2c49f45302f0092)

# Diffusion Language Models

> A survey of modern approaches to text generation via iterative denoising (Diffusion LM) as an alternative to autoregressive transformers. Four key architectures are examined: Gemini Diffusion (Google), Mercury Coder (Inception Labs), LLaDA (Chinese researchers), and the hybrid Eso-LM (NVIDIA & Cornell).

---

## 🚀 Key Achievements

* **Parallel text generation:** Instead of token-by-token autoregression, entire text fragments are updated in a fixed number of steps, achieving **1000–2000 tokens/s** (5–15× faster) on modern GPUs.

* **Iterative self-correction:** The model can "rewrite" incorrectly generated tokens in subsequent steps, reducing the risk of error accumulation and hallucinations.

* **Flexibility and editability:** Native support for masked editing and infilling, ideal for interactive code or text editing within existing fragments.

* **Hybrid schemes (Eso-LM):** Combination of MDM phase (large-scale parallel restoration) and AR phase (precise autoregressive completion) enables balancing between speed and quality.

---

## ⚙️ Architecture Overview

| Model               | Paradigm                     | Parallel Steps | Key Features                                   |
| ------------------- | ---------------------------- | -------------- | ---------------------------------------------- |
| **Gemini Diffusion** | Full MDM                     | ~100           | Bidirectional attention, permutation schedules |
| **Mercury Coder**   | Score-based diffusion        | ~50            | Score-entropy, adaptive masking, high throughput |
| **LLaDA 8B**        | Masked diffusion (MLM-based) | ~50            | ELBO-max, scaled to 8B parameters              |
| **Eso-LM (A/B)**    | Hybrid MDM + AR              | 1 (MDM) + LAR (AR) | Causal masked attention, KV caching in diffusion |

---

## 🔬 Key Results

* **Perplexity (PPL)**

  * Gemini Diffusion: ≈26–28 (comparable to AR-Gemini)
  * LLaDA-8B: ≈22–24 (level of LLaMA2-7B / LLaMA3-8B)
  * Eso-LM (A): ≈25.9; (B): ≈27.3

* **Code Generation (HumanEval)**

  * Mercury Coder Small: 78% ✔ (~5× faster than GPT-4o Mini)
  * Mercury Coder Mini: 85% ✔ (~1100 tokens/s on H100)

* **Inference Speed**

  * Gemini Diffusion: 600–1300 tokens/s
  * Mercury Coder: 737–1109 tokens/s
  * Eso-LM (B) with KV caching: +65% speedup vs baseline MDM

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>