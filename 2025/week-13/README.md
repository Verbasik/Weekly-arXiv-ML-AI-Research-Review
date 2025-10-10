# DAPO: An RL Algorithm for Training Large Language Models 🚀🤖**

[![arXiv](https://img.shields.io/badge/arXiv-2501.12948-b31b1b.svg  )](https://arxiv.org/abs/2503.14476  )

Hello everyone! Today we are excited to introduce **DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization)** — an innovative algorithm from ByteDance that opens new horizons in training large language models using reinforcement learning (RLHF).

## What is DAPO?
**DAPO** is an open platform for scalable RL training of LLMs that addresses key challenges in modern reasoning model training: entropy collapse, reward noise, and inefficiency in processing long reasoning chains. The algorithm introduces four core innovations that transform the rules of reinforcement learning.

### Key Features of DAPO:
- **Clip-Higher**: Asymmetric clipping bounds that prevent entropy collapse and encourage exploration of low-probability tokens.
- **Dynamic Sampling**: A dynamic sampling method that excludes examples with zero gradients and accelerates model convergence.
- **Token-Level Policy Gradient Loss**: Gradient computation at the individual token level, enabling efficient training on long sequences.
- **Overlong Reward Shaping**: An intelligent length penalty system that reduces reward noise and stabilizes training.

---

## Why is this important?
DAPO achieves record-breaking results on the AIME 2024 test, reaching **50 points** using the Qwen2.5-32B base model, surpassing the previous record of DeepSeek-R1-Zero-Qwen-32B (47 points) with **half the number of training steps**. This approach unlocks new possibilities for developing models capable of solving complex mathematical problems and performing multi-step reasoning.

Moreover, the project is fully open-source: the code, data, and methodology are available to the research community. This makes DAPO a significant step toward reproducible research in RLHF.

---

## How does it work?
DAPO is built on the **verl** framework and includes the following key components:

### 🟢 **Clip-Higher**
- Splits the clipping range into a lower bound (`ε_low = 0.2`) and an upper bound (`ε_high = 0.28`).
- Enables increased probability for "long-tail" tokens, preserving generation diversity and preventing premature policy determinism.

### 🟠 **Dynamic Sampling**
- Excludes output groups with identical rewards (e.g., 0 or 1) that generate no useful gradients.
- Dynamically replenishes the batch with examples exhibiting intermediate accuracy, enhancing data utilization and accelerating convergence.

### 🔵 **Token-Level Policy Gradient Loss**
- Weights each token’s contribution to the loss function, instead of averaging over the entire sequence.
- Promotes effective learning on long reasoning chains, preventing suppression of meaningful patterns.

### 🟣 **Overlong Reward Shaping**
- Replaces a hard length penalty with a gradual linear function.
- Responses up to 16K tokens receive full reward; within the 16–20K token range, the penalty increases linearly from 0 to -1. This reduces noise and allows the model to learn from partially correct long solutions.

---

## Experimental Results
- **AIME 2024**: The DAPO-trained Qwen2.5-32B model achieved a record **50 points**, outperforming DeepSeek-R1-Zero-Qwen-32B (47 points) using **half the training steps**.
- **Elimination of KL Divergence**: Removing the KL penalty allowed the model greater freedom to develop complex reasoning chains.
- **Dynamic Model Evolution**: During training, the model not only reinforces existing reasoning patterns but also develops fundamentally new capabilities, such as self-checking and reconsidering prior steps.

---

## Open Access
The project is fully open to the community:
- **Code**: DAPO implementation is available on [GitHub](https://github.com/volcengine/verl  ).
- **Datasets**: The carefully curated **DAPO-Math-17K** dataset is included in the repository.
- **Framework**: The algorithm is integrated into the **verl** framework for ease of use.

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>