# 🧠 Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

[![arXiv](https://img.shields.io/badge/arXiv-2501.12948-b31b1b.svg  )](https://arxiv.org/pdf/2502.11089  )

### 📊 Overview

DeepSeek introduces **Native Sparse Attention (NSA)** — an innovative sparse attention architecture designed for efficient long-context processing in large language models (LLMs). NSA addresses the high computational complexity of standard attention mechanisms, delivering significant acceleration in both training and inference without compromising generation quality or reasoning capabilities.

---

## 📌 Research Objectives
- **Improve computational efficiency** when processing long sequences without sacrificing generation quality.
- **Develop an architecture supporting end-to-end training**, eliminating the need for pre-computations.
- **Integrate NSA into hardware-optimized computations**, reducing latency for 64k-token text processing.
- **Create a scalable architecture** compatible with existing LLMs with minimal modifications.
- **Optimize memory consumption**, enabling efficient utilization of GPUs and TPUs for long-sequence processing.

## 💡 Key Aspects of the Paper
- **Natively Designed Sparse Attention** – NSA enables end-to-end optimization of the sparse pattern during pre-training.
- **Hierarchical Sparse Attention Mechanism** – Balances global and local information processing to enhance modeling quality.
- **Dynamic Multi-Level Sparsity** – NSA combines:
  - **Compressed Attention** for information summarization and reduced computational complexity.
  - **Selected Attention** for preserving critical information in long contexts.
  - **Sliding Attention** for capturing local dependencies.
- **Computation Optimization** – NSA leverages hardware accelerations (Triton, Tensor Cores) to balance performance and computational cost.
- **Configurable Flexibility** – Supports various attention modes tailored to computational resources and task requirements.

### 🔬 Results and Efficiency
- **Outperforms full attention** in accuracy and computational speed on long-context tasks.
- **Supports up to 128k-token context** – NSA maintains efficiency even under extremely long sequences.
- **Supervised and RL fine-tuning** – NSA demonstrates enhanced reasoning capabilities in Chain-of-Thought benchmarks.
- **Energy Efficiency** – Significant reduction in energy consumption during long-sequence processing, making NSA ideal for cloud services and mobile devices.

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>