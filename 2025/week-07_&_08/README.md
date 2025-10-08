# 🧠 From Generation to Reasoning: The Evolution of Language Models from Generative Pre-trained Transformers to Reasoning Systems

[![arXiv](https://img.shields.io/badge/arXiv-2305.14705-b31b1b.svg  )](https://arxiv.org/abs/2501.12948  )
[![GitHub](https://img.shields.io/badge/GitHub-DeepSeek--R1-brightgreen  )](https://github.com/deepseek-ai/DeepSeek-R1  )
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow  )](https://huggingface.co/deepseek-ai/DeepSeek-R1  )

Modern large language models (LLMs) have undergone a long journey from simple generative mechanisms to sophisticated reasoning systems. This review explores key advancements in this field, with particular emphasis on methods that enable models not only to generate text but also to reason 🤖. At the center of attention is the DeepSeek-R1 architecture, which employs reinforcement learning (RL) to enhance the model's reasoning capabilities.

One of the key techniques examined in this review is **Chain-of-Thought (CoT)** — a method in which the model generates a sequence of intermediate steps before producing a final answer, thereby increasing the transparency of reasoning. An extension of this approach is **Self-Consistency (CoT-SC)**, which uses ensembles of possible reasoning paths to select the most consistent answer. Further advancing this paradigm, **Tree-of-Thoughts (ToT)** organizes the thinking process as a decision tree, enabling the model to dynamically explore multiple solution pathways 🌳.

Particular interest lies in the **DeepSeek-V3** architecture, which leverages **Mixture-of-Experts (MoE)** and **Multi-Head Latent Attention (MLA)** to scale computations and improve efficiency in processing long sequences. Another critical technology is **FP8 training**, which reduces memory consumption and accelerates computations ⚡, along with the **YaRN** algorithm, which extends the context window up to 128 thousand tokens.

Reasoning model training has advanced further with the emergence of **LLM Programs** — a paradigm in which LLMs are integrated into algorithmic structures, enabling the solution of more complex tasks. These approaches allow for decomposition of computations, minimization of errors, and determinism in logical operations. The review also addresses **reinforcement learning (RL)** and **Group Relative Policy Optimization (GRPO)**, which make models more adaptive to tasks requiring logical inference 🎯.

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>