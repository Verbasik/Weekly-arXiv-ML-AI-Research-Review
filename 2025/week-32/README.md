[![arXiv](https://img.shields.io/badge/arXiv-2507.18071-b31b1b.svg)](https://arxiv.org/abs/2507.18071)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/papers/2507.18071)

# 🔥 Sunday Dump: GSPO (Qwen RL Algorithm by Alibaba Cloud) 🚀

**Qwen** is back with another release. But this time, it's not a model—it's a new RL algorithm for training LLMs.

The method is called **Group Sequence Policy Optimization (GSPO)**, and it forms the foundation of the company's latest high-profile models: **Qwen3 Instruct**, **Coder**, and **Thinking**. The paper was published just a few days ago, but everyone is already talking about it. So it's time we dive in too.

Today, one of the most popular RL algorithms for LLMs is **GRPO (by DeepSeek)**. If you're unfamiliar with it, read the breakdown here. GRPO works well and is fairly stable—but at the token level.

In GRPO, we:
- Compute the reward for the entire sequence
- Calculate the importance weight for each token and apply clipping individually to each token
- Update the policy at the token level.

In **GSPO**, everything happens at the sequence level:
- Compute the reward
- Calculate a single **importance weight** for the entire sequence and apply clipping to the full response with length normalization
- Update the policy.

### What are the advantages of this approach?

1. **No need for elaborate workarounds** when working with MoE. With GRPO, MoE architecture struggles, but here it works out of the box.
2. **Gradients are less noisy** due to reduced variance. Consequently—more stable training. Consequently—better metrics with the same resources.
3. **Much simpler to implement engineering-wise**.

In short, it looks extremely attractive and is likely to become the next big thing in RL for LLMs (especially in open source).

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>