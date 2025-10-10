[![arXiv](https://img.shields.io/badge/arXiv-2501.12948-b31b1b.svg  )](https://arxiv.org/abs/2504.03624  )
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow  )](https://huggingface.co/nvidia/Nemotron-H-47B-Base-8K  )

# Nemotron-H: A Hybrid Transformer+Mamba for Long Sequences

**Nemotron-H** from NVIDIA combines Mamba-2 (SSM) layers and limited self-attention to:
- 🚀 Accelerate inference by up to **3×**  
- 🎯 Maintain or surpass the accuracy of Llama-3.1 and Qwen-2.5  
- 💾 Reduce memory requirements via **FP8**  
- 🔧 Create compact versions using **MiniPuzzle**

## Why Hybrid?
- **Mamba-2** provides constant O(1) computation and memory per token  
- **Self-attention** (8% of layers) captures global context for in-context learning  

## Key Features
- **FP8 Training:** E4M3/E5M2 + BF16 in critical layers  
- **MiniPuzzle:** Pruning and distillation from 56B → 47B  
- **Flexibility:** VLM, instruction tuning, and long-context support  
- **Checkpoints:** 8B, 47B, and 56B available on Hugging Face and NGC

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>