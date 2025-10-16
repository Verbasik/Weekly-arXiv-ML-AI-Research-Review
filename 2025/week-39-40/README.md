# GPT-OSS: OpenAI's First Open Models Since GPT-2 🚀

[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--OSS-green)](https://openai.com/index/introducing-gpt-oss/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-blue)](https://huggingface.co/openai)

## 📝 Description

This week we examine the release of **GPT-OSS** — OpenAI’s first fully open models since GPT-2 in 2019. Two models, `gpt-oss-20b` and `gpt-oss-120b`, are modern LLMs with a Mixture-of-Experts (MoE) architecture, optimized for reasoning and capable of running on a single GPU thanks to MXFP4 quantization.

## 🔍 Key Features

- **Open weights**: First fully open models from OpenAI in six years under the Apache 2.0 license
- **MoE architecture**: 32 experts with 4 activated per token (20B) and 128 experts with 8 activated (120B)
- **MXFP4 optimization**: The 120B model fits on a single H100 (80GB); the 20B model runs on an RTX 50xx (16GB)
- **Configurable reasoning effort**: Control via the parameter "Reasoning effort: low/medium/high"
- **Modern architecture**: RoPE, RMSNorm, SwiGLU, Grouped Query Attention, Sliding Window Attention
- **Reasoning specialization**: Trained with emphasis on STEM, programming, and mathematics

## 📈 Results and Performance

- **Model sizes**: 20B and 120B parameters with efficient quantization
- **Memory usage**: 13.5GB for the 20B model on a Mac Mini; 80GB for the 120B model on an H100
- **Training time**: 2.1 million H100 hours (including SFT and RLHF)
- **Benchmark performance**: Comparable to Qwen3 and proprietary OpenAI models
- **Generation speed**: High, due to the wide architecture and optimizations

![](assets/Figure-01.png)
![](assets/Figure-02.jpg)

## 🧠 Architectural Evolution from GPT-2

### Key Changes:
- **Dropout removal**: Unnecessary under single-epoch training on massive datasets
- **RoPE instead of absolute positions**: More efficient positional encoding
- **SwiGLU activations**: Replacement of GELU with a more computationally efficient function
- **RMSNorm**: Replacement of LayerNorm for improved training stability
- **Grouped Query Attention**: Reduced memory consumption without sacrificing quality
- **Sliding Window Attention**: Context limited to 128 tokens in every second layer

![](assets/Figure-05.jpg)
![](assets/Figure-06.jpg)

## 🆚 Comparison with Modern Architectures

### GPT-OSS vs Qwen3:
- **Width vs depth**: GPT-OSS is wider (2880 dim); Qwen3 is deeper (48 layers vs 24)
- **MoE configuration**: GPT-OSS uses fewer but larger experts
- **Attention**: GPT-OSS employs sliding window; Qwen3 uses full attention
- **License**: Both under Apache 2.0, but Qwen3 provides base models

### Technical Advantages:
- **Attention sinks**: Learnable bias logits to stabilize long contexts
- **Attention bias**: An unusual feature in modern models
- **Scaling**: Only depth and number of experts change when moving from 20B to 120B

![](assets/Figure-13.png)
![](assets/Figure-14.png)

## 🌟 Practical Applications

- **Local deployment**: Models run on consumer hardware thanks to MXFP4
- **Reasoning**: Built-in support for varying levels of reasoning complexity
- **Research**: Open weights enable experimentation and distillation
- **Commercial use**: Apache 2.0 license with no restrictions
- **Tool integration**: Optimized for working with external APIs

## 🔗 Links

- Models on Hugging Face: [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)
- Official announcement: [OpenAI Blog](https://openai.com/index/introducing-gpt-oss/)

## 📜 Citation

```bibtex
@misc{openai2025gptoss,
  title        = {From GPT-2 to gpt-oss: Analyzing the Architectural Advances},
  author       = {Sebastian Raschka},
  year         = {2025},
  url          = {https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the}
}
```

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>