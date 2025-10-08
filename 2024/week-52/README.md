# 🔬 Study of "Super Weights" in Large Language Models (LLMs)

[![arXiv](https://img.shields.io/badge/arXiv-2411.07191-b31b1b.svg)](https://arxiv.org/abs/2411.07191)

> **"One Parameter Can Break an LLM: How Super Weights Govern Generation Quality"**

## 🧩 Key Findings

- 🎯 **1 parameter > 7000 others** – Removing a single super weight destroys model quality  
- 🔍 **Data-free detection** – Method for identification without training data  
- ⚡ **Practical application** – Improves model quantization by 42%  
- 📊 **Universality** – Results validated across Llama, Mistral, and Phi-3  

## 📋 Key Results

| Metric                     | Without SW | With SW | Delta   |
|---------------------------|------------|---------|---------|
| Accuracy (Zero-Shot)       | 0%         | 54.2%   | +54.2   |
| Perplexity                 | 562.1      | 12.3    | -549.8  |
| Stop-word probability      | +850%      | Normal  | -       |

## 🛠️ Methodology

### 3-stage SW identification:
1. **Activation analysis**  
   Outlier detection in `mlp.down_proj` distributions  
2. **Cross-validation**  
   Verification across diverse input prompts  
3. **Impact verification**  
   Testing the effect of parameter removal  

## 💡 Practical Applications

- 🧮 **Enhanced quantization**  
  Preserving SW yields +42% quality  
- 🔧 **Model optimization**  
  Targeted intervention on critical parameters  
- 🚀 **Efficient engineering**  
  Data-free approach for rapid analysis  

## 📊 Experimental Results

| Model                   | # of SW | Accuracy (INT4) |
|-------------------------|---------|-----------------|
| Llama-7B                | 1       | 82.3%           |
| Mistral-7B              | 2       | 85.1%           |
| Phi-3-mini-4k-instr     | 6       | 79.8%           |

## 📜 Citation

```bibtex
@article{superweights2024,
  title={The Super Weight in Large Language Models}, 
  author={Zheng-Xin Yong and Cyril Zhang},
  year={2024},
  eprint={2411.07191},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

---
<p align="center">Made with ❤️ for the AI research community</p>