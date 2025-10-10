[![arXiv](https://img.shields.io/badge/arXiv-2501.12948-b31b1b.svg  )](https://arxiv.org/abs/2503.21676  )

# How Do LLMs Learn Facts and Why Do They Hallucinate?

![Figure 2](https://raw.githubusercontent.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/refs/heads/develop/2025/week-15/assets/Figure_02.png  )

## 📝 Description

This repository contains a detailed overview of the research **"How Do Language Models Learn Facts? Dynamics, Curricula, and Hallucinations"** conducted by researchers from Google DeepMind and ETH Zürich. It examines the process by which large language models (LLMs) acquire factual knowledge and the reasons behind their propensity for hallucinations. This study provides an in-depth analysis of the learning dynamics that occur as language models learn to associate entities with their attributes.

## 🔍 Key Features of the Research

- **Three-Phase Learning Process**: Initial language understanding → Performance plateau → Knowledge emergence;
- **Neural Mechanisms for Knowledge Storage and Retrieval**: Distribution of information across attention layers and MLPs;
- **Impact of Data Distribution**: How the frequency of individual occurrences affects learning speed and accuracy;
- **Data Curriculum Strategies**: Effective approaches for optimizing training;
- **Hallucinations and Knowledge Distortion**: Causes of false fact generation and methods to minimize them.

## 📈 Key Findings of the Research

### 1. Three-Phase Learning Process

The study identifies three primary phases through which language models progress during learning:

- **Phase 1: Initial Language Understanding** — The model learns the overall statistics of attribute values;
- **Phase 2: Performance Plateau** — A critical period during which neural circuits for subsequent knowledge acquisition are formed;
- **Phase 3: Knowledge Emergence** — Rapid development of the ability to link individuals with specific attributes.

The duration of the plateau depends on the number of individuals in the dataset:

```
Plateau_Duration ≈ 0.43 × (Number_of_Individuals)^0.81
```

### 2. Neural Mechanisms for Knowledge Storage and Retrieval

Knowledge is distributed across several model components:
- **Early Attention Layers**: Process name tokens to form a query;
- **Middle MLP Layers**: Act as associative memory;
- **Final Attention Layers**: Retrieve specific attributes for queried individuals.

### 3. Hallucinations and Knowledge Distortion

The study shows that hallucinations emerge simultaneously with knowledge acquisition. Models begin confidently generating incorrect information about unfamiliar individuals, even though they initially correctly express uncertainty.

### 4. Fine-Tuning Challenges

Fine-tuning on new data can lead to:
- Distortion of existing knowledge;
- Vulnerability of associative memory;
- Stability of attention patterns.

## 🛠️ Practical Implications

The study offers several recommendations for LLM developers:
- **Optimizing Data Curricula** to reduce training time;
- **Methods for Mitigating Hallucinations**;
- **Alternative Fine-Tuning Approaches**, such as sparse fine-tuning or architectural modifications.

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>