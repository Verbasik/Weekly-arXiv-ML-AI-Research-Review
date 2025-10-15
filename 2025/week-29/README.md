[![arXiv](https://img.shields.io/badge/arXiv-2506.01928-b31b1b.svg)](https://arxiv.org/abs/2506.06105)
[![GitHub](https://img.shields.io/badge/GitHub-kimi-k2-black?logo=github)](https://github.com/MoonshotAI/Kimi-K2)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/moonshotai/kimi-k2-6871243b990f2af5ba60617d)
[![Kimi](https://img.shields.io/badge/Kimi-K2-purple)](https://www.kimi.com/)

# Kimi-K2: A MoE Monster with a Trillion Parameters

> 🔥 Sunday dump: The MoE Monster Kimi-K2! A new open-source star from China setting records and breaking price tags!

**Kimi-K2** from Moonshot AI is a revolutionary language model with a Mixture-of-Experts (MoE) architecture, boasting **1 trillion parameters** with only **32 billion activated**. The model was trained on an unprecedented scale (15.5 trillion tokens) using the innovative MuonClip optimizer, ensuring training stability with zero instabilities. Kimi-K2 sets new benchmarks for open models, especially in programming, mathematics, and agentic tasks.

---

## 🌟 Key Features

* **🔥 MoE Architecture:** 1 trillion parameters with 32 billion activated (384 experts, top-8 activation)
* **🤖 Agentic Pro:** The world's best open model for agentic tasks without reasoning
* **👨‍💻 Coding King:** SOTA among non-reasoning models, outperforms GPT-4.1 on SWE-bench (65.8%)
* **📈 Mathematical Power:** Outstanding results on MATH (70.2%) and GSM8K (92.1%)
* **⚡ Stability:** Trained on 15.5T tokens with ZERO loss spikes thanks to MuonClip
* **🌍 Openness:** Fully open-source under the Modified MIT License

---

## 🏗️ Architecture

### Mixture-of-Experts (MoE)

* **1 trillion parameters with 32 billion activated (384 experts, top-8 activation)**

### Multi-Head Latent Attention (MLA)
* **KV-cache compression:** 60× less memory vs standard MHA
* **FlashAttention integration:** For efficient long-context processing
* **RoPE scaling:** Supports 128k tokens without quality degradation

---

## 🚀 Performance

### Academic Benchmarks
* **MMLU (57 subjects):** 87.8% (surpasses GPT-4 ~86.4%)
* **C-Eval (Chinese):** 92.5% (new record for Chinese language)
* **MATH (advanced math):** 70.2% (significant leap)
* **GSM8K (arithmetic):** 92.1% (nearly error-free)

### Programming
* **LiveCodeBench v6:** 53.7% pass@1 (outperforms GPT-4.1 ~44.7%)
* **MultiPL-E:** 85–86% accuracy in multilingual programming
* **SWE-bench Verified:** 65.8% (comparable to proprietary models)

### Agentic Tasks
* **Tau & AceBench:** Top rankings among open models
* **Tool-use scenarios:** 70–75% success rate (approaching Claude 2)

---

## 🛠️ Technological Innovations

### MuonClip Optimizer

**1. Enhanced scalability for large models**  
- Early versions of Muon showed strong results on medium-sized models, but their efficiency on architectures with billions of parameters required refinement.  
- The updated version eliminates bottlenecks related to memory management and computational efficiency, enabling stable training of massive models, including MoE (Mixture of Experts).  

**2. Solved attention logit instability**  
- In the original Muon, aggressive optimization of **Query** and **Key** matrices sometimes led to abnormally high weight values, causing attention logits to explode (reaching 10³–10⁵). This disrupted the attention mechanism and triggered sharp loss spikes.  
- The new version (**MuonClip**) introduces **QK-clip**, a technique that **explicitly bounds weight norms** after each update. This prevents uncontrolled logit growth and stabilizes training, especially during later stages.  

**3. Improved compatibility with MoE architectures**  
- The exploding logits problem was particularly critical in expert-based models (MoE), where imbalance in one layer could disrupt the entire system.  
- MuonClip ensures **more predictable training dynamics** in such models, as confirmed by experiments (e.g., in the **Kimi K2** model).  

Now Muon combines the advantages of **high token efficiency** and **stability**, even when training the largest language models.

### Distributed Training
* **DeepSpeed + ZeRO:** Efficient parameter distribution
* **Expert parallelism:** Natural scaling for MoE
* **Gradient checkpointing:** Memory savings on long sequences

### Post-training
* **Large-Scale Agentic Data Synthesis:** Synthetic agentic tasks
* **Self-critique RL:** Training with self-evaluation without human feedback
* **Reflex-grade model:** Fast optimal responses without prolonged reasoning

---

## 📊 Comparison with Predecessors

| Model | Parameters | MMLU | MATH | SWE-bench | Status |
|-------|------------|------|------|-----------|--------|
| **Kimi-K2** | 1T (32B) | 87.8% | 70.2% | 65.8% | Open |
| Kimi k1.5 | 389B (52B) | ~82% | ~60% | ~45% | API |
| GPT-4.1 | ~1.8T | ~86.4% | ~85% | ~70% | Closed |
| Claude 4 | ~1.5T | ~87% | ~80% | ~68% | Closed |

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>