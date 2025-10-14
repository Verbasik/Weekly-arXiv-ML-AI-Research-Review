[![arXiv](https://img.shields.io/badge/arXiv-2506.06105-b31b1b.svg)](https://arxiv.org/abs/2506.06105)
[![GitHub](https://img.shields.io/badge/GitHub-text-to-lora-black?logo=github)](https://github.com/SakanaAI/text-to-lora)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/SakanaAI)
[![Telegram Channel](https://img.shields.io/badge/Telegram-TheWeeklyBrief-blue)](https://t.me/TheWeeklyBrief)

# Text-to-LoRA: Generate Adapters from Task Descriptions

> A revolutionary method from Sakana AI for instant language model adaptation via text prompts. Forget fine-tuning—just describe the task, and the system generates optimal LoRA weights!

Researchers at Sakana AI have developed **Text-to-LoRA (T2L)**, a hypernetwork that dynamically generates Low-Rank Adaptation (LoRA) weights for large language models based on natural language descriptions of target tasks. This method enables efficient, zero-shot adaptation, surpassing established baselines and achieving performance comparable to fine-tuned adapters on previously unseen tasks.

---

## 🌟 Key Capabilities

* **Zero-shot adaptation:** Obtain functional adapters for new tasks without training (up to +8% accuracy vs. base models)
* **Speed:** Adapter generation in a single forward pass (~0.5 sec on A100)
* **Universality:** Support for Mistral-7B, Llama-3-8B, and Gemma-2B
* **Semantic understanding:** Responds to nuances in task descriptions ("sentiment analysis" vs. "sarcasm detection")
* **Efficiency:** 4× fewer FLOPs than in-context few-shot learning

---

## 🛠️ How It Works

```python
from text_to_lora import T2LGenerator

# Initialize generator
generator = T2LGenerator("sakana-ai/t2l-large")

# Generate LoRA from description
lora_weights = generator.generate(
    "Model for analyzing medical reports with emphasis on detecting contradictions"
)

# Apply to base model
model.apply_lora(lora_weights)
```

---

## 📊 Architecture Comparison

| Version  | Parameters | Accuracy (Avg) | Generation Time |
|----------|------------|----------------|-----------------|
| **T2L-L** | 142M       | 78.2%          | 520 ms          |
| **T2L-M** | 89M        | 77.1%          | 340 ms          |
| **T2L-S** | 47M        | 75.3%          | 210 ms          |

*Results on 10 benchmarks (MMLU, GSM8K, HumanEval)*

---

## 🚀 Performance

* **Code Generation (HumanEval):**
  - Base model: 68.5%
  - T2L-L: 76.9% (+8.4%)
  - Manual LoRA: 77.2%

* **Mathematics (GSM8K):**
  - Few-shot: 72.1%
  - T2L-L: 79.4% (+7.3%)

* **Deployment Speed:**
  - Traditional LoRA: 15–60 min training
  - T2L: <1 sec generation

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>