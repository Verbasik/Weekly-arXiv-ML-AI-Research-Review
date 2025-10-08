# ModernBERT: A New Generation of Efficient NLP Encoders 🚀

[![arXiv](https://img.shields.io/badge/arXiv-2406.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2412.13663)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/answerdotai/ModernBERT-base)

## **"BERT evolution is finally here — faster, smarter, with long-context support"**

## 📌 Key Features

- 🚀 **2-4 times faster** than DeBERTaV3
- 📏 **Up to 8k token context length** (16 times longer than BERT)
- 💻 **Code understanding**
- ⚡ **Efficient memory usage** (<1/5 of DeBERTa)
- 🧩 **Hybrid attention** (local + global)

## 📊 Performance Comparison

| Metric               | BERT | RoBERTa | DeBERTaV3 | ModernBERT |
|----------------------|------|---------|-----------|------------|
| GLUE Score           | 78.3 | 88.5    | 91.2      | **92.1**   |
| Inference Speed (tokens/s)| 1420 | 1630    | 980       | **2400**   |
| Memory Usage (GB)    | 1.2  | 1.5     | 5.8       | **1.1**    |
| Context Length       | 512  | 512     | 512       | **8192**   |

## 🧠 Architectural Innovations

1. **Rotary Positional Embedding (RoPE)**  
   Provides better positional understanding for long contexts.

2. **GeGLU Activation**  
   Enhances the model's non-linear capabilities.

3. **Hybrid Attention Mechanism**  
   Alternating layers of global and local attention.

4. **Fill-Free Training**  
   Sequence packing for 20% higher efficiency.

## 🌟 Primary Applications

- 🔍 **Long-context RAG systems**
- 💻 **Code search and analysis**
- 📰 **Document understanding**
- 📊 **Semantic search**

## 📜 Citation

```bibtex
@misc{modernbert,
      title={Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference}, 
      author={Benjamin Warner and Antoine Chaffin and Benjamin Clavié and Orion Weller and Oskar Hallström and Said Taghadouini and Alexis Gallagher and Raja Biswas and Faisal Ladhak and Tom Aarsen and Nathan Cooper and Griffin Adams and Jeremy Howard and Iacopo Poli},
      year={2024},
      eprint={2412.13663},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.13663}, 
}
```

---

<div align="center">

**Explore with us 🚀**

⭐ Star this repository if you found it helpful

</div>