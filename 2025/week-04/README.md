# ModernBERT: Новое поколение моделей-кодировщиков для эффективного NLP 🚀

[![arXiv](https://img.shields.io/badge/arXiv-2406.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2412.13663)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Verbasik/Weekly-arXiv-ML-AI-Research-Review/blob/main/2024/week-XX/experiments.ipynb) 
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/answerdotai/ModernBERT-base)
[![Telegram](https://img.shields.io/badge/📢_Telegram_Channel-2CA5E0?style=flat)](https://t.me/TheWeeklyBrief)

> p.s. чуть позже актуализирую ссылку на тетрадку с кодом дообучением ModerBERT

## **«Эволюция BERT наконец здесь — быстрее, умнее, с поддержкой длинного контекста»**

## 📌 Основные особенности

- 🚀 **В 2-4 раза быстрее**, чем DeBERTaV3
- 📏 **Длина контекста до 8k токенов** (в 16 раз больше, чем у BERT)
- 💻 **Понимание кода**
- ⚡ **Эффективное использование памяти** (<1/5 от DeBERTa)
- 🧩 **Гибридное внимание** (локальное + глобальное)

## 📊 Сравнение производительности

| Метрика               | BERT | RoBERTa | DeBERTaV3 | ModernBERT |
|----------------------|------|---------|-----------|------------|
| GLUE Score           | 78.3 | 88.5    | 91.2      | **92.1**   |
| Скорость вывода (токен/с)| 1420 | 1630    | 980       | **2400**   |
| Использование памяти (ГБ)| 1.2  | 1.5     | 5.8       | **1.1**    |
| Длина контекста       | 512  | 512     | 512       | **8192**   |

## 🧠 Инновации в архитектуре

1. **Rotary Position Embedding (RoPE)**  
   Обеспечивает лучшее понимание позиций для длинных контекстов.

2. **GeGLU Activation**  
   Улучшает нелинейные возможности модели.

3. **Гибридный механизм внимания**  
   Чередование слоев глобального и локального внимания.

4. **Обучение без заполнения**  
   Упаковка последовательностей для повышения эффективности на 20%.

## 🌟 Основные применения

- 🔍 **RAG-системы с длинным контекстом**
- 💻 **Поиск и анализ кода**
- 📰 **Понимание документов**
- 📊 **Семантический поиск**

## 📜 Цитирование

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

<p align="center">⚡ Преобразите ваш NLP-пайплайн с ModernBERT уже сегодня!</p>