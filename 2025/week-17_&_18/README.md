[![arXiv](https://img.shields.io/badge/arXiv-2501.12948-b31b1b.svg)](https://arxiv.org/abs/2504.03624)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/nvidia/Nemotron-H-47B-Base-8K)
[![Telegram Channel](https://img.shields.io/badge/Telegram-TheWeeklyBrief-blue)](https://t.me/TheWeeklyBrief)

# Nemotron-H: гибрид Transformer+Mamba для длинных последовательностей

**Nemotron-H** от NVIDIA сочетает слои Mamba-2 (SSM) и ограниченный self-attention, чтобы:
- 🚀 Ускорить инференс до **3×**  
- 🎯 Сохранить или превзойти точность Llama-3.1 и Qwen-2.5  
- 💾 Снизить требования к памяти благодаря **FP8**  
- 🔧 Создавать компактные версии через **MiniPuzzle**

## Почему гибрид?
- **Mamba-2** даёт постоянные O(1) вычисления и память на токен  
- **Self-attention** (8% слоев) ловит глобальный контекст для in-context learning  

## Ключевые особенности
- **FP8-тренинг:** E4M3/E5M2 + BF16 в критичных слоях  
- **MiniPuzzle:** прунинг + дистилляция из 56B → 47B  
- **Гибкость:** VLM, инструктивное обучение и длинный контекст  
- **Чекпоинты:** 8B, 47B, 56B в Hugging Face и NGC

## 🌟 Поддержите проект

- Понравилось? Поставьте звезду и присоединяйтесь к обсуждению!

---

<p align="center">Исследуйте вместе с нами 🚀</p>