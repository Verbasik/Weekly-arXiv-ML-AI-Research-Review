[![arXiv](https://img.shields.io/badge/arXiv-2506.01928-b31b1b.svg)](https://arxiv.org/abs/2506.01928)
[![GitHub](https://img.shields.io/badge/GitHub-Eso-LMs-brightgreen)](https://github.com/s-sahoo/Eso-LMs)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/sahoo-diffusion/eso-lms-6838e86cb2c49f45302f0092)
[![Telegram Channel](https://img.shields.io/badge/Telegram-TheWeeklyBrief-blue)](https://t.me/TheWeeklyBrief)

# Диффузионные языковые модели

> Обзор современных подходов к генерации текста через итеративный денойзинг (Diffusion LM) как альтернативы авторегрессионным трансформерам. Рассмотрены четыре ключевые архитектуры: Gemini Diffusion (Google), Mercury Coder (Inception Labs), LLaDA (китайские исследователи) и гибрид Eso-LM (NVIDIA & Cornell).

---

## 🚀 Основные достижения

* **Параллельная генерация текста:** вместо по-токенной автогрессии, весь фрагмент текста обновляется за фиксированное число шагов, что даёт **1000–2000 токенов/с** (в 5–15× быстрее) на современных GPU.

* **Итеративное самокорректирование:** модель может «переписать» ошибочно сгенерированные токены на следующих шагах, снижая риск накопления ошибок и галлюцинаций.

* **Гибкость и редактируемость:** естественная поддержка masked editing и infilling, идеально для интерактивного редактирования кода или текста внутри уже написанного фрагмента.

* **Гибридные схемы (Eso-LM):** комбинация MDM-фазы (масштабное параллельное восстановление) и AR-фазы (точное автодополнение) позволяет балансировать между скоростью и качеством.

---

## ⚙️ Обзор архитектур

| Модель               | Парадигма                    | Параллельных шагов | Ключевые особенности                             |
| -------------------- | ---------------------------- | ------------------ | ------------------------------------------------ |
| **Gemini Diffusion** | Полная MDM                   | \~100              | Bidirectional attention, permutation schedules   |
| **Mercury Coder**    | Score-based diffusion        | \~50               | Score‐entropy, adaptive masking, high throughput |
| **LLaDA 8B**         | Masked diffusion (MLM-based) | \~50               | ELBO‐мах, масштабирование до 8B параметров       |
| **Eso-LM (A/B)**     | Гибрид MDM + AR              | 1 (MDM) + LAR (AR) | Causal masked attention, KV-кеш в диффузии       |

---

## 🔬 Ключевые результаты

* **Перплексия (PPL)**

  * Gemini Diffusion: ≈26–28 (сравнимо с AR-Gemini)
  * LLaDA-8B: ≈22–24 (уровень LLaMA2-7B/ LLaMA3-8B)
  * Eso-LM (A): ≈25.9; (B): ≈27.3

* **Кодогенерация (HumanEval)**

  * Mercury Coder Small: 78% ✔ (\~5× быстрее GPT-4o Mini)
  * Mercury Coder Mini: 85% ✔ (\~1100 токенов/с на H100)

* **Скорость инференса**

  * Gemini Diffusion: 600–1300 токенов/с
  * Mercury Coder: 737–1109 токенов/с
  * Eso-LM (B) с KV-кешем: +65% ускорения vs базовый MDM

---

⭐ **Понравился обзор?**
Не забудьте поставить ★ и подписаться на канал в Telegram, чтобы не пропустить новые разборы!

<p align="center">Исследуйте вместе с нами 🚀</p>
