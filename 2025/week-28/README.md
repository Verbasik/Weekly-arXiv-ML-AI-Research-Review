[![Telegram Channel](https://img.shields.io/badge/Telegram-TheWeeklyBrief-blue)](https://t.me/TheWeeklyBrief)

# OpenAI Agents SDK: Быстрый старт или дорогой «черный ящик»?

> Глубокий разбор архитектуры агентских инструментов OpenAI от команды The Weekly Brief. Узнайте, когда стоит выбирать этот инструмент, а когда лучше использовать альтернативы вроде LangChain.

🚀 **Новый выпуск в рубрике [#AgentsUnderHood](https://github.com/Verbasik/Weekly-arXiv-ML-AI-Research-Review/tree/develop/agents-under-hood/openai-cs-agents-demo)!**  

---

## 🔍 Основные выводы

* **Архитектура в два уровня**: Assistants API (исполнение) + Agents SDK (оркестрация)
* **Zero-shot мультиагентность**: передача задач между агентами через `Handoffs`
* **Скорость разработки**: рабочий прототип с RAG и тулами за 2 часа
* **Ценовой компромисс**: удобство vs высокая стоимость и vendor lock-in

---

## 🏗️ Архитектурные компоненты

```python
from agents import Agent, Runner, function_tool

# Создание агента с инструментом
@function_tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny"

agent = Agent(
    name="WeatherBot",
    instructions="You provide weather forecasts",
    tools=[get_weather]
)

# Запуск мультиагентного workflow
result = Runner.run_sync(agent, "What's the weather in Tokyo?")
```

---

## ⚖️ Сравнение с LangChain

| Критерий        | Agents SDK                     | LangChain                      |
|-----------------|--------------------------------|--------------------------------|
| Порог входа     | ★★★★★ (очень низкий)          | ★★★☆☆ (средний)               |
| Гибкость        | ★★☆☆☆ (ограниченная)           | ★★★★★ (полная)                |
| Стоимость       | ★★☆☆☆ (высокая)                | ★★★★☆ (контролируемая)        |
| Мультиагентность| Встроенная через Handoffs      | Через LangGraph                |

---

## 💡 Когда использовать?

**✅ Идеально для:**
- Быстрого прототипирования MVP
- Проектов внутри экосистемы OpenAI
- Знакомства с концепцией агентов

**❌ Проблемные сценарии:**
- Продакшен с требованием к контролю затрат
- Системы с hybrid-архитектурой (разные LLM)
- Проекты с кастомными RAG-пайплайнами

---

## 📌 Ключевые особенности

* **Автоматическое управление состоянием** через Threads
* **Встроенные инструменты**: Code Interpreter, File Search
* **Трассировка выполнения** для отладки сложных workflow
* **Декоратор @function_tool** для интеграции любых функций


## ⭐ **Понравился разбор?**

Подписывайтесь на [Telegram-канал](https://t.me/TheWeeklyBrief) и ставьте звезду в репозитории!

<p align="center">Исследуем технологии ИИ вместе! 🚀</p>