# A2A: Agent-to-Agent Communication Framework

<div align="center">
<alt="A2A Framework Overview" />

![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Version](https://img.shields.io/badge/Version-1.0.0-green.svg)
</div>

## 📋 Содержание

- [Обзор проекта](#-обзор-проекта)
- [Структура репозитория](#-структура-репозитория)
- [Установка](#-установка)
- [Быстрый старт](#-быстрый-старт)
- [Компоненты](#-компоненты)
  - [Общая библиотека (common)](#общая-библиотека-common)
  - [Агенты (agents)](#агенты-agents)
  - [Клиенты (hosts)](#клиенты-hosts)
- [Примеры использования](#-примеры-использования)
- [Разработка](#-разработка)
- [Лицензия](#-лицензия)

## 🔍 Обзор проекта

**A2A** (Agent-to-Agent) — это фреймворк для обеспечения унифицированного, стандартизированного взаимодействия между различными AI-агентами через единый протокол коммуникации. Фреймворк позволяет создавать экосистему взаимодействующих автономных агентов, которые могут обмениваться данными, делегировать задачи и совместно решать сложные проблемы.

### Ключевые особенности

- **Унифицированный протокол**: стандартизированное взаимодействие между агентами с использованием JSON-RPC 2.0
- **Обнаружение возможностей**: механизм для определения возможностей агентов через карточки агентов (AgentCard)
- **Потоковая передача**: поддержка потоковой передачи данных в режиме реального времени через SSE
- **Push-уведомления**: асинхронные уведомления с JWT-аутентификацией
- **Различные реализации**: готовые реализации агентов на различных фреймворках (LangGraph, Google ADK)
- **Клиентские приложения**: CLI и программные интерфейсы для взаимодействия с агентами

A2A обеспечивает модульный, расширяемый подход к созданию AI-систем, где различные специализированные агенты могут работать вместе, обмениваться информацией и координировать свои действия.

<div align="center" style="padding: 20px; border-radius: 10px; background: linear-gradient(145deg, #eef5f9, #dcedff); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
    <p style="font-size: 24px; font-weight: bold; color: #333; margin: 0;">A2A Ecosystem</p>
</div>

## 🗂 Структура репозитория

Репозиторий организован в виде трех основных модулей:

```
.A2A/
├── common/          # Основная библиотека протокола A2A
│   ├── client/      # Клиентские компоненты (A2AClient, A2ACardResolver)
│   ├── server/      # Серверные компоненты (A2AServer, TaskManager)
│   ├── utils/       # Утилиты (кэш, аутентификация push-уведомлений)
│   └── types.py     # Типы данных протокола A2A
├── agents/          # Различные реализации A2A агентов
│   ├── langgraph/   # Агент конвертации валют на базе LangGraph
│   ├── crewai/      # Агент генерации изображений на базе CrewAI
│   └── google_adk/  # Агент возмещения расходов на базе Google ADK
└── hosts/           # Клиентские приложения для взаимодействия с агентами
    ├── cli/         # Интерактивный CLI клиент
    └── multiagent/  # Мультиагентный оркестратор на базе Google ADK
```

## 🚀 Установка

### Предварительные требования

- Python 3.9+
- pip или другой менеджер пакетов Python

### Установка из репозитория

```bash
# Клонирование репозитория
git clone https://github.com/repo-name/A2A.git
cd A2A

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Установка зависимостей
pip install -r requirements.txt

# Опционально: установка в режиме разработки
pip install -e .
```

### Настройка API ключей

Для работы с агентами, использующими LLM, требуется настройка API ключей. Создайте файл `.env` в корневой директории:

```bash
# Пример содержимого .env файла
GOOGLE_API_KEY=ваш_ключ_google_api
```

## 🏁 Быстрый старт

### Запуск агента конвертации валют

```bash
# Запуск агента конвертации валют на порту 10000
python -m A2A.agents.langgraph --host localhost --port 10000
```

### Взаимодействие с агентом через CLI клиент

```bash
# Запуск CLI клиента для взаимодействия с агентом
python -m A2A.hosts.cli --agent http://localhost:10000
```

### Программное взаимодействие с агентом

```python
from A2A.common.client import A2AClient, A2ACardResolver
from A2A.common.types  import Message, TextPart

# Получение карточки агента
resolver = A2ACardResolver("http://localhost:10000")
agent_card = resolver.get_agent_card()

# Создание клиента
client = A2AClient(agent_card=agent_card)

# Отправка запроса
response = await client.send_task({
    "id": "task123",
    "sessionId": "session456",
    "message": Message(
        role="user",
        parts=[TextPart(text="Какой текущий курс USD к EUR?")]
    )
})

# Вывод результата
print(response.result.status.message.parts[0].text)
```

## 🧩 Компоненты

### Общая библиотека (common)

Основа фреймворка A2A - библиотека `common`, которая содержит:

- **Клиент A2A** (`A2AClient`) - для отправки запросов к агентам
- **Сервер A2A** (`A2AServer`) - для обработки входящих запросов
- **Карточки агентов** (`AgentCard`) - метаданные с описанием возможностей агентов
- **Менеджеры задач** (`TaskManager`) - для управления жизненным циклом задач
- **Типы данных** - модели Pydantic для формализации запросов и ответов

[Подробнее о библиотеке common](./common/README.md)

### Агенты (agents)

В репозитории представлены примеры агентов, реализованных на различных фреймворках:

- **Агент конвертации валют** - на базе фреймворка LangGraph и Google Gemini, предоставляет информацию о курсах валют

[Подробнее об агенте](./agents/README.md)

### Клиенты (hosts)

Для взаимодействия с агентами предоставляются различные клиентские реализации:

- **CLI клиент** - интерактивный терминальный интерфейс для прямого взаимодействия с агентами
- **Multi-Agent оркестратор** - интеллектуальный координатор на базе Google ADK, маршрутизирующий запросы между агентами

[Подробнее о клиентах](./hosts/README.md)

## 💡 Примеры использования

### Сценарий 1: Работа с агентом конвертации валют

Агент конвертации валют предоставляет информацию о курсах обмена валют:

```bash
# Запуск агента
python -m A2A.agents.langgraph

# Взаимодействие через CLI
python -m A2A.hosts.cli --agent http://localhost:10000
```

Примеры запросов:
- "Какой текущий курс обмена USD к EUR?"
- "Сконвертируй 100 USD в JPY"
- "Каков был курс евро к доллару 1 января 2020 года?"

### Сценарий 2: Делегирование задач через Multi-Agent оркестратор

Оркестратор маршрутизирует запросы к специализированным агентам:

```bash
# Запуск нескольких агентов
python -m A2A.agents.langgraph --port 10000
python -m A2A.agents.crewai --port 10001
python -m A2A.agents.google_adk --port 10002

# Использование Multi-Agent оркестратора в коде
from A2A.hosts.multiagent.agent import root_agent
from google.genai import types

# Создание запроса
content = types.Content(
    role="user", 
    parts=[types.Part.from_text(text="Сгенерируй изображение и укажи его стоимость в евро")]
)

# Выполнение запроса
events = root_agent.run(
    user_id="user123",
    session_id="session456",
    new_message=content
)
```

### Сценарий 3: Интеграция A2A в существующие приложения

```python
from A2A.common.client import A2AClient, A2ACardResolver
from A2A.common.types import Message, TextPart, PushNotificationConfig

# Настройка вебхука для push-уведомлений
notification_config = PushNotificationConfig(
    url="https://your-app.com/webhooks/a2a-notifications"
)

# Отправка запроса с настройкой push-уведомлений
response = await client.send_task({
    "id": "task123",
    "sessionId": "session456",
    "pushNotification": notification_config,
    "message": Message(
        role="user",
        parts=[TextPart(text="Обработай этот запрос и уведоми о завершении")]
    )
})
```

## 🛠 Разработка

### Создание собственного агента

1. Создайте новый пакет в директории `agents/`
2. Реализуйте класс агента, унаследовав или адаптировав `TaskManager` из `common.server`
3. Создайте точку входа, которая инициализирует `A2AServer` с вашим агентом

```python
# Пример базовой структуры
from A2A.common.server import A2AServer, InMemoryTaskManager
from A2A.common.types import AgentCard, AgentCapabilities, AgentSkill

# Создание менеджера задач
class MyTaskManager(InMemoryTaskManager):
    async def on_send_task(self, request):
        # Реализация обработки запроса
        # ...

# Создание карточки агента
agent_card = AgentCard(
    name="MyAgent",
    url="http://localhost:9000",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True),
    skills=[
        AgentSkill(
            id="my_skill",
            name="My Skill",
            description="Description of my agent's capability"
        )
    ]
)

# Запуск сервера
server = A2AServer(
    host="0.0.0.0",
    port=9000,
    agent_card=agent_card,
    task_manager=MyTaskManager()
)
server.start()
```