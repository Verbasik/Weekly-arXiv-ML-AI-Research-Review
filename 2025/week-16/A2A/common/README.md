# A2A: Agent-to-Agent API

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.7+-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

## 📖 Описание

A2A (Agent-to-Agent) — это библиотека для организации взаимодействия между автономными агентами через унифицированный API. Библиотека обеспечивает надежную и гибкую коммуникацию между различными сервисами с поддержкой асинхронных операций, потоковой передачи данных и push-уведомлений.

### Основные возможности

- ✅ **JSON-RPC 2.0 API** для стандартизированного взаимодействия между агентами
- ✅ **Потоковая передача данных** через Server-Sent Events (SSE)
- ✅ **Управление задачами** — создание, получение статуса, отмена задач
- ✅ **Push-уведомления** о статусе задач с поддержкой аутентификации JWT
- ✅ **Типизированный интерфейс** с использованием Pydantic для валидации данных
- ✅ **Расширяемая архитектура** для добавления новых типов агентов

## 🏗️ Архитектура

A2A использует модель клиент-сервер для взаимодействия между агентами:

- **Клиент (A2AClient)** — отправляет запросы к API агентов
- **Сервер (A2AServer)** — обрабатывает входящие запросы от других агентов
- **Карточка агента (AgentCard)** — содержит метаданные и возможности агента
- **Менеджер задач (TaskManager)** — управляет жизненным циклом задач

Взаимодействие происходит в следующей последовательности:

1. Клиент обнаруживает агента через его карточку
2. Клиент отправляет задачу агенту
3. Сервер агента обрабатывает задачу
4. Клиент получает результаты через прямой ответ, потоковую передачу или push-уведомления

## 🧩 Основные компоненты

### Клиентская часть

#### A2AClient

Клиент для взаимодействия с Agent-to-Agent API. Поддерживает отправку задач, получение результатов, управление уведомлениями.

```python
from common.client import A2AClient
from common.client import A2ACardResolver

# Получение карточки агента
resolver = A2ACardResolver("http://localhost:8001")
agent_card = resolver.get_agent_card()

# Создание клиента
client = A2AClient(agent_card=agent_card)

# Отправка задачи
response = await client.send_task({
    "id": "task123",
    "sessionId": "session456",
    "message": {"text": "Привет, мир!"}
})

# Получение статуса задачи
task_status = await client.get_task({"id": "task123"})
```

#### A2ACardResolver

Класс для получения метаданных агентов (карточек) через их API.

```python
from common.client import A2ACardResolver

# Создание резолвера
resolver = A2ACardResolver("http://localhost:8001")

# Получение карточки агента
agent_card = resolver.get_agent_card()
print(f"Имя агента: {agent_card.name}")
print(f"Описание: {agent_card.description}")
```

### Серверная часть

#### A2AServer

Сервер для обработки запросов Agent-to-Agent API. Поддерживает стандартные и потоковые запросы, а также управление задачами.

```python
from common.server import A2AServer
from common.server import InMemoryTaskManager
from common.types  import AgentCard, AgentCapabilities, AgentSkill

# Создание менеджера задач
task_manager = InMemoryTaskManager()

# Определение карточки агента
agent_card = AgentCard(
    name="MyAgent",
    url="http://localhost:5000",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=True, pushNotifications=True),
    skills=[
        AgentSkill(
            id="calculator",
            name="Калькулятор",
            description="Выполняет математические расчеты"
        )
    ]
)

# Создание и запуск сервера
server = A2AServer(
    host="0.0.0.0",
    port=5000,
    agent_card=agent_card,
    task_manager=task_manager
)
server.start()
```

#### TaskManager и InMemoryTaskManager

Абстрактный базовый класс и его базовая реализация для управления жизненным циклом задач.

```python
from common.server import InMemoryTaskManager
from common.types  import TaskState, TaskStatus, Message, TextPart

class MyTaskManager(InMemoryTaskManager):
    async def on_send_task(self, request):
        # Получение и обработка параметров запроса
        task_params = request.params
        
        # Создание или обновление задачи
        task = await self.upsert_task(task_params)
        
        # Обновление статуса задачи
        task = await self.update_store(
            task.id,
            TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(
                    role="agent",
                    parts=[TextPart(text="Задача выполнена успешно!")]
                )
            ),
            None  # Артефакты
        )
        
        # Возврат ответа
        return self.create_response(request.id, task)
```

### Утилиты

#### InMemoryCache

Потокобезопасный Singleton-класс для управления данными кэша с поддержкой TTL.

```python
from common.utils.in_memory_cache import InMemoryCache

# Получение экземпляра кэша
cache = InMemoryCache()

# Сохранение данных
cache.set("user_profile", {"name": "John", "age": 30})
cache.set("session_token", "abc123", ttl=3600)  # Срок действия 1 час

# Получение данных
profile = cache.get("user_profile")
token = cache.get("session_token", "default_token")

# Удаление данных
cache.delete("session_token")

# Очистка всего кэша
cache.clear()
```

#### PushNotificationAuth

Классы для отправки и приема аутентифицированных push-уведомлений с использованием JWT.

```python
from common.utils.push_notification_auth import PushNotificationSenderAuth

# Создание отправителя уведомлений
sender = PushNotificationSenderAuth()
sender.generate_jwk()

# Проверка URL для push-уведомлений
is_valid = await PushNotificationSenderAuth.verify_push_notification_url(
    "https://example.com/webhook"
)

# Отправка уведомления
success = await sender.send_push_notification(
    "https://example.com/webhook",
    {"event": "update", "id": "task123"}
)
```

## 📚 API Reference

### Клиентские классы

- `A2AClient` - Основной клиент для взаимодействия с API агентов
- `A2ACardResolver` - Класс для получения карточек агентов

### Серверные классы

- `A2AServer` - Сервер для обработки запросов API
- `TaskManager` - Абстрактный класс для управления задачами
- `InMemoryTaskManager` - Базовая реализация менеджера задач

### Типы данных

- `Message` - Сообщение, обмениваемое между пользователем и агентом
- `Task` - Задача в рамках A2A API
- `TaskStatus` - Статус задачи
- `Artifact` - Артефакт, создаваемый агентом
- `AgentCard` - Карточка агента с его метаданными

### Утилиты

- `InMemoryCache` - Класс для кэширования данных в памяти
- `PushNotificationSenderAuth` - Класс для отправки аутентифицированных push-уведомлений
- `PushNotificationReceiverAuth` - Класс для приема и проверки аутентифицированных push-уведомлений

## 📋 Требования

- Python 3.7+
- Pydantic
- Starlette
- httpx
- jwt
- jwcrypto