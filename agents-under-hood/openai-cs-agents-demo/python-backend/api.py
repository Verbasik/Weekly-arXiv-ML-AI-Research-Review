"""
FastAPI веб-сервер для системы AI-агентов клиентского сервиса авиакомпании.

Этот модуль предоставляет REST API для взаимодействия с multi-agent системой
обработки запросов клиентов авиакомпании. Сервер обрабатывает HTTP запросы,
управляет состоянием разговоров, маршрутизирует запросы между агентами,
обрабатывает guardrails и возвращает структурированные ответы.

Основные компоненты:
- FastAPI приложение с CORS middleware
- Модели данных для запросов и ответов
- In-memory хранилище состояния разговоров
- Основной chat endpoint для обработки сообщений
- Система обработки событий и guardrails
- Интеграция с OpenAI Agents SDK

API предоставляет единственный endpoint /chat для POST запросов с сообщениями
пользователей и возвращает полную информацию о состоянии разговора, активном
агенте, сообщениях, событиях и результатах проверки guardrails.
"""

# Стандартные библиотеки Python
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

# Библиотеки для веб-фреймворка
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Библиотеки для валидации данных
from pydantic import BaseModel

# Библиотеки для работы с переменными окружения
from dotenv import load_dotenv

# Локальные модули проекта - агенты системы
from main import (
    cancellation_agent,
    create_initial_context,
    faq_agent,
    flight_status_agent,
    seat_booking_agent,
    triage_agent,
)

# Локальные модули проекта - OpenAI Agents SDK
from agents import (
    Handoff,
    HandoffOutputItem,
    InputGuardrailTripwireTriggered,
    ItemHelpers,
    MessageOutputItem,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
)

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка системы логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Проверка загрузки OpenAI API ключа
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY not found in environment variables!")
    logger.error("Make sure you have set OPENAI_API_KEY in your .env file")
else:
    logger.info("OPENAI_API_KEY loaded successfully")

# Создание экземпляра FastAPI приложения
app = FastAPI()

# Настройка CORS middleware для кросс-доменных запросов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Разрешенные домены
    allow_credentials=True,                   # Поддержка credentials
    allow_methods=["*"],                      # Все HTTP методы
    allow_headers=["*"],                      # Все заголовки
)

# =========================
# МОДЕЛИ ДАННЫХ ДЛЯ API
# =========================
class ChatRequest(BaseModel):
    """
    Description:
    ---------------
        Модель запроса для chat endpoint.
        Структура данных для входящих запросов от клиентов,
        содержащая идентификатор разговора и текст сообщения.
    """
    conversation_id: Optional[str] = None
    message: str


class MessageResponse(BaseModel):
    """
    Description:
    ---------------
        Модель ответного сообщения от агента.
        Структура данных для сообщений от агентов системы,
        содержащая текст ответа и имя агента-отправителя.
    """
    content: str
    agent: str


class AgentEvent(BaseModel):
    """
    Description:
    ---------------
        Модель события в системе агентов.
        Структура данных для отслеживания событий в процессе
        работы агентов: сообщения, передачи управления, вызовы tools.
    """
    id: str
    type: str
    agent: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None


class GuardrailCheck(BaseModel):
    """
    Description:
    ---------------
        Модель результата проверки guardrail.
        Структура данных для результатов проверки системы безопасности,
        включающая информацию о прохождении проверки и обоснование.
    """
    id: str
    name: str
    input: str
    reasoning: str
    passed: bool
    timestamp: float


class ChatResponse(BaseModel):
    """
    Description:
    ---------------
        Модель полного ответа chat endpoint.
        Комплексная структура данных, содержащая все информацию
        о состоянии разговора после обработки сообщения пользователя.
    """
    conversation_id: str
    current_agent: str
    messages: List[MessageResponse]
    events: List[AgentEvent]
    context: Dict[str, Any]
    agents: List[Dict[str, Any]]
    guardrails: List[GuardrailCheck] = []


# =========================
# ХРАНИЛИЩЕ СОСТОЯНИЯ РАЗГОВОРОВ
# =========================
class ConversationStore:
    """
    Description:
    ---------------
        Абстрактный базовый класс для хранилища разговоров.
        Определяет интерфейс для сохранения и получения
        состояния разговоров между запросами пользователей.
    """
    
    # Получение состояния разговора по идентификатору
    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Description:
        ---------------
            Получение состояния разговора.
            Абстрактный метод для получения сохраненного состояния
            разговора по его уникальному идентификатору.
        """
        pass

    # Сохранение состояния разговора
    def save(self, conversation_id: str, state: Dict[str, Any]) -> None:
        """
        Description:
        ---------------
            Сохранение состояния разговора.
            Абстрактный метод для сохранения текущего состояния
            разговора для последующего восстановления.
        """
        pass


class InMemoryConversationStore(ConversationStore):
    """
    Description:
    ---------------
        In-memory реализация хранилища разговоров.
        Простая реализация хранилища состояний разговоров в памяти.
        Подходит для демо и разработки, но не для production из-за
        потери данных при перезапуске сервера.
    """
    
    _conversations: Dict[str, Dict[str, Any]] = {}

    # Получение состояния разговора из памяти
    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Description:
        ---------------
            Получение состояния разговора из памяти.
            Извлекает сохраненное состояние разговора из
            внутреннего словаря в памяти.
        """
        return self._conversations.get(conversation_id)

    # Сохранение состояния разговора в памяти
    def save(self, conversation_id: str, state: Dict[str, Any]) -> None:
        """
        Description:
        ---------------
            Сохранение состояния разговора в памяти.
            Сохраняет текущее состояние разговора во внутренний
            словарь в памяти для последующего использования.
        """
        self._conversations[conversation_id] = state


# TODO: при масштабировании заменить на production-ready решение
# Глобальный экземпляр хранилища разговоров
conversation_store = InMemoryConversationStore()

# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================
# Получение агента по имени
def _get_agent_by_name(name: str):
    """
    Description:
    ---------------
        Получение экземпляра агента по имени.
        Возвращает соответствующий объект агента по его имени
        из предопределенного списка доступных агентов.
    """
    # Словарь всех доступных агентов системы
    agents = {
        triage_agent.name: triage_agent,
        faq_agent.name: faq_agent,
        seat_booking_agent.name: seat_booking_agent,
        flight_status_agent.name: flight_status_agent,
        cancellation_agent.name: cancellation_agent,
    }
    # Возвращаем агента или triage_agent по умолчанию
    return agents.get(name, triage_agent)


# Извлечение понятного имени guardrail
def _get_guardrail_name(g) -> str:
    """
    Description:
    ---------------
        Извлечение понятного имени guardrail.
        Пытается извлечь читаемое имя guardrail из различных
        атрибутов объекта для отображения в пользовательском интерфейсе.
    """
    # Пытаемся получить атрибут name
    name_attr = getattr(g, "name", None)
    if isinstance(name_attr, str) and name_attr:
        return name_attr
    
    # Пытаемся получить имя функции guardrail
    guard_fn = getattr(g, "guardrail_function", None)
    if guard_fn is not None and hasattr(guard_fn, "__name__"):
        return guard_fn.__name__.replace("_", " ").title()
    
    # Пытаемся получить имя самого объекта
    fn_name = getattr(g, "__name__", None)
    if isinstance(fn_name, str) and fn_name:
        return fn_name.replace("_", " ").title()
    
    # Возвращаем строковое представление объекта
    return str(g)


# Построение списка доступных агентов
def _build_agents_list() -> List[Dict[str, Any]]:
    """
    Description:
    ---------------
        Построение списка всех доступных агентов и их метаданных.
        Создает структурированный список всех агентов системы
        с их описаниями, доступными handoffs, tools и guardrails
        для отображения в пользовательском интерфейсе.
    """
    
    # Функция для создания словаря метаданных агента
    def make_agent_dict(agent):
        """
        Description:
        ---------------
            Создание словаря метаданных для агента.
            Извлекает и структурирует метаданные агента включая
            имя, описание, доступные handoffs, tools и guardrails.
        """
        return {
            "name": agent.name,
            "description": getattr(agent, "handoff_description", ""),
            "handoffs": [
                getattr(h, "agent_name", getattr(h, "name", ""))
                for h in getattr(agent, "handoffs", [])
            ],
            "tools": [
                getattr(t, "name", getattr(t, "__name__", ""))
                for t in getattr(agent, "tools", [])
            ],
            "input_guardrails": [
                _get_guardrail_name(g)
                for g in getattr(agent, "input_guardrails", [])
            ],
        }
    
    # Создаем список метаданных для всех агентов
    return [
        make_agent_dict(triage_agent),
        make_agent_dict(faq_agent),
        make_agent_dict(seat_booking_agent),
        make_agent_dict(flight_status_agent),
        make_agent_dict(cancellation_agent),
    ]


# =========================
# ОСНОВНОЙ CHAT ENDPOINT
# =========================
# Главный endpoint для обработки чата
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """
    Description:
    ---------------
        Основной endpoint для обработки чата с агентами.
        Обрабатывает входящие сообщения пользователей, управляет
        состоянием разговоров, маршрутизирует запросы между агентами,
        обрабатывает guardrails и возвращает полную информацию
        о результате обработки.
    """
    # Определяем, новый ли это разговор или продолжение существующего
    is_new = (not req.conversation_id or 
              conversation_store.get(req.conversation_id) is None)
    
    if is_new:
        # Инициализация нового разговора
        conversation_id: str = uuid4().hex
        ctx = create_initial_context()
        current_agent_name = triage_agent.name
        
        # Создаем начальное состояние разговора
        state: Dict[str, Any] = {
            "input_items": [],
            "context": ctx,
            "current_agent": current_agent_name,
        }
        
        # Обработка пустого сообщения (инициализация интерфейса)
        if req.message.strip() == "":
            conversation_store.save(conversation_id, state)
            return ChatResponse(
                conversation_id=conversation_id,
                current_agent=current_agent_name,
                messages=[],
                events=[],
                context=ctx.model_dump(),
                agents=_build_agents_list(),
                guardrails=[],
            )
    else:
        # Восстановление существующего разговора
        conversation_id = req.conversation_id  # type: ignore
        state = conversation_store.get(conversation_id)

    # Получаем текущий активный агент
    current_agent = _get_agent_by_name(state["current_agent"])
    
    # Добавляем пользовательское сообщение в историю
    state["input_items"].append({"content": req.message, "role": "user"})
    
    # Сохраняем текущий контекст для отслеживания изменений
    old_context = state["context"].model_dump().copy()
    guardrail_checks: List[GuardrailCheck] = []

    try:
        # Запускаем обработку сообщения агентом
        result = await Runner.run(
            current_agent, state["input_items"], context=state["context"]
        )
    except InputGuardrailTripwireTriggered as e:
        # Обработка срабатывания guardrails
        failed = e.guardrail_result.guardrail
        gr_output = e.guardrail_result.output.output_info
        gr_reasoning = getattr(gr_output, "reasoning", "")
        gr_input = req.message
        gr_timestamp = time.time() * 1000
        
        # Создаем записи о проверках guardrails
        for g in current_agent.input_guardrails:
            guardrail_checks.append(GuardrailCheck(
                id=uuid4().hex,
                name=_get_guardrail_name(g),
                input=gr_input,
                reasoning=(gr_reasoning if g == failed else ""),
                passed=(g != failed),
                timestamp=gr_timestamp,
            ))
        
        # Возвращаем отказ в обслуживании
        refusal = "Извините, я могу отвечать только на вопросы, связанные с авиаперелетами."
        state["input_items"].append({"role": "assistant", "content": refusal})
        
        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=[MessageResponse(content=refusal, agent=current_agent.name)],
            events=[],
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=guardrail_checks,
        )

    # Инициализируем списки для сообщений и событий
    messages: List[MessageResponse] = []
    events: List[AgentEvent] = []

    # Обрабатываем каждый элемент результата выполнения агента
    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            # Обработка текстового сообщения от агента
            text = ItemHelpers.text_message_output(item)
            messages.append(MessageResponse(content=text, agent=item.agent.name))
            events.append(AgentEvent(
                id=uuid4().hex, 
                type="message", 
                agent=item.agent.name, 
                content=text
            ))
            
        elif isinstance(item, HandoffOutputItem):
            # Обработка передачи управления между агентами
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="handoff",
                    agent=item.source_agent.name,
                    content=f"{item.source_agent.name} -> {item.target_agent.name}",
                    metadata={
                        "source_agent": item.source_agent.name, 
                        "target_agent": item.target_agent.name
                    },
                )
            )
            
            # Поиск callback функции для handoff
            from_agent = item.source_agent
            to_agent = item.target_agent
            
            # Находим объект Handoff соответствующий передаче управления
            ho = next(
                (h for h in getattr(from_agent, "handoffs", [])
                 if isinstance(h, Handoff) and 
                 getattr(h, "agent_name", None) == to_agent.name),
                None,
            )
            
            # Если есть callback функция, записываем ее как tool call
            if ho:
                fn = ho.on_invoke_handoff
                fv = fn.__code__.co_freevars
                cl = fn.__closure__ or []
                if "on_handoff" in fv:
                    idx = fv.index("on_handoff")
                    if idx < len(cl) and cl[idx].cell_contents:
                        cb = cl[idx].cell_contents
                        cb_name = getattr(cb, "__name__", repr(cb))
                        events.append(
                            AgentEvent(
                                id=uuid4().hex,
                                type="tool_call",
                                agent=to_agent.name,
                                content=cb_name,
                            )
                        )
            
            # Обновляем текущий активный агент
            current_agent = item.target_agent
            
        elif isinstance(item, ToolCallItem):
            # Обработка вызова инструмента агентом
            tool_name = getattr(item.raw_item, "name", None)
            raw_args = getattr(item.raw_item, "arguments", None)
            tool_args: Any = raw_args
            
            # Парсим аргументы инструмента если они в JSON формате
            if isinstance(raw_args, str):
                try:
                    tool_args = json.loads(raw_args)
                except Exception:
                    pass
                    
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_call",
                    agent=item.agent.name,
                    content=tool_name or "",
                    metadata={"tool_args": tool_args},
                )
            )
            
            # Специальная обработка для инструмента карты мест
            if tool_name == "display_seat_map":
                messages.append(
                    MessageResponse(
                        content="DISPLAY_SEAT_MAP",
                        agent=item.agent.name,
                    )
                )
                
        elif isinstance(item, ToolCallOutputItem):
            # Обработка результата выполнения инструмента
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_output",
                    agent=item.agent.name,
                    content=str(item.output),
                    metadata={"tool_result": item.output},
                )
            )

    # Отслеживание изменений контекста
    new_context = state["context"].dict()
    changes = {
        k: new_context[k] 
        for k in new_context 
        if old_context.get(k) != new_context[k]
    }
    
    # Добавляем событие об изменении контекста если есть изменения
    if changes:
        events.append(
            AgentEvent(
                id=uuid4().hex,
                type="context_update",
                agent=current_agent.name,
                content="",
                metadata={"changes": changes},
            )
        )

    # Обновляем состояние разговора
    state["input_items"] = result.to_input_list()
    state["current_agent"] = current_agent.name
    conversation_store.save(conversation_id, state)

    # Формируем результаты guardrails
    final_guardrails: List[GuardrailCheck] = []
    for g in getattr(current_agent, "input_guardrails", []):
        name = _get_guardrail_name(g)
        failed = next((gc for gc in guardrail_checks if gc.name == name), None)
        
        if failed:
            # Добавляем информацию о неудачной проверке
            final_guardrails.append(failed)
        else:
            # Добавляем информацию об успешной проверке
            final_guardrails.append(GuardrailCheck(
                id=uuid4().hex,
                name=name,
                input=req.message,
                reasoning="",
                passed=True,
                timestamp=time.time() * 1000,
            ))

    # Возвращаем полный ответ
    return ChatResponse(
        conversation_id=conversation_id,
        current_agent=current_agent.name,
        messages=messages,
        events=events,
        context=state["context"].dict(),
        agents=_build_agents_list(),
        guardrails=final_guardrails,
    )
