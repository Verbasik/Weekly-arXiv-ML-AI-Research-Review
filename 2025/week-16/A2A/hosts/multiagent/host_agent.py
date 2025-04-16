# A2A/hosts/multiagent/host_agent.py
"""
Модуль реализует агент-хост для оркестрации взаимодействия с удалёнными агентами.

Позволяет выбирать подходящих удалённых агентов для выполнения задач,
координировать их работу и обрабатывать результаты. Поддерживает как
потоковый, так и обычный режим отправки задач.
"""

# Стандартные библиотеки
import sys
import asyncio
import functools
import json
import uuid
import threading
from typing import List, Optional, Callable, Dict, Any, Union, Generator

# Сторонние библиотеки
import base64
from google.genai import types
from google.adk import Agent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext

# Внутренние модули
from .remote_agent_connection import (
    RemoteAgentConnections,
    TaskUpdateCallback
)
from common.client import A2ACardResolver
from common.types import (
    AgentCard,
    Message,
    TaskState,
    Task,
    TaskSendParams,
    TextPart,
    DataPart,
    Part,
    TaskStatusUpdateEvent,
)


class HostAgent:
    """
    Description:
    ---------------
        Агент-хост, ответственный за выбор удалённых агентов для выполнения задач
        и координацию их работы.
        
    Args:
    ---------------
        remote_agent_addresses: Список адресов удалённых агентов
        task_callback: Функция обратного вызова для обновлений задач
        
    Raises:
    ---------------
        ValueError: При ошибке подключения к удалённому агенту
        
    Examples:
    ---------------
        >>> host_agent = HostAgent(["http://localhost:8001", "http://localhost:8002"])
        >>> agent = host_agent.create_agent()
    """

    def __init__(
        self,
        remote_agent_addresses: List[str],
        task_callback: Optional[TaskUpdateCallback] = None
    ) -> None:
        """
        Description:
        ---------------
            Инициализирует агент-хост и устанавливает соединения с удалёнными агентами.
            
        Args:
        ---------------
            remote_agent_addresses: Список адресов удалённых агентов
            task_callback: Функция обратного вызова для обновлений задач
            
        Returns:
        ---------------
            None
        """
        self.task_callback = task_callback
        self.remote_agent_connections: Dict[str, RemoteAgentConnections] = {}
        self.cards: Dict[str, AgentCard] = {}
        
        # Устанавливаем соединения со всеми удалёнными агентами
        for address in remote_agent_addresses:
            card_resolver = A2ACardResolver(address)
            card = card_resolver.get_agent_card()
            remote_connection = RemoteAgentConnections(card)
            self.remote_agent_connections[card.name] = remote_connection
            self.cards[card.name] = card
            
        # Формируем информацию о доступных агентах
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = '\n'.join(agent_info)

    def register_agent_card(self, card: AgentCard) -> None:
        """
        Description:
        ---------------
            Регистрирует карточку удалённого агента и устанавливает соединение.
            
        Args:
        ---------------
            card: Карточка агента для регистрации
            
        Returns:
        ---------------
            None
            
        Examples:
        ---------------
            >>> card = A2ACardResolver("http://localhost:8003").get_agent_card()
            >>> host_agent.register_agent_card(card)
        """
        remote_connection = RemoteAgentConnections(card)
        self.remote_agent_connections[card.name] = remote_connection
        self.cards[card.name] = card
        
        # Обновляем информацию о доступных агентах
        agent_info = []
        for ra in self.list_remote_agents():
            agent_info.append(json.dumps(ra))
        self.agents = '\n'.join(agent_info)

    def create_agent(self) -> Agent:
        """
        Description:
        ---------------
            Создаёт экземпляр агента с моделью Gemini и необходимыми инструкциями.
            
        Returns:
        ---------------
            Agent: Экземпляр агента Google ADK
            
        Examples:
        ---------------
            >>> agent = host_agent.create_agent()
        """
        return Agent(
            model="gemini-2.0-flash-001",
            name="host_agent",
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description=(
                "This agent orchestrates the decomposition of the user request into"
                " tasks that can be performed by the child agents."
            ),
            tools=[
                self.list_remote_agents,
                self.send_task,
            ],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        """
        Description:
        ---------------
            Формирует основные инструкции для агента-хоста.
            
        Args:
        ---------------
            context: Контекст только для чтения
            
        Returns:
        ---------------
            str: Текст инструкции для агента
            
        Examples:
        ---------------
            >>> instruction = host_agent.root_instruction(context)
        """
        current_agent = self.check_state(context)
        return f"""You are a expert delegator that can delegate the user request to the
appropriate remote agents.

Discovery:
- You can use `list_remote_agents` to list the available remote agents you
can use to delegate the task.

Execution:
- For actionable tasks, you can use `create_task` to assign tasks to remote agents to perform.
Be sure to include the remote agent name when you response to the user.

You can use `check_pending_task_states` to check the states of the pending
tasks.

Please rely on tools to address the request, don't make up the response. If you are not sure, please ask the user for more details.
Focus on the most recent parts of the conversation primarily.

If there is an active agent, send the request to that agent with the update task tool.

Agents:
{self.agents}

Current agent: {current_agent['active_agent']}
"""

    def check_state(self, context: ReadonlyContext) -> Dict[str, str]:
        """
        Description:
        ---------------
            Проверяет текущее состояние агента и сессии.
            
        Args:
        ---------------
            context: Контекст только для чтения
            
        Returns:
        ---------------
            Dict[str, str]: Словарь с информацией об активном агенте
            
        Examples:
        ---------------
            >>> state = host_agent.check_state(context)
            >>> active_agent = state['active_agent']
        """
        state = context.state
        if ('session_id' in state and
                'session_active' in state and
                state['session_active'] and
                'agent' in state):
            return {"active_agent": f'{state["agent"]}'}
        return {"active_agent": "None"}

    def before_model_callback(self, callback_context: CallbackContext, llm_request: Any) -> None:
        """
        Description:
        ---------------
            Выполняется перед отправкой запроса к модели LLM.
            Инициализирует сессию, если она не активна.
            
        Args:
        ---------------
            callback_context: Контекст обратного вызова
            llm_request: Запрос к LLM-модели
            
        Returns:
        ---------------
            None
            
        Examples:
        ---------------
            >>> host_agent.before_model_callback(callback_context, llm_request)
        """
        state = callback_context.state
        if 'session_active' not in state or not state['session_active']:
            if 'session_id' not in state:
                state['session_id'] = str(uuid.uuid4())
            state['session_active'] = True

    def list_remote_agents(self) -> List[Dict[str, str]]:
        """
        Description:
        ---------------
            Перечисляет доступные удалённые агенты, которые можно использовать для делегирования задач.
            
        Returns:
        ---------------
            List[Dict[str, str]]: Список словарей с информацией об агентах
            
        Examples:
        ---------------
            >>> agents = host_agent.list_remote_agents()
            >>> for agent in agents:
            ...     print(f"Агент: {agent['name']}, Описание: {agent['description']}")
        """
        if not self.remote_agent_connections:
            return []

        remote_agent_info = []
        for card in self.cards.values():
            remote_agent_info.append(
                {"name": card.name, "description": card.description}
            )
        return remote_agent_info

    async def send_task(
            self,
            agent_name: str,
            message: str,
            tool_context: ToolContext) -> List[Union[str, Dict[str, Any]]]:
        """
        Description:
        ---------------
            Отправляет задачу удалённому агенту в потоковом (если поддерживается) 
            или обычном режиме.
            
        Args:
        ---------------
            agent_name: Имя агента, которому отправляется задача
            message: Сообщение для отправки агенту
            tool_context: Контекст инструмента
            
        Returns:
        ---------------
            List[Union[str, Dict[str, Any]]]: Список частей ответа от агента
            
        Raises:
        ---------------
            ValueError: Если агент не найден, клиент недоступен или задача отменена/не выполнена
            
        Examples:
        ---------------
            >>> response = await host_agent.send_task("calculator_agent", "Вычисли 2+2", tool_context)
        """
        # Проверяем наличие агента
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
            
        # Обновляем состояние
        state = tool_context.state
        state['agent'] = agent_name
        
        # Получаем карточку и клиент агента
        card = self.cards[agent_name]
        client = self.remote_agent_connections[agent_name]
        if not client:
            raise ValueError(f"Client not available for {agent_name}")
            
        # Формируем идентификаторы задачи и сессии
        if 'task_id' in state:
            task_id = state['task_id']
        else:
            task_id = str(uuid.uuid4())
        session_id = state['session_id']
        
        # Обрабатываем метаданные и идентификатор сообщения
        message_id = ""
        metadata = {}
        if 'input_message_metadata' in state:
            metadata.update(**state['input_message_metadata'])
            if 'message_id' in state['input_message_metadata']:
                message_id = state['input_message_metadata']['message_id']
        if not message_id:
            message_id = str(uuid.uuid4())
            
        # Обновляем метаданные
        metadata.update(**{
            'conversation_id': session_id, 
            'message_id': message_id
        })
        
        # Формируем запрос на отправку задачи
        request: TaskSendParams = TaskSendParams(
            id=task_id,
            sessionId=session_id,
            message=Message(
                role="user",
                parts=[TextPart(text=message)],
                metadata=metadata,
            ),
            acceptedOutputModes=["text", "text/plain", "image/png"],
            # pushNotification=None,
            metadata={'conversation_id': session_id},
        )
        
        # Отправляем задачу
        task = await client.send_task(request, self.task_callback)
        
        # Обновляем состояние сессии на основе статуса задачи
        state['session_active'] = task.status.state not in [
            TaskState.COMPLETED,
            TaskState.CANCELED,
            TaskState.FAILED,
            TaskState.UNKNOWN,
        ]
        
        # Обрабатываем различные статусы задачи
        if task.status.state == TaskState.INPUT_REQUIRED:
            # Требуется ввод пользователя
            tool_context.actions.skip_summarization = True
            tool_context.actions.escalate = True
        elif task.status.state == TaskState.CANCELED:
            # Задача отменена
            raise ValueError(f"Agent {agent_name} task {task.id} is cancelled")
        elif task.status.state == TaskState.FAILED:
            # Задача не выполнена
            raise ValueError(f"Agent {agent_name} task {task.id} failed")
            
        # Формируем ответ
        response = []
        if task.status.message:
            # Используем информацию из сообщения задачи
            response.extend(convert_parts(task.status.message.parts, tool_context))
        if task.artifacts:
            for artifact in task.artifacts:
                response.extend(convert_parts(artifact.parts, tool_context))
                
        return response


def convert_parts(parts: List[Part], tool_context: ToolContext) -> List[Union[str, Dict[str, Any]]]:
    """
    Description:
    ---------------
        Преобразует список частей сообщения в формат, понятный для Gemini.
        
    Args:
    ---------------
        parts: Список частей сообщения для преобразования
        tool_context: Контекст инструмента
        
    Returns:
    ---------------
        List[Union[str, Dict[str, Any]]]: Преобразованные части сообщения
        
    Examples:
    ---------------
        >>> converted = convert_parts(message_parts, tool_context)
    """
    rval = []
    for p in parts:
        rval.append(convert_part(p, tool_context))
    return rval


def convert_part(part: Part, tool_context: ToolContext) -> Union[str, Dict[str, Any]]:
    """
    Description:
    ---------------
        Преобразует одну часть сообщения в формат, понятный для Gemini.
        
    Args:
    ---------------
        part: Часть сообщения для преобразования
        tool_context: Контекст инструмента
        
    Returns:
    ---------------
        Union[str, Dict[str, Any]]: Преобразованная часть сообщения
        
    Examples:
    ---------------
        >>> converted_text = convert_part(text_part, tool_context)
        >>> converted_file = convert_part(file_part, tool_context)
    """
    if part.type == "text":
        return part.text
    elif part.type == "data":
        return part.data
    elif part.type == "file":
        # Перепаковываем A2A FilePart в google.genai Blob
        # Не рассматриваем обычный текст как файлы    
        file_id = part.file.name
        file_bytes = base64.b64decode(part.file.bytes)    
        file_part = types.Part(
            inline_data=types.Blob(
                mime_type=part.file.mimeType,
                data=file_bytes
            )
        )
        tool_context.save_artifact(file_id, file_part)
        tool_context.actions.skip_summarization = True
        tool_context.actions.escalate = True
        return DataPart(data={"artifact-file-id": file_id})
    return f"Unknown type: {part.type}"