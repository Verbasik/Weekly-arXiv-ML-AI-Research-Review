# A2A/hosts/multiagent/remote_agent_connection.py
"""
Модуль реализует соединения с удаленными агентами через A2A API.

Обеспечивает отправку задач удаленным агентам, обработку обновлений статуса задач
и управление метаданными запросов и ответов. Поддерживает как потоковый, так и
стандартный режим отправки задач.
"""

# Стандартные библиотеки
import uuid
from typing import Callable, Optional, Union, Dict, Any, Set, AsyncGenerator

# Внутренние модули
from common.types import (
    AgentCard,
    Task,
    TaskSendParams,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    TaskStatus,
    TaskState,
    Message,
)
from common.client import A2AClient


# Определение типов для аннотаций
TaskCallbackArg = Union[Task, TaskStatusUpdateEvent, TaskArtifactUpdateEvent]
TaskUpdateCallback = Callable[[TaskCallbackArg], Task]


class RemoteAgentConnections:
    """
    Description:
    ---------------
        Класс для управления соединениями с удаленными агентами через A2A API.
        Поддерживает отправку задач и обработку ответов.
        
    Args:
    ---------------
        agent_card: Карточка агента, содержащая его метаданные и возможности
        
    Examples:
    ---------------
        >>> agent_card = A2ACardResolver("http://localhost:8001").get_agent_card()
        >>> connection = RemoteAgentConnections(agent_card)
        >>> task = await connection.send_task(task_params, callback_function)
    """

    def __init__(self, agent_card: AgentCard) -> None:
        """
        Description:
        ---------------
            Инициализирует соединение с удаленным агентом.
            
        Args:
        ---------------
            agent_card: Карточка агента, содержащая его метаданные и возможности
            
        Returns:
        ---------------
            None
        """
        self.agent_client = A2AClient(agent_card)
        self.card = agent_card

        # Параметры для управления разговором
        self.conversation_name: Optional[str] = None
        self.conversation: Optional[Any] = None
        self.pending_tasks: Set[str] = set()

    def get_agent(self) -> AgentCard:
        """
        Description:
        ---------------
            Возвращает карточку агента, связанного с данным соединением.
            
        Returns:
        ---------------
            AgentCard: Карточка агента
            
        Examples:
        ---------------
            >>> agent_card = connection.get_agent()
            >>> print(agent_card.name)
        """
        return self.card

    async def send_task(
            self,
            request: TaskSendParams,
            task_callback: Optional[TaskUpdateCallback] = None,
    ) -> Optional[Task]:
        """
        Description:
        ---------------
            Отправляет задачу удаленному агенту и обрабатывает ответ.
            Поддерживает как потоковый, так и стандартный режим отправки.
            
        Args:
        ---------------
            request: Параметры задачи для отправки
            task_callback: Функция обратного вызова для обработки обновлений задачи
            
        Returns:
        ---------------
            Optional[Task]: Результат выполнения задачи, или None если обработка выполняется через callback
            
        Examples:
        ---------------
            >>> result = await connection.send_task(
            ...     TaskSendParams(
            ...         id="task_id_123",
            ...         sessionId="session_456",
            ...         message=Message(role="user", parts=[TextPart(text="Привет")])
            ...     ),
            ...     callback_function
            ... )
        """
        # Потоковый режим отправки
        if self.card.capabilities.streaming:
            task = None
            
            # Отправляем начальное уведомление о статусе задачи, если есть callback
            if task_callback:
                task_callback(Task(
                    id=request.id,
                    sessionId=request.sessionId,
                    status=TaskStatus(
                        state=TaskState.SUBMITTED,
                        message=request.message,
                    ),
                    history=[request.message],
                ))
                
            # Обрабатываем поток ответов
            async for response in self.agent_client.send_task_streaming(request.model_dump()):
                # Объединяем метаданные запроса с ответом
                merge_metadata(response.result, request)
                
                # Обрабатываем метаданные сообщений в обновлениях статуса
                if (hasattr(response.result, 'status') and
                        hasattr(response.result.status, 'message') and
                        response.result.status.message):
                    merge_metadata(response.result.status.message, request.message)
                    message = response.result.status.message
                    
                    # Инициализируем метаданные, если их нет
                    if not message.metadata:
                        message.metadata = {}
                        
                    # Сохраняем предыдущий идентификатор сообщения
                    if 'message_id' in message.metadata:
                        message.metadata['last_message_id'] = message.metadata['message_id']
                        
                    # Генерируем новый идентификатор сообщения
                    message.metadata['message_id'] = str(uuid.uuid4())
                    
                # Вызываем callback для обработки результата, если он предоставлен
                if task_callback:
                    task = task_callback(response.result)
                    
                # Завершаем обработку, если это финальное сообщение
                if hasattr(response.result, 'final') and response.result.final:
                    break
                    
            return task
        else:  # Нестриминговый режим
            # Отправляем запрос и получаем ответ
            response = await self.agent_client.send_task(request.model_dump())
            
            # Объединяем метаданные запроса с ответом
            merge_metadata(response.result, request)
            
            # Обрабатываем метаданные сообщений в обновлениях статуса
            if (hasattr(response.result, 'status') and
                    hasattr(response.result.status, 'message') and
                    response.result.status.message):
                merge_metadata(response.result.status.message, request.message)
                message = response.result.status.message
                
                # Инициализируем метаданные, если их нет
                if not message.metadata:
                    message.metadata = {}
                    
                # Сохраняем предыдущий идентификатор сообщения
                if 'message_id' in message.metadata:
                    message.metadata['last_message_id'] = message.metadata['message_id']
                    
                # Генерируем новый идентификатор сообщения
                message.metadata['message_id'] = str(uuid.uuid4())

            # Вызываем callback для обработки результата, если он предоставлен
            if task_callback:
                task_callback(response.result)
                
            return response.result


def merge_metadata(target: Any, source: Any) -> None:
    """
    Description:
    ---------------
        Объединяет метаданные из источника в целевой объект, если оба объекта имеют
        атрибут metadata.
        
    Args:
    ---------------
        target: Целевой объект, в который будут добавлены метаданные
        source: Исходный объект, из которого будут взяты метаданные
        
    Returns:
    ---------------
        None
        
    Examples:
    ---------------
        >>> merge_metadata(response.result, request)
    """
    # Проверяем наличие атрибутов metadata у обоих объектов
    if not hasattr(target, 'metadata') or not hasattr(source, 'metadata'):
        return
        
    # Если оба объекта имеют метаданные, объединяем их
    if target.metadata and source.metadata:
        target.metadata.update(source.metadata)
    # Если только источник имеет метаданные, копируем их в цель
    elif source.metadata:
        target.metadata = dict(**source.metadata)