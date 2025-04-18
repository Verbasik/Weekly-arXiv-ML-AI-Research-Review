# A2A/common/server/task_manager.py
"""Базовые классы менеджеров задач.

Этот модуль содержит абстрактные классы и базовую реализацию для менеджеров задач,
которые управляют жизненным циклом задач, включая их создание, обновление,
отмену и настройку push-уведомлений.
"""

# Стандартные библиотеки Python
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterable, Dict, List, Optional, Union

# Внутренние модули проекта
import common.server.utils as utils
from common.types import (
    Artifact,
    CancelTaskRequest,
    CancelTaskResponse,
    GetTaskPushNotificationRequest,
    GetTaskPushNotificationResponse,
    GetTaskRequest,
    GetTaskResponse,
    InternalError,
    JSONRPCError,
    JSONRPCResponse,
    PushNotificationConfig,
    PushNotificationNotSupportedError,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    SetTaskPushNotificationRequest,
    SetTaskPushNotificationResponse,
    Task,
    TaskIdParams,
    TaskNotCancelableError,
    TaskNotFoundError,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskResubscriptionRequest,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)

# Настройка логирования
logger = logging.getLogger(__name__)


class TaskManager(ABC):
    """
    Description:
    ---------------
        Абстрактный базовый класс для менеджеров задач, определяющий интерфейс
        для управления жизненным циклом задач.

    Examples:
    ---------------
        >>> class MyTaskManager(TaskManager):
        ...     async def on_get_task(self, request):
        ...         # Реализация метода
        ...         pass
        ...     # Реализация других абстрактных методов
    """

    @abstractmethod
    async def on_get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        """
        Description:
        ---------------
            Обрабатывает запрос на получение информации о задаче.

        Args:
        ---------------
            request: Запрос на получение задачи.

        Returns:
        ---------------
            GetTaskResponse: Ответ с информацией о задаче или ошибкой.
        """
        pass

    @abstractmethod
    async def on_cancel_task(self, request: CancelTaskRequest) -> CancelTaskResponse:
        """
        Description:
        ---------------
            Обрабатывает запрос на отмену задачи.

        Args:
        ---------------
            request: Запрос на отмену задачи.

        Returns:
        ---------------
            CancelTaskResponse: Ответ с результатом отмены задачи или ошибкой.
        """
        pass

    @abstractmethod
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Description:
        ---------------
            Обрабатывает запрос на отправку задачи агенту.

        Args:
        ---------------
            request: Запрос на отправку задачи.

        Returns:
        ---------------
            SendTaskResponse: Ответ с результатом обработки задачи или ошибкой.
        """
        pass

    @abstractmethod
    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        """
        Description:
        ---------------
            Обрабатывает запрос на подписку на поток обновлений задачи.

        Args:
        ---------------
            request: Запрос на подписку на поток задачи.

        Returns:
        ---------------
            AsyncIterable[SendTaskStreamingResponse]: Поток обновлений задачи.
            JSONRPCResponse: Ответ с ошибкой, если запрос не валиден.
        """
        pass

    @abstractmethod
    async def on_set_task_push_notification(
        self, request: SetTaskPushNotificationRequest
    ) -> SetTaskPushNotificationResponse:
        """
        Description:
        ---------------
            Обрабатывает запрос на настройку push-уведомлений для задачи.

        Args:
        ---------------
            request: Запрос на настройку push-уведомлений.

        Returns:
        ---------------
            SetTaskPushNotificationResponse: Ответ с результатом настройки
                                            push-уведомлений или ошибкой.
        """
        pass

    @abstractmethod
    async def on_get_task_push_notification(
        self, request: GetTaskPushNotificationRequest
    ) -> GetTaskPushNotificationResponse:
        """
        Description:
        ---------------
            Обрабатывает запрос на получение информации о настройках
            push-уведомлений для задачи.

        Args:
        ---------------
            request: Запрос на получение информации о push-уведомлениях.

        Returns:
        ---------------
            GetTaskPushNotificationResponse: Ответ с информацией о настройках
                                           push-уведомлений или ошибкой.
        """
        pass

    @abstractmethod
    async def on_resubscribe_to_task(
        self, request: TaskResubscriptionRequest
    ) -> Union[AsyncIterable[SendTaskResponse], JSONRPCResponse]:
        """
        Description:
        ---------------
            Обрабатывает запрос на повторную подписку на поток обновлений задачи.

        Args:
        ---------------
            request: Запрос на повторную подписку.

        Returns:
        ---------------
            AsyncIterable[SendTaskResponse]: Поток обновлений задачи.
            JSONRPCResponse: Ответ с ошибкой, если запрос не валиден.
        """
        pass


class InMemoryTaskManager(TaskManager):
    """
    Description:
    ---------------
        Базовая реализация менеджера задач, хранящая задачи в памяти.
        Обеспечивает основные операции по управлению задачами и поддержку
        push-уведомлений и SSE (Server-Sent Events).

    Attributes:
    ---------------
        tasks: Словарь задач, индексированных по их ID.
        push_notification_infos: Словарь настроек push-уведомлений для задач.
        lock: Блокировка для безопасного доступа к данным задач.
        task_sse_subscribers: Словарь подписчиков на SSE для каждой задачи.
        subscriber_lock: Блокировка для безопасного доступа к подписчикам.

    Examples:
    ---------------
        >>> task_manager = InMemoryTaskManager()
        >>> # Использование методов task_manager
    """

    def __init__(self):
        """
        Description:
        ---------------
            Инициализирует менеджер задач с пустыми хранилищами для задач,
            настроек push-уведомлений и подписчиков на SSE.

        Returns:
        ---------------
            None
        """
        # Хранилище задач
        self.tasks: Dict[str, Task] = {}
        # Хранилище настроек push-уведомлений
        self.push_notification_infos: Dict[str, PushNotificationConfig] = {}
        # Блокировка для безопасного доступа к данным задач
        self.lock = asyncio.Lock()
        # Хранилище подписчиков на SSE для каждой задачи
        self.task_sse_subscribers: Dict[str, List[asyncio.Queue]] = {}
        # Блокировка для безопасного доступа к подписчикам
        self.subscriber_lock = asyncio.Lock()

    async def on_get_task(self, request: GetTaskRequest) -> GetTaskResponse:
        """
        Description:
        ---------------
            Обрабатывает запрос на получение информации о задаче.
            Возвращает задачу с ограничением истории, если она существует.

        Args:
        ---------------
            request: Запрос на получение задачи.

        Returns:
        ---------------
            GetTaskResponse: Ответ с информацией о задаче или ошибкой.

        Examples:
        ---------------
            >>> async def example():
            ...     request = GetTaskRequest(id="request123", params=TaskQueryParams(id="task456"))
            ...     response = await task_manager.on_get_task(request)
            ...     if "error" not in response:
            ...         print(f"Task state: {response.result.status.state}")
        """
        logger.info(f"Getting task {request.params.id}")
        # Получение параметров запроса
        task_query_params: TaskQueryParams = request.params

        # Получение задачи из хранилища с блокировкой для безопасного доступа
        async with self.lock:
            task = self.tasks.get(task_query_params.id)
            if task is None:
                # Возврат ошибки, если задача не найдена
                return GetTaskResponse(id=request.id, error=TaskNotFoundError())

            # Формирование ответа с ограничением истории задачи
            task_result = self.append_task_history(
                task, task_query_params.historyLength
            )

        # Возврат ответа с информацией о задаче
        return GetTaskResponse(id=request.id, result=task_result)

    async def on_cancel_task(self, request: CancelTaskRequest) -> CancelTaskResponse:
        """
        Description:
        ---------------
            Обрабатывает запрос на отмену задачи.
            В данной реализации задачи не поддерживают отмену.

        Args:
        ---------------
            request: Запрос на отмену задачи.

        Returns:
        ---------------
            CancelTaskResponse: Ответ с ошибкой, указывающей, что задача не может быть отменена,
                               или что задача не найдена.

        Examples:
        ---------------
            >>> async def example():
            ...     request = CancelTaskRequest(id="request123", params=TaskIdParams(id="task456"))
            ...     response = await task_manager.on_cancel_task(request)
            ...     print(f"Error: {response.error}")
        """
        logger.info(f"Cancelling task {request.params.id}")
        # Получение параметров запроса
        task_id_params: TaskIdParams = request.params

        # Проверка существования задачи
        async with self.lock:
            task = self.tasks.get(task_id_params.id)
            if task is None:
                # Возврат ошибки, если задача не найдена
                return CancelTaskResponse(id=request.id, error=TaskNotFoundError())

        # Возврат ошибки, указывающей, что задача не может быть отменена
        return CancelTaskResponse(id=request.id, error=TaskNotCancelableError())

    @abstractmethod
    async def on_send_task(self, request: SendTaskRequest) -> SendTaskResponse:
        """
        Description:
        ---------------
            Абстрактный метод для обработки запроса на отправку задачи агенту.
            Должен быть реализован в подклассах.

        Args:
        ---------------
            request: Запрос на отправку задачи.

        Returns:
        ---------------
            SendTaskResponse: Ответ с результатом обработки задачи или ошибкой.
        """
        pass

    @abstractmethod
    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        """
        Description:
        ---------------
            Абстрактный метод для обработки запроса на подписку на поток обновлений задачи.
            Должен быть реализован в подклассах.

        Args:
        ---------------
            request: Запрос на подписку на поток задачи.

        Returns:
        ---------------
            AsyncIterable[SendTaskStreamingResponse]: Поток обновлений задачи.
            JSONRPCResponse: Ответ с ошибкой, если запрос не валиден.
        """
        pass

    async def set_push_notification_info(
        self, task_id: str, notification_config: PushNotificationConfig
    ) -> None:
        """
        Description:
        ---------------
            Устанавливает настройки push-уведомлений для указанной задачи.

        Args:
        ---------------
            task_id: Идентификатор задачи.
            notification_config: Конфигурация push-уведомлений.

        Returns:
        ---------------
            None

        Raises:
        ---------------
            ValueError: Если задача с указанным ID не найдена.

        Examples:
        ---------------
            >>> async def example():
            ...     config = PushNotificationConfig(url="https://example.com/notify")
            ...     await task_manager.set_push_notification_info("task123", config)
        """
        # Проверка существования задачи и сохранение настроек с блокировкой
        async with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                raise ValueError(f"Task not found for {task_id}")

            # Сохранение настроек push-уведомлений
            self.push_notification_infos[task_id] = notification_config

    async def get_push_notification_info(
        self, task_id: str
    ) -> PushNotificationConfig:
        """
        Description:
        ---------------
            Получает настройки push-уведомлений для указанной задачи.

        Args:
        ---------------
            task_id: Идентификатор задачи.

        Returns:
        ---------------
            PushNotificationConfig: Конфигурация push-уведомлений.

        Raises:
        ---------------
            ValueError: Если задача с указанным ID не найдена.

        Examples:
        ---------------
            >>> async def example():
            ...     config = await task_manager.get_push_notification_info("task123")
            ...     print(f"Notification URL: {config.url}")
        """
        # Проверка существования задачи и получение настроек с блокировкой
        async with self.lock:
            task = self.tasks.get(task_id)
            if task is None:
                raise ValueError(f"Task not found for {task_id}")

            # Возврат настроек push-уведомлений
            return self.push_notification_infos[task_id]

    async def has_push_notification_info(self, task_id: str) -> bool:
        """
        Description:
        ---------------
            Проверяет, настроены ли push-уведомления для указанной задачи.

        Args:
        ---------------
            task_id: Идентификатор задачи.

        Returns:
        ---------------
            bool: True, если настройки push-уведомлений существуют, False в противном случае.

        Examples:
        ---------------
            >>> async def example():
            ...     has_notifications = await task_manager.has_push_notification_info("task123")
            ...     print(f"Has notifications: {has_notifications}")
        """
        # Проверка наличия настроек push-уведомлений с блокировкой
        async with self.lock:
            return task_id in self.push_notification_infos

    async def on_set_task_push_notification(
        self, request: SetTaskPushNotificationRequest
    ) -> SetTaskPushNotificationResponse:
        """
        Description:
        ---------------
            Обрабатывает запрос на настройку push-уведомлений для задачи.

        Args:
        ---------------
            request: Запрос на настройку push-уведомлений.

        Returns:
        ---------------
            SetTaskPushNotificationResponse: Ответ с результатом настройки
                                            push-уведомлений или ошибкой.

        Examples:
        ---------------
            >>> async def example():
            ...     config = PushNotificationConfig(url="https://example.com/notify")
            ...     params = TaskPushNotificationConfig(id="task123", pushNotificationConfig=config)
            ...     request = SetTaskPushNotificationRequest(id="request123", params=params)
            ...     response = await task_manager.on_set_task_push_notification(request)
            ...     if "error" not in response:
            ...         print("Push notifications configured successfully")
        """
        logger.info(f"Setting task push notification {request.params.id}")
        # Получение параметров запроса
        task_notification_params: TaskPushNotificationConfig = request.params

        try:
            # Установка настроек push-уведомлений
            await self.set_push_notification_info(
                task_notification_params.id, 
                task_notification_params.pushNotificationConfig
            )
        except Exception as e:
            # Логирование и обработка ошибок
            logger.error(f"Error while setting push notification info: {e}")
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message="An error occurred while setting push notification info"
                ),
            )
            
        # Возврат успешного ответа
        return SetTaskPushNotificationResponse(
            id=request.id, result=task_notification_params
        )

    async def on_get_task_push_notification(
        self, request: GetTaskPushNotificationRequest
    ) -> GetTaskPushNotificationResponse:
        """
        Description:
        ---------------
            Обрабатывает запрос на получение информации о настройках
            push-уведомлений для задачи.

        Args:
        ---------------
            request: Запрос на получение информации о push-уведомлениях.

        Returns:
        ---------------
            GetTaskPushNotificationResponse: Ответ с информацией о настройках
                                           push-уведомлений или ошибкой.

        Examples:
        ---------------
            >>> async def example():
            ...     request = GetTaskPushNotificationRequest(
            ...         id="request123", params=TaskIdParams(id="task123")
            ...     )
            ...     response = await task_manager.on_get_task_push_notification(request)
            ...     if "error" not in response:
            ...         print(f"Notification URL: {response.result.pushNotificationConfig.url}")
        """
        logger.info(f"Getting task push notification {request.params.id}")
        # Получение параметров запроса
        task_params: TaskIdParams = request.params

        try:
            # Получение настроек push-уведомлений
            notification_info = await self.get_push_notification_info(task_params.id)
        except Exception as e:
            # Логирование и обработка ошибок
            logger.error(f"Error while getting push notification info: {e}")
            return GetTaskPushNotificationResponse(
                id=request.id,
                error=InternalError(
                    message="An error occurred while getting push notification info"
                ),
            )
        
        # Возврат успешного ответа с настройками
        return GetTaskPushNotificationResponse(
            id=request.id, 
            result=TaskPushNotificationConfig(
                id=task_params.id, 
                pushNotificationConfig=notification_info
            )
        )

    async def upsert_task(self, task_send_params: TaskSendParams) -> Task:
        """
        Description:
        ---------------
            Создает новую задачу или обновляет существующую на основе параметров запроса.

        Args:
        ---------------
            task_send_params: Параметры отправки задачи.

        Returns:
        ---------------
            Task: Созданная или обновленная задача.

        Examples:
        ---------------
            >>> async def example():
            ...     # Предполагается, что task_params - это экземпляр TaskSendParams
            ...     task = await task_manager.upsert_task(task_params)
            ...     print(f"Task ID: {task.id}")
        """
        logger.info(f"Upserting task {task_send_params.id}")
        # Создание или обновление задачи с блокировкой
        async with self.lock:
            task = self.tasks.get(task_send_params.id)
            if task is None:
                # Создание новой задачи, если она не существует
                task = Task(
                    id=task_send_params.id,
                    sessionId=task_send_params.sessionId,
                    messages=[task_send_params.message],
                    status=TaskStatus(state=TaskState.SUBMITTED),
                    history=[task_send_params.message],
                )
                self.tasks[task_send_params.id] = task
            else:
                # Обновление истории существующей задачи
                task.history.append(task_send_params.message)

            return task

    async def on_resubscribe_to_task(
        self, request: TaskResubscriptionRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        """
        Description:
        ---------------
            Обрабатывает запрос на повторную подписку на поток обновлений задачи.
            В данной базовой реализации возвращает ошибку о том, что функциональность
            не реализована.

        Args:
        ---------------
            request: Запрос на повторную подписку.

        Returns:
        ---------------
            JSONRPCResponse: Ответ с ошибкой о том, что функциональность не реализована.

        Examples:
        ---------------
            >>> async def example():
            ...     request = TaskResubscriptionRequest(
            ...         id="request123", params=TaskIdParams(id="task123")
            ...     )
            ...     response = await task_manager.on_resubscribe_to_task(request)
            ...     print(f"Error: {response.error}")
        """
        # Возврат ошибки о том, что функциональность не реализована
        return utils.new_not_implemented_error(request.id)

    async def update_store(
        self, task_id: str, status: TaskStatus, artifacts: Optional[List[Artifact]]
    ) -> Task:
        """
        Description:
        ---------------
            Обновляет информацию о задаче в хранилище, включая статус и артефакты.

        Args:
        ---------------
            task_id: Идентификатор задачи.
            status: Новый статус задачи.
            artifacts: Список артефактов для добавления к задаче.

        Returns:
        ---------------
            Task: Обновленная задача.

        Raises:
        ---------------
            ValueError: Если задача с указанным ID не найдена.

        Examples:
        ---------------
            >>> async def example():
            ...     status = TaskStatus(state=TaskState.COMPLETED)
            ...     artifacts = [Artifact(parts=[{"type": "text", "text": "Result"}])]
            ...     task = await task_manager.update_store("task123", status, artifacts)
            ...     print(f"Task state: {task.status.state}")
        """
        # Обновление задачи с блокировкой
        async with self.lock:
            try:
                # Получение задачи из хранилища
                task = self.tasks[task_id]
            except KeyError:
                # Логирование и обработка ошибки, если задача не найдена
                logger.error(f"Task {task_id} not found for updating the task")
                raise ValueError(f"Task {task_id} not found")

            # Обновление статуса задачи
            task.status = status

            # Обновление истории задачи, если есть сообщение в статусе
            if status.message is not None:
                task.history.append(status.message)

            # Добавление артефактов, если они предоставлены
            if artifacts is not None:
                if task.artifacts is None:
                    task.artifacts = []
                task.artifacts.extend(artifacts)

            return task

    def append_task_history(
        self, task: Task, historyLength: Optional[int]
    ) -> Task:
        """
        Description:
        ---------------
            Создает копию задачи с ограниченной историей сообщений.

        Args:
        ---------------
            task: Исходная задача.
            historyLength: Максимальное количество сообщений в истории.

        Returns:
        ---------------
            Task: Копия задачи с ограниченной историей.

        Examples:
        ---------------
            >>> task_copy = task_manager.append_task_history(task, 5)
            >>> print(f"History length: {len(task_copy.history)}")
        """
        # Создание копии задачи
        new_task = task.model_copy()
        
        # Ограничение истории, если указано количество сообщений
        if historyLength is not None and historyLength > 0:
            new_task.history = new_task.history[-historyLength:]
        else:
            # Очистка истории, если количество не указано или равно 0
            new_task.history = []

        return new_task        

    async def setup_sse_consumer(
        self, task_id: str, is_resubscribe: bool = False
    ) -> asyncio.Queue:
        """
        Description:
        ---------------
            Настраивает потребителя Server-Sent Events (SSE) для задачи.

        Args:
        ---------------
            task_id: Идентификатор задачи.
            is_resubscribe: Флаг повторной подписки.

        Returns:
        ---------------
            asyncio.Queue: Очередь событий для SSE.

        Raises:
        ---------------
            ValueError: Если задача не найдена при повторной подписке.

        Examples:
        ---------------
            >>> async def example():
            ...     queue = await task_manager.setup_sse_consumer("task123")
            ...     # Использование очереди для получения обновлений
        """
        # Настройка потребителя SSE с блокировкой
        async with self.subscriber_lock:
            if task_id not in self.task_sse_subscribers:
                if is_resubscribe:
                    # Ошибка, если задача не найдена при повторной подписке
                    raise ValueError("Task not found for resubscription")
                else:
                    # Создание списка подписчиков для новой задачи
                    self.task_sse_subscribers[task_id] = []

            # Создание очереди без ограничения размера и добавление в список подписчиков
            sse_event_queue = asyncio.Queue(maxsize=0)  # <=0 означает неограниченный размер
            self.task_sse_subscribers[task_id].append(sse_event_queue)
            return sse_event_queue

    async def enqueue_events_for_sse(
        self, task_id: str, task_update_event: Any
    ) -> None:
        """
        Description:
        ---------------
            Отправляет событие обновления задачи всем подписчикам.

        Args:
        ---------------
            task_id: Идентификатор задачи.
            task_update_event: Событие обновления задачи.

        Returns:
        ---------------
            None

        Examples:
        ---------------
            >>> async def example():
            ...     event = TaskStatusUpdateEvent(
            ...         id="task123", 
            ...         status=TaskStatus(state=TaskState.COMPLETED), 
            ...         final=True
            ...     )
            ...     await task_manager.enqueue_events_for_sse("task123", event)
        """
        # Отправка события всем подписчикам с блокировкой
        async with self.subscriber_lock:
            if task_id not in self.task_sse_subscribers:
                # Прекращение обработки, если нет подписчиков
                return

            # Получение списка текущих подписчиков
            current_subscribers = self.task_sse_subscribers[task_id]
            # Отправка события каждому подписчику
            for subscriber in current_subscribers:
                await subscriber.put(task_update_event)

    async def dequeue_events_for_sse(
        self, request_id: str, task_id: str, sse_event_queue: asyncio.Queue
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        """
        Description:
        ---------------
            Извлекает события из очереди SSE и формирует поток ответов.
            Продолжает извлечение до получения финального события или ошибки.

        Args:
        ---------------
            request_id: Идентификатор запроса.
            task_id: Идентификатор задачи.
            sse_event_queue: Очередь событий SSE.

        Returns:
        ---------------
            AsyncIterable[SendTaskStreamingResponse]: Поток ответов с событиями обновления.
            JSONRPCResponse: Ответ с ошибкой, если произошла ошибка обработки.

        Examples:
        ---------------
            >>> async def example():
            ...     # Предполагается, что queue - это очередь событий
            ...     async for response in task_manager.dequeue_events_for_sse(
            ...         "request123", "task123", queue
            ...     ):
            ...         print(response)
        """
        try:
            # Извлечение событий из очереди в бесконечном цикле
            while True:                
                # Ожидание следующего события из очереди
                event = await sse_event_queue.get()
                
                # Проверка, является ли событие ошибкой
                if isinstance(event, JSONRPCError):
                    # Возврат ответа с ошибкой
                    yield SendTaskStreamingResponse(id=request_id, error=event)
                    # Прерывание цикла после отправки ошибки
                    break
                                                
                # Возврат ответа с событием
                yield SendTaskStreamingResponse(id=request_id, result=event)
                
                # Проверка, является ли событие финальным обновлением статуса
                if isinstance(event, TaskStatusUpdateEvent) and event.final:
                    # Прерывание цикла после финального события
                    break
        finally:
            # Очистка ресурсов при выходе из цикла
            async with self.subscriber_lock:
                # Удаление очереди из списка подписчиков
                if task_id in self.task_sse_subscribers:
                    self.task_sse_subscribers[task_id].remove(sse_event_queue)

