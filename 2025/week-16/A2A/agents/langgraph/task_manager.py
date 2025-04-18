# A2A/agents/langgraph/task_manager.py
"""Менеджер задач для агента конвертации валют.

Этот модуль предоставляет реализацию менеджера задач, управляющего обработкой
запросов к агенту конвертации валют, включая потоковую обработку и отправку
push-уведомлений о статусе задач.
"""

# Стандартные библиотеки Python
import asyncio
import logging
import traceback
from typing import Any, AsyncIterable, Dict, List, Optional, Union

# Внутренние модули проекта
from .agent import CurrencyAgent
from common.server.task_manager import InMemoryTaskManager
from common.types import (
    Artifact,
    InternalError,
    InvalidParamsError,
    JSONRPCResponse,
    Message,
    PushNotificationConfig,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    SetTaskPushNotificationRequest,
    SetTaskPushNotificationResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskNotFoundError,
    TaskPushNotificationConfig,
    TaskSendParams,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from common.utils.push_notification_auth import PushNotificationSenderAuth
import common.server.utils as utils

# Настройка логирования
logger = logging.getLogger(__name__)


class AgentTaskManager(InMemoryTaskManager):
    """
    Description:
    ---------------
        Менеджер задач для агента конвертации валют, обрабатывающий запросы, 
        управляющий потоковой обработкой и отправкой push-уведомлений.

    Args:
    ---------------
        agent: Экземпляр агента конвертации валют для обработки запросов.
        notification_sender_auth: Аутентификация для отправки push-уведомлений.

    Examples:
    ---------------
        >>> from agents.langgraph.agent import CurrencyAgent
        >>> from common.utils.push_notification_auth import PushNotificationSenderAuth
        >>> agent = CurrencyAgent()
        >>> auth = PushNotificationSenderAuth()
        >>> task_manager = AgentTaskManager(agent=agent, notification_sender_auth=auth)
    """

    def __init__(
        self, 
        agent: CurrencyAgent, 
        notification_sender_auth: PushNotificationSenderAuth
    ):
        """
        Description:
        ---------------
            Инициализирует менеджер задач агента конвертации валют.

        Args:
        ---------------
            agent: Экземпляр агента конвертации валют.
            notification_sender_auth: Аутентификация для отправки push-уведомлений.

        Returns:
        ---------------
            None
        """
        super().__init__()
        self.agent = agent
        self.notification_sender_auth = notification_sender_auth

    async def _run_streaming_agent(self, request: SendTaskStreamingRequest) -> None:
        """
        Description:
        ---------------
            Запускает агента в потоковом режиме и обрабатывает промежуточные результаты.
            Обновляет состояние задачи и отправляет уведомления по мере обработки.

        Args:
        ---------------
            request: Запрос на потоковую обработку задачи.

        Returns:
        ---------------
            None

        Raises:
        ---------------
            Exception: При ошибке в процессе потоковой обработки.

        Examples:
        ---------------
            >>> async def example():
            ...     await task_manager._run_streaming_agent(request)
        """
        # Получение параметров задачи и текста запроса
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)

        try:
            # Обработка потока ответов от агента
            async for item in self.agent.stream(query, task_send_params.sessionId):
                # Извлечение информации о состоянии задачи из ответа агента
                is_task_complete = item["is_task_complete"]
                require_user_input = item["require_user_input"]
                artifact = None
                message = None
                parts = [{"type": "text", "text": item["content"]}]
                end_stream = False

                # Определение состояния задачи на основе ответа агента
                if not is_task_complete and not require_user_input:
                    # Задача в процессе выполнения
                    task_state = TaskState.WORKING
                    message = Message(role="agent", parts=parts)
                elif require_user_input:
                    # Требуется ввод пользователя
                    task_state = TaskState.INPUT_REQUIRED
                    message = Message(role="agent", parts=parts)
                    end_stream = True
                else:
                    # Задача завершена
                    task_state = TaskState.COMPLETED
                    artifact = Artifact(parts=parts, index=0, append=False)
                    end_stream = True

                # Обновление статуса задачи и отправка уведомлений
                task_status = TaskStatus(state=task_state, message=message)
                latest_task = await self.update_store(
                    task_send_params.id,
                    task_status,
                    None if artifact is None else [artifact],
                )
                await self.send_task_notification(latest_task)

                # Отправка уведомления об артефакте, если он создан
                if artifact:
                    task_artifact_update_event = TaskArtifactUpdateEvent(
                        id=task_send_params.id, artifact=artifact
                    )
                    await self.enqueue_events_for_sse(
                        task_send_params.id, task_artifact_update_event
                    )                    
                
                # Отправка уведомления об обновлении статуса задачи
                task_update_event = TaskStatusUpdateEvent(
                    id=task_send_params.id, status=task_status, final=end_stream
                )
                await self.enqueue_events_for_sse(
                    task_send_params.id, task_update_event
                )

        except Exception as e:
            # Обработка ошибок при потоковой обработке
            logger.error(f"An error occurred while streaming the response: {e}")
            await self.enqueue_events_for_sse(
                task_send_params.id,
                InternalError(
                    message=f"An error occurred while streaming the response: {e}"
                )                
            )

    def _validate_request(
        self, request: Union[SendTaskRequest, SendTaskStreamingRequest]
    ) -> Optional[JSONRPCResponse]:
        """
        Description:
        ---------------
            Проверяет валидность запроса, включая совместимость модальностей
            и настройки push-уведомлений.

        Args:
        ---------------
            request: Запрос на обработку задачи или потоковую обработку.

        Returns:
        ---------------
            Optional[JSONRPCResponse]: Ответ с ошибкой, если запрос не валиден,
                                       или None, если запрос валиден.

        Examples:
        ---------------
            >>> error = task_manager._validate_request(request)
            >>> if error:
            ...     print("Request validation failed")
            ... else:
            ...     print("Request is valid")
        """
        # Получение параметров задачи
        task_send_params: TaskSendParams = request.params
        
        # Проверка совместимости типов контента
        if not utils.are_modalities_compatible(
            task_send_params.acceptedOutputModes, 
            CurrencyAgent.SUPPORTED_CONTENT_TYPES
        ):
            # Логирование предупреждения о несовместимости
            logger.warning(
                "Unsupported output mode. Received %s, Support %s",
                task_send_params.acceptedOutputModes,
                CurrencyAgent.SUPPORTED_CONTENT_TYPES,
            )
            # Возврат ошибки несовместимости типов
            return utils.new_incompatible_types_error(request.id)
        
        # Проверка настроек push-уведомлений
        if task_send_params.pushNotification and not task_send_params.pushNotification.url:
            logger.warning("Push notification URL is missing")
            return JSONRPCResponse(
                id=request.id, 
                error=InvalidParamsError(message="Push notification URL is missing")
            )
        
        # Запрос валиден
        return None
        
    async def on_send_task(
        self, request: SendTaskRequest
    ) -> SendTaskResponse:
        """
        Description:
        ---------------
            Обрабатывает запрос на отправку задачи агенту.
            Проверяет запрос, настраивает push-уведомления и вызывает агента.

        Args:
        ---------------
            request: Запрос на отправку задачи.

        Returns:
        ---------------
            SendTaskResponse: Ответ с результатом обработки задачи.

        Raises:
        ---------------
            ValueError: Если возникает ошибка при вызове агента.

        Examples:
        ---------------
            >>> async def example():
            ...     response = await task_manager.on_send_task(request)
            ...     print(f"Task state: {response.result.status.state}")
        """
        # Проверка валидности запроса
        validation_error = self._validate_request(request)
        if validation_error:
            return SendTaskResponse(id=request.id, error=validation_error.error)
        
        # Настройка push-уведомлений, если они указаны
        if request.params.pushNotification:
            if not await self.set_push_notification_info(
                request.params.id, request.params.pushNotification
            ):
                return SendTaskResponse(
                    id=request.id, 
                    error=InvalidParamsError(message="Push notification URL is invalid")
                )

        # Сохранение задачи и обновление ее статуса на "в процессе"
        await self.upsert_task(request.params)
        task = await self.update_store(
            request.params.id, TaskStatus(state=TaskState.WORKING), None
        )
        await self.send_task_notification(task)

        # Получение текста запроса и вызов агента
        task_send_params: TaskSendParams = request.params
        query = self._get_user_query(task_send_params)
        try:
            agent_response = self.agent.invoke(query, task_send_params.sessionId)
        except Exception as e:
            logger.error(f"Error invoking agent: {e}")
            raise ValueError(f"Error invoking agent: {e}")
        
        # Обработка ответа агента и возврат результата
        return await self._process_agent_response(request, agent_response)

    async def on_send_task_subscribe(
        self, request: SendTaskStreamingRequest
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        """
        Description:
        ---------------
            Обрабатывает запрос на подписку на потоковые обновления задачи.
            Настраивает push-уведомления и запускает агента в потоковом режиме.

        Args:
        ---------------
            request: Запрос на подписку на потоковые обновления.

        Returns:
        ---------------
            AsyncIterable[SendTaskStreamingResponse]: Поток обновлений задачи.
            JSONRPCResponse: Ответ с ошибкой, если запрос не валиден.

        Examples:
        ---------------
            >>> async def example():
            ...     response = await task_manager.on_send_task_subscribe(request)
            ...     if isinstance(response, JSONRPCResponse):
            ...         print("Subscription failed")
            ...     else:
            ...         async for update in response:
            ...             print("Received update")
        """
        try:
            # Проверка валидности запроса
            error = self._validate_request(request)
            if error:
                return error

            # Сохранение задачи
            await self.upsert_task(request.params)

            # Настройка push-уведомлений, если они указаны
            if request.params.pushNotification:
                if not await self.set_push_notification_info(
                    request.params.id, request.params.pushNotification
                ):
                    return JSONRPCResponse(
                        id=request.id, 
                        error=InvalidParamsError(message="Push notification URL is invalid")
                    )

            # Настройка очереди событий для Server-Sent Events (SSE)
            task_send_params: TaskSendParams = request.params
            sse_event_queue = await self.setup_sse_consumer(task_send_params.id, False)            

            # Запуск агента в потоковом режиме в отдельной задаче
            asyncio.create_task(self._run_streaming_agent(request))

            # Возврат очереди событий для потоковой передачи клиенту
            return self.dequeue_events_for_sse(
                request.id, task_send_params.id, sse_event_queue
            )
        except Exception as e:
            # Обработка ошибок
            logger.error(f"Error in SSE stream: {e}")
            print(traceback.format_exc())
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message="An error occurred while streaming the response"
                ),
            )

    async def _process_agent_response(
        self, request: SendTaskRequest, agent_response: Dict[str, Any]
    ) -> SendTaskResponse:
        """
        Description:
        ---------------
            Обрабатывает ответ агента и обновляет хранилище задач.
            Определяет статус задачи и формирует артефакты на основе ответа агента.

        Args:
        ---------------
            request: Запрос на отправку задачи.
            agent_response: Ответ от агента.

        Returns:
        ---------------
            SendTaskResponse: Ответ с результатом обработки задачи.

        Examples:
        ---------------
            >>> async def example():
            ...     agent_response = {"content": "Result", "require_user_input": False}
            ...     response = await task_manager._process_agent_response(request, agent_response)
            ...     print(f"Task state: {response.result.status.state}")
        """
        # Получение параметров задачи
        task_send_params: TaskSendParams = request.params
        task_id = task_send_params.id
        history_length = task_send_params.historyLength
        task_status = None

        # Формирование частей ответа
        parts = [{"type": "text", "text": agent_response["content"]}]
        artifact = None
        
        # Определение статуса задачи в зависимости от ответа агента
        if agent_response["require_user_input"]:
            # Требуется ввод пользователя
            task_status = TaskStatus(
                state=TaskState.INPUT_REQUIRED,
                message=Message(role="agent", parts=parts),
            )
        else:
            # Задача завершена
            task_status = TaskStatus(state=TaskState.COMPLETED)
            artifact = Artifact(parts=parts)
        
        # Обновление хранилища задач и отправка уведомления
        task = await self.update_store(
            task_id, task_status, None if artifact is None else [artifact]
        )
        task_result = self.append_task_history(task, history_length)
        await self.send_task_notification(task)
        
        # Формирование и возврат ответа
        return SendTaskResponse(id=request.id, result=task_result)
    
    def _get_user_query(self, task_send_params: TaskSendParams) -> str:
        """
        Description:
        ---------------
            Извлекает текстовый запрос пользователя из параметров задачи.

        Args:
        ---------------
            task_send_params: Параметры отправки задачи.

        Returns:
        ---------------
            str: Текстовый запрос пользователя.

        Raises:
        ---------------
            ValueError: Если часть сообщения не является текстовой.

        Examples:
        ---------------
            >>> query = task_manager._get_user_query(task_params)
            >>> print(f"User query: {query}")
        """
        # Получение первой части сообщения
        part = task_send_params.message.parts[0]
        
        # Проверка, что часть является текстовой
        if not isinstance(part, TextPart):
            raise ValueError("Only text parts are supported")
        
        # Возврат текста запроса
        return part.text
    
    async def send_task_notification(self, task: Task) -> None:
        """
        Description:
        ---------------
            Отправляет push-уведомление о состоянии задачи, если
            настроен URL для уведомлений.

        Args:
        ---------------
            task: Задача, о которой нужно отправить уведомление.

        Returns:
        ---------------
            None

        Examples:
        ---------------
            >>> async def example():
            ...     task = await task_manager.get_task("task123")
            ...     await task_manager.send_task_notification(task)
        """
        # Проверка наличия информации о push-уведомлениях для задачи
        if not await self.has_push_notification_info(task.id):
            logger.info(f"No push notification info found for task {task.id}")
            return
        
        # Получение информации о push-уведомлениях
        push_info = await self.get_push_notification_info(task.id)

        # Отправка уведомления
        logger.info(f"Notifying for task {task.id} => {task.status.state}")
        await self.notification_sender_auth.send_push_notification(
            push_info.url,
            data=task.model_dump(exclude_none=True)
        )

    async def on_resubscribe_to_task(
        self, request: Any
    ) -> Union[AsyncIterable[SendTaskStreamingResponse], JSONRPCResponse]:
        """
        Description:
        ---------------
            Обрабатывает запрос на повторную подписку на задачу после разрыва соединения.
            Восстанавливает поток обновлений для существующей задачи.

        Args:
        ---------------
            request: Запрос на повторную подписку.

        Returns:
        ---------------
            AsyncIterable[SendTaskStreamingResponse]: Поток обновлений задачи.
            JSONRPCResponse: Ответ с ошибкой, если возникла проблема.

        Examples:
        ---------------
            >>> async def example():
            ...     response = await task_manager.on_resubscribe_to_task(request)
            ...     if isinstance(response, JSONRPCResponse):
            ...         print("Resubscription failed")
            ...     else:
            ...         async for update in response:
            ...             print("Received update")
        """
        # Получение параметров идентификатора задачи
        task_id_params: TaskIdParams = request.params
        try:
            # Настройка очереди событий для SSE с флагом повторного подключения
            sse_event_queue = await self.setup_sse_consumer(task_id_params.id, True)
            
            # Возврат очереди событий для потоковой передачи клиенту
            return self.dequeue_events_for_sse(
                request.id, task_id_params.id, sse_event_queue
            )
        except Exception as e:
            # Обработка ошибок
            logger.error(f"Error while reconnecting to SSE stream: {e}")
            return JSONRPCResponse(
                id=request.id,
                error=InternalError(
                    message=f"An error occurred while reconnecting to stream: {e}"
                ),
            )
    
    async def set_push_notification_info(
        self, task_id: str, push_notification_config: PushNotificationConfig
    ) -> bool:
        """
        Description:
        ---------------
            Проверяет валидность URL для push-уведомлений и сохраняет настройки
            для указанной задачи при успешной проверке.

        Args:
        ---------------
            task_id: Идентификатор задачи.
            push_notification_config: Конфигурация push-уведомлений.

        Returns:
        ---------------
            bool: True, если URL успешно проверен и настройки сохранены,
                  False в противном случае.

        Examples:
        ---------------
            >>> async def example():
            ...     config = PushNotificationConfig(url="https://example.com/notify")
            ...     success = await task_manager.set_push_notification_info("task123", config)
            ...     print(f"Configuration saved: {success}")
        """
        # Проверка владения URL для уведомлений путем отправки запроса с вызовом
        is_verified = await self.notification_sender_auth.verify_push_notification_url(
            push_notification_config.url
        )
        
        # Если URL не верифицирован, возвращаем False
        if not is_verified:
            return False
        
        # Сохранение информации о push-уведомлениях для задачи
        await super().set_push_notification_info(task_id, push_notification_config)
        return True