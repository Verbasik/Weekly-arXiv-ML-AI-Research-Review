# A2A/common/client/client.py
"""
Модуль реализует клиент для взаимодействия с Agent-to-Agent API.

Обеспечивает отправку задач, получение результатов, управление уведомлениями и 
другие операции через A2A API. Поддерживает как стандартный, так и потоковый 
режим коммуникации с агентами.
"""

# Стандартные библиотеки
import json
from typing import Any, AsyncIterable, Dict, Optional, Union

# Сторонние библиотеки
import httpx
from httpx_sse import connect_sse

# Внутренние модули
from common.types import (
    AgentCard,
    GetTaskRequest,
    SendTaskRequest,
    SendTaskResponse,
    JSONRPCRequest,
    GetTaskResponse,
    CancelTaskResponse,
    CancelTaskRequest,
    SetTaskPushNotificationRequest,
    SetTaskPushNotificationResponse,
    GetTaskPushNotificationRequest,
    GetTaskPushNotificationResponse,
    A2AClientHTTPError,
    A2AClientJSONError,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
)


class A2AClient:
    """
    Description:
    ---------------
        Клиент для взаимодействия с Agent-to-Agent API.
        Поддерживает различные операции с задачами и уведомлениями.
        
    Args:
    ---------------
        agent_card: Карточка агента, содержащая URL для взаимодействия
        url: Прямой URL для взаимодействия с API (используется, если agent_card не указан)
        
    Raises:
    ---------------
        ValueError: Если не указаны ни agent_card, ни url
        
    Examples:
    ---------------
        >>> agent_card = A2ACardResolver("http://localhost:8001").get_agent_card()
        >>> client = A2AClient(agent_card=agent_card)
        >>> response = await client.send_task({"id": "task123", "message": {"text": "Привет"}})
    """

    def __init__(
        self, 
        agent_card: Optional[AgentCard] = None, 
        url: Optional[str] = None
    ) -> None:
        """
        Description:
        ---------------
            Инициализирует клиент A2A API.
            
        Args:
        ---------------
            agent_card: Карточка агента, содержащая URL для взаимодействия
            url: Прямой URL для взаимодействия с API (используется, если agent_card не указан)
            
        Returns:
        ---------------
            None
            
        Raises:
        ---------------
            ValueError: Если не указаны ни agent_card, ни url
        """
        if agent_card:
            self.url = agent_card.url
        elif url:
            self.url = url
        else:
            raise ValueError("Must provide either agent_card or url")

    async def send_task(self, payload: Dict[str, Any]) -> SendTaskResponse:
        """
        Description:
        ---------------
            Отправляет задачу агенту и получает ответ.
            
        Args:
        ---------------
            payload: Данные задачи для отправки
            
        Returns:
        ---------------
            SendTaskResponse: Ответ агента на отправленную задачу
            
        Raises:
        ---------------
            A2AClientHTTPError: При ошибке HTTP-запроса
            A2AClientJSONError: При ошибке декодирования JSON
            
        Examples:
        ---------------
            >>> response = await client.send_task({
            ...     "id": "task123",
            ...     "sessionId": "session456",
            ...     "message": {"text": "Привет"}
            ... })
            >>> print(response.result.status)
        """
        request = SendTaskRequest(params=payload)
        return SendTaskResponse(**await self._send_request(request))

    async def send_task_streaming(
        self, 
        payload: Dict[str, Any]
    ) -> AsyncIterable[SendTaskStreamingResponse]:
        """
        Description:
        ---------------
            Отправляет задачу агенту и получает ответ в потоковом режиме (Server-Sent Events).
            
        Args:
        ---------------
            payload: Данные задачи для отправки
            
        Returns:
        ---------------
            AsyncIterable[SendTaskStreamingResponse]: Поток ответов от агента
            
        Raises:
        ---------------
            A2AClientJSONError: При ошибке декодирования JSON
            A2AClientHTTPError: При ошибке HTTP-запроса
            
        Examples:
        ---------------
            >>> async for response in client.send_task_streaming({
            ...     "id": "task123",
            ...     "sessionId": "session456",
            ...     "message": {"text": "Привет"}
            ... }):
            ...     print(response.result.status)
        """
        # Создаем запрос на потоковую передачу
        request = SendTaskStreamingRequest(params=payload)
        
        # Используем неасинхронный клиент для работы с SSE
        with httpx.Client(timeout=None) as client:
            # Подключаемся к потоку событий
            with connect_sse(
                client, "POST", self.url, json=request.model_dump()
            ) as event_source:
                try:
                    # Итерируемся по событиям SSE
                    for sse in event_source.iter_sse():
                        yield SendTaskStreamingResponse(**json.loads(sse.data))
                except json.JSONDecodeError as e:
                    # Обрабатываем ошибки JSON
                    raise A2AClientJSONError(str(e)) from e
                except httpx.RequestError as e:
                    # Обрабатываем ошибки HTTP
                    raise A2AClientHTTPError(400, str(e)) from e

    async def _send_request(self, request: JSONRPCRequest) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Вспомогательный метод для отправки запросов к API.
            
        Args:
        ---------------
            request: Объект JSON-RPC запроса
            
        Returns:
        ---------------
            Dict[str, Any]: Ответ от API в виде словаря
            
        Raises:
        ---------------
            A2AClientHTTPError: При ошибке HTTP-запроса
            A2AClientJSONError: При ошибке декодирования JSON
        """
        async with httpx.AsyncClient() as client:
            try:
                # Генерация изображений может занять время, добавляем таймаут
                response = await client.post(
                    self.url, json=request.model_dump(), timeout=30
                )
                # Проверяем статус ответа
                response.raise_for_status()
                # Парсим JSON из ответа
                return response.json()
            except httpx.HTTPStatusError as e:
                # Обрабатываем ошибки HTTP
                raise A2AClientHTTPError(e.response.status_code, str(e)) from e
            except json.JSONDecodeError as e:
                # Обрабатываем ошибки JSON
                raise A2AClientJSONError(str(e)) from e

    async def get_task(self, payload: Dict[str, Any]) -> GetTaskResponse:
        """
        Description:
        ---------------
            Получает информацию о задаче по её идентификатору.
            
        Args:
        ---------------
            payload: Данные для запроса информации о задаче
            
        Returns:
        ---------------
            GetTaskResponse: Ответ с информацией о задаче
            
        Raises:
        ---------------
            A2AClientHTTPError: При ошибке HTTP-запроса
            A2AClientJSONError: При ошибке декодирования JSON
            
        Examples:
        ---------------
            >>> response = await client.get_task({"id": "task123"})
            >>> print(response.result.status)
        """
        request = GetTaskRequest(params=payload)
        return GetTaskResponse(**await self._send_request(request))

    async def cancel_task(self, payload: Dict[str, Any]) -> CancelTaskResponse:
        """
        Description:
        ---------------
            Отменяет выполнение задачи.
            
        Args:
        ---------------
            payload: Данные для запроса отмены задачи
            
        Returns:
        ---------------
            CancelTaskResponse: Ответ на запрос отмены задачи
            
        Raises:
        ---------------
            A2AClientHTTPError: При ошибке HTTP-запроса
            A2AClientJSONError: При ошибке декодирования JSON
            
        Examples:
        ---------------
            >>> response = await client.cancel_task({"id": "task123"})
            >>> print(response.result.status)
        """
        request = CancelTaskRequest(params=payload)
        return CancelTaskResponse(**await self._send_request(request))

    async def set_task_callback(
        self, 
        payload: Dict[str, Any]
    ) -> SetTaskPushNotificationResponse:
        """
        Description:
        ---------------
            Устанавливает callback для уведомлений о статусе задачи.
            
        Args:
        ---------------
            payload: Данные для установки callback
            
        Returns:
        ---------------
            SetTaskPushNotificationResponse: Ответ на запрос установки callback
            
        Raises:
        ---------------
            A2AClientHTTPError: При ошибке HTTP-запроса
            A2AClientJSONError: При ошибке декодирования JSON
            
        Examples:
        ---------------
            >>> response = await client.set_task_callback({
            ...     "id": "task123",
            ...     "pushNotification": {"url": "http://example.com/notify"}
            ... })
        """
        request = SetTaskPushNotificationRequest(params=payload)
        return SetTaskPushNotificationResponse(**await self._send_request(request))

    async def get_task_callback(
        self, 
        payload: Dict[str, Any]
    ) -> GetTaskPushNotificationResponse:
        """
        Description:
        ---------------
            Получает информацию о callback для уведомлений о статусе задачи.
            
        Args:
        ---------------
            payload: Данные для запроса информации о callback
            
        Returns:
        ---------------
            GetTaskPushNotificationResponse: Ответ с информацией о callback
            
        Raises:
        ---------------
            A2AClientHTTPError: При ошибке HTTP-запроса
            A2AClientJSONError: При ошибке декодирования JSON
            
        Examples:
        ---------------
            >>> response = await client.get_task_callback({"id": "task123"})
            >>> if response.result.pushNotification:
            ...     print(response.result.pushNotification.url)
        """
        request = GetTaskPushNotificationRequest(params=payload)
        return GetTaskPushNotificationResponse(**await self._send_request(request))