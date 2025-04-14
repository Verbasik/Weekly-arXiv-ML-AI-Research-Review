# A2A/common/server/server.py
"""
Модуль реализует сервер для Agent-to-Agent API.

Обеспечивает взаимодействие агентов через HTTP API, обработку запросов в формате JSON-RPC,
потоковую передачу данных через Server-Sent Events (SSE) и управление задачами.
Поддерживает различные операции с задачами и уведомлениями.
"""

# Стандартные библиотеки
import json
import logging
from typing import Any, AsyncIterable, Dict, Optional, Union

# Сторонние библиотеки
from pydantic import ValidationError
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

# Внутренние модули
from common.types import (
    A2ARequest,
    JSONRPCResponse,
    InvalidRequestError,
    JSONParseError,
    GetTaskRequest,
    CancelTaskRequest,
    SendTaskRequest,
    SetTaskPushNotificationRequest,
    GetTaskPushNotificationRequest,
    InternalError,
    AgentCard,
    TaskResubscriptionRequest,
    SendTaskStreamingRequest,
)
from common.server.task_manager import TaskManager


# Настройка логирования
logger = logging.getLogger(__name__)


class A2AServer:
    """
    Description:
    ---------------
        Сервер для обработки запросов Agent-to-Agent API.
        Поддерживает стандартные и потоковые запросы, а также управление задачами.
        
    Args:
    ---------------
        host: Хост для запуска сервера
        port: Порт для запуска сервера
        endpoint: Основная конечная точка API
        agent_card: Карточка агента с информацией о его возможностях
        task_manager: Менеджер задач для обработки запросов
        
    Raises:
    ---------------
        ValueError: Если agent_card или task_manager не определены при запуске
        
    Examples:
    ---------------
        >>> server = A2AServer(
        ...     host="0.0.0.0",
        ...     port=5000,
        ...     agent_card=agent_card,
        ...     task_manager=task_manager
        ... )
        >>> server.start()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5000,
        endpoint: str = "/",
        agent_card: Optional[AgentCard] = None,
        task_manager: Optional[TaskManager] = None,
    ) -> None:
        """
        Description:
        ---------------
            Инициализирует сервер A2A API.
            
        Args:
        ---------------
            host: Хост для запуска сервера
            port: Порт для запуска сервера
            endpoint: Основная конечная точка API
            agent_card: Карточка агента с информацией о его возможностях
            task_manager: Менеджер задач для обработки запросов
            
        Returns:
        ---------------
            None
        """
        self.host = host
        self.port = port
        self.endpoint = endpoint
        self.task_manager = task_manager
        self.agent_card = agent_card
        
        # Инициализация приложения Starlette
        self.app = Starlette()
        
        # Добавление маршрутов
        self.app.add_route(self.endpoint, self._process_request, methods=["POST"])
        self.app.add_route(
            "/.well-known/agent.json", self._get_agent_card, methods=["GET"]
        )

    def start(self) -> None:
        """
        Description:
        ---------------
            Запускает сервер A2A API.
            
        Returns:
        ---------------
            None
            
        Raises:
        ---------------
            ValueError: Если agent_card или task_manager не определены
            
        Examples:
        ---------------
            >>> server.start()
        """
        # Проверка наличия необходимых компонентов
        if self.agent_card is None:
            raise ValueError("agent_card is not defined")

        if self.task_manager is None:
            raise ValueError("request_handler is not defined")

        # Запуск сервера uvicorn
        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)

    def _get_agent_card(self, request: Request) -> JSONResponse:
        """
        Description:
        ---------------
            Обработчик GET-запроса для получения карточки агента.
            
        Args:
        ---------------
            request: Объект HTTP-запроса
            
        Returns:
        ---------------
            JSONResponse: Ответ с данными карточки агента
            
        Examples:
        ---------------
            >>> await _get_agent_card(request)
        """
        return JSONResponse(self.agent_card.model_dump(exclude_none=True))

    async def _process_request(
        self, 
        request: Request
    ) -> Union[JSONResponse, EventSourceResponse]:
        """
        Description:
        ---------------
            Обрабатывает входящие POST-запросы к API.
            Определяет тип запроса и перенаправляет его соответствующему обработчику.
            
        Args:
        ---------------
            request: Объект HTTP-запроса
            
        Returns:
        ---------------
            Union[JSONResponse, EventSourceResponse]: Ответ на запрос (JSON или поток событий)
            
        Raises:
        ---------------
            ValueError: Если тип запроса не поддерживается
            
        Examples:
        ---------------
            >>> await _process_request(request)
        """
        try:
            # Получаем данные запроса
            body = await request.json()
            json_rpc_request = A2ARequest.validate_python(body)

            # Определяем тип запроса и вызываем соответствующий обработчик
            if isinstance(json_rpc_request, GetTaskRequest):
                result = await self.task_manager.on_get_task(json_rpc_request)
            elif isinstance(json_rpc_request, SendTaskRequest):
                result = await self.task_manager.on_send_task(json_rpc_request)
            elif isinstance(json_rpc_request, SendTaskStreamingRequest):
                result = await self.task_manager.on_send_task_subscribe(
                    json_rpc_request
                )
            elif isinstance(json_rpc_request, CancelTaskRequest):
                result = await self.task_manager.on_cancel_task(json_rpc_request)
            elif isinstance(json_rpc_request, SetTaskPushNotificationRequest):
                result = await self.task_manager.on_set_task_push_notification(json_rpc_request)
            elif isinstance(json_rpc_request, GetTaskPushNotificationRequest):
                result = await self.task_manager.on_get_task_push_notification(json_rpc_request)
            elif isinstance(json_rpc_request, TaskResubscriptionRequest):
                result = await self.task_manager.on_resubscribe_to_task(
                    json_rpc_request
                )
            else:
                # Логирование неожиданного типа запроса
                logger.warning(f"Unexpected request type: {type(json_rpc_request)}")
                raise ValueError(f"Unexpected request type: {type(request)}")

            # Создаем и возвращаем ответ
            return self._create_response(result)

        except Exception as e:
            # Обрабатываем исключения
            return self._handle_exception(e)

    def _handle_exception(self, e: Exception) -> JSONResponse:
        """
        Description:
        ---------------
            Обрабатывает исключения, возникающие при обработке запросов.
            Преобразует исключения в соответствующие JSON-RPC ошибки.
            
        Args:
        ---------------
            e: Объект исключения
            
        Returns:
        ---------------
            JSONResponse: Ответ с информацией об ошибке
            
        Examples:
        ---------------
            >>> _handle_exception(ValueError("Invalid request"))
        """
        # Определяем тип ошибки и создаем соответствующий ответ
        if isinstance(e, json.decoder.JSONDecodeError):
            json_rpc_error = JSONParseError()
        elif isinstance(e, ValidationError):
            json_rpc_error = InvalidRequestError(data=json.loads(e.json()))
        else:
            # Логируем необработанные исключения
            logger.error(f"Unhandled exception: {e}")
            json_rpc_error = InternalError()

        # Создаем и возвращаем ответ с ошибкой
        response = JSONRPCResponse(id=None, error=json_rpc_error)
        return JSONResponse(response.model_dump(exclude_none=True), status_code=400)

    def _create_response(
        self, 
        result: Any
    ) -> Union[JSONResponse, EventSourceResponse]:
        """
        Description:
        ---------------
            Создает HTTP-ответ на основе результата обработки запроса.
            Поддерживает как обычные JSON-ответы, так и потоковые ответы SSE.
            
        Args:
        ---------------
            result: Результат обработки запроса
            
        Returns:
        ---------------
            Union[JSONResponse, EventSourceResponse]: Ответ на запрос
            
        Raises:
        ---------------
            ValueError: Если тип результата не поддерживается
            
        Examples:
        ---------------
            >>> _create_response(json_rpc_response)
            >>> _create_response(event_stream)
        """
        # Если результат - асинхронный итератор, создаем потоковый ответ
        if isinstance(result, AsyncIterable):
            async def event_generator(result: AsyncIterable) -> AsyncIterable[Dict[str, str]]:
                """
                Внутренний генератор событий для SSE.
                
                Args:
                    result: Асинхронный итератор с результатами
                    
                Returns:
                    AsyncIterable[Dict[str, str]]: Генератор событий для SSE
                """
                async for item in result:
                    yield {"data": item.model_dump_json(exclude_none=True)}

            return EventSourceResponse(event_generator(result))
        # Если результат - JSON-RPC ответ, создаем обычный JSON-ответ
        elif isinstance(result, JSONRPCResponse):
            return JSONResponse(result.model_dump(exclude_none=True))
        else:
            # Логируем и поднимаем исключение для неподдерживаемых типов
            logger.error(f"Unexpected result type: {type(result)}")
            raise ValueError(f"Unexpected result type: {type(result)}")