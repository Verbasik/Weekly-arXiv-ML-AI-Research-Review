# A2A/common/types.py
"""
Модуль определяет типы данных для Agent-to-Agent (A2A) API.

Содержит классы для описания задач, сообщений, артефактов, статусов, запросов и ответов
в формате JSON-RPC, а также типы для управления уведомлениями и карточками агентов.
Все классы основаны на Pydantic для валидации и сериализации данных.
"""

# Стандартные библиотеки
from datetime import datetime
from enum import Enum
import uuid
from typing import (
    Annotated, 
    Any, 
    Dict, 
    List, 
    Literal, 
    Optional, 
    Union
)

# Сторонние библиотеки
from pydantic import (
    BaseModel,
    ConfigDict,
    Field, 
    TypeAdapter, 
    field_serializer, 
    model_validator
)
from typing_extensions import Self


class TaskState(str, Enum):
    """
    Description:
    ---------------
        Перечисление возможных состояний задачи в рамках A2A API.
        
    Examples:
    ---------------
        >>> TaskState.WORKING
        <TaskState.WORKING: 'working'>
        >>> TaskState.COMPLETED
        <TaskState.COMPLETED: 'completed'>
    """
    SUBMITTED = "submitted"  # Задача отправлена
    WORKING = "working"      # Задача выполняется
    INPUT_REQUIRED = "input-required"  # Требуется ввод от пользователя
    COMPLETED = "completed"  # Задача успешно завершена
    CANCELED = "canceled"    # Задача отменена
    FAILED = "failed"        # Задача завершилась с ошибкой
    UNKNOWN = "unknown"      # Неизвестное состояние


# Классы частей сообщений

class TextPart(BaseModel):
    """
    Description:
    ---------------
        Текстовая часть сообщения.
        
    Args:
    ---------------
        text: Текстовое содержимое
        metadata: Метаданные текстовой части (опционально)
        
    Examples:
    ---------------
        >>> text_part = TextPart(text="Привет, мир!")
    """
    type: Literal["text"] = "text"
    text: str
    metadata: Optional[Dict[str, Any]] = None


class FileContent(BaseModel):
    """
    Description:
    ---------------
        Содержимое файла для передачи в части сообщения.
        
    Args:
    ---------------
        name: Имя файла (опционально)
        mimeType: MIME-тип файла (опционально)
        bytes: Содержимое файла в кодировке Base64 (опционально)
        uri: URI для доступа к файлу (опционально)
        
    Raises:
    ---------------
        ValueError: Если не указаны или указаны одновременно bytes и uri
        
    Examples:
    ---------------
        >>> file_content = FileContent(
        ...     name="image.png", 
        ...     mimeType="image/png", 
        ...     bytes="base64_encoded_data"
        ... )
    """
    name: Optional[str] = None
    mimeType: Optional[str] = None
    bytes: Optional[str] = None
    uri: Optional[str] = None

    @model_validator(mode="after")
    def check_content(self) -> Self:
        """
        Description:
        ---------------
            Проверяет, что указан только один из параметров bytes или uri.
            
        Returns:
        ---------------
            Self: Экземпляр этого же объекта
            
        Raises:
        ---------------
            ValueError: Если не указаны или указаны одновременно bytes и uri
        """
        if not (self.bytes or self.uri):
            raise ValueError("Either 'bytes' or 'uri' must be present in the file data")
        if self.bytes and self.uri:
            raise ValueError(
                "Only one of 'bytes' or 'uri' can be present in the file data"
            )
        return self


class FilePart(BaseModel):
    """
    Description:
    ---------------
        Файловая часть сообщения.
        
    Args:
    ---------------
        file: Содержимое файла
        metadata: Метаданные файловой части (опционально)
        
    Examples:
    ---------------
        >>> file_part = FilePart(
        ...     file=FileContent(
        ...         name="image.png", 
        ...         mimeType="image/png", 
        ...         bytes="base64_encoded_data"
        ...     )
        ... )
    """
    type: Literal["file"] = "file"
    file: FileContent
    metadata: Optional[Dict[str, Any]] = None


class DataPart(BaseModel):
    """
    Description:
    ---------------
        Часть сообщения с произвольными структурированными данными.
        
    Args:
    ---------------
        data: Словарь с данными
        metadata: Метаданные части с данными (опционально)
        
    Examples:
    ---------------
        >>> data_part = DataPart(data={"key": "value"})
    """
    type: Literal["data"] = "data"
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


# Объединенный тип для частей сообщения
Part = Annotated[Union[TextPart, FilePart, DataPart], Field(discriminator="type")]


class Message(BaseModel):
    """
    Description:
    ---------------
        Сообщение, обмениваемое между пользователем и агентом.
        
    Args:
    ---------------
        role: Роль отправителя сообщения ("user" или "agent")
        parts: Список частей сообщения
        metadata: Метаданные сообщения (опционально)
        
    Examples:
    ---------------
        >>> message = Message(
        ...     role="user",
        ...     parts=[TextPart(text="Привет, как дела?")]
        ... )
    """
    role: Literal["user", "agent"]
    parts: List[Part]
    metadata: Optional[Dict[str, Any]] = None


class TaskStatus(BaseModel):
    """
    Description:
    ---------------
        Статус задачи в A2A API.
        
    Args:
    ---------------
        state: Состояние задачи
        message: Сообщение, связанное со статусом (опционально)
        timestamp: Временная метка изменения статуса
        
    Examples:
    ---------------
        >>> status = TaskStatus(
        ...     state=TaskState.WORKING,
        ...     message=Message(
        ...         role="agent",
        ...         parts=[TextPart(text="Обработка запроса...")]
        ...     )
        ... )
    """
    state: TaskState
    message: Optional[Message] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_serializer("timestamp")
    def serialize_dt(self, dt: datetime, _info) -> str:
        """
        Description:
        ---------------
            Сериализует временную метку в формат ISO 8601.
            
        Args:
        ---------------
            dt: Объект даты и времени
            _info: Информация о сериализации (не используется)
            
        Returns:
        ---------------
            str: Дата и время в формате ISO 8601
        """
        return dt.isoformat()


class Artifact(BaseModel):
    """
    Description:
    ---------------
        Артефакт, создаваемый агентом в процессе выполнения задачи.
        
    Args:
    ---------------
        name: Имя артефакта (опционально)
        description: Описание артефакта (опционально)
        parts: Список частей, составляющих артефакт
        metadata: Метаданные артефакта (опционально)
        index: Индекс артефакта
        append: Флаг добавления к существующему артефакту (опционально)
        lastChunk: Флаг последнего фрагмента артефакта (опционально)
        
    Examples:
    ---------------
        >>> artifact = Artifact(
        ...     name="result.txt",
        ...     description="Результат обработки данных",
        ...     parts=[TextPart(text="Результат: 42")]
        ... )
    """
    name: Optional[str] = None
    description: Optional[str] = None
    parts: List[Part]
    metadata: Optional[Dict[str, Any]] = None
    index: int = 0
    append: Optional[bool] = None
    lastChunk: Optional[bool] = None


class Task(BaseModel):
    """
    Description:
    ---------------
        Задача в A2A API.
        
    Args:
    ---------------
        id: Уникальный идентификатор задачи
        sessionId: Идентификатор сессии (опционально)
        status: Статус задачи
        artifacts: Список артефактов, созданных в ходе выполнения задачи (опционально)
        history: История сообщений в рамках задачи (опционально)
        metadata: Метаданные задачи (опционально)
        
    Examples:
    ---------------
        >>> task = Task(
        ...     id="task_123",
        ...     sessionId="session_456",
        ...     status=TaskStatus(state=TaskState.WORKING)
        ... )
    """
    id: str
    sessionId: Optional[str] = None
    status: TaskStatus
    artifacts: Optional[List[Artifact]] = None
    history: Optional[List[Message]] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskStatusUpdateEvent(BaseModel):
    """
    Description:
    ---------------
        Событие обновления статуса задачи.
        
    Args:
    ---------------
        id: Идентификатор задачи
        status: Новый статус задачи
        final: Флаг финального обновления
        metadata: Метаданные события (опционально)
        
    Examples:
    ---------------
        >>> event = TaskStatusUpdateEvent(
        ...     id="task_123",
        ...     status=TaskStatus(state=TaskState.COMPLETED),
        ...     final=True
        ... )
    """
    id: str
    status: TaskStatus
    final: bool = False
    metadata: Optional[Dict[str, Any]] = None


class TaskArtifactUpdateEvent(BaseModel):
    """
    Description:
    ---------------
        Событие обновления артефакта задачи.
        
    Args:
    ---------------
        id: Идентификатор задачи
        artifact: Обновленный артефакт
        metadata: Метаданные события (опционально)
        
    Examples:
    ---------------
        >>> event = TaskArtifactUpdateEvent(
        ...     id="task_123",
        ...     artifact=Artifact(
        ...         name="result.txt",
        ...         parts=[TextPart(text="Обновленный результат")]
        ...     )
        ... )
    """
    id: str
    artifact: Artifact    
    metadata: Optional[Dict[str, Any]] = None


class AuthenticationInfo(BaseModel):
    """
    Description:
    ---------------
        Информация об аутентификации для API.
        
    Args:
    ---------------
        schemes: Список поддерживаемых схем аутентификации
        credentials: Учетные данные (опционально)
        
    Examples:
    ---------------
        >>> auth_info = AuthenticationInfo(schemes=["bearer"])
    """
    model_config = ConfigDict(extra="allow")

    schemes: List[str]
    credentials: Optional[str] = None


class PushNotificationConfig(BaseModel):
    """
    Description:
    ---------------
        Конфигурация push-уведомлений.
        
    Args:
    ---------------
        url: URL для отправки уведомлений
        token: Токен для аутентификации (опционально)
        authentication: Информация об аутентификации (опционально)
        
    Examples:
    ---------------
        >>> config = PushNotificationConfig(
        ...     url="https://example.com/webhook",
        ...     authentication=AuthenticationInfo(schemes=["bearer"])
        ... )
    """
    url: str
    token: Optional[str] = None
    authentication: Optional[AuthenticationInfo] = None


class TaskIdParams(BaseModel):
    """
    Description:
    ---------------
        Параметры запроса с идентификатором задачи.
        
    Args:
    ---------------
        id: Идентификатор задачи
        metadata: Метаданные запроса (опционально)
        
    Examples:
    ---------------
        >>> params = TaskIdParams(id="task_123")
    """
    id: str
    metadata: Optional[Dict[str, Any]] = None


class TaskQueryParams(TaskIdParams):
    """
    Description:
    ---------------
        Параметры запроса задачи с возможностью указания длины истории.
        
    Args:
    ---------------
        id: Идентификатор задачи
        historyLength: Количество сообщений истории для включения (опционально)
        metadata: Метаданные запроса (опционально)
        
    Examples:
    ---------------
        >>> params = TaskQueryParams(id="task_123", historyLength=5)
    """
    historyLength: Optional[int] = None


class TaskSendParams(BaseModel):
    """
    Description:
    ---------------
        Параметры для отправки задачи.
        
    Args:
    ---------------
        id: Идентификатор задачи
        sessionId: Идентификатор сессии
        message: Сообщение для обработки
        acceptedOutputModes: Список принимаемых форматов вывода (опционально)
        pushNotification: Конфигурация push-уведомлений (опционально)
        historyLength: Количество сообщений истории для включения (опционально)
        metadata: Метаданные запроса (опционально)
        
    Examples:
    ---------------
        >>> params = TaskSendParams(
        ...     id="task_123",
        ...     sessionId="session_456",
        ...     message=Message(
        ...         role="user",
        ...         parts=[TextPart(text="Привет, мир!")]
        ...     ),
        ...     acceptedOutputModes=["text", "image/png"]
        ... )
    """
    id: str
    sessionId: str = Field(default_factory=lambda: uuid.uuid4().hex)
    message: Message
    acceptedOutputModes: Optional[List[str]] = None
    pushNotification: Optional[PushNotificationConfig] = None
    historyLength: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskPushNotificationConfig(BaseModel):
    """
    Description:
    ---------------
        Конфигурация push-уведомлений для задачи.
        
    Args:
    ---------------
        id: Идентификатор задачи
        pushNotificationConfig: Конфигурация push-уведомлений
        
    Examples:
    ---------------
        >>> config = TaskPushNotificationConfig(
        ...     id="task_123",
        ...     pushNotificationConfig=PushNotificationConfig(
        ...         url="https://example.com/webhook"
        ...     )
        ... )
    """
    id: str
    pushNotificationConfig: PushNotificationConfig


# JSON-RPC сообщения

class JSONRPCMessage(BaseModel):
    """
    Description:
    ---------------
        Базовое сообщение JSON-RPC 2.0.
        
    Args:
    ---------------
        jsonrpc: Версия протокола (всегда "2.0")
        id: Идентификатор сообщения
        
    Examples:
    ---------------
        >>> message = JSONRPCMessage()
    """
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[int, str]] = Field(default_factory=lambda: uuid.uuid4().hex)


class JSONRPCRequest(JSONRPCMessage):
    """
    Description:
    ---------------
        Запрос JSON-RPC.
        
    Args:
    ---------------
        method: Имя метода для вызова
        params: Параметры метода (опционально)
        
    Examples:
    ---------------
        >>> request = JSONRPCRequest(method="tasks/get", params={"id": "task_123"})
    """
    method: str
    params: Optional[Dict[str, Any]] = None


class JSONRPCError(BaseModel):
    """
    Description:
    ---------------
        Ошибка JSON-RPC.
        
    Args:
    ---------------
        code: Код ошибки
        message: Сообщение об ошибке
        data: Дополнительные данные об ошибке (опционально)
        
    Examples:
    ---------------
        >>> error = JSONRPCError(code=-32600, message="Invalid Request")
    """
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCResponse(JSONRPCMessage):
    """
    Description:
    ---------------
        Ответ JSON-RPC.
        
    Args:
    ---------------
        result: Результат вызова метода (опционально)
        error: Информация об ошибке (опционально)
        
    Examples:
    ---------------
        >>> response = JSONRPCResponse(result={"status": "ok"})
        >>> error_response = JSONRPCResponse(
        ...     error=JSONRPCError(code=-32600, message="Invalid Request")
        ... )
    """
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None


class SendTaskRequest(JSONRPCRequest):
    """
    Description:
    ---------------
        Запрос на отправку задачи.
        
    Args:
    ---------------
        params: Параметры для отправки задачи
        
    Examples:
    ---------------
        >>> request = SendTaskRequest(params=TaskSendParams(...))
    """
    method: Literal["tasks/send"] = "tasks/send"
    params: TaskSendParams


class SendTaskResponse(JSONRPCResponse):
    """
    Description:
    ---------------
        Ответ на запрос отправки задачи.
        
    Args:
    ---------------
        result: Информация о созданной задаче (опционально)
        
    Examples:
    ---------------
        >>> response = SendTaskResponse(result=Task(...))
    """
    result: Optional[Task] = None


class SendTaskStreamingRequest(JSONRPCRequest):
    """
    Description:
    ---------------
        Запрос на отправку задачи с потоковым получением результатов.
        
    Args:
    ---------------
        params: Параметры для отправки задачи
        
    Examples:
    ---------------
        >>> request = SendTaskStreamingRequest(params=TaskSendParams(...))
    """
    method: Literal["tasks/sendSubscribe"] = "tasks/sendSubscribe"
    params: TaskSendParams


class SendTaskStreamingResponse(JSONRPCResponse):
    """
    Description:
    ---------------
        Ответ на запрос отправки задачи с потоковой передачей.
        
    Args:
    ---------------
        result: Событие обновления статуса или артефакта (опционально)
        
    Examples:
    ---------------
        >>> response = SendTaskStreamingResponse(
        ...     result=TaskStatusUpdateEvent(...)
        ... )
    """
    result: Optional[Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]] = None


class GetTaskRequest(JSONRPCRequest):
    """
    Description:
    ---------------
        Запрос на получение информации о задаче.
        
    Args:
    ---------------
        params: Параметры запроса задачи
        
    Examples:
    ---------------
        >>> request = GetTaskRequest(params=TaskQueryParams(id="task_123"))
    """
    method: Literal["tasks/get"] = "tasks/get"
    params: TaskQueryParams


class GetTaskResponse(JSONRPCResponse):
    """
    Description:
    ---------------
        Ответ на запрос получения информации о задаче.
        
    Args:
    ---------------
        result: Информация о задаче (опционально)
        
    Examples:
    ---------------
        >>> response = GetTaskResponse(result=Task(...))
    """
    result: Optional[Task] = None


class CancelTaskRequest(JSONRPCRequest):
    """
    Description:
    ---------------
        Запрос на отмену задачи.
        
    Args:
    ---------------
        params: Параметры запроса с идентификатором задачи
        
    Examples:
    ---------------
        >>> request = CancelTaskRequest(params=TaskIdParams(id="task_123"))
    """
    method: Literal["tasks/cancel"] = "tasks/cancel"
    params: TaskIdParams


class CancelTaskResponse(JSONRPCResponse):
    """
    Description:
    ---------------
        Ответ на запрос отмены задачи.
        
    Args:
    ---------------
        result: Информация о задаче после отмены (опционально)
        
    Examples:
    ---------------
        >>> response = CancelTaskResponse(result=Task(...))
    """
    result: Optional[Task] = None


class SetTaskPushNotificationRequest(JSONRPCRequest):
    """
    Description:
    ---------------
        Запрос на установку push-уведомлений для задачи.
        
    Args:
    ---------------
        params: Конфигурация push-уведомлений для задачи
        
    Examples:
    ---------------
        >>> request = SetTaskPushNotificationRequest(
        ...     params=TaskPushNotificationConfig(...)
        ... )
    """
    method: Literal["tasks/pushNotification/set"] = "tasks/pushNotification/set"
    params: TaskPushNotificationConfig


class SetTaskPushNotificationResponse(JSONRPCResponse):
    """
    Description:
    ---------------
        Ответ на запрос установки push-уведомлений.
        
    Args:
    ---------------
        result: Конфигурация push-уведомлений для задачи (опционально)
        
    Examples:
    ---------------
        >>> response = SetTaskPushNotificationResponse(
        ...     result=TaskPushNotificationConfig(...)
        ... )
    """
    result: Optional[TaskPushNotificationConfig] = None


class GetTaskPushNotificationRequest(JSONRPCRequest):
    """
    Description:
    ---------------
        Запрос на получение конфигурации push-уведомлений для задачи.
        
    Args:
    ---------------
        params: Параметры запроса с идентификатором задачи
        
    Examples:
    ---------------
        >>> request = GetTaskPushNotificationRequest(
        ...     params=TaskIdParams(id="task_123")
        ... )
    """
    method: Literal["tasks/pushNotification/get"] = "tasks/pushNotification/get"
    params: TaskIdParams


class GetTaskPushNotificationResponse(JSONRPCResponse):
    """
    Description:
    ---------------
        Ответ на запрос получения конфигурации push-уведомлений.
        
    Args:
    ---------------
        result: Конфигурация push-уведомлений для задачи (опционально)
        
    Examples:
    ---------------
        >>> response = GetTaskPushNotificationResponse(
        ...     result=TaskPushNotificationConfig(...)
        ... )
    """
    result: Optional[TaskPushNotificationConfig] = None


class TaskResubscriptionRequest(JSONRPCRequest):
    """
    Description:
    ---------------
        Запрос на повторную подписку на обновления задачи.
        
    Args:
    ---------------
        params: Параметры запроса с идентификатором задачи
        
    Examples:
    ---------------
        >>> request = TaskResubscriptionRequest(
        ...     params=TaskIdParams(id="task_123")
        ... )
    """
    method: Literal["tasks/resubscribe"] = "tasks/resubscribe"
    params: TaskIdParams


# Адаптер типов для валидации запросов A2A
A2ARequest = TypeAdapter(
    Annotated[
        Union[
            SendTaskRequest,
            GetTaskRequest,
            CancelTaskRequest,
            SetTaskPushNotificationRequest,
            GetTaskPushNotificationRequest,
            TaskResubscriptionRequest,
            SendTaskStreamingRequest,
        ],
        Field(discriminator="method"),
    ]
)


# Типы ошибок

class JSONParseError(JSONRPCError):
    """
    Description:
    ---------------
        Ошибка разбора JSON.
        
    Examples:
    ---------------
        >>> error = JSONParseError()
    """
    code: int = -32700
    message: str = "Invalid JSON payload"
    data: Optional[Any] = None


class InvalidRequestError(JSONRPCError):
    """
    Description:
    ---------------
        Ошибка валидации запроса.
        
    Examples:
    ---------------
        >>> error = InvalidRequestError(data={"field": "error message"})
    """
    code: int = -32600
    message: str = "Request payload validation error"
    data: Optional[Any] = None


class MethodNotFoundError(JSONRPCError):
    """
    Description:
    ---------------
        Ошибка отсутствия метода.
        
    Examples:
    ---------------
        >>> error = MethodNotFoundError()
    """
    code: int = -32601
    message: str = "Method not found"
    data: None = None


class InvalidParamsError(JSONRPCError):
    """
    Description:
    ---------------
        Ошибка некорректных параметров.
        
    Examples:
    ---------------
        >>> error = InvalidParamsError()
    """
    code: int = -32602
    message: str = "Invalid parameters"
    data: Optional[Any] = None


class InternalError(JSONRPCError):
    """
    Description:
    ---------------
        Внутренняя ошибка сервера.
        
    Examples:
    ---------------
        >>> error = InternalError()
    """
    code: int = -32603
    message: str = "Internal error"
    data: Optional[Any] = None


class TaskNotFoundError(JSONRPCError):
    """
    Description:
    ---------------
        Ошибка отсутствия задачи.
        
    Examples:
    ---------------
        >>> error = TaskNotFoundError()
    """
    code: int = -32001
    message: str = "Task not found"
    data: None = None


class TaskNotCancelableError(JSONRPCError):
    """
    Description:
    ---------------
        Ошибка невозможности отмены задачи.
        
    Examples:
    ---------------
        >>> error = TaskNotCancelableError()
    """
    code: int = -32002
    message: str = "Task cannot be canceled"
    data: None = None


class PushNotificationNotSupportedError(JSONRPCError):
    """
    Description:
    ---------------
        Ошибка отсутствия поддержки push-уведомлений.
        
    Examples:
    ---------------
        >>> error = PushNotificationNotSupportedError()
    """
    code: int = -32003
    message: str = "Push Notification is not supported"
    data: None = None


class UnsupportedOperationError(JSONRPCError):
    """
    Description:
    ---------------
        Ошибка неподдерживаемой операции.
        
    Examples:
    ---------------
        >>> error = UnsupportedOperationError()
    """
    code: int = -32004
    message: str = "This operation is not supported"
    data: None = None


class ContentTypeNotSupportedError(JSONRPCError):
    """
    Description:
    ---------------
        Ошибка неподдерживаемого типа контента.
        
    Examples:
    ---------------
        >>> error = ContentTypeNotSupportedError()
    """
    code: int = -32005
    message: str = "Incompatible content types"
    data: None = None


# Классы карточки агента

class AgentProvider(BaseModel):
    """
    Description:
    ---------------
        Информация о провайдере агента.
        
    Args:
    ---------------
        organization: Название организации
        url: URL организации (опционально)
        
    Examples:
    ---------------
        >>> provider = AgentProvider(
        ...     organization="Example Corp",
        ...     url="https://example.com"
        ... )
    """
    organization: str
    url: Optional[str] = None


class AgentCapabilities(BaseModel):
    """
    Description:
    ---------------
        Возможности агента.
        
    Args:
    ---------------
        streaming: Поддержка потоковой передачи данных
        pushNotifications: Поддержка push-уведомлений
        stateTransitionHistory: Поддержка истории переходов состояний
        
    Examples:
    ---------------
        >>> capabilities = AgentCapabilities(
        ...     streaming=True,
        ...     pushNotifications=True
        ... )
    """
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


class AgentAuthentication(BaseModel):
    """
    Description:
    ---------------
        Информация об аутентификации агента.
        
    Args:
    ---------------
        schemes: Список поддерживаемых схем аутентификации
        credentials: Учетные данные (опционально)
        
    Examples:
    ---------------
        >>> auth = AgentAuthentication(schemes=["bearer"])
    """
    schemes: List[str]
    credentials: Optional[str] = None


class AgentSkill(BaseModel):
    """
    Description:
    ---------------
        Описание навыка агента.
        
    Args:
    ---------------
        id: Идентификатор навыка
        name: Название навыка
        description: Описание навыка (опционально)
        tags: Список тегов для навыка (опционально)
        examples: Примеры использования навыка (опционально)
        inputModes: Поддерживаемые форматы ввода (опционально)
        outputModes: Поддерживаемые форматы вывода (опционально)
        
    Examples:
    ---------------
        >>> skill = AgentSkill(
        ...     id="calculator",
        ...     name="Калькулятор",
        ...     description="Выполняет математические расчеты",
        ...     tags=["math", "calculation"]
        ... )
    """
    id: str
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    inputModes: Optional[List[str]] = None
    outputModes: Optional[List[str]] = None


class AgentCard(BaseModel):
    """
    Description:
    ---------------
        Карточка агента, описывающая его возможности и метаданные.
        
    Args:
    ---------------
        name: Имя агента
        description: Описание агента (опционально)
        url: URL для доступа к API агента
        provider: Информация о провайдере агента (опционально)
        version: Версия агента
        documentationUrl: URL с документацией по агенту (опционально)
        capabilities: Возможности агента
        authentication: Информация об аутентификации (опционально)
        defaultInputModes: Список поддерживаемых форматов ввода по умолчанию
        defaultOutputModes: Список поддерживаемых форматов вывода по умолчанию
        skills: Список навыков агента
        
    Examples:
    ---------------
        >>> card = AgentCard(
        ...     name="CalculatorAgent",
        ...     url="https://calculator.example.com/api",
        ...     version="1.0.0",
        ...     capabilities=AgentCapabilities(streaming=True),
        ...     skills=[AgentSkill(id="calculator", name="Калькулятор")]
        ... )
    """
    name: str
    description: Optional[str] = None
    url: str
    provider: Optional[AgentProvider] = None
    version: str
    documentationUrl: Optional[str] = None
    capabilities: AgentCapabilities
    authentication: Optional[AgentAuthentication] = None
    defaultInputModes: List[str] = ["text"]
    defaultOutputModes: List[str] = ["text"]
    skills: List[AgentSkill]


# Классы исключений

class A2AClientError(Exception):
    """
    Description:
    ---------------
        Базовое исключение для ошибок клиента A2A API.
        
    Examples:
    ---------------
        >>> try:
        ...     # Код, который может вызвать ошибку
        ...     pass
        ... except A2AClientError as e:
        ...     print(f"Произошла ошибка: {e}")
    """
    pass


class A2AClientHTTPError(A2AClientError):
    """
    Description:
    ---------------
        Исключение для ошибок HTTP.
        
    Args:
    ---------------
        status_code: HTTP-код статуса
        message: Сообщение об ошибке
        
    Examples:
    ---------------
        >>> raise A2AClientHTTPError(404, "Resource not found")
    """
    def __init__(self, status_code: int, message: str) -> None:
        """
        Description:
        ---------------
            Инициализирует исключение HTTP-ошибки.
            
        Args:
        ---------------
            status_code: HTTP-код статуса
            message: Сообщение об ошибке
            
        Returns:
        ---------------
            None
        """
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP Error {status_code}: {message}")


class A2AClientJSONError(A2AClientError):
    """
    Description:
    ---------------
        Исключение для ошибок разбора JSON.
        
    Args:
    ---------------
        message: Сообщение об ошибке
        
    Examples:
    ---------------
        >>> raise A2AClientJSONError("Invalid JSON format")
    """
    def __init__(self, message: str) -> None:
        """
        Description:
        ---------------
            Инициализирует исключение ошибки JSON.
            
        Args:
        ---------------
            message: Сообщение об ошибке
            
        Returns:
        ---------------
            None
        """
        self.message = message
        super().__init__(f"JSON Error: {message}")


class MissingAPIKeyError(Exception):
    """
    Description:
    ---------------
        Исключение для отсутствующего API-ключа.
        
    Examples:
    ---------------
        >>> raise MissingAPIKeyError()
    """
    pass