# A2A/agents/langgraph/agent.py
"""Модуль агента конвертации валют.

Предоставляет агента на базе Langgraph для обработки запросов о курсах обмена
и конвертации валют с использованием внешнего API Frankfurter.
"""

# Стандартные библиотеки Python
import logging
from typing import Any, AsyncIterable, Dict, Literal, Optional

# Сторонние библиотеки
import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

# Настройка логирования
logger = logging.getLogger(__name__)

# Создание хранилища памяти для сохранения состояний разговора
memory = MemorySaver()


@tool
def get_exchange_rate(
    currency_from: str = "USD",
    currency_to: str = "EUR",
    currency_date: str = "latest",
) -> Dict[str, Any]:
    """
    Description:
    ---------------
        Получает текущий или исторический курс обмена между двумя валютами
        с использованием API Frankfurter.

    Args:
    ---------------
        currency_from: Валюта, из которой конвертировать (например, "USD").
        currency_to: Валюта, в которую конвертировать (например, "EUR").
        currency_date: Дата для получения курса или "latest" для текущего курса.

    Returns:
    ---------------
        Dict[str, Any]: Словарь с данными о курсе обмена или сообщением об ошибке.

    Raises:
    ---------------
        Не выбрасывает исключения, все ошибки обрабатываются внутри и возвращаются
        в формате словаря с ключом "error".

    Examples:
    ---------------
        >>> result = get_exchange_rate("USD", "EUR")
        >>> "rates" in result
        True
        >>> result = get_exchange_rate("USD", "EUR", "2023-01-01")
        >>> "date" in result
        True
    """    
    try:
        # Выполнение HTTP-запроса к API Frankfurter
        response = httpx.get(
            f"https://api.frankfurter.app/{currency_date}",
            params={"from": currency_from, "to": currency_to},
        )
        response.raise_for_status()

        # Обработка JSON-ответа
        data = response.json()
        if "rates" not in data:
            return {"error": "Invalid API response format."}
        return data
    except httpx.HTTPError as e:
        # Обработка ошибок HTTP-запроса
        logger.error(f"API request failed: {e}")
        return {"error": f"API request failed: {e}"}
    except ValueError:
        # Обработка ошибок парсинга JSON
        logger.error("Invalid JSON response from API")
        return {"error": "Invalid JSON response from API."}


class ResponseFormat(BaseModel):
    """
    Description:
    ---------------
        Модель формата ответа агента пользователю.
        Определяет структуру и формат ответа агента.

    Attributes:
    ---------------
        status: Статус обработки запроса ("input_required", "completed", "error").
        message: Текстовое сообщение для пользователя.

    Examples:
    ---------------
        >>> response = ResponseFormat(status="completed", message="Exchange rate is 1.2 USD to EUR")
        >>> response.status
        'completed'
    """
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class CurrencyAgent:
    """
    Description:
    ---------------
        Агент конвертации валют, построенный на базе Langgraph.
        Обрабатывает запросы о курсах обмена и конвертации валют.

    Attributes:
    ---------------
        SUPPORTED_CONTENT_TYPES: Поддерживаемые типы контента для ввода/вывода.
        SYSTEM_INSTRUCTION: Системная инструкция для модели LLM.
        model: Модель Gemini, используемая для обработки запросов.
        tools: Инструменты, доступные агенту для выполнения задач.
        graph: Граф Langgraph для выполнения цепочки рассуждений агента.

    Examples:
    ---------------
        >>> agent = CurrencyAgent()
        >>> response = agent.invoke("What is the exchange rate from USD to EUR?", "session123")
        >>> response["is_task_complete"]
        True
    """

    # Системная инструкция для модели
    SYSTEM_INSTRUCTION = (
        "Вы являетесь специализированным помощником для конвертации валют. "
        "Ваша единственная задача — использовать инструмент 'get_exchange_rate' для ответов на вопросы о курсах обмена валют. "
        "Если пользователь задает вопрос, не связанный с конвертацией валют или курсами обмена, "
        "вежливо сообщите, что вы не можете помочь с этим вопросом и можете помочь только с запросами, касающимися валют. "
        "Не пытайтесь отвечать на вопросы, не связанные с темой, или использовать инструменты для других целей."
        "Установите статус ответа в 'input_required', если пользователю нужно предоставить больше информации. "
        "Установите статус ответа в 'error', если произошла ошибка при обработке запроса. "
        "Установите статус ответа в 'completed', если запрос выполнен."
    )
    
    # Поддерживаемые типы контента
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
     
    def __init__(self):
        """
        Description:
        ---------------
            Инициализирует агента конвертации валют.
            Настраивает модель LLM, инструменты и граф Langgraph.

        Returns:
        ---------------
            None
        """
        # Инициализация модели Gemini
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        
        # Установка доступных инструментов
        self.tools = [get_exchange_rate]

        # Создание графа Langgraph для агента ReAct
        self.graph = create_react_agent(
            self.model, 
            tools=self.tools, 
            checkpointer=memory, 
            prompt=self.SYSTEM_INSTRUCTION, 
            response_format=ResponseFormat
        )

    def invoke(self, query: str, sessionId: str) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Обрабатывает запрос пользователя и возвращает ответ.
            Использует граф Langgraph для выполнения цепочки рассуждений.

        Args:
        ---------------
            query: Текстовый запрос пользователя о курсе обмена валют.
            sessionId: Идентификатор сессии для сохранения состояния разговора.

        Returns:
        ---------------
            Dict[str, Any]: Словарь с информацией о результате обработки запроса.

        Examples:
        ---------------
            >>> agent = CurrencyAgent()
            >>> response = agent.invoke("Convert 100 USD to EUR", "session456")
            >>> print(isinstance(response, dict))
            True
        """
        # Настройка конфигурации с идентификатором сессии
        config = {"configurable": {"thread_id": sessionId}}
        
        # Вызов графа для обработки запроса
        self.graph.invoke({"messages": [("user", query)]}, config)  
        
        # Получение и возврат ответа агента
        return self.get_agent_response(config)

    async def stream(
        self, query: str, sessionId: str
    ) -> AsyncIterable[Dict[str, Any]]:
        """
        Description:
        ---------------
            Асинхронно обрабатывает запрос и возвращает поток промежуточных
            и окончательных результатов.

        Args:
        ---------------
            query: Текстовый запрос пользователя о курсе обмена валют.
            sessionId: Идентификатор сессии для сохранения состояния разговора.

        Returns:
        ---------------
            AsyncIterable[Dict[str, Any]]: Поток словарей с информацией о ходе
                                          и результатах обработки запроса.

        Examples:
        ---------------
            >>> agent = CurrencyAgent()
            >>> async for response in agent.stream("What is the USD to JPY rate?", "session789"):
            ...     print(response["is_task_complete"])
        """
        # Подготовка входных данных и конфигурации
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": sessionId}}

        # Обработка потока событий от графа
        for item in self.graph.stream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            
            # Если сообщение - вызов инструмента
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Looking up the exchange rates...",
                }
            # Если сообщение - результат работы инструмента
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing the exchange rates..",
                }            
        
        # Возврат окончательного ответа
        yield self.get_agent_response(config)

    def get_agent_response(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Получает и форматирует ответ агента на основе текущего состояния графа.

        Args:
        ---------------
            config: Конфигурация графа, содержащая идентификатор сессии.

        Returns:
        ---------------
            Dict[str, Any]: Словарь с информацией о результате обработки запроса.

        Examples:
        ---------------
            >>> agent = CurrencyAgent()
            >>> # Предположим, что граф уже вызван с некоторым запросом
            >>> response = agent.get_agent_response({"configurable": {"thread_id": "session123"}})
            >>> "is_task_complete" in response
            True
        """
        # Получение текущего состояния графа
        current_state = self.graph.get_state(config)   
        
        # Получение структурированного ответа
        structured_response = current_state.values.get('structured_response')
        
        # Обработка ответа в зависимости от его статуса
        if structured_response and isinstance(structured_response, ResponseFormat): 
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message
                }
            elif structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message
                }
            elif structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message
                }

        # Возврат сообщения по умолчанию в случае проблемы
        logger.warning("Unable to get a proper structured response from the agent")
        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "We are unable to process your request at the moment. Please try again.",
        }