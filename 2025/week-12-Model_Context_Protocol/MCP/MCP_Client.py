#!/usr/bin/env python3

# Стандартные библиотеки Python
import os
import re
import sys
import json
import logging
import asyncio
import shutil
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pathlib import Path
from contextlib import AsyncExitStack
from types import SimpleNamespace

# Сторонние библиотеки
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServerConnectType(str, Enum):
    """
    Description:
    ---------------
        Перечисление типов подключения к серверу.
    """
    EXECUTABLE = "executable"  # Запуск сервера как процесса
    MCP_LOOKUP = "mcp_lookup"  # Использование имени из конфигурации MCP
    HTTP = "http"              # Подключение к серверу по HTTP


class LLMConfig:
    """
    Description:
    ---------------
        Класс конфигурации для языковой модели (LLM).
        
    Args:
    ---------------
        api_url: URL для API LLM
        api_key: Ключ API для аутентификации (опционально)
        model: Название модели для использования
        headers: Пользовательские HTTP-заголовки (опционально)
        is_openai_compatible: Флаг совместимости с OpenAI API
        max_tokens: Максимальное количество токенов для генерации
        temperature: Температура (креативность) генерации
        
    Examples:
    ---------------
        >>> config = LLMConfig(
        ...     api_url="https://api.openai.com/v1/chat/completions",
        ...     api_key="sk-...",
        ...     model="gpt-3.5-turbo"
        ... )
    """
    def __init__(
        self, 
        api_url: str,
        api_key: Optional[str] = None,
        model: str = "default",
        headers: Optional[Dict[str, str]] = None,
        is_openai_compatible: bool = True,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.is_openai_compatible = is_openai_compatible
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Настройка заголовков
        self.headers = headers or {}
        if api_key and "Authorization" not in self.headers:
            self.headers["Authorization"] = f"Bearer {api_key}"
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"


class ServerConfig:
    """
    Description:
    ---------------
        Класс конфигурации для MCP сервера.
        
    Args:
    ---------------
        connect_type: Тип подключения к серверу
        name: Имя сервера (для MCP_LOOKUP)
        executable: Путь к исполняемому файлу (для EXECUTABLE)
        args: Аргументы командной строки для сервера
        env: Переменные окружения для сервера
        host: Хост сервера (для HTTP)
        port: Порт сервера (для HTTP)
        
    Examples:
    ---------------
        >>> config = ServerConfig(
        ...     connect_type=ServerConnectType.EXECUTABLE,
        ...     executable="python3",
        ...     args=["server.py"]
        ... )
        >>> config_http = ServerConfig(
        ...     connect_type=ServerConnectType.HTTP,
        ...     host="127.0.0.1",
        ...     port=8080
        ... )
    """
    def __init__(
        self,
        connect_type: ServerConnectType,
        name: Optional[str] = None,
        executable: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        self.connect_type = connect_type
        self.name = name
        self.executable = executable
        self.args = args or []
        self.env = env or {}
        self.host = host
        self.port = port


class MCPHttpClient:
    """
    Description:
    ---------------
        Класс для взаимодействия с MCP сервером по HTTP.
        
    Args:
    ---------------
        host: Хост сервера
        port: Порт сервера
    """
    def __init__(self, host: str, port: int):
        self.base_url = f"http://{host}:{port}"
        self.http_client = httpx.AsyncClient()
        
    async def initialize(self):
        """
        Description:
        ---------------
            Инициализация клиента.
        """
        # Проверка доступности сервера
        try:
            response = await self.http_client.get(f"{self.base_url}/status")
            if response.status_code != 200:
                raise Exception(f"Сервер недоступен: {response.status_code}")
        except Exception as e:
            raise Exception(f"Ошибка при подключении к серверу: {str(e)}")
    
    async def list_tools(self):
        """
        Description:
        ---------------
            Получение списка доступных инструментов.
            
        Returns:
        ---------------
            Список доступных инструментов
        """
        response = await self.http_client.get(f"{self.base_url}/tools")
        if response.status_code == 200:
            data = response.json()
            return SimpleNamespace(tools=data["tools"])
        else:
            raise Exception(f"Ошибка при получении списка инструментов: {response.status_code}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """
        Description:
        ---------------
            Вызов инструмента.
            
        Args:
        ---------------
            tool_name: Имя инструмента
            arguments: Аргументы инструмента
            
        Returns:
        ---------------
            Результат вызова инструмента
        """
        payload = {
            "tool": tool_name,
            "arguments": arguments
        }
        response = await self.http_client.post(f"{self.base_url}/call", json=payload)
        if response.status_code == 200:
            data = response.json()
            # Преобразуем список текстовых ответов в объекты TextContent
            content = [TextContent(text=item) for item in data.get("content", [])]
            return SimpleNamespace(content=content)
        else:
            raise Exception(f"Ошибка при вызове инструмента: {response.status_code}")
    
    async def close(self):
        """
        Description:
        ---------------
            Закрытие клиента.
        """
        await self.http_client.aclose()


def find_python_executable() -> str:
    """
    Description:
    ---------------
        Находит доступный исполняемый файл Python в системе.
        
    Returns:
    ---------------
        str: Команда для запуска Python
        
    Examples:
    ---------------
        >>> find_python_executable()
        'python3'
    """
    # Проверяем возможные варианты
    python_variants = [
        "python3", "python", "python3.10", 
        "python3.11", "python3.12", "python3.13"
    ]
    
    for cmd in python_variants:
        if shutil.which(cmd):
            logger.info(f"Найден исполняемый файл Python: {shutil.which(cmd)}")
            return cmd
    
    # Если никакой вариант не найден, пробуем использовать sys.executable
    if sys.executable:
        logger.info(f"Используем текущий Python: {sys.executable}")
        return sys.executable
    
    # Последняя попытка - просто вернуть "python3"
    logger.warning(
        "Не удалось найти Python, используем 'python3' по умолчанию"
    )
    return "python3"


class MCPGitClient:
    """
    Description:
    ---------------
        Клиент для работы с Git через MCP и языковую модель.
        
    Args:
    ---------------
        llm_config: Конфигурация для языковой модели
        
    Examples:
    ---------------
        >>> llm_config = LLMConfig(
        ...     api_url="https://api.openai.com/v1/chat/completions",
        ...     api_key="sk-..."
        ... )
        >>> client = MCPGitClient(llm_config)
    """
    def __init__(self, llm_config: LLMConfig):
        """
        Description:
        ---------------
            Инициализация клиента для работы с LLM и MCP Git сервером.
            
        Args:
        ---------------
            llm_config: Конфигурация для LLM
        """
        self.session = None
        self.exit_stack = AsyncExitStack()
        
        # Настройка для LLM
        self.llm_config = llm_config
        self.http_client = httpx.AsyncClient(headers=llm_config.headers)
        self.available_tools = []
        
    async def connect_to_server(self, server_config: ServerConfig):
        """
        Description:
        ---------------
            Подключение к MCP Git серверу.
            
        Args:
        ---------------
            server_config: Конфигурация сервера
            
        Raises:
        ---------------
            FileNotFoundError: Если исполняемый файл не найден
            ValueError: Если неверный тип подключения или отсутствует 
                        обязательный параметр
        """
        logger.info(
            f"Подключение к серверу: {server_config.name or 'Unnamed'}"
        )
        
        if server_config.connect_type == ServerConnectType.HTTP:
            if not server_config.host or not server_config.port:
                raise ValueError(
                    "Для типа подключения HTTP необходимо указать хост и порт сервера"
                )
            
            logger.info(f"Подключение к HTTP серверу: {server_config.host}:{server_config.port}")
            
            # Создаем HTTP-клиент
            self.mcp_client = MCPHttpClient(server_config.host, server_config.port)
            await self.mcp_client.initialize()
            
            # Получаем список доступных инструментов
            response = await self.mcp_client.list_tools()
            self.available_tools = response.tools
            
        elif server_config.connect_type == ServerConnectType.EXECUTABLE:
            if not server_config.executable:
                # Автоматически определяем Python
                logger.info(
                    "Исполняемый файл не указан, пытаемся определить "
                    "Python автоматически"
                )
                server_config.executable = find_python_executable()
                
            # Проверяем, существует ли исполняемый файл
            executable_path = shutil.which(server_config.executable)
            if not executable_path:
                raise FileNotFoundError(
                    f"Исполняемый файл не найден: {server_config.executable}"
                )
            
            logger.info(f"Исполняемый файл найден: {executable_path}")
            
            # Проверяем, существуют ли файлы скрипта сервера
            if server_config.args and not os.path.exists(server_config.args[0]):
                raise FileNotFoundError(
                    f"Файл скрипта сервера не найден: {server_config.args[0]}"
                )
                
            server_params = StdioServerParameters(
                command=executable_path,
                args=server_config.args,
                env=server_config.env
            )
            
            logger.info(
                f"Запуск сервера: {executable_path} "
                f"{' '.join(server_config.args)}"
            )
            try:
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Ошибка при запуске сервера: {str(e)}\n"
                    f"Проверьте путь к исполняемому файлу и аргументы."
                )
            
            # Инициализация сессии для stdio
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            await self.session.initialize()

            # Получаем список доступных инструментов
            response = await self.session.list_tools()
            self.available_tools = response.tools
                
        elif server_config.connect_type == ServerConnectType.MCP_LOOKUP:
            if not server_config.name:
                raise ValueError(
                    "Для типа подключения MCP_LOOKUP необходимо "
                    "указать имя сервера"
                )
                
            # Поиск сервера в конфигурации Claude Desktop или MCP-клиента
            config_paths = [
                Path.home() / ".config" / "mcp" / "config.json"
            ]
            
            server_found = False
            for config_path in config_paths:
                if config_path.exists():
                    logger.info(f"Найдена конфигурация MCP: {config_path}")
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            
                        if ("mcpServers" in config and 
                                server_config.name in config["mcpServers"]):
                            server_info = config["mcpServers"][
                                server_config.name
                            ]
                            command = server_info.get("command")
                            
                            # Проверяем наличие команды
                            command_path = shutil.which(command)
                            if not command_path:
                                logger.warning(
                                    f"Команда '{command}' не найдена, "
                                    f"пытаемся определить Python автоматически"
                                )
                                command = find_python_executable()
                            
                            server_params = StdioServerParameters(
                                command=command,
                                args=server_info.get("args", []),
                                env=server_info.get("env", {})
                            )
                            
                            logger.info(
                                f"Используется сервер из конфигурации: "
                                f"{server_config.name}"
                            )
                            try:
                                stdio_transport = (
                                    await self.exit_stack.enter_async_context(
                                        stdio_client(server_params)
                                    )
                                )
                                # Инициализация сессии для stdio
                                self.stdio, self.write = stdio_transport
                                self.session = await self.exit_stack.enter_async_context(
                                    ClientSession(self.stdio, self.write)
                                )
                                await self.session.initialize()

                                # Получаем список доступных инструментов
                                response = await self.session.list_tools()
                                self.available_tools = response.tools
                                server_found = True
                                break
                            except FileNotFoundError as e:
                                logger.error(
                                    f"Ошибка при запуске сервера из "
                                    f"конфигурации: {str(e)}"
                                )
                    except Exception as e:
                        logger.error(
                            f"Ошибка при чтении конфигурации {config_path}: {e}"
                        )
                        
            if not server_found:
                raise ValueError(
                    f"Сервер с именем '{server_config.name}' не найден "
                    f"в конфигурации MCP или не удалось запустить"
                )
                
        else:
            raise ValueError(
                f"Неизвестный тип подключения: {server_config.connect_type}"
            )
        
        logger.info(
            f"Подключено к серверу. Доступные инструменты: "
            f"{[tool.name for tool in self.available_tools]}"
        )
        
    async def process_query(self, query: str) -> str:
        """
        Description:
        ---------------
            Обработка запроса с использованием LLM и доступных инструментов.
            
        Args:
        ---------------
            query: Текст запроса от пользователя
            
        Returns:
        ---------------
            str: Результат обработки запроса
            
        Raises:
        ---------------
            Exception: При ошибке обработки запроса
        """
        print(f"🔍 Получен запрос: '{query}'")
        
        # Составляем системное сообщение с инструкциями
        system_message = self._create_system_message()
        
        # Инициализируем диалог
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
        print(f"💬 All content for LLM: {messages}")
        
        # Преобразуем инструменты в формат для LLM
        tools = self._format_tools_for_llm()
        
        # Основной цикл обработки
        max_iterations = 5  # Ограничиваем количество итераций
        final_text = []
        
        for i in range(max_iterations):
            print(f"\n⭐ Итерация {i+1}/{max_iterations} ⭐")
            
            # Вызываем LLM
            llm_response = await self._call_llm(messages, tools)
            print(f"✅ Получен ответ от модели: {llm_response}")
            
            # Проверяем наличие вызовов инструментов
            tool_calls = llm_response.get("tool_calls", [])
            content = llm_response.get("content", "")
            
            # Добавляем текстовый ответ
            if content:
                print("📝 Получен текстовый ответ от модели")
                final_text.append(content)
            
            if not tool_calls:
                # Если нет вызовов инструментов, завершаем обработку
                break
            
            # Обрабатываем вызовы инструментов
            assistant_message = {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls
            }
            messages.append(assistant_message)
            
            tool_results = []
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                tool_name = function.get("name", "")
                tool_call_id = tool_call.get("id", "")
                
                try:
                    # Парсим аргументы
                    arguments = json.loads(function.get("arguments", "{}"))
                    
                    # Вызываем инструмент через соответствующий клиент
                    if hasattr(self, "mcp_client"):
                        # Для HTTP-клиента
                        result = await self.mcp_client.call_tool(tool_name, arguments)
                    else:
                        # Для stdio-клиента
                        result = await self.session.call_tool(tool_name, arguments)
                    
                    # Преобразуем результат в текст
                    tool_result = self._format_tool_result(result.content)
                    
                    final_text.append(f"\n[Вызов инструмента {tool_name}]")
                    final_text.append(f"Результат: {tool_result}")
                    tool_results.append(tool_result)
                    
                    # Добавляем результат в сообщения
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result
                    })
                    
                except Exception as e:
                    error_message = (
                        f"Ошибка при вызове инструмента {tool_name}: {str(e)}"
                    )
                    print(f"❌ {error_message}")
                    final_text.append(f"\n{error_message}")
                    
                    # Добавляем сообщение об ошибке
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": error_message
                    })
            
            # Если последняя итерация и были вызовы, получаем финальный ответ
            if i == max_iterations - 1 and tool_results:
                final_response = await self._call_llm(messages, tools)
                final_content = final_response.get("content", "")
                if final_content:
                    final_text.append(f"\nИтоговый ответ: {final_content}")
        
        return "\n".join([text for text in final_text if text])
    
    def _format_tool_result(self, content_list: List[Any]) -> str:
        """
        Description:
        ---------------
            Форматирует результат вызова инструмента в текстовый формат.
            
        Args:
        ---------------
            content_list: Список объектов с текстовым содержимым
            
        Returns:
        ---------------
            str: Форматированный результат в виде текста
        """
        return "\n".join(
            [item.text for item in content_list if hasattr(item, 'text')]
        )
    
    def _create_system_message(self) -> str:
        """
        Description:
        ---------------
            Создает системное сообщение с описанием инструментов.
            
        Returns:
        ---------------
            str: Текст системного сообщения
        """
        tools_description = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in self.available_tools
        ])
        
        return f"""Ты ассистент, который помогает пользователю работать с Git \
репозиториями.
У тебя есть доступ к следующим инструментам Git:

{tools_description}

Когда пользователь задает вопрос о Git или просит выполнить Git-операцию:
1. Определи, какой инструмент нужно использовать
2. Объясни, что ты собираешься сделать
3. Вызови соответствующий инструмент
4. После получения результата, объясни его пользователю

Всегда вызывай инструменты с правильными аргументами согласно схеме:
1. Для инструментов, требующих repo_path, нужно указать полный путь к \
репозиторию
2. Следи за типами данных аргументов (строки, числа, списки)
3. Если путь к репозиторию не указан, сначала вызови list_repositories()
"""
    
    def _format_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Description:
        ---------------
            Форматирует инструменты в формат, понятный LLM API.
            
        Returns:
        ---------------
            List[Dict[str, Any]]: Список инструментов в формате для LLM
        """
        llm_tools = []
        
        for tool in self.available_tools:
            # Преобразуем схему инструмента в формат, понятный LLM
            input_schema = tool.inputSchema or {}
            
            function_spec = {
                "name": tool.name,
                "description": tool.description,
                "parameters": input_schema
            }
            
            llm_tools.append({
                "type": "function",
                "function": function_spec
            })
            
        return llm_tools
    
    async def _call_llm(
        self, 
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Вызывает LLM API с заданными сообщениями и инструментами.
            
        Args:
        ---------------
            messages: Список сообщений диалога
            tools: Список инструментов
            
        Returns:
        ---------------
            Dict[str, Any]: Ответ от LLM
            
        Raises:
        ---------------
            Exception: При ошибке вызова API
        """
        try:
            # Формируем запрос в зависимости от типа API
            if self.llm_config.is_openai_compatible:
                payload = {
                    "model": self.llm_config.model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": self.llm_config.temperature,
                    "max_tokens": self.llm_config.max_tokens
                }
            else:
                # Для API, не совместимых с OpenAI
                payload = {
                    "model": self.llm_config.model,
                    "prompt": self._format_messages_for_custom_llm(messages),
                    "tools": tools,
                    "temperature": self.llm_config.temperature,
                    "max_tokens": self.llm_config.max_tokens
                }
            
            response = await self.http_client.post(
                self.llm_config.api_url,
                json=payload,
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Обработка ответа в зависимости от типа API
                if self.llm_config.is_openai_compatible:
                    choices = result.get("choices", [])
                    if choices:
                        message = choices[0].get("message", {})
                        return message
                    return {"content": "Получен пустой ответ от LLM"}
                else:
                    # Для API, не совместимых с OpenAI
                    return self._parse_custom_llm_response(result)
            else:
                return {
                    "content": (
                        f"Ошибка при вызове LLM: "
                        f"{response.status_code} - {response.text}"
                    )
                }
                
        except Exception as e:
            return {"content": f"Ошибка при обращении к LLM: {str(e)}"}
    
    def _format_messages_for_custom_llm(
        self, 
        messages: List[Dict[str, Any]]
    ) -> str:
        """
        Description:
        ---------------
            Форматирует сообщения для пользовательской LLM.
            
        Args:
        ---------------
            messages: Список сообщений диалога
            
        Returns:
        ---------------
            str: Отформатированный текст промпта
        """
        formatted_messages = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                formatted_messages.append(f"### Инструкции:\n{content}")
            elif role == "user":
                formatted_messages.append(f"### Пользователь:\n{content}")
            elif role == "assistant":
                formatted_messages.append(f"### Ассистент:\n{content}")
            elif role == "tool":
                tool_call_id = message.get("tool_call_id", "")
                formatted_messages.append(
                    f"### Результат инструмента ({tool_call_id}):\n{content}"
                )
        
        formatted_messages.append("### Ассистент:")
        return "\n\n".join(formatted_messages)
    
    def _parse_custom_llm_response(
            self, 
            response: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Description:
            ---------------
                Обрабатывает ответ от пользовательской LLM.
                
            Args:
            ---------------
                response: Ответ от API
                
            Returns:
            ---------------
                Dict[str, Any]: Обработанный ответ
            """
            if "response" in response:
                content = response["response"]
                
                # Проверяем, есть ли вызовы инструментов в тексте
                tool_calls = []
                
                # Ищем паттерны вызова инструментов в тексте
                tool_call_pattern = (
                    r'Вызов инструмента (\w+)\s*с аргументами\s*\{([^}]*)\}'
                )
                matches = re.findall(tool_call_pattern, content)
                
                for i, (tool_name, args_str) in enumerate(matches):
                    try:
                        # Преобразуем строку аргументов в словарь JSON
                        args_dict = {}
                        for arg_pair in args_str.split(','):
                            if ':' in arg_pair:
                                key, value = arg_pair.split(':', 1)
                                key = key.strip().strip('"\'')
                                value = value.strip().strip('"\'')
                                args_dict[key] = value
                        
                        tool_calls.append({
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(args_dict)
                            }
                        })
                    except Exception:
                        pass
                
                return {
                    "content": content,
                    "tool_calls": tool_calls
                }
            
            return {"content": "Не удалось обработать ответ от LLM"}
        
    async def chat_loop(self):
        """
        Description:
        ---------------
            Запускает интерактивный цикл чата с пользователем.
            
        Raises:
        ---------------
            Exception: При ошибке обработки запроса
            
        Examples:
        ---------------
            >>> await client.chat_loop()
            MCP Git Client запущен!
            Введите запрос или 'quit' для выхода.
        """
        print("\nMCP Git Client запущен!")
        print("Введите запрос или 'quit' для выхода.")

        while True:
            try:
                query = input("\nЗапрос: ").strip()

                if query.lower() in ('quit', 'exit', 'выход'):
                    break

                print("Обработка запроса...")
                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nОшибка: {str(e)}")
                if "--debug" in sys.argv:
                    import traceback
                    traceback.print_exc()

    async def cleanup(self):
        """
        Description:
        ---------------
            Освобождает ресурсы клиента.
            
        Examples:
        ---------------
            >>> await client.cleanup()
        """
        await self.http_client.aclose()
        
        if hasattr(self, "mcp_client"):
            await self.mcp_client.close()
        
        if hasattr(self, "exit_stack"):
            await self.exit_stack.aclose()


def load_config(config_path: str) -> Tuple[ServerConfig, LLMConfig]:
    """
    Description:
    ---------------
        Загружает конфигурацию из файла JSON или YAML.
        
    Args:
    ---------------
        config_path: Путь к файлу конфигурации
        
    Returns:
    ---------------
        Tuple[ServerConfig, LLMConfig]: Конфигурации для сервера и LLM
        
    Raises:
    ---------------
        ImportError: Если требуется YAML, но библиотека не установлена
        ValueError: Если формат файла не поддерживается
        Exception: При ошибке загрузки конфигурации
        
    Examples:
    ---------------
        >>> server_config, llm_config = load_config("config.json")
    """
    try:
        with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Загрузка конфигурации сервера
        server_config_data = config.get('server', {})
        server_connect_type = ServerConnectType(
            server_config_data.get('connect_type', 'executable')
        )
        
        if server_connect_type == ServerConnectType.HTTP:
            server_config = ServerConfig(
                connect_type=server_connect_type,
                host=server_config_data.get('host', '127.0.0.1'),
                port=server_config_data.get('port', 8080)
            )
        else:
            # Обработка пути к исполняемому файлу
            executable = server_config_data.get('executable')
            if executable == "python" and sys.platform == "darwin":
                # На macOS автоматически используем python3
                logger.info("Обнаружена macOS, меняем 'python' на 'python3'")
                executable = "python3"
            
            server_config = ServerConfig(
                connect_type=server_connect_type,
                name=server_config_data.get('name'),
                executable=executable,
                args=server_config_data.get('args', []),
                env=server_config_data.get('env', {})
            )
        
        # Загрузка конфигурации LLM
        llm_config_data = config.get('llm', {})
        
        # Проверяем наличие API ключа в переменных окружения
        api_key = llm_config_data.get('api_key')
        if not api_key:
            api_key = os.environ.get("LLM_API_KEY", "")
            if api_key:
                logger.info("Использую API ключ из переменной окружения LLM_API_KEY")
        
        llm_config = LLMConfig(
            api_url=llm_config_data.get('api_url', ''),
            api_key=api_key,
            model=llm_config_data.get('model', 'default'),
            is_openai_compatible=llm_config_data.get(
                'is_openai_compatible', True
            ),
            max_tokens=llm_config_data.get('max_tokens', 1000),
            temperature=llm_config_data.get('temperature', 0.7)
        )
        
        return server_config, llm_config
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {e}")
        raise


async def main():
    """
    Description:
    ---------------
        Основная функция программы.
        
    Raises:
    ---------------
        Exception: При ошибке выполнения
        
    Examples:
    ---------------
        >>> asyncio.run(main())
    """
    # Путь к конфигурационному файлу
    config_path = "config.json"
    
    # Проверка аргументов командной строки для опции debug
    if "--debug" in sys.argv:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Проверка наличия пути к конфигурации в аргументах
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break
    
    try:
        # Загрузка конфигурации
        server_config, llm_config = load_config(config_path)
        
        # Создание и запуск клиента
        client = MCPGitClient(llm_config)
        try:
            await client.connect_to_server(server_config)
            await client.chat_loop()
        finally:
            await client.cleanup()
    except Exception as e:
        logger.error(f"Ошибка при запуске приложения: {e}")
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())