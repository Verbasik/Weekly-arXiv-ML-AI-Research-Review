# A2A/agents/langgraph/__main__.py
"""Запуск сервера агента валютных конвертаций.

Этот модуль инициализирует и запускает сервер A2A с агентом конвертации валют,
который помогает пользователям получать информацию о курсах обмена между
различными валютами и выполнять конвертацию.
"""

# Стандартные библиотеки Python
import os
import logging

# Сторонние библиотеки и фреймворки
import click
from dotenv import load_dotenv

# Внутренние модули приложения
from .agent import CurrencyAgent
from .task_manager import AgentTaskManager
from common.server import A2AServer
from common.types import AgentCapabilities, AgentCard, AgentSkill, MissingAPIKeyError
from common.utils.push_notification_auth import PushNotificationSenderAuth

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логирования для отслеживания работы приложения
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Точка входа в приложение, инициализирует и запускает сервер с агентом конвертации валют
@click.command()
@click.option("--host", "host", default="localhost")
@click.option("--port", "port", default=10000)
def main(host: str, port: int) -> None:
    """
    Description:
    ---------------
        Запускает сервер агента конвертации валют.
        Инициализирует необходимые компоненты и запускает сервер на указанном хосте и порту.

    Args:
    ---------------
        host: Хост для запуска сервера (например, "localhost" или "0.0.0.0")
        port: Порт для запуска сервера (целое число)

    Returns:
    ---------------
        None: Функция не возвращает значений, но запускает сервер,
              который работает до принудительной остановки

    Raises:
    ---------------
        MissingAPIKeyError: Если не установлена переменная окружения GOOGLE_API_KEY
        Exception: При других ошибках инициализации или запуска сервера

    Examples:
    ---------------
        >>> main("localhost", 10000)  # Запуск сервера на localhost:10000
        >>> main("0.0.0.0", 8080)     # Запуск сервера на всех интерфейсах, порт 8080
    """
    try:
        # Проверка наличия необходимого API ключа
        if not os.getenv("GOOGLE_API_KEY"):
            raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")

        # Настройка возможностей агента, включая поддержку потоковой передачи и push-уведомлений
        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        
        # Определение навыков агента конвертации валют
        skill = AgentSkill(
            id="convert_currency",
            name="Currency Exchange Rates Tool",
            description="Helps with exchange values between various currencies",
            tags=["currency conversion", "currency exchange"],
            examples=["What is exchange rate between USD and GBP?"],
        )
        
        # Создание карточки агента с метаданными для регистрации в системе
        agent_card = AgentCard(
            name="Currency Agent",
            description="Helps with exchange rates for currencies",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=CurrencyAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=CurrencyAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        # Инициализация аутентификации для отправки push-уведомлений
        notification_sender_auth = PushNotificationSenderAuth()
        notification_sender_auth.generate_jwk()
        
        # Инициализация сервера с настроенным агентом и аутентификацией
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(
                agent=CurrencyAgent(), 
                notification_sender_auth=notification_sender_auth
            ),
            host=host,
            port=port,
        )

        # Добавление маршрута для JWKS (JSON Web Key Set) конечной точки
        server.app.add_route(
            "/.well-known/jwks.json", 
            notification_sender_auth.handle_jwks_endpoint, 
            methods=["GET"]
        )

        # Запуск сервера
        logger.info(f"Starting server on {host}:{port}")
        server.start()
        
    except MissingAPIKeyError as e:
        # Обработка ошибки отсутствия API ключа
        logger.error(f"Error: {e}")
        logger.error("Please set the GOOGLE_API_KEY environment variable and try again.")
        exit(1)
    except Exception as e:
        # Обработка прочих исключений при запуске сервера
        logger.error(f"An error occurred during server startup: {e}")
        logger.error("Check configuration and try again.")
        exit(1)


# Проверка, что скрипт запущен напрямую, а не импортирован
if __name__ == "__main__":
    main()