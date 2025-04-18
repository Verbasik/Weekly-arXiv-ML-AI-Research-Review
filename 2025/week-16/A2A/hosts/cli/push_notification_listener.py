# A2A/hosts/cli/push_notification_listener.py
"""
Модуль реализует слушатель push-уведомлений для A2A API.

Создает HTTP-сервер, который принимает уведомления от A2A агента и выводит их в консоль.
Поддерживает проверку подлинности уведомлений и валидацию токенов.
"""

# Стандартные библиотеки
import asyncio
import threading
import traceback

# Сторонние библиотеки
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response

# Внутренние модули
from common.utils.push_notification_auth import PushNotificationReceiverAuth


class PushNotificationListener:
    """
    Description:
    ---------------
        Класс для прослушивания и обработки push-уведомлений от A2A агента.
        Запускает HTTP-сервер в отдельном потоке.
        
    Args:
    ---------------
        host: Хост для запуска HTTP-сервера
        port: Порт для запуска HTTP-сервера
        notification_receiver_auth: Объект для аутентификации уведомлений
        
    Raises:
    ---------------
        Exception: При ошибке запуска сервера или обработки уведомлений
        
    Examples:
    ---------------
        >>> auth = PushNotificationReceiverAuth()
        >>> await auth.load_jwks("http://localhost:10000/.well-known/jwks.json")
        >>> listener = PushNotificationListener("localhost", 5000, auth)
        >>> listener.start()
    """
    
    def __init__(
        self, 
        host: str, 
        port: int, 
        notification_receiver_auth: PushNotificationReceiverAuth
    ) -> None:
        """
        Description:
        ---------------
            Инициализирует слушатель push-уведомлений.
            
        Args:
        ---------------
            host: Хост для запуска HTTP-сервера
            port: Порт для запуска HTTP-сервера
            notification_receiver_auth: Объект для аутентификации уведомлений
            
        Returns:
        ---------------
            None
        """
        self.host = host
        self.port = port
        self.notification_receiver_auth = notification_receiver_auth
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=lambda loop: loop.run_forever(), 
            args=(self.loop,)
        )
        self.thread.daemon = True
        self.thread.start()
        self.app = None
        self.server = None

    def start(self) -> None:
        """
        Description:
        ---------------
            Запускает сервер для прослушивания push-уведомлений в отдельном потоке.
            
        Returns:
        ---------------
            None
            
        Raises:
        ---------------
            Exception: При ошибке запуска сервера
            
        Examples:
        ---------------
            >>> listener.start()
        """
        try:
            # Необходимо запустить сервер в отдельном потоке, так как текущий поток
            # будет заблокирован при ожидании ввода пользователя
            asyncio.run_coroutine_threadsafe(
                self.start_server(),
                self.loop,
            )
            print("======= push notification listener started =======")
        except Exception as e:
            print(e)

    async def start_server(self) -> None:
        """
        Description:
        ---------------
            Асинхронно запускает HTTP-сервер на указанном хосте и порту.
            
        Returns:
        ---------------
            None
            
        Raises:
        ---------------
            Exception: При ошибке инициализации или запуска сервера
            
        Examples:
        ---------------
            >>> await listener.start_server()
        """
        import uvicorn

        self.app = Starlette()
        self.app.add_route(
            "/notify", self.handle_notification, methods=["POST"]
        )
        self.app.add_route(
            "/notify", self.handle_validation_check, methods=["GET"]
        )
        
        config = uvicorn.Config(
            self.app, 
            host=self.host, 
            port=self.port, 
            log_level="critical"
        )
        self.server = uvicorn.Server(config)
        await self.server.serve()
    
    async def handle_validation_check(self, request: Request) -> Response:
        """
        Description:
        ---------------
            Обрабатывает GET-запросы для валидации токена уведомлений.
            
        Args:
        ---------------
            request: Объект HTTP-запроса
            
        Returns:
        ---------------
            Response: HTTP-ответ с валидационным токеном или ошибкой
            
        Examples:
        ---------------
            >>> await handle_validation_check(request)
            <Response status_code=200>
        """
        validation_token = request.query_params.get("validationToken")
        print(f"\npush notification verification received => \n{validation_token}\n")

        if not validation_token:
            return Response(status_code=400)
            
        return Response(content=validation_token, status_code=200)
    
    async def handle_notification(self, request: Request) -> Response:
        """
        Description:
        ---------------
            Обрабатывает POST-запросы с push-уведомлениями.
            Проверяет аутентификацию и выводит содержимое уведомления.
            
        Args:
        ---------------
            request: Объект HTTP-запроса с данными уведомления
            
        Returns:
        ---------------
            Response: HTTP-ответ с кодом 200 при успешной обработке
            
        Raises:
        ---------------
            Exception: При ошибке проверки аутентификации или обработки JSON
            
        Examples:
        ---------------
            >>> await handle_notification(request)
            <Response status_code=200>
        """
        data = await request.json()
        try:
            # Проверяем подлинность уведомления
            if not await self.notification_receiver_auth.verify_push_notification(request):
                print("push notification verification failed")
                return Response(status_code=401)
        except Exception as e:
            print(f"error verifying push notification: {e}")
            print(traceback.format_exc())
            return Response(status_code=500)
            
        print(f"\npush notification received => \n{data}\n")
        return Response(status_code=200)