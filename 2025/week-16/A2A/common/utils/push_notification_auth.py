# A2A/common/utils/push_notification_auth.py
"""Модуль аутентификации push-уведомлений.

Предоставляет классы для проверки и отправки push-уведомлений с использованием
JWT (JSON Web Tokens) для обеспечения целостности сообщений и защиты от атак.
"""

# Стандартные библиотеки Python
import hashlib
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

# Сторонние библиотеки
import httpx
import jwt
from jwcrypto import jwk
from jwt import PyJWK, PyJWKClient
from starlette.requests import Request
from starlette.responses import JSONResponse

# Константы
AUTH_HEADER_PREFIX = 'Bearer '

# Настройка логирования
logger = logging.getLogger(__name__)


class PushNotificationAuth:
    """
    Description:
    ---------------
        Базовый класс для аутентификации push-уведомлений.
        Содержит общую функциональность для отправителей и получателей уведомлений.

    Examples:
    ---------------
        >>> auth = PushNotificationAuth()
        >>> sha256 = auth._calculate_request_body_sha256({"data": "value"})
        >>> isinstance(sha256, str)
        True
    """

    def _calculate_request_body_sha256(self, data: Dict[str, Any]) -> str:
        """
        Description:
        ---------------
            Вычисляет хеш SHA256 для тела запроса.
            Эта логика должна быть одинаковой как для агента, который подписывает
            полезную нагрузку, так и для клиента-верификатора.

        Args:
        ---------------
            data: Словарь данных для хеширования.

        Returns:
        ---------------
            str: Шестнадцатеричное представление хеша SHA256.

        Examples:
        ---------------
            >>> auth = PushNotificationAuth()
            >>> sha256 = auth._calculate_request_body_sha256({"test": "data"})
            >>> len(sha256) == 64  # SHA256 всегда имеет длину 64 символа в hex-формате
            True
        """
        # Преобразование словаря в строку JSON с определенным форматированием
        body_str = json.dumps(
            data,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        )
        
        # Вычисление хеша SHA256 и возврат в шестнадцатеричном формате
        return hashlib.sha256(body_str.encode()).hexdigest()


class PushNotificationSenderAuth(PushNotificationAuth):
    """
    Description:
    ---------------
        Класс для отправки аутентифицированных push-уведомлений.
        Генерирует пары ключей JWK, создает JWT-токены и отправляет уведомления
        с подписью для обеспечения их подлинности.

    Attributes:
    ---------------
        public_keys: Список открытых ключей для проверки подписи.
        private_key_jwk: Закрытый ключ JWK для подписи токенов.

    Examples:
    ---------------
        >>> sender = PushNotificationSenderAuth()
        >>> sender.generate_jwk()
        >>> # Теперь можно использовать для отправки уведомлений
    """

    def __init__(self):
        """
        Description:
        ---------------
            Инициализирует экземпляр отправителя push-уведомлений.

        Returns:
        ---------------
            None
        """
        # Инициализация списка открытых ключей и закрытого ключа
        self.public_keys = []
        self.private_key_jwk: Optional[PyJWK] = None

    @staticmethod
    async def verify_push_notification_url(url: str) -> bool:
        """
        Description:
        ---------------
            Проверяет URL для push-уведомлений, отправляя валидационный токен
            и ожидая такой же ответ для подтверждения владения URL.

        Args:
        ---------------
            url: URL для проверки.

        Returns:
        ---------------
            bool: True, если URL верифицирован успешно, False в противном случае.

        Examples:
        ---------------
            >>> async def example():
            ...     is_valid = await PushNotificationSenderAuth.verify_push_notification_url("https://example.com/webhook")
            ...     print(f"URL валиден: {is_valid}")
        """
        # Создание HTTP-клиента с таймаутом
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                # Генерация случайного валидационного токена
                validation_token = str(uuid.uuid4())
                
                # Отправка GET-запроса с токеном в качестве параметра
                response = await client.get(
                    url,
                    params={"validationToken": validation_token}
                )
                response.raise_for_status()
                
                # Проверка, что ответ содержит тот же токен
                is_verified = response.text == validation_token

                # Логирование результата проверки
                logger.info(f"Verified push-notification URL: {url} => {is_verified}")            
                return is_verified                
            except Exception as e:
                # Логирование ошибки при проверке URL
                logger.warning(f"Error during sending push-notification for URL {url}: {e}")

        return False

    def generate_jwk(self) -> None:
        """
        Description:
        ---------------
            Генерирует новую пару ключей JWK и сохраняет их.
            Открытый ключ добавляется в список публичных ключей,
            а закрытый ключ сохраняется для подписи.

        Returns:
        ---------------
            None

        Examples:
        ---------------
            >>> sender = PushNotificationSenderAuth()
            >>> sender.generate_jwk()
            >>> len(sender.public_keys) > 0
            True
        """
        # Генерация новой пары ключей RSA с уникальным идентификатором
        key = jwk.JWK.generate(kty='RSA', size=2048, kid=str(uuid.uuid4()), use="sig")
        
        # Сохранение открытого ключа в список
        self.public_keys.append(key.export_public(as_dict=True))
        
        # Сохранение закрытого ключа для подписи
        self.private_key_jwk = PyJWK.from_json(key.export_private())
    
    def handle_jwks_endpoint(self, _request: Request) -> JSONResponse:
        """
        Description:
        ---------------
            Обработчик для эндпоинта JWKS, который позволяет клиентам
            получать открытые ключи для проверки подписи.

        Args:
        ---------------
            _request: Запрос Starlette.

        Returns:
        ---------------
            JSONResponse: Ответ, содержащий набор открытых ключей.

        Examples:
        ---------------
            >>> # Регистрация эндпоинта в приложении FastAPI/Starlette:
            >>> # app.add_route("/.well-known/jwks.json", auth.handle_jwks_endpoint, methods=["GET"])
        """
        # Возврат JSON-ответа с открытыми ключами
        return JSONResponse({
            "keys": self.public_keys
        })
    
    def _generate_jwt(self, data: Dict[str, Any]) -> str:
        """
        Description:
        ---------------
            Генерирует JWT-токен, подписывая как хеш SHA256 полезной нагрузки запроса,
            так и время генерации токена.

            Полезная нагрузка подписывается закрытым ключом, что обеспечивает целостность
            для клиента. Включение времени создания (iat) предотвращает атаки повторного
            воспроизведения.

        Args:
        ---------------
            data: Данные запроса для включения в токен.

        Returns:
        ---------------
            str: Подписанный JWT-токен.

        Raises:
        ---------------
            ValueError: Если закрытый ключ не был сгенерирован.

        Examples:
        ---------------
            >>> sender = PushNotificationSenderAuth()
            >>> sender.generate_jwk()
            >>> jwt_token = sender._generate_jwt({"message": "test"})
            >>> isinstance(jwt_token, str)
            True
        """
        # Проверка наличия закрытого ключа
        if not self.private_key_jwk:
            raise ValueError("Private key not generated. Call generate_jwk() first.")
            
        # Получение текущего времени в формате UNIX timestamp
        iat = int(time.time())

        # Создание и подпись JWT-токена
        return jwt.encode(
            {
                "iat": iat,  # Время создания токена
                "request_body_sha256": self._calculate_request_body_sha256(data)  # Хеш тела запроса
            },
            key=self.private_key_jwk.key,
            headers={"kid": self.private_key_jwk.key_id},
            algorithm="RS256"
        )

    async def send_push_notification(
        self, url: str, data: Dict[str, Any]
    ) -> bool:
        """
        Description:
        ---------------
            Отправляет аутентифицированное push-уведомление на указанный URL.
            Подписывает данные с помощью JWT и отправляет их в POST-запросе.

        Args:
        ---------------
            url: URL, на который отправляется уведомление.
            data: Данные для отправки в теле запроса.

        Returns:
        ---------------
            bool: True, если уведомление успешно отправлено, False в противном случае.

        Examples:
        ---------------
            >>> async def example():
            ...     sender = PushNotificationSenderAuth()
            ...     sender.generate_jwk()
            ...     success = await sender.send_push_notification(
            ...         "https://example.com/webhook",
            ...         {"event": "update", "id": "123"}
            ...     )
            ...     print(f"Отправлено успешно: {success}")
        """
        try:
            # Генерация JWT-токена для аутентификации
            jwt_token = self._generate_jwt(data)
            
            # Настройка заголовков с токеном авторизации
            headers = {'Authorization': f"{AUTH_HEADER_PREFIX}{jwt_token}"}
            
            # Отправка POST-запроса с данными и заголовками
            async with httpx.AsyncClient(timeout=10) as client: 
                response = await client.post(
                    url,
                    json=data,
                    headers=headers
                )
                response.raise_for_status()
                
                # Логирование успешной отправки
                logger.info(f"Push-notification sent for URL: {url}")
                return True
                                
        except Exception as e:
            # Логирование ошибки при отправке
            logger.warning(f"Error during sending push-notification for URL {url}: {e}")
            return False


class PushNotificationReceiverAuth(PushNotificationAuth):
    """
    Description:
    ---------------
        Класс для приема и проверки аутентифицированных push-уведомлений.
        Проверяет JWT-токены и целостность полученных данных.

    Attributes:
    ---------------
        public_keys_jwks: Список открытых ключей JWKS.
        jwks_client: Клиент для работы с JWKS.

    Examples:
    ---------------
        >>> receiver = PushNotificationReceiverAuth()
        >>> # Далее нужно загрузить JWKS и использовать для проверки уведомлений
    """

    def __init__(self):
        """
        Description:
        ---------------
            Инициализирует экземпляр получателя push-уведомлений.

        Returns:
        ---------------
            None
        """
        # Инициализация списка открытых ключей и клиента JWKS
        self.public_keys_jwks = []
        self.jwks_client = None

    async def load_jwks(self, jwks_url: str) -> None:
        """
        Description:
        ---------------
            Загружает набор ключей JWKS с указанного URL для проверки подписей.

        Args:
        ---------------
            jwks_url: URL эндпоинта JWKS, содержащего открытые ключи.

        Returns:
        ---------------
            None

        Examples:
        ---------------
            >>> async def example():
            ...     receiver = PushNotificationReceiverAuth()
            ...     await receiver.load_jwks("https://example.com/.well-known/jwks.json")
        """
        # Инициализация клиента JWKS с указанным URL
        self.jwks_client = PyJWKClient(jwks_url)
    
    async def verify_push_notification(self, request: Request) -> bool:
        """
        Description:
        ---------------
            Проверяет подлинность полученного push-уведомления.
            Извлекает и проверяет JWT-токен, проверяет хеш тела запроса
            и время создания токена.

        Args:
        ---------------
            request: Запрос Starlette, содержащий уведомление.

        Returns:
        ---------------
            bool: True, если уведомление аутентично, False в противном случае.

        Raises:
        ---------------
            ValueError: Если тело запроса не соответствует подписи или токен просрочен.
            Exception: Если возникли другие ошибки проверки.

        Examples:
        ---------------
            >>> async def example():
            ...     # В обработчике FastAPI/Starlette:
            ...     receiver = PushNotificationReceiverAuth()
            ...     await receiver.load_jwks("https://example.com/.well-known/jwks.json")
            ...     try:
            ...         is_valid = await receiver.verify_push_notification(request)
            ...         if is_valid:
            ...             # Обработка уведомления
            ...             pass
            ...     except ValueError as e:
            ...         # Обработка ошибки аутентификации
            ...         pass
        """
        # Проверка наличия и формата заголовка авторизации
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith(AUTH_HEADER_PREFIX):
            logger.warning("Invalid authorization header")
            return False
        
        try:
            # Извлечение токена из заголовка
            token = auth_header[len(AUTH_HEADER_PREFIX):]
            
            # Получение ключа для проверки подписи
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)

            # Проверка и декодирование токена
            decode_token = jwt.decode(
                token,
                signing_key.key,
                options={"require": ["iat", "request_body_sha256"]},
                algorithms=["RS256"],
            )

            # Проверка целостности тела запроса
            actual_body_sha256 = self._calculate_request_body_sha256(await request.json())
            if actual_body_sha256 != decode_token["request_body_sha256"]:
                # Подпись полезной нагрузки не соответствует хешу в токене
                raise ValueError("Invalid request body")
            
            # Проверка времени создания токена (не более 5 минут назад)
            if time.time() - decode_token["iat"] > 60 * 5:
                # Не принимать push-уведомления старше 5 минут
                # Это предотвращает атаки повторного воспроизведения
                raise ValueError("Token is expired")
            
            return True
            
        except Exception as e:
            # Логирование ошибки верификации
            logger.warning(f"Push notification verification failed: {e}")
            raise