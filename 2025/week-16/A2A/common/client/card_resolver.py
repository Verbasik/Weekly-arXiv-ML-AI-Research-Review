# A2A/common/client/card_resolver.py
"""
Модуль для получения информации о карточках агентов через A2A API.

Обеспечивает взаимодействие с API удаленных агентов для получения
их карточек (метаданных, возможностей и описания).
"""

# Стандартные библиотеки
import json
from typing import Optional

# Сторонние библиотеки
import httpx

# Внутренние модули
from common.types import (
    AgentCard,
    A2AClientJSONError,
)


class A2ACardResolver:
    """
    Description:
    ---------------
        Класс для получения карточек агентов по A2A API.
        
    Args:
    ---------------
        base_url: Базовый URL-адрес агента
        agent_card_path: Путь к карточке агента (по умолчанию "/.well-known/agent.json")
        
    Raises:
    ---------------
        httpx.HTTPStatusError: При ошибке HTTP-запроса
        A2AClientJSONError: При ошибке декодирования JSON
        
    Examples:
    ---------------
        >>> resolver = A2ACardResolver("http://localhost:10000")
        >>> agent_card = resolver.get_agent_card()
        >>> print(agent_card.name)
    """

    def __init__(
        self, 
        base_url: str, 
        agent_card_path: str = "/.well-known/agent.json"
    ) -> None:
        """
        Description:
        ---------------
            Инициализирует объект для получения карточки агента.
            
        Args:
        ---------------
            base_url: Базовый URL-адрес агента
            agent_card_path: Путь к карточке агента (по умолчанию "/.well-known/agent.json")
            
        Returns:
        ---------------
            None
        """
        # Удаляем завершающий слеш из базового URL, если он есть
        self.base_url = base_url.rstrip("/")
        # Удаляем начальный слеш из пути к карточке, если он есть
        self.agent_card_path = agent_card_path.lstrip("/")

    def get_agent_card(self) -> AgentCard:
        """
        Description:
        ---------------
            Получает карточку агента с удаленного сервера.
            
        Returns:
        ---------------
            AgentCard: Объект карточки агента с его метаданными
            
        Raises:
        ---------------
            httpx.HTTPStatusError: При ошибке HTTP-запроса
            A2AClientJSONError: При ошибке декодирования JSON
            
        Examples:
        ---------------
            >>> agent_card = resolver.get_agent_card()
            >>> print(f"Имя агента: {agent_card.name}")
            >>> print(f"Описание: {agent_card.description}")
        """
        # Создаем HTTP-клиент в контексте менеджера для автоматического закрытия
        with httpx.Client() as client:
            # Формируем полный URL и отправляем GET-запрос
            full_url = f"{self.base_url}/{self.agent_card_path}"
            response = client.get(full_url)
            # Проверяем статус ответа, при ошибке вызывает исключение
            response.raise_for_status()
            
            try:
                # Пытаемся распарсить JSON и создать объект AgentCard
                return AgentCard(**response.json())
            except json.JSONDecodeError as e:
                # Перехватываем ошибку декодирования JSON и заменяем на специфичную для A2A
                raise A2AClientJSONError(str(e)) from e