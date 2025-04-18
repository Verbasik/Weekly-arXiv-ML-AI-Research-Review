# A2A/common/server/utils.py
"""Вспомогательные функции для сервера A2A.

Этот модуль содержит служебные функции для проверки совместимости типов контента
и создания стандартизированных ответов об ошибках.
"""

# Стандартные библиотеки Python
from typing import List, Optional

# Внутренние модули проекта
from common.types import (
    ContentTypeNotSupportedError,
    JSONRPCResponse,
    UnsupportedOperationError,
)


def are_modalities_compatible(
    server_output_modes: Optional[List[str]], client_output_modes: Optional[List[str]]
) -> bool:
    """
    Description:
    ---------------
        Проверяет совместимость типов контента (модальностей) между сервером и клиентом.
        Модальности считаются совместимыми, если они оба не пусты и имеют хотя бы один
        общий элемент, или если хотя бы один из них пуст.

    Args:
    ---------------
        server_output_modes: Список типов контента, поддерживаемых сервером.
        client_output_modes: Список типов контента, поддерживаемых клиентом.

    Returns:
    ---------------
        bool: True, если модальности совместимы, False в противном случае.

    Examples:
    ---------------
        >>> are_modalities_compatible(["text/plain", "image/png"], ["text/plain"])
        True
        >>> are_modalities_compatible(["text/plain"], ["image/png"])
        False
        >>> are_modalities_compatible(None, ["text/plain"])
        True
        >>> are_modalities_compatible(["text/plain"], None)
        True
    """
    # Если клиент не указал типы контента, считаем, что он принимает любые
    if client_output_modes is None or len(client_output_modes) == 0:
        return True

    # Если сервер не указал типы контента, считаем, что он может отправлять любые
    if server_output_modes is None or len(server_output_modes) == 0:
        return True

    # Проверка наличия хотя бы одного общего типа контента
    return any(x in server_output_modes for x in client_output_modes)


def new_incompatible_types_error(request_id: str) -> JSONRPCResponse:
    """
    Description:
    ---------------
        Создает стандартизированный ответ об ошибке несовместимости типов контента.

    Args:
    ---------------
        request_id: Идентификатор запроса, для которого создается ответ об ошибке.

    Returns:
    ---------------
        JSONRPCResponse: Ответ с ошибкой несовместимости типов контента.

    Examples:
    ---------------
        >>> error_response = new_incompatible_types_error("request123")
        >>> isinstance(error_response.error, ContentTypeNotSupportedError)
        True
    """
    # Создание ответа с ошибкой несовместимости типов контента
    return JSONRPCResponse(id=request_id, error=ContentTypeNotSupportedError())


def new_not_implemented_error(request_id: str) -> JSONRPCResponse:
    """
    Description:
    ---------------
        Создает стандартизированный ответ об ошибке неподдерживаемой операции.

    Args:
    ---------------
        request_id: Идентификатор запроса, для которого создается ответ об ошибке.

    Returns:
    ---------------
        JSONRPCResponse: Ответ с ошибкой неподдерживаемой операции.

    Examples:
    ---------------
        >>> error_response = new_not_implemented_error("request123")
        >>> isinstance(error_response.error, UnsupportedOperationError)
        True
    """
    # Создание ответа с ошибкой неподдерживаемой операции
    return JSONRPCResponse(id=request_id, error=UnsupportedOperationError())