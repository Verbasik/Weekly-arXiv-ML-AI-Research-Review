# A2A/common/utils/in_memory_cache.py
"""Утилита для кэширования данных в памяти.

Этот модуль предоставляет инструменты для кэширования данных в памяти
приложения с поддержкой времени жизни кэша и потокобезопасных операций.
"""

# Стандартные библиотеки Python
import threading
import time
from typing import Any, Dict, Optional, Union, TypeVar, cast

# Определение типа для значения, возвращаемого методом get
T = TypeVar('T')


class InMemoryCache:
    """
    Description:
    ---------------
        Потокобезопасный Singleton-класс для управления данными кэша.
        Гарантирует существование только одного экземпляра кэша во всём приложении.
        Поддерживает механизм TTL (Time To Live) для автоматического устаревания данных.

    Attributes:
    ---------------
        _instance: Единственный экземпляр класса InMemoryCache.
        _lock: Блокировка для обеспечения потокобезопасности при создании экземпляра.
        _initialized: Флаг, указывающий, был ли инициализирован экземпляр.
        _cache_data: Хранилище данных кэша.
        _ttl: Словарь времени жизни для ключей кэша.
        _data_lock: Блокировка для потокобезопасного доступа к данным.

    Examples:
    ---------------
        >>> cache = InMemoryCache()
        >>> cache.set("key1", {"data": "value1"})
        >>> cache.get("key1")
        {'data': 'value1'}
        >>> cache.set("key2", "value2", ttl=5)  # Установка с временем жизни 5 секунд
        >>> cache.get("key2")  # В течение 5 секунд
        'value2'
        >>> # После 5 секунд
        >>> cache.get("key2")
        None
    """

    _instance: Optional["InMemoryCache"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls) -> "InMemoryCache":
        """
        Description:
        ---------------
            Переопределение метода __new__ для контроля создания экземпляра (паттерн Singleton).
            Использует блокировку для обеспечения потокобезопасности при первом создании.

        Returns:
        ---------------
            InMemoryCache: Единственный экземпляр класса InMemoryCache.

        Examples:
        ---------------
            >>> cache1 = InMemoryCache()
            >>> cache2 = InMemoryCache()
            >>> cache1 is cache2
            True
        """
        # Проверка наличия экземпляра и создание его при необходимости
        if cls._instance is None:
            # Блокировка для потокобезопасного создания экземпляра
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cast(InMemoryCache, cls._instance)

    def __init__(self) -> None:
        """
        Description:
        ---------------
            Инициализация хранилища кэша.
            Использует флаг (_initialized) для обеспечения выполнения этой логики
            только при самом первом создании экземпляра Singleton.

        Returns:
        ---------------
            None

        Examples:
        ---------------
            >>> cache = InMemoryCache()  # Инициализация происходит при первом создании
        """
        # Проверка, был ли уже инициализирован экземпляр
        if not self._initialized:
            # Блокировка для потокобезопасной инициализации
            with self._lock:
                if not self._initialized:
                    # Инициализация структур данных для хранения кэша
                    self._cache_data: Dict[str, Any] = {}
                    self._ttl: Dict[str, float] = {}
                    self._data_lock: threading.Lock = threading.Lock()
                    self._initialized = True

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Description:
        ---------------
            Устанавливает пару ключ-значение в кэш.
            При указании ttl данные будут автоматически удалены после истечения времени жизни.

        Args:
        ---------------
            key: Ключ для хранения данных.
            value: Данные для сохранения.
            ttl: Время жизни в секундах. Если None, данные не будут иметь срока действия.

        Returns:
        ---------------
            None

        Examples:
        ---------------
            >>> cache = InMemoryCache()
            >>> cache.set("user_profile", {"name": "John", "age": 30})
            >>> cache.set("session_token", "abc123", ttl=3600)  # Срок действия 1 час
        """
        # Блокировка для потокобезопасной записи данных
        with self._data_lock:
            # Сохранение данных в кэше
            self._cache_data[key] = value

            # Установка времени жизни, если указано
            if ttl is not None:
                self._ttl[key] = time.time() + ttl
            else:
                # Удаление времени жизни, если оно было установлено ранее
                if key in self._ttl:
                    del self._ttl[key]

    def get(self, key: str, default: Optional[T] = None) -> Union[Any, T]:
        """
        Description:
        ---------------
            Получает значение, связанное с ключом.
            Проверяет срок действия данных и возвращает значение по умолчанию,
            если данные истекли или не найдены.

        Args:
        ---------------
            key: Ключ для поиска данных в кэше.
            default: Значение, возвращаемое, если ключ не найден или срок действия истек.

        Returns:
        ---------------
            Any: Кэшированное значение или значение по умолчанию.

        Examples:
        ---------------
            >>> cache = InMemoryCache()
            >>> cache.set("counter", 42)
            >>> cache.get("counter")
            42
            >>> cache.get("nonexistent", "Not found")
            'Not found'
        """
        # Блокировка для потокобезопасного чтения данных
        with self._data_lock:
            # Проверка времени жизни и удаление устаревших данных
            if key in self._ttl and time.time() > self._ttl[key]:
                del self._cache_data[key]
                del self._ttl[key]
                return default
            # Возврат данных или значения по умолчанию
            return self._cache_data.get(key, default)

    def delete(self, key: str) -> bool:
        """
        Description:
        ---------------
            Удаляет указанную пару ключ-значение из кэша.

        Args:
        ---------------
            key: Ключ для удаления.

        Returns:
        ---------------
            bool: True, если ключ был найден и удален, False в противном случае.

        Examples:
        ---------------
            >>> cache = InMemoryCache()
            >>> cache.set("temporary", "value")
            >>> cache.delete("temporary")
            True
            >>> cache.delete("nonexistent")
            False
        """
        # Блокировка для потокобезопасного удаления данных
        with self._data_lock:
            # Проверка наличия ключа и удаление данных
            if key in self._cache_data:
                del self._cache_data[key]
                # Удаление времени жизни, если оно было установлено
                if key in self._ttl:
                    del self._ttl[key]
                return True
            return False

    def clear(self) -> bool:
        """
        Description:
        ---------------
            Удаляет все данные из кэша.

        Returns:
        ---------------
            bool: True, если данные были очищены, False в противном случае.

        Examples:
        ---------------
            >>> cache = InMemoryCache()
            >>> cache.set("key1", "value1")
            >>> cache.set("key2", "value2")
            >>> cache.clear()
            True
            >>> cache.get("key1")
            None
        """
        # Блокировка для потокобезопасной очистки данных
        with self._data_lock:
            # Очистка хранилища данных и времени жизни
            self._cache_data.clear()
            self._ttl.clear()
            return True
        # Эта строка никогда не будет выполнена из-за конструкции with
        # (добавлена для сохранения логики оригинала)
        return False