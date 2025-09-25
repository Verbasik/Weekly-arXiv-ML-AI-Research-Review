# Стандартная библиотека
from typing import Optional, Tuple, Union

# Сторонние библиотеки
import torch
import torch.nn as nn
import math


class RoPE(nn.Module):
    """
    Description:
    ---------------
        Rotary Position Embedding (RoPE) — метод позиционного кодирования,
        основанный на вращении векторов в комплексной плоскости.
        
        В отличие от абсолютных позиционных эмбеддингов, RoPE кодирует
        относительные позиции, что делает его особенно эффективным для
        моделей с длинным контекстом.
        
        Ключевые преимущества:
        - Инвариантность к сдвигу (shift invariance)
        - Эффективность вычислений
        - Хорошая экстраполяция на длины, превышающие обучающие
        - Совместимость с линейными attention механизмами
        
    Args:
    ---------------
        dim: Размерность эмбеддинга (должна быть четной для комплексного представления)
        base: База для вычисления частот (обычно 10000.0)
        max_position: Максимальная позиция для предварительного вычисления (кэширования)
        scale: Масштабирующий коэффициент для частот (используется для RoPE scaling)
        
    Examples:
    ---------------
        >>> import torch
        >>> rope = RoPE(dim=128)
        >>> q = torch.randn(2, 4, 128)  # [batch_size, seq_len, dim]
        >>> k = torch.randn(2, 4, 128)  # [batch_size, seq_len, dim]
        >>> q_pos, k_pos = rope(q, k)
        >>> q_pos.shape, k_pos.shape
        (torch.Size([2, 4, 128]), torch.Size([2, 4, 128]))
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_position: int = 2048,
        scale: float = 1.0
    ):
        super().__init__()
        
        # TODO: Проверьте, что dim четное (необходимо для комплексного представления)
        # TODO: Сохраните параметры (dim, base, max_position, scale)
        # TODO: Предварительно вычислите sin/cos таблицу частот для позиций [0, max_position)
        #       - Создайте тензор позиций;
        #       - Создайте тензор частот;
        #       - Вычислите углы для каждой позиции и частоты;
        #       - Вычислите sin и cos;
        #       - Сохраните кэш как буферы (не параметры).
        
        # Вопросы для размышления:
        # - Почему RoPE использует комплексное представление для кодирования позиций?
        # - Как выбор base влияет на частотные характеристики позиционного кодирования?
        # - Почему для длинных контекстов часто используют scale < 1.0?
        # pass

        if dim % 2 != 0:
            raise ValueError('Для корректной работы параметр dim должен быть четным')
        
        # Сохраняем основные параметры для использования в других методах
        self.dim = dim                    # Размерность эмбеддинга (должна быть четной)
        self.base = base                  # База для вычисления частот (обычно 10000.0)
        self.max_position = max_position  # Максимальная позиция для кэширования
        self.scale = scale                # Масштабирующий коэффициент для длинных контекстов

        # Создаем тензор позиций [0, 1, 2, ..., max_position-1]
        position = torch.arange(start = 0, end = max_position, step =1).float()

        # Создаем тензор с четными индексами [0, 2, 4, ...] для адресации пар измерений
        # Каждая пара (2i, 2i+1) будет обрабатываться как комплексное число
        idx = torch.arange(start=0, end=dim, step=2).float()

        # Вычисляем частоты для каждой пары измерений по формуле: ω_d = base^(-2d/D)
        # - Низкие частоты (начало тензора) меняются медленно с изменением позиции
        # - Высокие частоты (конец тензора) меняются быстрее
        # - scale < 1.0 замедляет вращение для лучшей экстраполяции на длинные контексты
        freqs = base ** (-idx / dim)

        # Вычисляем углы для каждой комбинации позиции и частоты
        # Применяем scale для поддержки длинных контекстов
        # Форма: (max_position, dim/2)
        angles = position.unsqueeze(1) * freqs.unsqueeze(0) / scale

        # Вычисляем sin и cos для каждого угла
        cos = torch.cos(angles)  # (max_position, dim/2)
        sin = torch.sin(angles)  # (max_position, dim/2)

        # Сохраняем sin и cos как буферы (не параметры)
        # Используем register_buffer для правильной работы с CUDA и сохранения/загрузки модели
        self.register_buffer('sin_cached', sin)
        self.register_buffer('cos_cached', cos)

        
    def _compute_rope_embeddings(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        is_query: bool = True
    ) -> torch.Tensor:
        """
        Description:
        ---------------
            Применяет ротационное позиционное кодирование к входному тензору.
        
        Args:
        ---------------
            x: Входной тензор формы (..., seq_len, dim)
            positions: Тензор позиций формы (..., seq_len)
            is_query: Флаг, указывающий, является ли вход query (True) или key (False)
            
        Returns:
        ---------------
            Тензор с примененным позиционным кодированием той же формы, что и x
        """
        # TODO: Получите форму входного тензора x
        # TODO: Извлеките sin и cos для заданных позиций из кэша или вычислите их на лету
        # TODO: Если positions выходят за пределы max_position, вычислите sin и cos динамически
        # TODO: Примените вращение к каждой паре соседних измерений (dim[i], dim[i+1])
        #       - Для четных индексов i: x[..., i] = x[..., i] * cos - x[..., i+1] * sin
        #       - Для нечетных индексов i: x[..., i] = x[..., i] * sin + x[..., i-1] * cos
        # TODO: Если is_query=False (для ключей), инвертируйте направление вращения
        # TODO: Верните тензор с примененным позиционным кодированием
        
        # Вопросы для размышления:
        # - Почему для query и key используются разные направления вращения?
        # - Как RoPE обеспечивает относительное позиционное кодирование?
        # - Как работает экстраполяция на позиции за пределами max_position?
        # pass

        # Получаем форму входного тензора
        x_shape = x.shape

        if positions.max() < self.max_position:
            sin = self.sin_cached[positions]
            cos = self.cos_cached[positions]
        else:
            # Так же как в __init__
            idx = torch.arange(start=0, end=self.dim, step=2).float()
            freqs = self.base ** (-idx / self.dim)
            angles = positions.unsqueeze(1) * freqs.unsqueeze(0) / self.scale

            cos = torch.cos(angles)
            sin = torch.sin(angles)

        # Создаем выходной тензор той же формы, что и входной
        x_out = torch.zeros_like(x)

        # Применяем вращение в зависимости от типа входа (query или key)
        if is_query:
            # Для query - положительное направление вращения
            x_out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
            x_out[..., 1::2] = x[..., 1::2] * cos + x[..., 0::2] * sin
        else:
            # Для key - отрицательное направление вращения
            x_out[..., 0::2] = x[..., 0::2] * cos + x[..., 1::2] * sin
            x_out[..., 1::2] = x[..., 1::2] * cos - x[..., 0::2] * sin

        return x_out

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Description:
        ---------------
            Применяет RoPE к query и key тензорам.
        
        Args:
        ---------------
            query: Query тензор формы (..., seq_len, dim)
            key: Key тензор формы (..., seq_len, dim)
            positions: Опциональный тензор позиций. Если None, используется torch.arange(seq_len)
            
        Returns:
            Кортеж (query_pos, key_pos) с примененным позиционным кодированием
        """
        # TODO: Проверьте, что последнее измерение query и key равно self.dim
        # TODO: Получите seq_len из формы query
        # TODO: Если positions не указаны, создайте их через torch
        # TODO: Примените _compute_rope_embeddings к query
        # TODO: Примените _compute_rope_embeddings к key
        # TODO: Верните кортеж (query_pos, key_pos)
        
        # Вопросы для размышления:
        # - Как RoPE влияет на взаимодействие между query и key в attention?
        # - Почему важно применять RoPE к обоим тензорам - query и key?
        # - Как можно оптимизировать вычисления RoPE для больших моделей?
        # pass

        if query.shape[-1] != self.dim or key.shape[-1] != self.dim:
            raise ValueError(f'Размерность query и key должна быть равна {self.dim}')
        
        # Проверяем, что формы query и key совпадают по seq_len
        if query.shape[-2] != key.shape[-2]:
            raise ValueError(f'Длины последовательностей query и key должны совпадать')
        
        seq_len = query.shape[-2]

        if positions is not None:
            query_rope = self._compute_rope_embeddings(query, positions, is_query=True)
            key_rope   = self._compute_rope_embeddings(key,   positions, is_query=False)

            return query_rope, key_rope

        else:
            # Создаем позиции от 0 до seq_len-1 на том же устройстве, что и query
            positions = torch.arange(seq_len, device=query.device)
            # Применяем RoPE к query и key
            query_rope = self._compute_rope_embeddings(query, positions, is_query=True)
            key_rope = self._compute_rope_embeddings(key, positions, is_query=False)
            
            return query_rope, key_rope


    def extra_repr(self) -> str:
        """Строковое представление модуля для отладки."""
        return f'dim={self.dim}, base={self.base}, max_position={self.max_position}, scale={self.scale}'