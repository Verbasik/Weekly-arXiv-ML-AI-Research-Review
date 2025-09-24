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
        print(f"Тензор позиций: {position}")

        # Создаем тензор с четными индексами [0, 2, 4, ...] для адресации пар измерений
        # Каждая пара (2i, 2i+1) будет обрабатываться как комплексное число
        idx = torch.arange(start=0, end=dim, step=2).float()
        print(f"Индексы четных измерений (2i): {idx}")

        # Вычисляем частоты для каждой пары измерений по формуле: ω_d = base^(-2d/D)
        # - Низкие частоты (начало тензора) меняются медленно с изменением позиции
        # - Высокие частоты (конец тензора) меняются быстрее
        # - scale < 1.0 замедляет вращение для лучшей экстраполяции на длинные контексты
        freqs = base ** (-idx / dim)
        print(f"Частоты для каждой пары измерений: {freqs}")

        print(f"Тензор позиций (unsqueeze(1)): {position.unsqueeze(1)}")
        print(f"Частоты (unsqueeze(0)): {freqs.unsqueeze(0)}")

        # Вычисляем углы для каждой комбинации позиции и частоты
        # Применяем scale для поддержки длинных контекстов
        # Форма: (max_position, dim/2)
        angles = position.unsqueeze(1) * freqs.unsqueeze(0) / scale
        print(f"Углы для каждой комбинации позиции и частоты: {angles}")

        
        
        
    def _compute_rope_embeddings(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        is_query: bool = True
    ) -> torch.Tensor:
        """
        Применяет ротационное позиционное кодирование к входному тензору.
        
        Args:
            x: Входной тензор формы (..., seq_len, dim)
            positions: Тензор позиций формы (..., seq_len)
            is_query: Флаг, указывающий, является ли вход query (True) или key (False)
            
        Returns:
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
        pass

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Применяет RoPE к query и key тензорам.
        
        Args:
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
        pass
    
    def extra_repr(self) -> str:
        """Строковое представление модуля для отладки."""
        return f'dim={self.dim}, base={self.base}, max_position={self.max_position}, scale={self.scale}'

if __name__ == '__main__':
    rope = RoPE(dim=8)
    print(rope)