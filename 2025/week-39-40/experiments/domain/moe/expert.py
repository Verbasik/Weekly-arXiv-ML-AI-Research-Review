# Стандартная библиотека
from typing import Optional

# Сторонние библиотеки
import torch
import torch.nn as nn

# Локальные импорты
from experiments.domain.activations.swiglu import SwiGLU


class Expert(nn.Module):
    """
    Description:
    ---------------
        Expert Network для Mixture-of-Experts архитектуры.

        Каждый эксперт - это независимая feed-forward сеть, которая обрабатывает
        токены, направленные к ней Router'ом.

        Архитектура:
        Input (hidden_size) → SwiGLU FFN → Output (hidden_size)

        Внутри SwiGLU:
        hidden_size → intermediate_size (с gating) → hidden_size

        Для модели 0.6B:
        - hidden_size = 512
        - intermediate_size = 2048 (обычно 4 * hidden_size)
        - num_experts = 8 (каждый с независимыми весами)

    Mathematical Flow:
    ---------------
        x ∈ ℝ^(batch×seq×hidden)
            ↓
        SwiGLU(x) = Swish(W1·x) ⊙ (W2·x)  [intermediate_dim]
            ↓
        W3·SwiGLU(x) + b3
            ↓
        output ∈ ℝ^(batch×seq×hidden)

    Args:
    ---------------
        hidden_size: Размерность входа и выхода (должна совпадать с hidden_size модели)
        intermediate_size: Размерность промежуточного слоя (обычно 4 * hidden_size)
        dropout: Dropout вероятность для регуляризации (default: 0.0)

    Returns (from forward):
    ---------------
        output: Тензор формы (batch_size, seq_len, hidden_size)
                Преобразованные скрытые состояния

    Example:
    ---------------
        >>> # Создание одного эксперта для модели 0.6B
        >>> expert = Expert(hidden_size=512, intermediate_size=2048)
        >>> x = torch.randn(2, 10, 512)  # (batch=2, seq=10, hidden=512)
        >>> output = expert(x)
        >>> output.shape  # torch.Size([2, 10, 512])

        >>> # Создание нескольких экспертов
        >>> num_experts = 8
        >>> experts = nn.ModuleList([
        ...     Expert(hidden_size=512, intermediate_size=2048)
        ...     for _ in range(num_experts)
        ... ])
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.0
    ):
        super().__init__()

        # TODO: Проверьте валидность параметров
        #       - hidden_size должен быть положительным целым числом
        #       - intermediate_size должен быть положительным целым числом
        #       - dropout должен быть в диапазоне [0.0, 1.0)
        # TODO: Сохраните параметры как атрибуты класса
        # TODO: Создайте self.ffn - экземпляр SwiGLU
        #       Параметры: input_dim=hidden_size, output_dim=hidden_size,
        #                  intermediate_dim=intermediate_size
        # TODO: Создайте self.dropout - слой Dropout с заданной вероятностью
        #       (даже если dropout=0.0, создайте слой для единообразия)

        # Вопросы для размышления:
        # - Почему intermediate_size обычно в 4 раза больше hidden_size?
        # - Зачем нужен dropout в экспертах?
        # - Как SwiGLU отличается от обычного ReLU FFN?
        # - Почему каждый эксперт должен иметь одинаковую архитектуру?
        pass


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Description:
        ---------------
            Применяет преобразование эксперта к входным скрытым состояниям.

            Процесс:
            1. Применение SwiGLU feed-forward сети
            2. Применение dropout для регуляризации

        Args:
        ---------------
            hidden_states: Входной тензор формы (batch_size, seq_len, hidden_size)
                          Скрытые состояния токенов, направленных к этому эксперту

        Returns:
        ---------------
            output: Тензор формы (batch_size, seq_len, hidden_size)
                   Преобразованные скрытые состояния
        """
        # TODO: Примените self.ffn к hidden_states
        # TODO: Примените self.dropout к результату
        # TODO: Верните результат

        # Вопросы для размышления:
        # - Нужен ли residual connection внутри эксперта?
        # - Когда применяется dropout - только при training или всегда?
        # - Как размерности входа и выхода связаны?
        pass
