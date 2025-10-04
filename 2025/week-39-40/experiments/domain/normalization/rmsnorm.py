# Стандартная библиотека
from typing import Optional

# Сторонние библиотеки
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Description:
    ---------------
        Root Mean Square Layer Normalization — современная альтернатива LayerNorm.
        Формула: RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight

        В отличие от LayerNorm, RMSNorm не центрирует данные (не вычитает среднее),
        что обеспечивает лучшую численную стабильность и производительность
        при обучении больших языковых моделей.

        Ключевые преимущества:
        - Меньше вычислений (нет центрирования)
        - Лучшая стабильность при больших моделях
        - Используется в современных архитектурах (LLaMA, Qwen, и др.)

    Args:
    ---------------
        normalized_shape: int или tuple размерностей для нормализации.
                          Обычно равен hidden_size модели.
        eps: Малая константа для численной устойчивости, предотвращает деление на ноль.
                          Добавляется под корень: sqrt(mean(x²) + eps).
        elementwise_affine: Если True, добавляет обучаемые параметры weight.
                            Если False, применяет только нормализацию без масштабирования.

    Examples:
    ---------------
        >>> import torch
        >>> rms_norm = RMSNorm(512)
        >>> x = torch.randn(10, 20, 512)
        >>> output = rms_norm(x)
        >>> output.shape
        torch.Size([10, 20, 512])

        >>> # Проверка нормализации: RMS должен быть близок к 1
        >>> rms_value = torch.sqrt(torch.mean(output**2, dim=-1))
        >>> print(f"RMS after normalization: {rms_value.mean():.4f}")
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True
    ):
        super().__init__()

        # TODO: Сохраните normalized_shape для использования в forward
        # TODO: Сохраните eps для численной стабильности
        # TODO: Если elementwise_affine=True, создайте Parameter weight с формой (normalized_shape)
        # TODO: Инициализируйте weight единицами: torch.ones()
        # TODO: Если elementwise_affine=False, зарегистрируйте weight как None

        # Вопросы для размышления:
        # - Почему weight инициализируется единицами, а не нулями?
        # - Что произойдет, если eps слишком большой или слишком маленький?
        # - Зачем нужен параметр elementwise_affine?
        # pass

        self.normalized_shape = normalized_shape
        self.eps = eps

        # Создаем обучаемый параметр масштабирования (g в формуле RMSNorm)
        if elementwise_affine == True:
            # Инициализируем weight единицами для стабильности начального обучения
            # weight соответствует вектору g в формуле: g ⊙ (x/RMS(x))
            self.weight = nn.Parameter(
                torch.ones(normalized_shape)
            )
        else:
            # Если масштабирование не требуется, регистрируем пустой параметр
            # для совместимости с механизмами PyTorch (state_dict и др.)
            self.register_parameter('weight', None)
                    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
        ---------------
            Реализует формулу RMSNorm: x / sqrt(mean(x²) + eps) * weight.
            Нормализует входной тензор по его среднеквадратичному значению,
            без центрирования (в отличие от LayerNorm).

        Args:
        ---------------
            x: Входной тензор формы (..., normalized_shape)

        Returns:
        ---------------
            Нормализованный тензор той же формы

        Raises:
        ---------------
            RuntimeError: Если последняя размерность x не совпадает с normalized_shape
        """
        # TODO: Проверьте, что последняя размерность x равна self.normalized_shape
        # TODO: Вычислите квадраты элементов: x_squared = x * x или x.pow(2)
        # TODO: Вычислите среднее квадратов по последней оси: torch.mean
        # TODO: Вычислите RMS: torch.sqrt
        # TODO: Нормализуйте по формуле: RMSNorm(x) = x / sqrt(mean(x²) + eps)
        # TODO: Если есть weight, примените масштабирование: output = normalized * self.weight
        # TODO: Верните результат

        # Вопросы для размышления:
        # - Почему важно использовать keepdim=True при вычислении среднего?
        # - Как поведет себя функция на тензорах разной размерности?
        # - Что произойдет с градиентами при обратном проходе?
        # - Как RMSNorm влияет на распределение активаций?
        # pass

        # Проверяем соответствие размерности входного тензора
        if x.shape[-1] != self.normalized_shape:
            raise RuntimeError(
                f"Последняя размерность x должна быть равна {self.normalized_shape}"
            )

        # Вычисляем квадраты элементов
        x_sqr = x * x

        # Вычисляем среднее квадратов по последней оси
        # keepdim=True сохраняет размерность для корректного вещания при делении
        mean_sqr = torch.mean(x_sqr, dim=-1, keepdim=True)

        # Вычисляем RMS (корень из среднего квадратов) с добавлением eps для стабильности
        rms = torch.sqrt(mean_sqr + self.eps)

        # Нормализуем входной тензор, деля на RMS
        x_norm = x / rms

        # Применяем масштабирование, если есть weight
        if self.weight is not None:
            # Поэлементное умножение на обучаемый параметр weight
            return x_norm * self.weight
        
        return x_norm
        

    def extra_repr(self) -> str:
        """Строковое представление модуля для отладки."""
        return f'normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.weight is not None}'