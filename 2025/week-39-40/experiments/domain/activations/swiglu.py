# Стандартная библиотека
from typing import Optional

# Сторонние библиотеки
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """
    Description:
    ---------------
        Swish активация: x * sigmoid(x)
        
        Предложена в статье "Searching for Activation Functions" (Ramachandran et al., 2017).
        Также известна как SiLU (Sigmoid Linear Unit) в PyTorch.
        
        Формула: Swish(x) = x * sigmoid(x)
        
        Преимущества:
        - Гладкая функция (все производные существуют)
        - Не ограничена сверху (в отличие от sigmoid)
        - Имеет нелинейность, близкую к ReLU для положительных значений
        - Имеет небольшое подавление для отрицательных значений
        
    Args:
    ---------------
        beta: Опциональный параметр для масштабирования: x * sigmoid(beta * x)
              По умолчанию beta=1.0 (стандартный Swish)
        
    Returns:
    ---------------
        Тензор той же формы, что и вход, с примененной Swish активацией
        
    Examples:
    ---------------
        >>> import torch
        >>> swish = Swish()
        >>> x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> swish(x)
        tensor([-0.2384, -0.2689, 0.0000, 0.7311, 1.7616])
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        # TODO: Сохраните beta параметр
        # pass

        self.beta = beta
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
        ---------------
            Применяет Swish активацию к входному тензору.
        
        Args:
        ---------------
            x: Входной тензор любой формы
            
        Returns:
        ---------------
            Тензор той же формы с примененной Swish активацией
        """
        # TODO: Реализуйте Swish активацию: x * sigmoid(beta * x)
        # pass

        # Сигмоида масштабирует значения тензора от 0 до 1, 
        # затем умножаем на исходный тензор, тем самым сглаживая значения
        return x * torch.sigmoid(self.beta * x)


class SwiGLU(nn.Module):
    """
    Description:
    ---------------
        SwiGLU (Swish-Gated Linear Unit) - активационная функция,
        используемая в современных языковых моделях, включая Qwen3.
        
        Сочетает Swish активацию и механизм гейтинга (GLU).
        
        Формула:
        SwiGLU(x, W1, W2, b1, b2) = Swish(W1*x + b1) ⊙ (W2*x + b2)
        
        где:
        - W1, W2 - весовые матрицы
        - b1, b2 - векторы смещения (опциональные)
        - ⊙ - поэлементное умножение
        
        Преимущества:
        - Лучшая производительность по сравнению с ReLU/GELU в глубоких моделях
        - Эффективный механизм гейтинга для контроля потока информации
        - Используется в современных LLM (Qwen, PaLM, LLaMA)
        
    Args:
    ---------------
        input_dim: Размерность входного вектора
        output_dim: Размерность выходного вектора
        intermediate_dim: Размерность промежуточных матриц W1 и W2 (как правило 4*input_dim)
        bias: Использовать ли смещение в линейных преобразованиях (по умолчанию True)
        
    Returns:
    ---------------
        Тензор формы (..., output_dim) - результат применения SwiGLU
        
    Examples:
    ---------------
        >>> import torch
        >>> swiglu = SwiGLU(512, 512)
        >>> x = torch.randn(2, 10, 512)
        >>> output = swiglu(x)
        >>> output.shape
        torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        intermediate_dim: Optional[int] = None,
        bias: bool = True
    ):
        super().__init__()
        
        # TODO: Если intermediate_dim не указан, установите его как 4*output_dim
        # TODO: Создайте линейный слой gate_proj для проекции входа в промежуточное представление
        # TODO: Создайте линейный слой value_proj для проекции входа в промежуточное представление
        # TODO: Создайте экземпляр Swish активации
        
        # Вопросы для размышления:
        # - Почему используется коэффициент 4 для промежуточной размерности?
        # - Какое преимущество дает механизм гейтинга по сравнению с простой активацией?
        # - Почему SwiGLU лучше работает в глубоких моделях по сравнению с ReLU/GELU?
        pass
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Description:
        ---------------
            Применяет SwiGLU активацию к входному тензору.
        
        Args:
        ---------------
            x: Входной тензор формы (..., input_dim)
            
        Returns:
        ---------------
            Тензор формы (..., output_dim) - результат применения SwiGLU
        """
        # TODO: Примените gate_proj к входу
        # TODO: Примените Swish активацию к результату gate_proj
        # TODO: Примените value_proj к входу
        # TODO: Перемножьте поэлементно результаты Swish(gate_proj(x)) и value_proj(x)
        # TODO: Верните результат
        
        # Вопросы для размышления:
        # - Как механизм гейтинга влияет на градиенты при обратном распространении?
        # - Почему важно использовать разные проекции для gate и value?
        # - Как SwiGLU способствует обучению более глубоких моделей?
        pass
