# Стандартная библиотека
import math

# Сторонние библиотеки
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Локальные импорты
import sys
import os
# Получаем путь к директории с swiglu.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from swiglu import Swish, SwiGLU


class TestSwish:
    """Тесты для проверки корректности Swish активации."""

    def test_swish_initialization(self):
        """
        Тест инициализации Swish модуля.

        Проверяет:
        - Корректность сохранения beta параметра
        """
        # TODO: Создайте Swish с beta=1.0 (по умолчанию)
        # TODO: Проверьте, что beta сохранился корректно
        
        # TODO: Создайте Swish с beta=2.0
        # TODO: Проверьте, что beta сохранился корректно
        
        pass

    def test_swish_forward(self):
        """
        Тест корректности вычисления Swish активации.

        Проверяет:
        - Математическую корректность формулы Swish(x) = x * sigmoid(x)
        - Работу с тензорами разных размерностей
        """
        # TODO: Создайте Swish с beta=1.0
        # TODO: Создайте тестовый тензор с известными значениями
        # TODO: Примените Swish активацию
        # TODO: Вычислите ожидаемый результат вручную
        # TODO: Сравните с помощью torch.allclose
        
        # TODO: Проверьте работу с тензорами разных размерностей (2D, 3D, 4D)
        
        pass

    def test_swish_vs_silu(self):
        """
        Тест сравнения Swish с встроенной SiLU активацией PyTorch.

        Проверяет:
        - Эквивалентность Swish(beta=1.0) и nn.SiLU()
        """
        # TODO: Создайте Swish с beta=1.0 и nn.SiLU()
        # TODO: Создайте случайный тензор
        # TODO: Примените обе активации
        # TODO: Сравните результаты с помощью torch.allclose
        
        pass

    def test_swish_gradient_flow(self):
        """
        Тест корректности градиентов через Swish.

        Проверяет:
        - Градиенты входного тензора
        - Отсутствие NaN или Inf в градиентах
        """
        # TODO: Создайте Swish
        # TODO: Создайте тензор с requires_grad=True
        # TODO: Примените Swish активацию
        # TODO: Вычислите скалярный loss и выполните backward pass
        # TODO: Проверьте, что градиенты не содержат NaN или Inf
        
        pass


class TestSwiGLU:
    """Тесты для проверки корректности SwiGLU активации."""

    def test_swiglu_initialization(self):
        """
        Тест инициализации SwiGLU модуля.

        Проверяет:
        - Корректность создания линейных слоев
        - Корректность размерностей
        """
        # TODO: Создайте SwiGLU с input_dim=512, output_dim=512
        # TODO: Проверьте, что gate_proj и value_proj созданы корректно
        # TODO: Проверьте, что размерности соответствуют ожидаемым
        
        # TODO: Создайте SwiGLU с input_dim=512, output_dim=256, intermediate_dim=1024
        # TODO: Проверьте, что размерности соответствуют ожидаемым
        
        # TODO: Создайте SwiGLU с bias=False
        # TODO: Проверьте, что bias отсутствует в линейных слоях
        
        pass

    def test_swiglu_forward_shape(self):
        """
        Тест сохранения формы тензора при forward pass.

        Проверяет различные формы входов:
        - 2D: (batch_size, input_dim)
        - 3D: (batch_size, seq_len, input_dim)
        """
        # TODO: Создайте SwiGLU с input_dim=512, output_dim=512
        # TODO: Создайте тензоры разных размерностей
        # TODO: Примените SwiGLU к тензорам
        # TODO: Проверьте, что выходные формы соответствуют ожидаемым
        
        pass

    def test_swiglu_mathematical_correctness(self):
        """
        Тест математической корректности SwiGLU.

        Проверяет:
        - Формулу: SwiGLU(x) = Swish(gate_proj(x)) ⊙ value_proj(x)
        """
        # TODO: Создайте SwiGLU с маленькими размерностями для простоты проверки
        # TODO: Создайте простой тензор с известными значениями
        # TODO: Получите значения gate_proj и value_proj напрямую
        # TODO: Вычислите ожидаемый результат вручную
        # TODO: Сравните с результатом forward pass
        
        pass

    def test_swiglu_vs_other_activations(self):
        """
        Сравнительный тест SwiGLU с другими активациями.

        Проверяет:
        - Отличие от простого линейного слоя
        - Отличие от других активаций (ReLU, GELU)
        """
        # TODO: Создайте SwiGLU, Linear+ReLU, Linear+GELU с одинаковыми размерностями
        # TODO: Создайте случайный тензор
        # TODO: Примените все активации
        # TODO: Проверьте, что результаты различаются
        
        pass

    def test_swiglu_gradient_flow(self):
        """
        Тест корректности градиентов через SwiGLU.

        Проверяет:
        - Градиенты входного тензора
        - Градиенты параметров gate_proj и value_proj
        - Отсутствие NaN или Inf в градиентах
        """
        # TODO: Создайте SwiGLU
        # TODO: Создайте тензор с requires_grad=True
        # TODO: Примените SwiGLU активацию
        # TODO: Вычислите скалярный loss и выполните backward pass
        # TODO: Проверьте, что градиенты не содержат NaN или Inf
        # TODO: Проверьте, что градиенты параметров имеют правильные формы
        
        pass

    @pytest.mark.parametrize("input_dim,output_dim", [(64, 64), (128, 256), (512, 512), (1024, 512)])
    def test_swiglu_different_sizes(self, input_dim, output_dim):
        """
        Параметризованный тест для различных размерностей.

        Проверяет корректность работы SwiGLU с разными размерностями.
        """
        # TODO: Создайте SwiGLU с заданными размерностями
        # TODO: Создайте случайный тензор подходящей формы
        # TODO: Примените SwiGLU активацию
        # TODO: Проверьте, что выход имеет ожидаемую форму
        
        pass


# Вопросы для размышления при написании тестов:
# 1. Как убедиться, что механизм гейтинга работает корректно?
# 2. Какие преимущества дает SwiGLU по сравнению с другими активациями?
# 3. Как проверить, что градиенты вычисляются правильно?
# 4. Как влияет параметр beta в Swish на форму активационной функции?
# 5. Почему промежуточная размерность обычно в 4 раза больше выходной?
