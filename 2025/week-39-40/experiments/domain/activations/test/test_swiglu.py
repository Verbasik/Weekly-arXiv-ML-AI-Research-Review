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
        
        # pass
        
        # Создаем Swish с beta=1.0 (по умолчанию)
        swish_default = Swish()
        # Проверяем, что beta сохранился корректно
        assert swish_default.beta == 1.0, "Beta параметр по умолчанию должен быть равен 1.0"
        
        # Создаем Swish с beta=2.0
        swish_custom = Swish(beta=2.0)
        # Проверяем, что beta сохранился корректно
        assert swish_custom.beta == 2.0, "Beta параметр должен быть равен переданному значению 2.0"

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
        
        # pass
        
        # Создаем Swish с beta=1.0
        swish = Swish(beta=1.0)
        
        # Создаем тестовый тензор с известными значениями
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # Применяем Swish активацию
        result = swish(x)
        
        # Вычисляем ожидаемый результат вручную
        sigmoid_x = torch.sigmoid(x)
        expected = x * sigmoid_x
        
        # Сравниваем с помощью torch.allclose
        assert torch.allclose(result, expected, rtol=1e-4), f"Ожидаемый результат: {expected}, получено: {result}"
        
        # Проверяем работу с тензорами разных размерностей
        # 2D тензор
        x_2d = torch.randn(3, 4)
        result_2d = swish(x_2d)
        expected_2d = x_2d * torch.sigmoid(x_2d)
        assert torch.allclose(result_2d, expected_2d), "Некорректная работа с 2D тензором"
        assert result_2d.shape == x_2d.shape, f"Некорректная форма выхода: {result_2d.shape}, ожидается: {x_2d.shape}"
        
        # 3D тензор
        x_3d = torch.randn(2, 3, 4)
        result_3d = swish(x_3d)
        expected_3d = x_3d * torch.sigmoid(x_3d)
        assert torch.allclose(result_3d, expected_3d), "Некорректная работа с 3D тензором"
        assert result_3d.shape == x_3d.shape, f"Некорректная форма выхода: {result_3d.shape}, ожидается: {x_3d.shape}"
        
        # 4D тензор
        x_4d = torch.randn(2, 3, 4, 5)
        result_4d = swish(x_4d)
        expected_4d = x_4d * torch.sigmoid(x_4d)
        assert torch.allclose(result_4d, expected_4d), "Некорректная работа с 4D тензором"
        assert result_4d.shape == x_4d.shape, f"Некорректная форма выхода: {result_4d.shape}, ожидается: {x_4d.shape}"

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
        
        # pass
        
        # Создаем Swish с beta=1.0 и nn.SiLU()
        swish = Swish(beta=1.0)
        silu = nn.SiLU()
        
        # Создаем случайный тензор
        x = torch.randn(100)
        
        # Применяем обе активации
        swish_result = swish(x)
        silu_result = silu(x)
        
        # Сравниваем результаты с помощью torch.allclose
        assert torch.allclose(swish_result, silu_result), "Swish(beta=1.0) должен быть эквивалентен nn.SiLU()"

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
        
        # pass
        
        # Создаем Swish
        swish = Swish()
        
        # Создаем тензор с requires_grad=True
        x = torch.randn(10, requires_grad=True)
        
        # Применяем Swish активацию
        y = swish(x)
        
        # Вычисляем скалярный loss и выполняем backward pass
        loss = y.sum()
        loss.backward()
        
        # Проверяем, что градиенты не содержат NaN или Inf
        assert torch.isfinite(x.grad).all(), "Градиенты содержат NaN или Inf"
        
        # Проверяем, что градиенты не нулевые (активация должна пропускать градиенты)
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Градиенты не должны быть нулевыми для всех элементов"


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
        
        # pass
        
        # Создаем SwiGLU с input_dim=512, output_dim=512
        swiglu = SwiGLU(input_dim=512, output_dim=512)
        
        # Проверяем, что gate_proj и value_proj созданы корректно
        assert isinstance(swiglu.gate_proj, nn.Linear), "gate_proj должен быть экземпляром nn.Linear"
        assert isinstance(swiglu.value_proj, nn.Linear), "value_proj должен быть экземпляром nn.Linear"
        assert isinstance(swiglu.output_proj, nn.Linear), "output_proj должен быть экземпляром nn.Linear"
        
        # Проверяем, что размерности соответствуют ожидаемым
        assert swiglu.gate_proj.in_features == 512, f"Входная размерность gate_proj должна быть 512, получено {swiglu.gate_proj.in_features}"
        assert swiglu.gate_proj.out_features == 2048, f"Выходная размерность gate_proj должна быть 2048 (4*512), получено {swiglu.gate_proj.out_features}"
        assert swiglu.value_proj.in_features == 512, f"Входная размерность value_proj должна быть 512, получено {swiglu.value_proj.in_features}"
        assert swiglu.value_proj.out_features == 2048, f"Выходная размерность value_proj должна быть 2048 (4*512), получено {swiglu.value_proj.out_features}"
        assert swiglu.output_proj.in_features == 2048, f"Входная размерность output_proj должна быть 2048, получено {swiglu.output_proj.in_features}"
        assert swiglu.output_proj.out_features == 512, f"Выходная размерность output_proj должна быть 512, получено {swiglu.output_proj.out_features}"
        
        # Создаем SwiGLU с input_dim=512, output_dim=256, intermediate_dim=1024
        swiglu_custom = SwiGLU(input_dim=512, output_dim=256, intermediate_dim=1024)
        
        # Проверяем, что размерности соответствуют ожидаемым
        assert swiglu_custom.gate_proj.in_features == 512, f"Входная размерность gate_proj должна быть 512, получено {swiglu_custom.gate_proj.in_features}"
        assert swiglu_custom.gate_proj.out_features == 1024, f"Выходная размерность gate_proj должна быть 1024, получено {swiglu_custom.gate_proj.out_features}"
        assert swiglu_custom.value_proj.in_features == 512, f"Входная размерность value_proj должна быть 512, получено {swiglu_custom.value_proj.in_features}"
        assert swiglu_custom.value_proj.out_features == 1024, f"Выходная размерность value_proj должна быть 1024, получено {swiglu_custom.value_proj.out_features}"
        assert swiglu_custom.output_proj.in_features == 1024, f"Входная размерность output_proj должна быть 1024, получено {swiglu_custom.output_proj.in_features}"
        assert swiglu_custom.output_proj.out_features == 256, f"Выходная размерность output_proj должна быть 256, получено {swiglu_custom.output_proj.out_features}"
        
        # Создаем SwiGLU с bias=False
        swiglu_no_bias = SwiGLU(input_dim=512, output_dim=512, bias=False)
        
        # Проверяем, что bias отсутствует в линейных слоях
        assert swiglu_no_bias.gate_proj.bias is None, "bias должен отсутствовать в gate_proj"
        assert swiglu_no_bias.value_proj.bias is None, "bias должен отсутствовать в value_proj"
        assert swiglu_no_bias.output_proj.bias is None, "bias должен отсутствовать в output_proj"

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
        
        # pass
        
        # Создаем SwiGLU с input_dim=512, output_dim=512
        swiglu = SwiGLU(input_dim=512, output_dim=512)
        
        # Создаем тензоры разных размерностей
        # 2D тензор: (batch_size, input_dim)
        x_2d = torch.randn(32, 512)
        # 3D тензор: (batch_size, seq_len, input_dim)
        x_3d = torch.randn(16, 24, 512)
        
        # Применяем SwiGLU к тензорам
        output_2d = swiglu(x_2d)
        output_3d = swiglu(x_3d)
        
        # Проверяем, что выходные формы соответствуют ожидаемым
        assert output_2d.shape == torch.Size([32, 512]), f"Неверная форма выхода для 2D тензора: {output_2d.shape}, ожидается: torch.Size([32, 512])"
        assert output_3d.shape == torch.Size([16, 24, 512]), f"Неверная форма выхода для 3D тензора: {output_3d.shape}, ожидается: torch.Size([16, 24, 512])"
        
        # Проверяем сохранение формы при другой выходной размерности
        swiglu_diff_out = SwiGLU(input_dim=512, output_dim=256)
        output_2d_diff = swiglu_diff_out(x_2d)
        output_3d_diff = swiglu_diff_out(x_3d)
        
        assert output_2d_diff.shape == torch.Size([32, 256]), f"Неверная форма выхода для 2D тензора: {output_2d_diff.shape}, ожидается: torch.Size([32, 256])"
        assert output_3d_diff.shape == torch.Size([16, 24, 256]), f"Неверная форма выхода для 3D тензора: {output_3d_diff.shape}, ожидается: torch.Size([16, 24, 256])"

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
        
        # pass
        
        # Создаем SwiGLU с маленькими размерностями для простоты проверки
        input_dim = 4
        output_dim = 2
        intermediate_dim = 8
        swiglu = SwiGLU(input_dim=input_dim, output_dim=output_dim, intermediate_dim=intermediate_dim)
        
        # Для детерминированного теста, устанавливаем веса вручную
        with torch.no_grad():
            # Устанавливаем простые веса для gate_proj
            swiglu.gate_proj.weight.fill_(0.1)
            if swiglu.gate_proj.bias is not None:
                swiglu.gate_proj.bias.fill_(0.01)
                
            # Устанавливаем простые веса для value_proj
            swiglu.value_proj.weight.fill_(0.2)
            if swiglu.value_proj.bias is not None:
                swiglu.value_proj.bias.fill_(0.02)
                
            # Устанавливаем простые веса для output_proj
            swiglu.output_proj.weight.fill_(0.3)
            if swiglu.output_proj.bias is not None:
                swiglu.output_proj.bias.fill_(0.03)
        
        # Создаем простой тензор с известными значениями
        x = torch.ones(1, input_dim)
        
        # Получаем результат forward pass
        output = swiglu(x)
        
        # Вычисляем ожидаемый результат вручную
        # 1. Применяем gate_proj к входу
        gate_out = swiglu.gate_proj(x)
        # 2. Применяем value_proj к входу
        value_out = swiglu.value_proj(x)
        # 3. Применяем Swish активацию к результату gate_proj
        swish_out = gate_out * torch.sigmoid(gate_out)
        # 4. Поэлементно умножаем результаты
        intermediate = swish_out * value_out
        # 5. Применяем output_proj
        expected_output = swiglu.output_proj(intermediate)
        
        # Сравниваем с результатом forward pass
        assert torch.allclose(output, expected_output), f"Ожидаемый результат: {expected_output}, получено: {output}"

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
        
        # pass
        
        # Задаем размерности
        input_dim = 64
        output_dim = 64
        
        # Создаем SwiGLU
        swiglu = SwiGLU(input_dim=input_dim, output_dim=output_dim)
        
        # Создаем простую линейную модель
        class LinearModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)
                
            def forward(self, x):
                return self.linear(x)
        
        linear_model = LinearModel(input_dim=input_dim, output_dim=output_dim)
        
        # Создаем модель с ReLU
        class ReLUModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                return self.relu(self.linear(x))
        
        relu_model = ReLUModel(input_dim=input_dim, output_dim=output_dim)
        
        # Создаем модель с GELU
        class GELUModel(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)
                self.gelu = nn.GELU()
                
            def forward(self, x):
                return self.gelu(self.linear(x))
        
        gelu_model = GELUModel(input_dim=input_dim, output_dim=output_dim)
        
        # Создаем случайный тензор
        x = torch.randn(10, input_dim)
        
        # Применяем все активации
        swiglu_output = swiglu(x)
        linear_output = linear_model(x)
        relu_output = relu_model(x)
        gelu_output = gelu_model(x)
        
        # Проверяем, что результаты различаются
        assert not torch.allclose(swiglu_output, linear_output), "SwiGLU не должен быть эквивалентен линейному слою"
        assert not torch.allclose(swiglu_output, relu_output), "SwiGLU не должен быть эквивалентен ReLU"
        assert not torch.allclose(swiglu_output, gelu_output), "SwiGLU не должен быть эквивалентен GELU"
        
        # Проверяем, что нелинейности отличаются друг от друга
        assert not torch.allclose(linear_output, relu_output), "Linear не должен быть эквивалентен ReLU"
        assert not torch.allclose(linear_output, gelu_output), "Linear не должен быть эквивалентен GELU"
        assert not torch.allclose(relu_output, gelu_output), "ReLU не должен быть эквивалентен GELU"

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
        
        # pass
        
        # Создаем SwiGLU
        input_dim = 32
        output_dim = 32
        swiglu = SwiGLU(input_dim=input_dim, output_dim=output_dim)
        
        # Создаем тензор с requires_grad=True
        x = torch.randn(5, input_dim, requires_grad=True)
        
        # Применяем SwiGLU активацию
        output = swiglu(x)
        
        # Вычисляем скалярный loss и выполняем backward pass
        loss = output.sum()
        loss.backward()
        
        # Проверяем, что градиенты входного тензора не содержат NaN или Inf
        assert torch.isfinite(x.grad).all(), "Градиенты входного тензора содержат NaN или Inf"
        
        # Проверяем, что градиенты параметров не содержат NaN или Inf
        assert torch.isfinite(swiglu.gate_proj.weight.grad).all(), "Градиенты gate_proj.weight содержат NaN или Inf"
        if swiglu.gate_proj.bias is not None:
            assert torch.isfinite(swiglu.gate_proj.bias.grad).all(), "Градиенты gate_proj.bias содержат NaN или Inf"
            
        assert torch.isfinite(swiglu.value_proj.weight.grad).all(), "Градиенты value_proj.weight содержат NaN или Inf"
        if swiglu.value_proj.bias is not None:
            assert torch.isfinite(swiglu.value_proj.bias.grad).all(), "Градиенты value_proj.bias содержат NaN или Inf"
            
        assert torch.isfinite(swiglu.output_proj.weight.grad).all(), "Градиенты output_proj.weight содержат NaN или Inf"
        if swiglu.output_proj.bias is not None:
            assert torch.isfinite(swiglu.output_proj.bias.grad).all(), "Градиенты output_proj.bias содержат NaN или Inf"
        
        # Проверяем, что градиенты параметров имеют правильные формы
        assert swiglu.gate_proj.weight.grad.shape == swiglu.gate_proj.weight.shape, f"Неверная форма градиентов gate_proj.weight: {swiglu.gate_proj.weight.grad.shape}, ожидается: {swiglu.gate_proj.weight.shape}"
        if swiglu.gate_proj.bias is not None:
            assert swiglu.gate_proj.bias.grad.shape == swiglu.gate_proj.bias.shape, f"Неверная форма градиентов gate_proj.bias: {swiglu.gate_proj.bias.grad.shape}, ожидается: {swiglu.gate_proj.bias.shape}"
            
        assert swiglu.value_proj.weight.grad.shape == swiglu.value_proj.weight.shape, f"Неверная форма градиентов value_proj.weight: {swiglu.value_proj.weight.grad.shape}, ожидается: {swiglu.value_proj.weight.shape}"
        if swiglu.value_proj.bias is not None:
            assert swiglu.value_proj.bias.grad.shape == swiglu.value_proj.bias.shape, f"Неверная форма градиентов value_proj.bias: {swiglu.value_proj.bias.grad.shape}, ожидается: {swiglu.value_proj.bias.shape}"
            
        assert swiglu.output_proj.weight.grad.shape == swiglu.output_proj.weight.shape, f"Неверная форма градиентов output_proj.weight: {swiglu.output_proj.weight.grad.shape}, ожидается: {swiglu.output_proj.weight.shape}"
        if swiglu.output_proj.bias is not None:
            assert swiglu.output_proj.bias.grad.shape == swiglu.output_proj.bias.shape, f"Неверная форма градиентов output_proj.bias: {swiglu.output_proj.bias.grad.shape}, ожидается: {swiglu.output_proj.bias.shape}"

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
        
        # pass
        
        # Создаем SwiGLU с заданными размерностями
        swiglu = SwiGLU(input_dim=input_dim, output_dim=output_dim)
        
        # Проверяем размерности линейных слоев
        assert swiglu.gate_proj.in_features == input_dim, f"Входная размерность gate_proj должна быть {input_dim}, получено {swiglu.gate_proj.in_features}"
        assert swiglu.gate_proj.out_features == 4 * input_dim, f"Выходная размерность gate_proj должна быть {4 * input_dim}, получено {swiglu.gate_proj.out_features}"
        assert swiglu.value_proj.in_features == input_dim, f"Входная размерность value_proj должна быть {input_dim}, получено {swiglu.value_proj.in_features}"
        assert swiglu.value_proj.out_features == 4 * input_dim, f"Выходная размерность value_proj должна быть {4 * input_dim}, получено {swiglu.value_proj.out_features}"
        assert swiglu.output_proj.in_features == 4 * input_dim, f"Входная размерность output_proj должна быть {4 * input_dim}, получено {swiglu.output_proj.in_features}"
        assert swiglu.output_proj.out_features == output_dim, f"Выходная размерность output_proj должна быть {output_dim}, получено {swiglu.output_proj.out_features}"
        
        # Создаем случайные тензоры разных размерностей
        batch_size = 8
        seq_len = 16
        
        # 2D тензор: (batch_size, input_dim)
        x_2d = torch.randn(batch_size, input_dim)
        # 3D тензор: (batch_size, seq_len, input_dim)
        x_3d = torch.randn(batch_size, seq_len, input_dim)
        
        # Применяем SwiGLU активацию
        output_2d = swiglu(x_2d)
        output_3d = swiglu(x_3d)
        
        # Проверяем, что выходные формы соответствуют ожидаемым
        assert output_2d.shape == torch.Size([batch_size, output_dim]), f"Неверная форма выхода для 2D тензора: {output_2d.shape}, ожидается: torch.Size([{batch_size}, {output_dim}])"
        assert output_3d.shape == torch.Size([batch_size, seq_len, output_dim]), f"Неверная форма выхода для 3D тензора: {output_3d.shape}, ожидается: torch.Size([{batch_size}, {seq_len}, {output_dim}])"


# Вопросы для размышления при написании тестов:
# 1. Как убедиться, что механизм гейтинга работает корректно?
# 2. Какие преимущества дает SwiGLU по сравнению с другими активациями?
# 3. Как проверить, что градиенты вычисляются правильно?
# 4. Как влияет параметр beta в Swish на форму активационной функции?
# 5. Почему промежуточная размерность обычно в 4 раза больше выходной?
