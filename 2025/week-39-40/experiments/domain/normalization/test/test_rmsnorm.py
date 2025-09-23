# Стандартная библиотека
import math

# Сторонние библиотеки
import pytest
import torch
import torch.nn as nn

# Локальные импорты
import sys
import os
# Получаем путь к директории с rmsnorm.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from rmsnorm import RMSNorm


class TestRMSNorm:
    """Тесты для проверки корректности RMSNorm реализации."""

    def test_rmsnorm_initialization(self):
        """
        Тест инициализации RMSNorm модуля.

        Проверяет:
        - Корректность сохранения параметров
        - Инициализацию weight единицами
        - Поведение при elementwise_affine=False
        """
        # TODO: Создайте RMSNorm с normalized_shape=512
        # TODO: Проверьте, что normalized_shape сохранился корректно
        # TODO: Проверьте, что eps имеет правильное значение
        # TODO: Проверьте, что weight инициализирован единицами
        # TODO: Проверьте форму weight: должна быть (512,)

        # TODO: Создайте RMSNorm с elementwise_affine=False
        # TODO: Проверьте, что weight равен None

        # pass
        
        # Создаем RMSNorm с normalized_shape=512 и проверяем параметры
        norm = RMSNorm(normalized_shape=512)
        assert norm.normalized_shape == 512, "normalized_shape не сохранился корректно"
        assert norm.eps == 1e-6, "eps не соответствует значению по умолчанию"
        assert isinstance(norm.weight, nn.Parameter), "weight должен быть Parameter"
        assert torch.all(norm.weight == 1.0), "weight должен быть инициализирован единицами"
        assert norm.weight.shape == (512,), f"Форма weight должна быть (512,), получено {norm.weight.shape}"
        
        # Создаем RMSNorm с elementwise_affine=False
        norm_no_affine = RMSNorm(normalized_shape=512, elementwise_affine=False)
        assert norm_no_affine.weight is None, "weight должен быть None при elementwise_affine=False"

    def test_rmsnorm_forward_shape(self):
        """
        Тест сохранения формы тензора при forward pass.

        Проверяет различные формы входов:
        - 2D: (batch_size, hidden_size)
        - 3D: (batch_size, seq_len, hidden_size)
        - 4D: (batch_size, num_heads, seq_len, head_dim)
        """
        rmsnorm = RMSNorm(256)

        # TODO: Тест 2D тензора формы (10, 256)
        # TODO: Проверьте, что выход имеет ту же форму

        # TODO: Тест 3D тензора формы (5, 20, 256)
        # TODO: Проверьте, что выход имеет ту же форму

        # TODO: Тест 4D тензора формы (2, 8, 15, 32) с RMSNorm(32)
        # TODO: Проверьте, что выход имеет ту же форму

        # pass
        
        # Тест 2D тензора формы (10, 256)
        x_2d = torch.randn(10, 256)
        out_2d = rmsnorm(x_2d)
        assert out_2d.shape == x_2d.shape, f"Выход должен иметь ту же форму {x_2d.shape}, получено {out_2d.shape}"
        
        # Тест 3D тензора формы (5, 20, 256)
        x_3d = torch.randn(5, 20, 256)
        out_3d = rmsnorm(x_3d)
        assert out_3d.shape == x_3d.shape, f"Выход должен иметь ту же форму {x_3d.shape}, получено {out_3d.shape}"
        
        # Тест 4D тензора формы (2, 8, 15, 32) с RMSNorm(32)
        rmsnorm_small = RMSNorm(32)
        x_4d = torch.randn(2, 8, 15, 32)
        out_4d = rmsnorm_small(x_4d)
        assert out_4d.shape == x_4d.shape, f"Выход должен иметь ту же форму {x_4d.shape}, получено {out_4d.shape}"

    def test_rmsnorm_mathematical_correctness(self):
        """
        Тест математической корректности RMSNorm.

        Проверяет:
        - Формулу: output = x / sqrt(mean(x²) + eps) * weight
        - RMS значение выхода близко к 1 (при weight=1)
        - Численную стабильность
        """
        eps = 1e-6
        rmsnorm = RMSNorm(4, eps=eps)

        # Простой тест случай
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        # TODO: Примените rmsnorm к x
        # TODO: Вычислите ожидаемый результат вручную:
        #       mean_square = (1² + 2² + 3² + 4²) / 4 = 7.5
        #       rms = sqrt(7.5 + eps)
        #       expected = [1/rms, 2/rms, 3/rms, 4/rms]
        # TODO: Сравните с помощью torch.allclose с толерантностью 1e-5

        # TODO: Проверьте, что RMS выхода близок к 1:
        #       output_rms = sqrt(mean(output²))
        #       assert abs(output_rms - 1.0) < 1e-5

        # pass
        
        # Применяем rmsnorm к x
        output = rmsnorm(x)
        
        # Вычисляем ожидаемый результат вручную
        mean_square = (1**2 + 2**2 + 3**2 + 4**2) / 4  # = 7.5
        rms = torch.sqrt(torch.tensor(mean_square + eps))
        expected = torch.tensor([[1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms]])
        
        # Сравниваем с помощью torch.allclose
        assert torch.allclose(output, expected, rtol=1e-5), \
            f"Ожидалось {expected}, получено {output}"
        
        # Проверяем, что RMS выхода близок к 1
        output_rms = torch.sqrt(torch.mean(output**2))
        assert abs(output_rms.item() - 1.0) < 1e-5, \
            f"RMS выхода должен быть близок к 1, получено {output_rms.item()}"

    def test_rmsnorm_vs_layernorm(self):
        """
        Сравнительный тест RMSNorm vs LayerNorm.

        Показывает ключевые различия:
        - LayerNorm центрирует (среднее ≈ 0)
        - RMSNorm не центрирует (среднее может быть ≠ 0)
        - Оба нормализуют дисперсию
        """
        torch.manual_seed(42)
        x = torch.randn(3, 5, 128)

        rmsnorm = RMSNorm(128)
        layernorm = nn.LayerNorm(128)

        # TODO: Примените обе нормализации к x
        # TODO: Проверьте, что среднее LayerNorm выхода близко к 0
        # TODO: Проверьте, что среднее RMSNorm выхода может быть ≠ 0
        # TODO: Проверьте, что дисперсии обоих выходов близки к 1

        # pass
        
        # Применяем обе нормализации к x
        rms_output = rmsnorm(x)
        ln_output = layernorm(x)
        
        # Вычисляем средние значения по последней оси
        rms_mean = torch.mean(rms_output, dim=-1)
        ln_mean = torch.mean(ln_output, dim=-1)
        
        # Проверяем, что среднее LayerNorm выхода близко к 0
        assert torch.allclose(ln_mean, torch.zeros_like(ln_mean), atol=1e-5), \
            f"Среднее LayerNorm должно быть близко к 0, получено {ln_mean.mean().item()}"
        
        # Проверяем, что среднее RMSNorm выхода может быть ≠ 0
        # Мы не ожидаем, что оно будет точно 0, поэтому проверяем, что оно не близко к 0
        assert not torch.allclose(rms_mean, torch.zeros_like(rms_mean), atol=1e-3), \
            f"Среднее RMSNorm не должно быть близко к 0, получено {rms_mean.mean().item()}"
        
        # Вычисляем дисперсии выходов
        rms_var = torch.var(rms_output, dim=-1)
        ln_var = torch.var(ln_output, dim=-1)
        
        # Проверяем, что дисперсии обоих выходов близки к 1
        assert torch.allclose(rms_var, torch.ones_like(rms_var), atol=1e-1), \
            f"Дисперсия RMSNorm должна быть близка к 1, получено {rms_var.mean().item()}"
        assert torch.allclose(ln_var, torch.ones_like(ln_var), atol=1e-1), \
            f"Дисперсия LayerNorm должна быть близка к 1, получено {ln_var.mean().item()}"

    def test_rmsnorm_gradient_flow(self):
        """
        Тест корректности градиентов через RMSNorm.

        Проверяет:
        - Градиенты weight параметра
        - Градиенты входного тензора
        - Отсутствие NaN или Inf в градиентах
        """
        rmsnorm = RMSNorm(64)
        x = torch.randn(2, 10, 64, requires_grad=True)

        # TODO: Выполните forward pass
        # TODO: Вычислите скалярный loss: loss = output.sum()
        # TODO: Выполните backward pass: loss.backward()

        # TODO: Проверьте, что x.grad не None и не содержит NaN
        # TODO: Проверьте, что rmsnorm.weight.grad не None и не содержит NaN
        # TODO: Проверьте, что градиенты имеют правильные формы

        # pass
        
        # Выполняем forward pass
        output = rmsnorm(x)
        
        # Вычисляем скалярный loss для обратного распространения
        loss = output.sum()
        
        # Выполняем backward pass
        loss.backward()
        
        # Проверяем, что x.grad не None и не содержит NaN или Inf
        assert x.grad is not None, "Градиент для x не должен быть None"
        assert not torch.isnan(x.grad).any(), "Градиент для x не должен содержать NaN"
        assert not torch.isinf(x.grad).any(), "Градиент для x не должен содержать Inf"
        
        # Проверяем, что rmsnorm.weight.grad не None и не содержит NaN или Inf
        assert rmsnorm.weight.grad is not None, "Градиент для weight не должен быть None"
        assert not torch.isnan(rmsnorm.weight.grad).any(), "Градиент для weight не должен содержать NaN"
        assert not torch.isinf(rmsnorm.weight.grad).any(), "Градиент для weight не должен содержать Inf"
        
        # Проверяем, что градиенты имеют правильные формы
        assert x.grad.shape == x.shape, f"Форма градиента x должна быть {x.shape}, получено {x.grad.shape}"
        assert rmsnorm.weight.grad.shape == rmsnorm.weight.shape, \
            f"Форма градиента weight должна быть {rmsnorm.weight.shape}, получено {rmsnorm.weight.grad.shape}"

    def test_rmsnorm_numerical_stability(self):
        """
        Тест численной стабильности RMSNorm.

        Проверяет поведение на:
        - Очень малых значениях
        - Очень больших значениях
        - Нулевых значениях
        - Различных eps значениях
        """
        # TODO: Тест на очень малых значениях (порядка 1e-8)
        # TODO: Проверьте, что выход не содержит NaN или Inf

        # TODO: Тест на очень больших значениях (порядка 1e8)
        # TODO: Проверьте численную стабильность

        # TODO: Тест на нулевом тензоре
        # TODO: Проверьте, что eps предотвращает деление на ноль

        # TODO: Сравните поведение с разными eps (1e-6, 1e-8, 1e-4)

        pass
        
        # Тест на очень малых значениях (порядка 1e-8)
        rmsnorm = RMSNorm(32)
        x_small = torch.ones(5, 32) * 1e-8
        output_small = rmsnorm(x_small)
        
        # Проверяем, что выход не содержит NaN или Inf
        assert not torch.isnan(output_small).any(), "Выход не должен содержать NaN для малых значений"
        assert not torch.isinf(output_small).any(), "Выход не должен содержать Inf для малых значений"
        
        # Тест на очень больших значениях (порядка 1e8)
        x_large = torch.ones(5, 32) * 1e8
        output_large = rmsnorm(x_large)
        
        # Проверяем численную стабильность
        assert not torch.isnan(output_large).any(), "Выход не должен содержать NaN для больших значений"
        assert not torch.isinf(output_large).any(), "Выход не должен содержать Inf для больших значений"
        
        # Проверяем, что нормализация работает правильно даже для больших значений
        assert torch.allclose(output_large, torch.ones_like(output_large), atol=1e-5), \
            "Для однородных больших входов выход должен быть близок к единичному вектору"
        
        # Тест на нулевом тензоре
        x_zero = torch.zeros(5, 32)
        output_zero = rmsnorm(x_zero)
        
        # Проверяем, что eps предотвращает деление на ноль
        assert not torch.isnan(output_zero).any(), "Выход не должен содержать NaN для нулевого тензора"
        assert not torch.isinf(output_zero).any(), "Выход не должен содержать Inf для нулевого тензора"
        
        # Сравниваем поведение с разными eps
        eps_values = [1e-6, 1e-8, 1e-4]
        x_test = torch.randn(5, 32)
        
        outputs = []
        for eps in eps_values:
            rmsnorm_eps = RMSNorm(32, eps=eps)
            outputs.append(rmsnorm_eps(x_test))
        
        # Проверяем, что все выходы не содержат NaN или Inf
        for i, output in enumerate(outputs):
            assert not torch.isnan(output).any(), f"Выход с eps={eps_values[i]} не должен содержать NaN"
            assert not torch.isinf(output).any(), f"Выход с eps={eps_values[i]} не должен содержать Inf"
        
        # Проверяем, что разные eps дают немного разные результаты
        # Но не слишком разные (должны быть похожи)
        assert not torch.allclose(outputs[0], outputs[2], atol=1e-5), \
            "Выходы с разными eps должны немного отличаться"

    def test_rmsnorm_without_elementwise_affine(self):
        """
        Тест RMSNorm без обучаемых параметров.

        При elementwise_affine=False weight должен быть None,
        и применяется только нормализация без масштабирования.
        """
        rmsnorm = RMSNorm(32, elementwise_affine=False)
        x = torch.randn(5, 32)

        # TODO: Проверьте, что weight равен None
        # TODO: Выполните forward pass
        # TODO: Проверьте, что RMS выхода близок к 1
        # TODO: Убедитесь, что нет обучаемых параметров

        # pass
        
        # Проверяем, что weight равен None
        assert rmsnorm.weight is None, "weight должен быть None при elementwise_affine=False"
        
        # Выполняем forward pass
        output = rmsnorm(x)
        
        # Проверяем, что RMS выхода близок к 1
        output_rms = torch.sqrt(torch.mean(output**2, dim=-1))
        assert torch.allclose(output_rms, torch.ones_like(output_rms), atol=1e-5), \
            f"RMS выхода должен быть близок к 1, получено {output_rms.mean().item()}"
        
        # Убеждаемся, что нет обучаемых параметров
        assert sum(p.numel() for p in rmsnorm.parameters() if p.requires_grad) == 0, \
            "При elementwise_affine=False не должно быть обучаемых параметров"
        
        # Дополнительно проверяем, что выход отличается от входа (нормализация произошла)
        assert not torch.allclose(x, output), \
            "Выход должен отличаться от входа после нормализации"

    @pytest.mark.parametrize("normalized_shape", [64, 128, 512, 1024])
    def test_rmsnorm_different_sizes(self, normalized_shape):
        """
        Параметризованный тест для различных размеров normalized_shape.

        Проверяет корректность работы RMSNorm с разными размерностями.
        """
        # TODO: Создайте RMSNorm с заданным normalized_shape
        # TODO: Создайте случайный тензор подходящей формы
        # TODO: Проверьте корректность forward pass
        # TODO: Проверьте, что RMS выхода близок к 1

        # pass
        
        # Создаем RMSNorm с заданным normalized_shape
        rmsnorm = RMSNorm(normalized_shape)
        
        # Создаем случайный тензор подходящей формы
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, normalized_shape)
        
        # Проверяем корректность forward pass
        output = rmsnorm(x)
        assert output.shape == x.shape, f"Форма выхода должна быть {x.shape}, получено {output.shape}"
        
        # Проверяем, что RMS выхода близок к 1
        output_rms = torch.sqrt(torch.mean(output**2, dim=-1))
        assert torch.allclose(output_rms, torch.ones_like(output_rms), atol=1e-5), \
            f"RMS выхода должен быть близок к 1 для normalized_shape={normalized_shape}, " \
            f"получено {output_rms.mean().item()}"
        
        # Дополнительно проверяем, что параметры имеют правильную форму
        assert rmsnorm.weight.shape == (normalized_shape,), \
            f"Форма weight должна быть ({normalized_shape},), получено {rmsnorm.weight.shape}"


# Вопросы для размышления при написании тестов:
# 1. Как убедиться, что реализация математически корректна?
# 2. Какие крайние случаи важно протестировать?
# 3. Как проверить, что градиенты вычисляются правильно?
# 4. В чем ключевые различия между RMSNorm и LayerNorm?
# 5. Как eps влияет на численную стабильность?