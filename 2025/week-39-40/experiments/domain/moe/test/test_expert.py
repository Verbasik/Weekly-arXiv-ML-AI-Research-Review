# Стандартная библиотека
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь поиска модулей
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Сторонние библиотеки
import pytest
import torch

# Тестируемый модуль
from experiments.domain.moe.expert import Expert


class TestExpertInitialization:
    """Тесты инициализации Expert Network"""

    def test_basic_initialization(self):
        """Тест базовой инициализации эксперта"""
        expert = Expert(hidden_size=512, intermediate_size=2048)

        assert expert.hidden_size == 512
        assert expert.intermediate_size == 2048
        assert expert.dropout_prob == 0.0
        assert hasattr(expert, 'ffn')
        assert hasattr(expert, 'dropout')

    def test_custom_dropout(self):
        """Тест инициализации с кастомным dropout"""
        expert = Expert(hidden_size=512, intermediate_size=2048, dropout=0.1)

        assert expert.dropout_prob == 0.1

    def test_parameter_validation(self):
        """Тест валидации параметров"""
        # Невалидный hidden_size
        with pytest.raises(AssertionError):
            Expert(hidden_size=-1, intermediate_size=2048)

        # Невалидный intermediate_size
        with pytest.raises(AssertionError):
            Expert(hidden_size=512, intermediate_size=0)

        # Невалидный dropout
        with pytest.raises(AssertionError):
            Expert(hidden_size=512, intermediate_size=2048, dropout=1.5)


class TestExpertForward:
    """Тесты forward pass Expert Network"""

    def test_forward_output_shape(self):
        """Тест размерности выхода"""
        expert = Expert(hidden_size=512, intermediate_size=2048)
        x = torch.randn(2, 10, 512)  # (batch=2, seq=10, hidden=512)

        output = expert(x)

        assert output.shape == (2, 10, 512), \
            f"Expected shape (2, 10, 512), got {output.shape}"

    def test_forward_different_batch_sizes(self):
        """Тест с разными batch sizes"""
        expert = Expert(hidden_size=512, intermediate_size=2048)

        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 10, 512)
            output = expert(x)
            assert output.shape == (batch_size, 10, 512)

    def test_forward_different_sequence_lengths(self):
        """Тест с разными sequence lengths"""
        expert = Expert(hidden_size=512, intermediate_size=2048)

        for seq_len in [1, 10, 50, 128]:
            x = torch.randn(2, seq_len, 512)
            output = expert(x)
            assert output.shape == (2, seq_len, 512)

    def test_forward_preserves_dtype(self):
        """Тест сохранения типа данных"""
        expert = Expert(hidden_size=512, intermediate_size=2048)

        # Float32
        x_f32 = torch.randn(2, 10, 512, dtype=torch.float32)
        output_f32 = expert(x_f32)
        assert output_f32.dtype == torch.float32

        # Float64
        expert_f64 = Expert(hidden_size=512, intermediate_size=2048).double()
        x_f64 = torch.randn(2, 10, 512, dtype=torch.float64)
        output_f64 = expert_f64(x_f64)
        assert output_f64.dtype == torch.float64


class TestExpertGradientFlow:
    """Тесты градиентного потока через эксперта"""

    def test_gradient_flow(self):
        """Тест распространения градиентов"""
        expert = Expert(hidden_size=512, intermediate_size=2048)
        x = torch.randn(2, 10, 512, requires_grad=True)

        output = expert(x)
        loss = output.sum()
        loss.backward()

        # Проверяем, что градиенты вычислены
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_parameters_require_grad(self):
        """Тест, что параметры требуют градиенты"""
        expert = Expert(hidden_size=512, intermediate_size=2048)

        for name, param in expert.named_parameters():
            assert param.requires_grad, f"Parameter {name} does not require grad"


class TestExpertNumericalStability:
    """Тесты численной стабильности"""

    def test_numerical_stability_large_values(self):
        """Тест стабильности с большими значениями"""
        expert = Expert(hidden_size=512, intermediate_size=2048)
        x = torch.randn(2, 10, 512) * 100  # Большие значения

        output = expert(x)

        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_numerical_stability_small_values(self):
        """Тест стабильности с маленькими значениями"""
        expert = Expert(hidden_size=512, intermediate_size=2048)
        x = torch.randn(2, 10, 512) * 0.001  # Маленькие значения

        output = expert(x)

        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


class TestExpertDropout:
    """Тесты dropout функциональности"""

    def test_dropout_in_training_mode(self):
        """Тест, что dropout работает в режиме обучения"""
        torch.manual_seed(42)
        expert = Expert(hidden_size=512, intermediate_size=2048, dropout=0.5)
        expert.train()

        x = torch.randn(2, 10, 512)

        # Два forward pass должны дать разные результаты из-за dropout
        output1 = expert(x)
        output2 = expert(x)

        # С dropout=0.5 результаты должны отличаться
        assert not torch.allclose(output1, output2), \
            "Dropout should produce different outputs in training mode"

    def test_no_dropout_in_eval_mode(self):
        """Тест, что dropout не работает в режиме eval"""
        torch.manual_seed(42)
        expert = Expert(hidden_size=512, intermediate_size=2048, dropout=0.5)
        expert.eval()

        x = torch.randn(2, 10, 512)

        # В eval режиме результаты должны быть одинаковыми
        output1 = expert(x)
        output2 = expert(x)

        assert torch.allclose(output1, output2), \
            "Outputs should be identical in eval mode"


class TestExpertIntegration:
    """Интеграционные тесты"""

    def test_multiple_experts_independent(self):
        """Тест, что множественные эксперты независимы"""
        num_experts = 8
        experts = torch.nn.ModuleList([
            Expert(hidden_size=512, intermediate_size=2048)
            for _ in range(num_experts)
        ])

        x = torch.randn(2, 10, 512)

        outputs = [expert(x) for expert in experts]

        # Каждый эксперт должен дать разный результат
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                assert not torch.allclose(outputs[i], outputs[j]), \
                    f"Expert {i} and {j} produced identical outputs"

    def test_deterministic_output(self):
        """Тест детерминированности выхода"""
        torch.manual_seed(42)
        expert1 = Expert(hidden_size=512, intermediate_size=2048)
        expert1.eval()

        torch.manual_seed(42)
        expert2 = Expert(hidden_size=512, intermediate_size=2048)
        expert2.eval()

        x = torch.randn(2, 10, 512)

        output1 = expert1(x)
        output2 = expert2(x)

        assert torch.allclose(output1, output2), \
            "Identical experts with same seed should produce identical outputs"
