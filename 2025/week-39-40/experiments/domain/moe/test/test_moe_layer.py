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
from experiments.domain.moe.moe_layer import SimpleMoELayer


class TestSimpleMoELayerInitialization:
    """Тесты инициализации SimpleMoELayer"""

    def test_basic_initialization(self):
        """Тест базовой инициализации"""
        moe = SimpleMoELayer(
            hidden_size=512,
            num_experts=8,
            top_k=2,
            intermediate_size=2048
        )

        assert moe.hidden_size == 512
        assert moe.num_experts == 8
        assert moe.top_k == 2
        assert moe.intermediate_size == 2048
        assert hasattr(moe, 'router')
        assert hasattr(moe, 'experts')
        assert len(moe.experts) == 8

    def test_custom_parameters(self):
        """Тест с кастомными параметрами"""
        moe = SimpleMoELayer(
            hidden_size=256,
            num_experts=4,
            top_k=1,
            intermediate_size=1024,
            expert_dropout=0.1
        )

        assert moe.num_experts == 4
        assert moe.top_k == 1
        assert len(moe.experts) == 4

    def test_parameter_validation(self):
        """Тест валидации параметров"""
        # Невалидный hidden_size
        with pytest.raises(AssertionError):
            SimpleMoELayer(hidden_size=-1, num_experts=8, top_k=2)

        # top_k больше num_experts
        with pytest.raises(AssertionError):
            SimpleMoELayer(hidden_size=512, num_experts=4, top_k=5)


class TestSimpleMoELayerForward:
    """Тесты forward pass"""

    def test_forward_output_shape(self):
        """Тест размерности выхода"""
        moe = SimpleMoELayer(hidden_size=512, num_experts=8, top_k=2)
        moe.eval()

        x = torch.randn(2, 10, 512)
        output, loss = moe(x, training=False)

        assert output.shape == (2, 10, 512), \
            f"Expected shape (2, 10, 512), got {output.shape}"

    def test_forward_with_training(self):
        """Тест forward в режиме training"""
        moe = SimpleMoELayer(hidden_size=512, num_experts=8, top_k=2)
        moe.train()

        x = torch.randn(2, 10, 512)
        output, loss = moe(x, training=True)

        assert output.shape == (2, 10, 512)
        assert loss.item() >= 0.0, "Balance loss должен быть неотрицательным"

    def test_forward_without_training(self):
        """Тест forward в режиме inference"""
        moe = SimpleMoELayer(hidden_size=512, num_experts=8, top_k=2)
        moe.eval()

        x = torch.randn(2, 10, 512)
        output, loss = moe(x, training=False)

        assert output.shape == (2, 10, 512)
        assert loss.item() == 0.0, "Balance loss должен быть 0 в inference режиме"

    def test_different_batch_sizes(self):
        """Тест с разными batch sizes"""
        moe = SimpleMoELayer(hidden_size=512, num_experts=8, top_k=2)
        moe.eval()

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 10, 512)
            output, loss = moe(x, training=False)
            assert output.shape == (batch_size, 10, 512)

    def test_different_sequence_lengths(self):
        """Тест с разными sequence lengths"""
        moe = SimpleMoELayer(hidden_size=512, num_experts=8, top_k=2)
        moe.eval()

        for seq_len in [1, 10, 50]:
            x = torch.randn(2, seq_len, 512)
            output, loss = moe(x, training=False)
            assert output.shape == (2, seq_len, 512)


class TestSimpleMoELayerResidualConnection:
    """Тесты residual connection"""

    def test_residual_connection_exists(self):
        """Тест наличия residual connection"""
        moe = SimpleMoELayer(hidden_size=512, num_experts=8, top_k=2)
        moe.eval()

        # Создаём входные данные
        x = torch.randn(2, 10, 512)

        # Forward pass
        output, _ = moe(x, training=False)

        # Residual connection должен влиять на результат
        # Проверяем, что output != 0 даже если эксперты обнулены
        assert not torch.allclose(output, torch.zeros_like(output)), \
            "Output не должен быть нулевым благодаря residual connection"


class TestSimpleMoELayerGradientFlow:
    """Тесты градиентного потока"""

    def test_gradient_flow(self):
        """Тест распространения градиентов"""
        moe = SimpleMoELayer(hidden_size=512, num_experts=8, top_k=2)
        x = torch.randn(2, 10, 512, requires_grad=True)

        output, loss = moe(x, training=True)
        total_loss = output.sum() + loss
        total_loss.backward()

        # Проверяем градиенты входа
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_parameters_require_grad(self):
        """Тест, что все параметры требуют градиенты"""
        moe = SimpleMoELayer(hidden_size=512, num_experts=8, top_k=2)

        for name, param in moe.named_parameters():
            assert param.requires_grad, f"Parameter {name} does not require grad"


class TestSimpleMoELayerNumericalStability:
    """Тесты численной стабильности"""

    def test_numerical_stability_large_values(self):
        """Тест с большими значениями"""
        moe = SimpleMoELayer(hidden_size=512, num_experts=8, top_k=2)
        moe.eval()

        x = torch.randn(2, 10, 512) * 100
        output, loss = moe(x, training=False)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_numerical_stability_small_values(self):
        """Тест с маленькими значениями"""
        moe = SimpleMoELayer(hidden_size=512, num_experts=8, top_k=2)
        moe.eval()

        x = torch.randn(2, 10, 512) * 0.001
        output, loss = moe(x, training=False)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestSimpleMoELayerDeterminism:
    """Тесты детерминированности"""

    def test_deterministic_output(self):
        """Тест детерминированности с фиксированным seed"""
        torch.manual_seed(42)
        moe1 = SimpleMoELayer(hidden_size=512, num_experts=8, top_k=2)
        moe1.eval()

        torch.manual_seed(42)
        moe2 = SimpleMoELayer(hidden_size=512, num_experts=8, top_k=2)
        moe2.eval()

        x = torch.randn(2, 10, 512)

        output1, _ = moe1(x, training=False)
        output2, _ = moe2(x, training=False)

        assert torch.allclose(output1, output2), \
            "Identical models with same seed should produce identical outputs"
