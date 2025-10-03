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
from experiments.domain.transformer.moe_transformer_block import MoETransformerBlock


class TestMoETransformerBlockInitialization:
    """Тесты инициализации MoETransformerBlock"""

    def test_basic_initialization(self):
        """Тест базовой инициализации"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16,
            num_experts=8,
            top_k=2
        )

        assert block.hidden_size == 512
        assert block.num_query_groups == 8
        assert block.num_attention_heads == 16
        assert block.num_experts == 8
        assert block.top_k == 2
        assert hasattr(block, 'attention_norm')
        assert hasattr(block, 'attention')
        assert hasattr(block, 'ffn_norm')
        assert hasattr(block, 'moe_layer')

    def test_intermediate_size_calculation(self):
        """Тест автоматического вычисления intermediate_size"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )

        assert block.intermediate_size == 2048  # 4 * 512

    def test_custom_intermediate_size(self):
        """Тест с кастомным intermediate_size"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16,
            intermediate_size=1024
        )

        assert block.intermediate_size == 1024

    def test_parameter_validation(self):
        """Тест валидации параметров"""
        # Невалидный hidden_size
        with pytest.raises(AssertionError):
            MoETransformerBlock(
                hidden_size=-1,
                num_query_groups=8,
                num_attention_heads=16
            )

        # num_attention_heads не делится на num_query_groups
        with pytest.raises(AssertionError):
            MoETransformerBlock(
                hidden_size=512,
                num_query_groups=7,
                num_attention_heads=16
            )

        # top_k больше num_experts
        with pytest.raises(AssertionError):
            MoETransformerBlock(
                hidden_size=512,
                num_query_groups=8,
                num_attention_heads=16,
                num_experts=4,
                top_k=5
            )


class TestMoETransformerBlockForward:
    """Тесты forward pass"""

    def test_forward_output_shape(self):
        """Тест размерности выхода"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )
        block.eval()

        x = torch.randn(2, 10, 512)
        output, balance_loss = block(x, training=False)

        assert output.shape == (2, 10, 512), \
            f"Expected shape (2, 10, 512), got {output.shape}"

    def test_forward_returns_balance_loss(self):
        """Тест возврата balance_loss"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )

        x = torch.randn(2, 10, 512)
        output, balance_loss = block(x, training=True)

        assert isinstance(balance_loss, torch.Tensor)
        assert balance_loss.ndim == 0  # Скаляр
        assert balance_loss.item() >= 0.0

    def test_forward_with_training_mode(self):
        """Тест forward в режиме training"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )
        block.train()

        x = torch.randn(2, 10, 512)
        output, balance_loss = block(x, training=True)

        assert output.shape == (2, 10, 512)
        assert balance_loss.item() >= 0.0  # balance_loss должен быть неотрицательным

    def test_forward_with_inference_mode(self):
        """Тест forward в режиме inference"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )
        block.eval()

        x = torch.randn(2, 10, 512)
        output, balance_loss = block(x, training=False)

        assert output.shape == (2, 10, 512)
        assert balance_loss.item() == 0.0  # balance_loss должен быть 0 в inference

    def test_different_batch_sizes(self):
        """Тест с разными batch sizes"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )
        block.eval()

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 10, 512)
            output, balance_loss = block(x, training=False)
            assert output.shape == (batch_size, 10, 512)

    def test_different_sequence_lengths(self):
        """Тест с разными sequence lengths"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )
        block.eval()

        for seq_len in [1, 10, 50]:
            x = torch.randn(2, seq_len, 512)
            output, balance_loss = block(x, training=False)
            assert output.shape == (2, seq_len, 512)


class TestMoETransformerBlockResidualConnections:
    """Тесты residual connections"""

    def test_residual_connections_exist(self):
        """Тест наличия residual connections"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )
        block.eval()

        x = torch.randn(2, 10, 512)
        output, _ = block(x, training=False)

        # Residual connections должны влиять на результат
        assert not torch.allclose(output, torch.zeros_like(output)), \
            "Output не должен быть нулевым благодаря residual connections"


class TestMoETransformerBlockGradientFlow:
    """Тесты градиентного потока"""

    def test_gradient_flow(self):
        """Тест распространения градиентов"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )
        x = torch.randn(2, 10, 512, requires_grad=True)

        output, balance_loss = block(x, training=True)
        total_loss = output.sum() + balance_loss
        total_loss.backward()

        # Проверяем градиенты входа
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_balance_loss_gradient_flow(self):
        """Тест распространения градиентов через balance_loss"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )
        x = torch.randn(2, 10, 512)

        output, balance_loss = block(x, training=True)
        balance_loss.backward()

        # Проверяем, что градиенты есть у MoE Router
        for param in block.moe_layer.router.parameters():
            assert param.grad is not None or not param.requires_grad


class TestMoETransformerBlockNumericalStability:
    """Тесты численной стабильности"""

    def test_numerical_stability_large_values(self):
        """Тест с большими значениями"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )
        block.eval()

        x = torch.randn(2, 10, 512) * 100
        output, balance_loss = block(x, training=False)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_numerical_stability_small_values(self):
        """Тест с маленькими значениями"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )
        block.eval()

        x = torch.randn(2, 10, 512) * 0.001
        output, balance_loss = block(x, training=False)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestMoETransformerBlockDeterminism:
    """Тесты детерминированности"""

    def test_deterministic_output(self):
        """Тест детерминированности с фиксированным seed"""
        torch.manual_seed(42)
        block1 = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )
        block1.eval()

        torch.manual_seed(42)
        block2 = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )
        block2.eval()

        x = torch.randn(2, 10, 512)

        output1, _ = block1(x, training=False)
        output2, _ = block2(x, training=False)

        assert torch.allclose(output1, output2), \
            "Identical models with same seed should produce identical outputs"


class TestMoETransformerBlockWithCache:
    """Тесты с KV cache"""

    def test_output_with_use_cache(self):
        """Тест возврата с use_cache=True"""
        block = MoETransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )
        block.eval()

        x = torch.randn(2, 10, 512)
        result = block(x, use_cache=True, training=False)

        # Должен вернуть tuple: (hidden_states, balance_loss, present_key_value, attn_weights)
        assert isinstance(result, tuple)
        assert len(result) == 4
        hidden_states, balance_loss, present_kv, attn_weights = result
        assert hidden_states.shape == (2, 10, 512)
