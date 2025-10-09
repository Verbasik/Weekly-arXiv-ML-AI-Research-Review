"""
Tests for Qwen3 MoE Model

Проверяет полную языковую модель с генерацией текста.
"""
import pytest
import torch
import torch.nn as nn

from experiments.domain.model.config import Qwen3Config
from experiments.domain.model.qwen3_model import Qwen3MoEModel


@pytest.fixture
def small_config():
    """Маленькая конфигурация для быстрых тестов."""
    return Qwen3Config(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,  # Только 2 слоя для скорости
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,  # Меньше экспертов
        top_k=2,
        max_position_embeddings=128,
    )


@pytest.fixture
def model(small_config):
    """Модель для тестирования."""
    torch.manual_seed(42)
    return Qwen3MoEModel(small_config)


# ================================
# Initialization Tests
# ================================


class TestInitialization:
    """Тесты инициализации модели."""

    def test_model_creation(self, small_config):
        """Тест: модель создаётся без ошибок."""
        model = Qwen3MoEModel(small_config)
        assert isinstance(model, nn.Module)

    def test_embedding_layer(self, model, small_config):
        """Тест: embedding layer правильно инициализирован."""
        assert hasattr(model, "embed_tokens")
        assert isinstance(model.embed_tokens, nn.Embedding)
        assert model.embed_tokens.num_embeddings == small_config.vocab_size
        assert model.embed_tokens.embedding_dim == small_config.hidden_size

    def test_transformer_layers(self, model, small_config):
        """Тест: правильное количество transformer блоков."""
        assert hasattr(model, "layers")
        assert isinstance(model.layers, nn.ModuleList)
        assert len(model.layers) == small_config.num_layers

    def test_final_norm(self, model):
        """Тест: финальная нормализация присутствует."""
        assert hasattr(model, "norm")
        # RMSNorm не имеет стандартного базового класса, проверяем метод forward
        assert callable(model.norm.forward)

    def test_lm_head(self, model, small_config):
        """Тест: LM head правильно инициализирован."""
        assert hasattr(model, "lm_head")
        assert isinstance(model.lm_head, nn.Linear)
        assert model.lm_head.in_features == small_config.hidden_size
        assert model.lm_head.out_features == small_config.vocab_size
        assert model.lm_head.bias is None  # bias=False


# ================================
# Forward Pass Tests
# ================================


class TestForwardPass:
    """Тесты forward pass."""

    def test_forward_shape(self, model, small_config):
        """Тест: forward возвращает правильные размерности."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))

        logits, balance_loss = model(input_ids)

        # Проверка размерностей
        assert logits.shape == (batch_size, seq_len, small_config.vocab_size)
        assert balance_loss.ndim == 0  # Скаляр

    def test_forward_different_batch_sizes(self, model, small_config):
        """Тест: forward работает с разными batch sizes."""
        for batch_size in [1, 4, 8]:
            seq_len = 10
            input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
            logits, _ = model(input_ids)
            assert logits.shape == (batch_size, seq_len, small_config.vocab_size)

    def test_forward_different_seq_lengths(self, model, small_config):
        """Тест: forward работает с разными длинами последовательностей."""
        batch_size = 2
        for seq_len in [5, 20, 50]:
            if seq_len > small_config.max_position_embeddings:
                continue
            input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
            logits, _ = model(input_ids)
            assert logits.shape == (batch_size, seq_len, small_config.vocab_size)

    def test_balance_loss_is_positive(self, model, small_config):
        """Тест: balance loss неотрицательный."""
        input_ids = torch.randint(0, small_config.vocab_size, (2, 10))
        _, balance_loss = model(input_ids)
        assert balance_loss.item() >= 0

    def test_balance_loss_accumulation(self, model, small_config):
        """Тест: balance loss накапливается из всех слоёв."""
        input_ids = torch.randint(0, small_config.vocab_size, (2, 10))
        _, total_loss = model(input_ids)

        # Balance loss должен быть > 0 если есть MoE слои
        assert total_loss.item() > 0

    def test_output_dtype(self, model, small_config):
        """Тест: выходные тензоры имеют правильный dtype."""
        input_ids = torch.randint(0, small_config.vocab_size, (2, 10))
        logits, balance_loss = model(input_ids)

        assert logits.dtype == torch.float32
        assert balance_loss.dtype == torch.float32


# ================================
# Gradient Flow Tests
# ================================


class TestGradientFlow:
    """Тесты прохождения градиентов."""

    def test_gradients_flow_through_model(self, model, small_config):
        """Тест: градиенты проходят через всю модель."""
        input_ids = torch.randint(0, small_config.vocab_size, (2, 10))
        logits, balance_loss = model(input_ids)

        # Dummy loss для backward
        loss = logits.sum() + balance_loss
        loss.backward()

        # Проверяем, что градиенты есть в ключевых слоях
        assert model.embed_tokens.weight.grad is not None
        assert model.lm_head.weight.grad is not None

    def test_embedding_gradients(self, model, small_config):
        """Тест: градиенты проходят через embedding."""
        input_ids = torch.randint(0, small_config.vocab_size, (2, 10))
        logits, _ = model(input_ids)

        loss = logits.sum()
        loss.backward()

        assert model.embed_tokens.weight.grad is not None
        # Только используемые embeddings должны иметь ненулевые градиенты
        assert model.embed_tokens.weight.grad.abs().sum() > 0


# ================================
# Numerical Stability Tests
# ================================


class TestNumericalStability:
    """Тесты численной стабильности."""

    def test_no_nan_in_output(self, model, small_config):
        """Тест: нет NaN в выходах."""
        input_ids = torch.randint(0, small_config.vocab_size, (2, 10))
        logits, balance_loss = model(input_ids)

        assert not torch.isnan(logits).any()
        assert not torch.isnan(balance_loss).any()

    def test_no_inf_in_output(self, model, small_config):
        """Тест: нет Inf в выходах."""
        input_ids = torch.randint(0, small_config.vocab_size, (2, 10))
        logits, balance_loss = model(input_ids)

        assert not torch.isinf(logits).any()
        assert not torch.isinf(balance_loss).any()


# ================================
# Determinism Tests
# ================================


class TestDeterminism:
    """Тесты детерминизма."""

    def test_deterministic_forward(self, small_config):
        """Тест: одинаковый input → одинаковый output."""
        torch.manual_seed(42)
        model1 = Qwen3MoEModel(small_config)
        model1.eval()

        torch.manual_seed(42)
        model2 = Qwen3MoEModel(small_config)
        model2.eval()

        input_ids = torch.randint(0, small_config.vocab_size, (2, 10))

        logits1, loss1 = model1(input_ids)
        logits2, loss2 = model2(input_ids)

        assert torch.allclose(logits1, logits2, atol=1e-6)
        assert torch.allclose(loss1, loss2, atol=1e-6)


# ================================
# Edge Cases Tests
# ================================


class TestEdgeCases:
    """Тесты граничных случаев."""

    def test_single_token_input(self, model, small_config):
        """Тест: работа с одним токеном."""
        input_ids = torch.randint(0, small_config.vocab_size, (1, 1))
        logits, balance_loss = model(input_ids)

        assert logits.shape == (1, 1, small_config.vocab_size)
        assert balance_loss.item() >= 0

    def test_max_sequence_length(self, model, small_config):
        """Тест: максимальная длина последовательности."""
        max_len = small_config.max_position_embeddings
        input_ids = torch.randint(0, small_config.vocab_size, (1, max_len))

        logits, balance_loss = model(input_ids)
        assert logits.shape == (1, max_len, small_config.vocab_size)

    def test_batch_size_one(self, model, small_config):
        """Тест: batch size = 1."""
        input_ids = torch.randint(0, small_config.vocab_size, (1, 10))
        logits, balance_loss = model(input_ids)

        assert logits.shape == (1, 10, small_config.vocab_size)
