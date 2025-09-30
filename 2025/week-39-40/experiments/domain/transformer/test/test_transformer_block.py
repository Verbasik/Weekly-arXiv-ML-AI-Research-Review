"""
Тесты для TransformerBlock - базового блока архитектуры Qwen3.

Эти тесты проверяют:
1. Правильность инициализации и размерностей
2. Forward pass с различными конфигурациями
3. Интеграцию всех компонентов (RMSNorm, GQA, SwiGLU)
4. Градиентный поток через residual connections
5. Кэширование и attention weights
6. Совместимость с различными размерами batch и последовательностей
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Добавляем родительскую директорию в путь для импорта
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from transformer_block import TransformerBlock


class TestTransformerBlockInitialization:
    """Тесты инициализации TransformerBlock"""

    def test_basic_initialization(self):
        """Тест базовой инициализации с минимальными параметрами"""
        block = TransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )

        # Проверяем основные атрибуты
        assert block.hidden_size == 512
        assert block.num_query_groups == 8
        assert block.num_attention_heads == 16
        assert block.intermediate_size == 4 * 512  # Автоматически вычисленное значение

        # Проверяем создание всех компонентов
        assert hasattr(block, 'attention_norm')
        assert hasattr(block, 'attention')
        assert hasattr(block, 'ffn_norm')
        assert hasattr(block, 'feed_forward')

    def test_custom_intermediate_size(self):
        """Тест с заданным intermediate_size"""
        block = TransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16,
            intermediate_size=1024
        )

        assert block.intermediate_size == 1024

    def test_parameter_validation(self):
        """Тест валидации параметров"""
        # Тест невалидных значений
        with pytest.raises(AssertionError):
            TransformerBlock(
                hidden_size=0,  # Невалидное значение
                num_query_groups=8,
                num_attention_heads=16
            )

        with pytest.raises(AssertionError):
            TransformerBlock(
                hidden_size=512,
                num_query_groups=8,
                num_attention_heads=15  # Не делится на num_query_groups
            )


class TestTransformerBlockForward:
    """Тесты forward pass TransformerBlock"""

    @pytest.fixture
    def block(self):
        """Создает стандартный TransformerBlock для тестов"""
        return TransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16,
            intermediate_size=2048
        )

    def test_forward_shape(self, block):
        """Тест корректности размерностей выхода"""
        batch_size, seq_len, hidden_size = 2, 10, 512
        x = torch.randn(batch_size, seq_len, hidden_size)

        output = block(x)

        # Проверяем размерности
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert output.dtype == x.dtype
        assert output.device == x.device

    def test_forward_with_attention_mask(self, block):
        """Тест forward с attention mask"""
        batch_size, seq_len, hidden_size = 2, 10, 512
        x = torch.randn(batch_size, seq_len, hidden_size)

        # Создаем causal mask
        attention_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        output = block(x, attention_mask=attention_mask)

        assert output.shape == (batch_size, seq_len, hidden_size)

    def test_forward_with_cache(self, block):
        """Тест forward с кэшированием"""
        batch_size, seq_len, hidden_size = 2, 5, 512
        x = torch.randn(batch_size, seq_len, hidden_size)

        # Первый проход с кэшированием
        output, past_key_value, _ = block(x, use_cache=True)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert past_key_value is not None
        assert len(past_key_value) == 2  # key и value

        # Второй проход с кэшем
        new_x = torch.randn(batch_size, 3, hidden_size)  # Новые токены
        output2, past_key_value2, _ = block(
            new_x,
            past_key_value=past_key_value,
            use_cache=True
        )

        assert output2.shape == (batch_size, 3, hidden_size)
        assert past_key_value2 is not None

    def test_forward_with_attention_weights(self, block):
        """Тест вывода attention weights"""
        batch_size, seq_len, hidden_size = 2, 8, 512
        x = torch.randn(batch_size, seq_len, hidden_size)

        output, _, attention_weights = block(x, output_attentions=True)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert attention_weights is not None
        # Attention weights должны иметь размерность (batch, heads, seq, seq)
        expected_shape = (batch_size, 8, seq_len, seq_len)  # 8 query groups
        assert attention_weights.shape == expected_shape


class TestTransformerBlockIntegration:
    """Тесты интеграции компонентов"""

    def test_residual_connections(self):
        """Тест работы residual connections"""
        block = TransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )

        # Создаем входной тензор
        x = torch.randn(2, 10, 512)

        # Включаем режим обучения для точного тестирования
        block.train()

        # Forward pass
        output = block(x)

        # Residual connections должны сохранять градиентный поток
        # Проверяем что выход не идентичен входу (есть изменения)
        assert not torch.allclose(output, x, atol=1e-6)

        # Но форма должна быть сохранена
        assert output.shape == x.shape

    def test_gradient_flow(self):
        """Тест прохождения градиентов через блок"""
        block = TransformerBlock(
            hidden_size=256,
            num_query_groups=4,
            num_attention_heads=8,
            intermediate_size=1024
        )

        # Входной тензор с требованием градиентов
        x = torch.randn(1, 5, 256, requires_grad=True)

        # Forward pass
        output = block(x)

        # Создаем фиктивную loss функцию
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Проверяем что градиенты есть во всех компонентах
        assert x.grad is not None

        # Проверяем градиенты в основных компонентах
        for name, param in block.named_parameters():
            assert param.grad is not None, f"Градиент отсутствует в параметре: {name}"

    def test_different_sequence_lengths(self):
        """Тест с различными длинами последовательностей"""
        block = TransformerBlock(
            hidden_size=256,
            num_query_groups=4,
            num_attention_heads=8
        )

        sequence_lengths = [1, 5, 10, 50, 100]

        for seq_len in sequence_lengths:
            x = torch.randn(2, seq_len, 256)
            output = block(x)
            assert output.shape == (2, seq_len, 256)

    def test_different_batch_sizes(self):
        """Тест с различными размерами batch"""
        block = TransformerBlock(
            hidden_size=256,
            num_query_groups=4,
            num_attention_heads=8
        )

        batch_sizes = [1, 2, 4, 8, 16]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 10, 256)
            output = block(x)
            assert output.shape == (batch_size, 10, 256)


class TestTransformerBlockPerformance:
    """Тесты производительности и стабильности"""

    def test_numerical_stability(self):
        """Тест численной стабильности"""
        block = TransformerBlock(
            hidden_size=512,
            num_query_groups=8,
            num_attention_heads=16
        )

        # Тест с малыми значениями
        x_small = torch.randn(2, 10, 512) * 1e-6
        output_small = block(x_small)
        assert torch.isfinite(output_small).all()

        # Тест с большими значениями
        x_large = torch.randn(2, 10, 512) * 1e6
        output_large = block(x_large)
        assert torch.isfinite(output_large).all()

    def test_memory_efficiency(self):
        """Базовый тест использования памяти"""
        block = TransformerBlock(
            hidden_size=256,
            num_query_groups=4,
            num_attention_heads=8
        )

        # Подсчитаем примерное количество параметров
        total_params = sum(p.numel() for p in block.parameters())

        # Для данной конфигурации ожидаем разумное количество параметров
        # (это эвристическая проверка)
        assert total_params > 100_000  # Минимум
        assert total_params < 10_000_000  # Максимум для данной конфигурации

    def test_deterministic_output(self):
        """Тест детерминированности выхода"""
        # Фиксируем seed для воспроизводимости
        torch.manual_seed(42)

        block1 = TransformerBlock(
            hidden_size=256,
            num_query_groups=4,
            num_attention_heads=8
        )

        torch.manual_seed(42)

        block2 = TransformerBlock(
            hidden_size=256,
            num_query_groups=4,
            num_attention_heads=8
        )

        # При одинаковых seed и входах должны получить одинаковые результаты
        x = torch.randn(2, 10, 256)

        block1.eval()
        block2.eval()

        with torch.no_grad():
            output1 = block1(x)
            output2 = block2(x)

        assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == "__main__":
    # Можно запустить тесты напрямую
    pytest.main([__file__, "-v"])