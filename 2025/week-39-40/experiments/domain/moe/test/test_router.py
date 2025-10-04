"""
Тесты для MoERouter - компонента маршрутизации экспертов.

Эти тесты проверяют:
1. Правильность инициализации и параметров
2. Forward pass с корректными размерностями
3. Top-K selection работает правильно
4. Load balancing loss вычисляется корректно
5. Нормализация весов (сумма = 1)
6. Граничные случаи и численная стабильность
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Добавляем родительскую директорию в путь для импорта
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from router import MoERouter


class TestMoERouterInitialization:
    """Тесты инициализации MoERouter"""

    def test_basic_initialization(self):
        """Тест базовой инициализации с минимальными параметрами"""
        router = MoERouter(
            hidden_size=512,
            num_experts=128,
            top_k=8
        )

        # Проверяем основные атрибуты
        assert router.hidden_size == 512
        assert router.num_experts == 128
        assert router.top_k == 8
        assert router.capacity_factor == 1.25  # Default value
        assert router.balance_loss_coef == 0.01  # Default value

        # Проверяем создание gate layer
        assert hasattr(router, 'gate')
        assert isinstance(router.gate, nn.Linear)
        assert router.gate.in_features == 512
        assert router.gate.out_features == 128

    def test_custom_parameters(self):
        """Тест с кастомными параметрами"""
        router = MoERouter(
            hidden_size=256,
            num_experts=64,
            top_k=4,
            capacity_factor=1.5,
            balance_loss_coef=0.02
        )

        assert router.capacity_factor == 1.5
        assert router.balance_loss_coef == 0.02

    def test_parameter_validation(self):
        """Тест валидации параметров"""
        # top_k должен быть <= num_experts
        with pytest.raises(AssertionError):
            MoERouter(
                hidden_size=512,
                num_experts=8,
                top_k=16  # top_k > num_experts - недопустимо!
            )

        # hidden_size должен быть положительным
        with pytest.raises(AssertionError):
            MoERouter(
                hidden_size=0,
                num_experts=128,
                top_k=8
            )


class TestMoERouterForward:
    """Тесты forward pass MoERouter"""

    @pytest.fixture
    def router(self):
        """Создает стандартный MoERouter для тестов"""
        return MoERouter(
            hidden_size=512,
            num_experts=128,
            top_k=8
        )

    def test_forward_output_shapes(self, router):
        """Тест корректности размерностей выхода"""
        batch_size, seq_len, hidden_size = 2, 10, 512
        x = torch.randn(batch_size, seq_len, hidden_size)

        routing_weights, selected_experts, balance_loss = router(x, training=True)

        # Проверяем размерности
        assert routing_weights.shape == (batch_size, seq_len, 8)  # top_k=8
        assert selected_experts.shape == (batch_size, seq_len, 8)
        assert balance_loss.dim() == 0  # Скаляр

    def test_routing_weights_normalization(self, router):
        """Тест нормализации весов (сумма должна быть 1 для каждого токена)"""
        x = torch.randn(2, 10, 512)
        routing_weights, _, _ = router(x)

        # Сумма весов по top_k экспертам должна быть близка к 1.0
        weights_sum = routing_weights.sum(dim=-1)
        assert torch.allclose(weights_sum, torch.ones_like(weights_sum), atol=1e-5)

    def test_expert_indices_valid_range(self, router):
        """Тест что индексы экспертов в допустимом диапазоне [0, num_experts)"""
        x = torch.randn(2, 10, 512)
        _, selected_experts, _ = router(x)

        # Все индексы должны быть в диапазоне [0, 128)
        assert (selected_experts >= 0).all()
        assert (selected_experts < 128).all()

    def test_top_k_selection(self, router):
        """Тест что выбираются действительно топ-K экспертов"""
        x = torch.randn(1, 5, 512)
        routing_weights, selected_experts, _ = router(x)

        # Веса должны быть отсортированы по убыванию для каждого токена
        # (так как мы выбрали Top-K)
        for b in range(routing_weights.shape[0]):
            for s in range(routing_weights.shape[1]):
                weights = routing_weights[b, s]
                # Проверяем что веса в порядке убывания (или равны)
                assert (weights[:-1] >= weights[1:]).all() or torch.allclose(
                    weights[:-1], weights[1:], atol=1e-5
                )

    def test_balance_loss_training_mode(self, router):
        """Тест что balance_loss вычисляется в режиме training"""
        x = torch.randn(2, 10, 512)

        # В режиме training должен быть ненулевой loss
        _, _, loss_train = router(x, training=True)
        assert loss_train.item() > 0

        # В режиме inference loss должен быть 0
        _, _, loss_eval = router(x, training=False)
        assert loss_eval.item() == 0.0


class TestMoERouterLoadBalancing:
    """Тесты load balancing механизма"""

    def test_balance_loss_properties(self):
        """Тест свойств balance loss"""
        router = MoERouter(
            hidden_size=256,
            num_experts=32,
            top_k=4,
            balance_loss_coef=0.01
        )

        x = torch.randn(4, 8, 256)
        _, _, balance_loss = router(x, training=True)

        # Balance loss должен быть положительным скаляром
        assert balance_loss.dim() == 0
        assert balance_loss.item() >= 0

    def test_expert_capacity_computation(self):
        """Тест вычисления емкости эксперта"""
        router = MoERouter(
            hidden_size=512,
            num_experts=128,
            top_k=8,
            capacity_factor=1.25
        )

        num_tokens = 100  # batch_size * seq_len
        capacity = router.expert_capacity(num_tokens)

        # Базовая формула: (100 / 128) * 1.25 * 8 = 7.8125 ≈ 8
        expected = int((num_tokens / 128) * 1.25 * 8)
        assert capacity >= expected  # Должно округляться вверх


class TestMoERouterIntegration:
    """Интеграционные тесты"""

    def test_gradient_flow(self):
        """Тест прохождения градиентов через router"""
        router = MoERouter(
            hidden_size=256,
            num_experts=64,
            top_k=4
        )

        x = torch.randn(2, 5, 256, requires_grad=True)
        routing_weights, selected_experts, balance_loss = router(x, training=True)

        # Backward через routing_weights
        loss = routing_weights.sum() + balance_loss
        loss.backward()

        # Проверяем наличие градиентов
        assert x.grad is not None
        assert router.gate.weight.grad is not None

    def test_different_batch_sizes(self):
        """Тест с различными размерами batch"""
        router = MoERouter(hidden_size=256, num_experts=64, top_k=4)

        batch_sizes = [1, 2, 4, 8, 16]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 10, 256)
            weights, experts, loss = router(x)

            assert weights.shape == (batch_size, 10, 4)
            assert experts.shape == (batch_size, 10, 4)

    def test_different_sequence_lengths(self):
        """Тест с различными длинами последовательностей"""
        router = MoERouter(hidden_size=256, num_experts=64, top_k=4)

        seq_lengths = [1, 5, 10, 50, 100]
        for seq_len in seq_lengths:
            x = torch.randn(2, seq_len, 256)
            weights, experts, loss = router(x)

            assert weights.shape == (2, seq_len, 4)
            assert experts.shape == (2, seq_len, 4)

    def test_numerical_stability(self):
        """Тест численной стабильности"""
        router = MoERouter(hidden_size=256, num_experts=64, top_k=4)

        # Тест с очень маленькими значениями
        x_small = torch.randn(2, 10, 256) * 1e-6
        weights_small, experts_small, loss_small = router(x_small)
        assert torch.isfinite(weights_small).all()
        assert torch.isfinite(loss_small)

        # Тест с большими значениями
        x_large = torch.randn(2, 10, 256) * 1e6
        weights_large, experts_large, loss_large = router(x_large)
        assert torch.isfinite(weights_large).all()
        assert torch.isfinite(loss_large)

    def test_deterministic_output(self):
        """Тест детерминированности выхода"""
        torch.manual_seed(42)
        router1 = MoERouter(hidden_size=256, num_experts=64, top_k=4)

        torch.manual_seed(42)
        router2 = MoERouter(hidden_size=256, num_experts=64, top_k=4)

        x = torch.randn(2, 10, 256)

        router1.eval()
        router2.eval()

        with torch.no_grad():
            weights1, experts1, _ = router1(x, training=False)
            weights2, experts2, _ = router2(x, training=False)

        assert torch.allclose(weights1, weights2, atol=1e-6)
        assert torch.equal(experts1, experts2)


if __name__ == "__main__":
    # Можно запустить тесты напрямую
    pytest.main([__file__, "-v"])
