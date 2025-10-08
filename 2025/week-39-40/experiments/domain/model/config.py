"""
Qwen3 MoE Model Configuration

Определяет все гиперпараметры модели в одном месте.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Qwen3Config:
    """
    Конфигурация для Qwen3 MoE модели.

    Архитектура:
    ------------
    - Vocabulary: GPT-2 tokenizer (50257 токенов)
    - Embedding: 1024-dim continuous vectors
    - Transformer: 12 MoE blocks с GQA + RoPE + SwiGLU
    - MoE: 8 экспертов, 2 активных per token (25% активация)
    - Output: Language modeling head (1024 → 50257)

    Параметры:
    ----------
    Model Architecture:
        vocab_size: Размер словаря (GPT-2 = 50257)
        hidden_size: Размерность скрытого слоя (embedding dimension)
        num_layers: Количество MoE Transformer блоков
        intermediate_size: Размерность FFN внутри каждого эксперта

    Attention:
        num_attention_heads: Количество Query голов (для GQA)
        num_key_value_heads: Количество Key/Value голов (GQA группировка)
        max_position_embeddings: Максимальная длина последовательности
        rope_theta: Базовая частота для RoPE (10000.0 стандарт)

    MoE Specific:
        num_experts: Общее количество экспертов в каждом MoE слое
        top_k: Количество активных экспертов per token
        balance_loss_coef: Коэффициент для load balancing loss (обычно 0.01)

    Regularization:
        dropout: Dropout rate для регуляризации (0.0 = отключен)

    Training:
        initializer_range: Стандартное отклонение для инициализации весов

    Примеры:
    --------
    >>> # Конфигурация по умолчанию (0.6B параметров)
    >>> config = Qwen3Config()
    >>> print(f"Model size: ~{config.vocab_size * config.hidden_size / 1e9:.2f}B parameters")

    >>> # Кастомная конфигурация
    >>> config = Qwen3Config(
    ...     hidden_size=2048,
    ...     num_layers=24,
    ...     num_experts=16
    ... )
    """

    # Model Architecture
    vocab_size: int = 50257  # GPT-2 tokenizer
    hidden_size: int = 1024
    num_layers: int = 12
    intermediate_size: int = 2048  # 2 * hidden_size для каждого эксперта

    # Attention Configuration
    num_attention_heads: int = 16  # Query heads
    num_key_value_heads: int = 4   # KV heads (GQA: 4x группировка)
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0

    # MoE Configuration
    num_experts: int = 8
    top_k: int = 2
    balance_loss_coef: float = 0.01

    # Regularization
    dropout: float = 0.1

    # Training
    initializer_range: float = 0.02

    def __post_init__(self):
        """Валидация конфигурации после инициализации."""
        # Базовые проверки
        assert self.vocab_size > 0, "vocab_size должен быть положительным"
        assert self.hidden_size > 0, "hidden_size должен быть положительным"
        assert self.num_layers > 0, "num_layers должен быть положительным"
        assert self.intermediate_size > 0, "intermediate_size должен быть положительным"

        # Attention проверки
        assert self.num_attention_heads > 0, "num_attention_heads должен быть положительным"
        assert self.num_key_value_heads > 0, "num_key_value_heads должен быть положительным"
        assert (
            self.num_attention_heads % self.num_key_value_heads == 0
        ), "num_attention_heads должен делиться на num_key_value_heads"
        assert (
            self.hidden_size % self.num_attention_heads == 0
        ), "hidden_size должен делиться на num_attention_heads"

        # MoE проверки
        assert self.num_experts > 0, "num_experts должен быть положительным"
        assert self.top_k > 0, "top_k должен быть положительным"
        assert self.top_k <= self.num_experts, "top_k не может быть больше num_experts"

        # Regularization проверки
        assert 0.0 <= self.dropout <= 1.0, "dropout должен быть в диапазоне [0, 1]"

    def to_dict(self):
        """Конвертация конфигурации в словарь."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "balance_loss_coef": self.balance_loss_coef,
            "dropout": self.dropout,
            "initializer_range": self.initializer_range,
        }
