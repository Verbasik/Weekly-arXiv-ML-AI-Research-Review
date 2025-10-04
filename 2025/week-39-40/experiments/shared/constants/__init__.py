"""
Константы проекта Qwen3 MoE

Центральное место для всех констант, используемых в проекте.
"""

# Математические константы
DEFAULT_EPS = 1e-6
DEFAULT_ROPE_THETA = 1000000.0

# Конфигурация модели Qwen3-30B-A3B
QWEN3_30B_CONFIG = {
    "vocab_size": 151000,
    "hidden_size": 2048,
    "intermediate_size": 768,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "max_position_embeddings": 262144,
    "rope_theta": DEFAULT_ROPE_THETA,
    "rms_norm_eps": DEFAULT_EPS
}

__all__ = ["DEFAULT_EPS", "DEFAULT_ROPE_THETA", "QWEN3_30B_CONFIG"]