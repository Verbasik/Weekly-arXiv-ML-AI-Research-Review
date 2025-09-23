"""
Qwen3 MoE Implementation - DDD Architecture

Проект организован по принципам Domain-Driven Design:

Domain Layer (Доменный слой):
- normalization/     : Домен нормализации (RMSNorm, LayerNorm)
- attention/         : Домен механизмов внимания (GQA, Multi-head)
- positional_encoding/ : Домен позиционного кодирования (RoPE)
- activations/       : Домен функций активации (SwiGLU, Swish)
- moe/              : Домен Mixture-of-Experts (Router, Experts)
- modeling/         : Домен сборки моделей (TransformerBlock, Model)

Infrastructure Layer (Инфраструктурный слой):
- config/           : Конфигурации и настройки
- utils/            : Утилиты и вспомогательные функции
- metrics/          : Метрики и мониторинг

Application Layer (Прикладной слой):
- training/         : Сценарии обучения
- inference/        : Сценарии инференса

Shared Kernel (Общее ядро):
- types/            : Общие типы данных
- constants/        : Константы проекта
- exceptions/       : Исключения
"""

__version__ = "0.1.0"
__author__ = "Learning Project - DDD Architecture"