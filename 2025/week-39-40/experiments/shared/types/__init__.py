"""
Общие типы для проекта Qwen3 MoE

Определяет базовые типы данных, используемые во всех доменах.
"""

from typing import Union, Tuple, Optional
import torch

# Базовые типы для размерностей
Shape = Union[int, Tuple[int, ...]]
Device = Union[str, torch.device]
Dtype = torch.dtype

# Типы для конфигураций
ConfigValue = Union[int, float, str, bool]

__all__ = ["Shape", "Device", "Dtype", "ConfigValue"]