# Стандартная библиотека
from typing import Optional, Tuple, Union

# Сторонние библиотеки
import torch
import torch.nn as nn

# Локальные импорты
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from normalization.rmsnorm import RMSNorm
from attention.gqa import GroupedQueryAttention
from activations.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """
    Description:
    ---------------
        Блок Transformer для архитектуры Qwen3 MoE. Использует Pre-Norm архитектуру
        с Grouped-Query Attention и SwiGLU активацией.

        Архитектура блока:
        Input → RMSNorm → GQA → Residual → RMSNorm → SwiGLU → Residual → Output

    Args:
    ---------------
        hidden_size: Размерность скрытого состояния (d_model)
        num_query_groups: Количество групп запросов для GQA
        num_attention_heads: Количество голов внимания для key/value
        intermediate_size: Размерность промежуточного слоя в SwiGLU (по умолчанию 4 * hidden_size)
    """

    def __init__(
        self,
        hidden_size: int,
        num_query_groups: int,
        num_attention_heads: int,
        intermediate_size: Optional[int] = None
    ):
        super().__init__()

        # TODO: Проверьте валидность параметров
        # TODO: Вычислите intermediate_size если не указан (4 * hidden_size)
        # TODO: Сохраните параметры как атрибуты
        # TODO: Создайте self.attention_norm = RMSNorm(hidden_size)
        # TODO: Создайте self.attention = GroupedQueryAttention(...)
        # TODO: Создайте self.ffn_norm = RMSNorm(hidden_size)
        # TODO: Создайте self.feed_forward = SwiGLU(...)

        # Вопросы для размышления:
        # - Почему используется Pre-Norm архитектура вместо Post-Norm?
        # - Как residual connections влияют на градиентный поток?
        # - Какая связь между hidden_size и intermediate_size?
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        """
        Description:
        ---------------
            Применяет Transformer блок к входному тензору.

        Args:
        ---------------
            hidden_states: Входной тензор формы (batch_size, seq_len, hidden_size)

        Returns:
        ---------------
            Тензор формы (batch_size, seq_len, hidden_size) - выход Transformer блока
        """
        # TODO: Сохраните входной тензор для первого residual connection
        # TODO: Примените attention_norm
        # TODO: Примените self.attention
        # TODO: Добавьте первый residual connection
        # TODO: Сохраните результат для второго residual connection
        # TODO: Примените ffn_norm
        # TODO: Примените self.feed_forward
        # TODO: Добавьте второй residual connection
        # TODO: Верните результат

        # Вопросы для размышления:
        # - Почему нормализация применяется ДО attention/ffn, а не после?
        # - Как residual connections помогают при обучении глубоких сетей?
        pass