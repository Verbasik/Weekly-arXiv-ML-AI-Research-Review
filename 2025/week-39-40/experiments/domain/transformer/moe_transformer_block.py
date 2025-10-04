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
from moe.moe_layer import SimpleMoELayer


class MoETransformerBlock(nn.Module):
    """
    Description:
    ---------------
        MoE Transformer Block для архитектуры Qwen3. Использует Pre-Norm архитектуру
        с Grouped-Query Attention и SimpleMoELayer вместо обычного FFN.

        Архитектура блока:
        Input → RMSNorm → GQA → Residual → RMSNorm → SimpleMoELayer → Residual → Output
                                                           ↓
                                                     balance_loss

        Отличие от обычного TransformerBlock:
        - SwiGLU FFN заменён на SimpleMoELayer
        - Forward возвращает (output, balance_loss) вместо просто output
        - balance_loss используется для обучения (предотвращение коллапса экспертов)

    Args:
    ---------------
        hidden_size: Размерность скрытого состояния (d_model)
        num_query_groups: Количество групп запросов для GQA
        num_attention_heads: Количество голов внимания для key/value
        num_experts: Количество экспертов в MoE Layer (default: 8)
        top_k: Количество активных экспертов per token (default: 2)
        intermediate_size: Размерность промежуточного слоя в экспертах (default: 4 * hidden_size)
        expert_dropout: Dropout для экспертов (default: 0.0)
        balance_loss_coef: Коэффициент для load balancing loss (default: 0.01)

    Returns (from forward):
    ---------------
        Если use_cache или output_attentions:
            (hidden_states, balance_loss, present_key_value, attn_weights)
        Иначе:
            (hidden_states, balance_loss)

    Example:
    ---------------
        >>> # Создание MoE Transformer Block
        >>> block = MoETransformerBlock(
        ...     hidden_size=512,
        ...     num_query_groups=8,
        ...     num_attention_heads=16,
        ...     num_experts=8,
        ...     top_k=2
        ... )
        >>> x = torch.randn(2, 10, 512)  # (batch, seq, hidden)
        >>> output, balance_loss = block(x)
        >>> output.shape  # torch.Size([2, 10, 512])
        >>> balance_loss.item()  # Скаляр loss

    Note:
    ---------------
        balance_loss должен быть добавлен к общему loss модели во время обучения:
        total_loss = language_model_loss + balance_loss
    """

    def __init__(
        self,
        hidden_size: int,
        num_query_groups: int,
        num_attention_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        intermediate_size: Optional[int] = None,
        expert_dropout: float = 0.0,
        balance_loss_coef: float = 0.01
    ):
        super().__init__()

        # TODO: Проверьте валидность параметров
        #       Подсказка: посмотрите TransformerBlock (строки 53-80)
        #       Добавьте проверку для MoE: top_k <= num_experts

        # TODO: Вычислите intermediate_size если не указан
        #       Подсказка: какое стандартное соотношение к hidden_size?

        # TODO: Сохраните параметры как атрибуты класса
        #       Подсказка: self.hidden_size = ..., self.num_experts = ..., и т.д.

        # TODO: Создайте 4 компонента (см. TransformerBlock строки 91-102):
        #       - self.attention_norm (какой тип нормализации?)
        #       - self.attention (какой механизм внимания?)
        #       - self.ffn_norm (снова нормализация)
        #       - self.moe_layer (вместо self.feed_forward!)
        #
        #       Вопрос: какие параметры нужны SimpleMoELayer?

        # Вопросы для размышления:
        # - Почему мы заменяем FFN на MoE Layer?
        # - Как balance_loss влияет на обучение модели?
        # - Что произойдёт, если не добавлять balance_loss к общему loss?
        pass


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        training: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Optional[Tuple], Optional[torch.Tensor]]]:
        """
        Description:
        ---------------
            Применяет MoE Transformer блок к входному тензору.

            Pipeline:
            1. RMSNorm → GQA → Residual
            2. RMSNorm → SimpleMoELayer → Residual
            3. Return (output, balance_loss)

        Args:
        ---------------
            hidden_states: Входной тензор формы (batch_size, seq_len, hidden_size)
            attention_mask: Маска внимания (optional)
            position_ids: Позиционные индексы для RoPE (optional)
            past_key_value: Кэш key/value для генерации (optional)
            output_attentions: Возвращать ли attention weights (default: False)
            use_cache: Использовать ли KV cache (default: False)
            training: Режим обучения для balance_loss (default: True)

        Returns:
        ---------------
            Если use_cache или output_attentions:
                (hidden_states, balance_loss, present_key_value, attn_weights)
            Иначе:
                (hidden_states, balance_loss)
        """
        # TODO: ПЕРВЫЙ RESIDUAL BLOCK - Self-Attention (GQA)
        #       Подсказка: скопируйте из TransformerBlock (строки 143-171)
        #       Структура: residual → norm → attention → residual_add
        #       Внимание: attention может вернуть tuple!

        # TODO: ВТОРОЙ RESIDUAL BLOCK - MoE Feed-Forward
        #       Подсказка: структура как в TransformerBlock (строки 177-187)
        #       НО: self.feed_forward → self.moe_layer
        #       ВАЖНО: moe_layer возвращает (output, balance_loss) - tuple!
        #       Не забудьте передать параметр training

        # TODO: RETURN
        #       Вопрос: сколько значений нужно вернуть?
        #       - Всегда: (hidden_states, balance_loss)
        #       - Если use_cache/output_attentions: добавьте present_kv и attn_weights
        #       Подсказка: посмотрите TransformerBlock строки 189-192

        # Вопросы для размышления:
        # - Почему balance_loss возвращается вместе с hidden_states?
        # - Как будет собираться balance_loss от всех слоёв модели?
        # - Чем отличается forward от обычного TransformerBlock?
        pass
