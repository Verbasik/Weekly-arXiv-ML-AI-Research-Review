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
        # pass

        # --- Валидация параметров -------------------------------------------------
        assert (
            isinstance(hidden_size, int) and hidden_size > 0
        ), "hidden_size должен быть положительным целым числом"
        assert (
            isinstance(num_query_groups, int) and num_query_groups > 0
        ), "num_query_groups должен быть положительным целым числом"
        assert (
            isinstance(num_attention_heads, int) and num_attention_heads > 0
        ), "num_attention_heads должен быть положительным целым числом"

        # MoE специфичные проверки
        assert (
            isinstance(num_experts, int) and num_experts > 0
        ), "num_experts должен быть положительным целым числом"
        assert (
            top_k > 0 and top_k <= num_experts
        ), "top_k должен быть > 0 и <= num_experts"

        # Ключевая проверка для GQA архитектуры:
        assert (
            num_attention_heads % num_query_groups == 0
        ), (
            "num_attention_heads должен делиться на num_query_groups для "
            "корректной работы GQA"
        )
        # Проверка делимости hidden_size на число голов
        # Тензор скрытого состояния должен равномерно делиться на число голов
        assert (
            hidden_size % num_attention_heads == 0
        ), "hidden_size должен делиться на num_attention_heads"

        # Проверка intermediate_size если указан
        if intermediate_size is not None:
            assert (
                isinstance(intermediate_size, int) and intermediate_size > 0
            ), "intermediate_size должен быть положительным целым числом"

        # --- Инициализация атрибутов ----------------------------------------------
        self.hidden_size         = hidden_size
        self.num_query_groups    = num_query_groups
        self.num_attention_heads = num_attention_heads
        self.num_experts         = num_experts
        self.top_k               = top_k
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else 4 * hidden_size
        )
        self.expert_dropout    = expert_dropout
        self.balance_loss_coef = balance_loss_coef

        # --- Компоненты нормализации и подблоков ----------------------------------
        self.attention_norm = RMSNorm(hidden_size)
        self.attention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_query_groups=num_query_groups,
            num_attention_heads=num_attention_heads,
        )
        self.ffn_norm  = RMSNorm(hidden_size)
        self.moe_layer = SimpleMoELayer(
            hidden_size = hidden_size,
            num_experts = num_experts,
            top_k = top_k,
            intermediate_size = self.intermediate_size,
            expert_dropout = expert_dropout,
            balance_loss_coef = balance_loss_coef
        )


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
        # pass

        # ──────────────────────────────────────────────────────────────────────────
        # ПЕРВЫЙ RESIDUAL BLOCK: Self-Attention (GQA)
        # ──────────────────────────────────────────────────────────────────────────

        # Сохраняем вход для остаточной связи, что бы потом прибавить к выходу блока
        # p.s. Помогает сохранить информацию о входном тензоре (векторах), чтобы не терять её.
        residual_1 = hidden_states

        # Нормализуем входной тензор
        normed = self.attention_norm(hidden_states)

        # Вызываем модуль группового внимания. Он может вернуть:
        # - только выход (Tensor), либо
        # - кортеж (att_output, present_key_value, attn_weights).
        att_output = self.attention(
            hidden_states=normed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        if isinstance(att_output, tuple):
            att_output, present_key_value, attn_weights = att_output
        else:
            present_key_value = None
            attn_weights = None

        # Первая residual-связь: складываем вход и выход подблока.
        hidden_states = att_output + residual_1

        # ──────────────────────────────────────────────────────────────────────────
        # ВТОРОЙ RESIDUAL BLOCK: MoE Feed-Forward
        # ──────────────────────────────────────────────────────────────────────────

        residual_2 = hidden_states

        # Нормализуем перед MoE по тем же причинам, что и перед вниманием.
        normed = self.ffn_norm(hidden_states)
        # Применяем MoE слой вместо обычного FFN.
        ffn_output, balance_loss = self.moe_layer(hidden_states=normed, training=training)
        # Вторая residual-связь.
        hidden_states = ffn_output + residual_2

        if use_cache or output_attentions:
            return hidden_states, balance_loss, present_key_value, attn_weights

        return hidden_states, balance_loss
