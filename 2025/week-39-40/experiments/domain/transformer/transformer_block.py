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

        # Ключевая проверка для GQA архитектуры:
        assert (
            num_attention_heads % num_query_groups == 0
        ), (
            "num_attention_heads должен делиться на num_query_groups для "
            "корректной работы GQA"
        )

        # Проверка делимости hidden_size на число голов:
        assert (
            hidden_size % num_attention_heads == 0
        ), "hidden_size должен делиться на num_attention_heads"

        # Проверка intermediate_size если указан
        if intermediate_size is not None:
            assert (
                isinstance(intermediate_size, int) and intermediate_size > 0
            ), "intermediate_size должен быть положительным целым числом"

        # --- Инициализация атрибутов ----------------------------------------------
        self.hidden_size = hidden_size
        self.num_query_groups = num_query_groups
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else 4 * hidden_size
        )

        # --- Компоненты нормализации и подблоков ----------------------------------
        self.attention_norm = RMSNorm(hidden_size)
        self.attention = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_query_groups=num_query_groups,
            num_attention_heads=num_attention_heads,
        )
        self.ffn_norm = RMSNorm(hidden_size)
        self.feed_forward = SwiGLU(
            input_dim=self.hidden_size,
            output_dim=self.hidden_size,
            intermediate_dim=self.intermediate_size,
        )
        
        
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
            Input → RMSNorm → GQA → Residual → RMSNorm → SwiGLU → Residual → Output

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

        # ──────────────────────────────────────────────────────────────────────────
        # ПЕРВЫЙ RESIDUAL BLOCK: Self-Attention (GQA)
        # ──────────────────────────────────────────────────────────────────────────

        # Сохраняем вход для остаточной связи (residual).
        residual_1 = hidden_states

        # Преднормализация улучшает устойчивость и качество внимания.
        normed = self.attention_norm(hidden_states)

        # Вызываем модуль внимания. Он может вернуть:
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
        # ВТОРОЙ RESIDUAL BLOCK: Feed-Forward (SwiGLU)
        # ──────────────────────────────────────────────────────────────────────────

        residual_2 = hidden_states

        # Преднормализация перед FFN по тем же причинам, что и перед вниманием.
        normed = self.ffn_norm(hidden_states)

        # Применяем нелинейную проекцию SwiGLU с расширением размерности
        # до intermediate_size и обратной проекцией к hidden_size.
        ffn_output = self.feed_forward(normed)

        # Вторая residual-связь.
        hidden_states = ffn_output + residual_2

        if use_cache or output_attentions:
            return hidden_states, present_key_value, attn_weights

        return hidden_states



