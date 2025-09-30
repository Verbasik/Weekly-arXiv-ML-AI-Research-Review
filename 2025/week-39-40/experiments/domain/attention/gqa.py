# Стандартная библиотека
from typing import Optional, Tuple, Union

# Сторонние библиотеки
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Локальные импорты
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from positional_encoding.rope import RoPE


class GroupedQueryAttention(nn.Module):
    """
    Description:
    ---------------
        Grouped-Query Attention (GQA) - оптимизированная версия Multi-Head Attention,
        используемая в современных языковых моделях, включая Qwen3.
        
        В отличие от Multi-Head Attention, где каждая голова имеет свои проекции
        для query, key и value, в GQA запросы (queries) группируются, а ключи (keys)
        и значения (values) используются совместно несколькими головами внимания.
        
        Это позволяет сократить вычислительные затраты и память, сохраняя при этом
        выразительную мощность механизма внимания.
        
        Формула:
        GQA(Q, K, V) = Softmax(QK^T/√d_k)V
        
        где:
        - Q разделен на G групп (меньше, чем количество голов для ключей и значений)
        - K и V имеют H голов (H ≥ G)
        - Каждая группа запросов использует несколько голов ключей и значений
        
        Преимущества:
        - Снижение вычислительных затрат и использования памяти
        - Сохранение выразительной мощности механизма внимания
        - Улучшение масштабируемости для больших моделей
        
    Args:
    ---------------
        hidden_size: Размерность скрытого состояния == d_model (example: LLaMA 70B hidden_size = d_model = 8192)
        num_query_groups: Количество групп запросов
        num_attention_heads: Количество голов внимания (для ключей и значений)
        head_dim: Размерность каждой головы внимания
        dropout: Вероятность дропаута (по умолчанию 0.0)
        bias: Использовать ли смещение в линейных преобразованиях (по умолчанию True)
        use_rope: Использовать ли RoPE для позиционного кодирования (по умолчанию True)
        rope_theta: База для RoPE (по умолчанию 10000.0)
        rope_scaling: Масштабирование для RoPE (по умолчанию 1.0)
        max_position: Максимальная длина последовательности для RoPE (по умолчанию 2048)
        
    Returns:
    ---------------
        Тензор формы (batch_size, seq_len, hidden_size) - результат применения GQA
        
    Examples:
    ---------------
        >>> import torch
        >>> gqa = GroupedQueryAttention(
        ...     hidden_size=512,
        ...     num_query_groups=8,
        ...     num_attention_heads=16,
        ...     head_dim=64
        ... )
        >>> x = torch.randn(2, 10, 512)
        >>> output = gqa(x)
        >>> output.shape
        torch.Size([2, 10, 512])
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_query_groups: int,
        num_attention_heads: int,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        rope_scaling: float = 1.0,
        max_position: int = 2048
    ):
        super().__init__()
        
        # TODO: Проверьте, что num_attention_heads делится на num_query_groups
        # TODO: Проверьте, что hidden_size делится на num_attention_heads
        # TODO: Вычислите head_dim, если он не указан
        # TODO: Сохраните все параметры как атрибуты класса
        # TODO: Создайте проекции для query, key и value
        # TODO: Создайте проекцию для выхода
        # TODO: Создайте dropout слой
        # TODO: Если use_rope=True, создайте RoPE модуль
        
        # Вопросы для размышления:
        # - Почему GQA эффективнее, чем обычный Multi-Head Attention?
        # - Как количество групп запросов влияет на производительность и качество модели?
        # - Какие преимущества дает совместное использование ключей и значений?
        # - Как RoPE интегрируется с GQA?
        # pass

        assert num_attention_heads % num_query_groups == 0, "Количество голов внимания должно делиться на количество групп запросов"
        assert hidden_size % num_attention_heads == 0, "Размерность скрытого состояния должна делиться на количество голов внимания"
        
        # Делим скрытую размерность d_model (hidden_size) на число голов внимания h (num_attention_heads)
        # Каждая голова обрабатывает кусок размерности d_head = d_model / h, а конкатенация h голов возвращает исходную размерность
        head_dim = hidden_size // num_attention_heads if head_dim is None else head_dim

        self.hidden_size = hidden_size
        self.num_query_groups = num_query_groups
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.bias = bias
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.max_position = max_position

        # Проекции для query, key и value
        self.query_proj = nn.Linear(in_features = hidden_size, 
                                    out_features = num_attention_heads * head_dim, 
                                    bias=bias)
        
        self.key_proj   = nn.Linear(in_features = hidden_size, 
                                    out_features = num_attention_heads * head_dim, 
                                    bias=bias)
        
        self.value_proj = nn.Linear(in_features = hidden_size, 
                                    out_features = num_attention_heads * head_dim, 
                                    bias=bias)
        
        self.output_proj = nn.Linear(in_features = num_attention_heads * head_dim, 
                                    out_features = hidden_size,
                                    bias=bias)
        
        self.dropout = nn.Dropout(dropout)


        self.rope = RoPE(dim=head_dim) if use_rope else None

        
    def _split_heads(
        self, 
        x: torch.Tensor, 
        num_heads: int
    ) -> torch.Tensor:
        """
        Description:
        ---------------
            Разделяет последнюю размерность тензора на несколько голов внимания.
        
        Args:
        ---------------
            x: Входной тензор формы (batch_size, seq_len, hidden_size)
            num_heads: Количество голов внимания
            
        Returns:
        ---------------
            Тензор формы (batch_size, num_heads, seq_len, head_dim)
        """
        # TODO: Получите новую форму тензора
        # TODO: Измените форму тензора и транспонируйте размерности
        # TODO: Верните результат
        # pass

        # batch_size - количество последовательностей, обрабатываемых одновременно
        # seq_len - количество токенов в последовательности
        # hidden_size - размерность скрытого состояния (эмбеддинги токенов)
        batch_size, seq_len, hidden_size = x.shape

        # .view() перестраивает тензор в новую форму, не изменяя данные.
        x = x.view(batch_size, seq_len, num_heads, head_dim)

        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> Union[
        Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]],
        torch.Tensor
    ]:
        """
        Description:
        ---------------
            Применяет Grouped-Query Attention к входному тензору.
        
        Args:
        ---------------
            hidden_states: Входной тензор формы (batch_size, seq_len, hidden_size)
            attention_mask: Маска внимания (опционально)
            position_ids: Позиционные индексы для RoPE (опционально)
            past_key_value: Кэшированные ключи и значения (опционально)
            output_attentions: Возвращать ли веса внимания (опционально)
            use_cache: Использовать ли кэширование ключей и значений (опционально)
            
        Returns:
        ---------------
            Тензор формы (batch_size, seq_len, hidden_size) - результат применения GQA
            Опционально: кэшированные ключи и значения, веса внимания
        """
        # TODO: Получите размерности входного тензора
        # TODO: Примените проекции для query, key и value
        # TODO: Разделите query на группы, а key и value на головы
        # TODO: Если use_rope=True, примените RoPE к query и key
        # TODO: Если past_key_value не None, объедините с текущими key и value
        # TODO: Если use_cache=True, подготовьте новый past_key_value
        # TODO: Вычислите скалярное произведение query и key
        # TODO: Масштабируйте скалярное произведение
        # TODO: Если attention_mask не None, примените маску
        # TODO: Примените softmax к весам внимания
        # TODO: Примените dropout к весам внимания
        # TODO: Вычислите взвешенную сумму значений
        # TODO: Объедините головы внимания
        # TODO: Примените выходную проекцию
        # TODO: Верните результат и опциональные выходы
        
        # Вопросы для размышления:
        # - Как attention_mask влияет на веса внимания?
        # - Как кэширование ключей и значений ускоряет генерацию?
        # - Какие преимущества дает использование RoPE в GQA?
        # pass

        # Тензор скрытого состояния
        batch_size, seq_len, hidden_size = hidden_states.shape()
        # Проекции для query, key и value
        # Используем фабрику линейных слоев для создания проекций
        query = self.query_proj(hidden_states)
        key   = self.key_proj(hidden_states)
        value = self.value_proj(hidden_states)
        # Разделяем query на группы, а key и value на головы
        # Получаем тензоры формы: (batch_size, num_query_groups, seq_len, head_dim)
        query = self._split_heads(query, self.num_query_groups)



if __name__ == "__main__":
    # Создаем GQA с параметрами
    hidden_size = 512
    num_query_groups = 8
    num_attention_heads = 16
    head_dim = 64
    
    gqa = GroupedQueryAttention(
        hidden_size=hidden_size,
        num_query_groups=num_query_groups,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim
    )
    
    # Создаем тестовый тензор
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Проверяем, что форма выхода соответствует ожидаемой
    print(f"Входной тензор: {x.shape}")
    # output = gqa(x)
    # print(f"Выходной тензор: {output.shape}")
