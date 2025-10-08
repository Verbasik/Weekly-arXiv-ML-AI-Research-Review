"""
Qwen3 MoE Language Model

Полная реализация генеративной языковой модели с MoE архитектурой.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import Qwen3Config
from ..normalization.rmsnorm import RMSNorm
from ..transformer.moe_transformer_block import MoETransformerBlock


class Qwen3MoEModel(nn.Module):
    """
    Description:
    ---------------
        Полная генеративная языковая модель Qwen3 с MoE архитектурой.

    Архитектура:
    ------------
    Input (batch, seq_len) — token IDs
        ↓
    Token Embedding (batch, seq_len, hidden_size)
        ↓
    N × MoE Transformer Blocks
        ├─ RMSNorm
        ├─ Grouped-Query Attention + RoPE
        ├─ RMSNorm
        └─ SimpleMoELayer (8 экспертов, 2 активных)
        ↓
    Final RMSNorm (batch, seq_len, hidden_size)
        ↓
    LM Head: Linear(hidden_size → vocab_size)
        ↓
    Output Logits (batch, seq_len, vocab_size)

    Args:
    -----
        config: Qwen3Config с параметрами модели

    Attributes:
    -----------
        embed_tokens: Token embedding layer (vocab_size → hidden_size)
        layers: nn.ModuleList из N MoE Transformer блоков
        norm: Final RMSNorm перед LM head
        lm_head: Language modeling head (hidden_size → vocab_size)

    Examples:
    ---------
        >>> config = Qwen3Config()
        >>> model = Qwen3MoEModel(config)
        >>>
        >>> # Forward pass
        >>> input_ids = torch.randint(0, config.vocab_size, (2, 10))
        >>> logits, balance_loss = model(input_ids)
        >>> print(logits.shape)  # (2, 10, 50257)
        >>>
        >>> # Generation
        >>> generated = model.generate(input_ids, max_length=50)
        >>> print(generated.shape)  # (2, 50)
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        # TODO: Преобразование token IDs → continuous vectors
        # Вопрос: Какой PyTorch слой создаёт lookup table размера (vocab_size, hidden_size)?
        # Подсказка: В SimpleMoELayer вы использовали nn.ModuleList. А для embeddings?

        # TODO: Стек из N transformer блоков
        # Вопрос: Как создать список из config.num_layers одинаковых блоков MoETransformerBlock(config)?
        # Подсказка: Вспомните, как в SimpleMoELayer создавались эксперты

        # TODO: Финальная нормализация скрытых состояний
        # Вопрос: Какой компонент нормализации вы реализовали на первых этапах?
        # Подсказка: Принимает один аргумент — размерность для нормализации

        # TODO: Проекция hidden_size → vocab_size для предсказания токенов
        # Вопрос: Какой слой проецирует векторы из одного пространства в другое?
        # Подсказка: В LM обычно используют bias=False в финальной проекции

        # Вопросы для размышления:
        # - Почему все четыре компонента должны быть атрибутами класса (self.*)?
        # - Что произойдёт, если использовать Python list вместо nn.ModuleList?
        # - Почему размерность embedding должна совпадать с hidden_size блоков?

        # Инициализация весов
        self._init_weights()

        # Token Embedding Layer: преобразование token IDs → continuous vectors
        # Создаёт lookup table размера (vocab_size, hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Стек из N transformer блоков
        # Каждый блок содержит: RMSNorm → GQA → RMSNorm → SimpleMoELayer
        self.layers = nn.ModuleList([
            MoETransformerBlock(config) for _ in range(config.num_layers)
        ])


    def _init_weights(self):
        """
        Инициализация весов модели.

        Стратегия:
        ----------
        - Embeddings: normal distribution N(0, initializer_range)
        - Linear layers: уже инициализированы в sub-модулях
        - LM Head: normal distribution N(0, initializer_range)
        """
        # Embedding initialization
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.config.initializer_range)

        # LM Head initialization
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass модели.

        Pipeline:
        ---------
        1. Token IDs → Embeddings (lookup)
        2. Embeddings → Transformer Blocks (N раз)
        3. Hidden States → Final Norm
        4. Normalized States → LM Head → Logits
        5. Accumulate balance loss из всех MoE блоков

        Args:
        -----
            input_ids: Тензор token IDs формы (batch_size, seq_len)
            attention_mask: Опциональная маска внимания (batch_size, seq_len)
                           1 = attend, 0 = ignore. По умолчанию None (все токены видимы)

        Returns:
        --------
            logits: Тензор логитов формы (batch_size, seq_len, vocab_size)
                   Вероятности следующего токена для каждой позиции
            balance_loss: Скалярный тензор, сумма balance losses из всех MoE слоёв
                         Используется для load balancing экспертов

        Shape Flow:
        -----------
            input_ids: (B, S) → embeddings: (B, S, H)
            → transformer blocks → hidden_states: (B, S, H)
            → norm → normalized: (B, S, H)
            → lm_head → logits: (B, S, V)

        Examples:
        ---------
            >>> model = Qwen3MoEModel(config)
            >>> input_ids = torch.randint(0, 50257, (4, 32))  # batch=4, seq=32
            >>> logits, loss = model(input_ids)
            >>> print(f"Logits: {logits.shape}, Loss: {loss.item():.4f}")
            Logits: torch.Size([4, 32, 50257]), Loss: 0.0234
        """
        # TODO: Преобразуйте input_ids → embeddings через self.embed_tokens
        # TODO: Инициализируйте total_balance_loss нулевым тензором на device embeddings
        # TODO: Пройдите циклом по self.layers, накапливая balance_loss
        # TODO: Примените финальную нормализацию self.norm
        # TODO: Спроецируйте через self.lm_head в vocab space
        # TODO: Верните (logits, total_balance_loss)

        # Вопросы для размышления:
        # - Почему важно указать device при создании total_balance_loss?
        # - Что возвращает каждый MoE блок?
        # - Чем logits отличаются от вероятностей?
        pass

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Автогрессивная генерация текста.

        Стратегия:
        ----------
        1. Начинаем с input_ids (prompt)
        2. В цикле (до max_length):
           a. Forward pass: получаем logits для следующего токена
           b. Применяем temperature/top-k/top-p
           c. Сэмплируем следующий токен
           d. Добавляем токен к последовательности
        3. Возвращаем сгенерированную последовательность

        Args:
        -----
            input_ids: Начальная последовательность (prompt) формы (batch, seq_len)
            max_length: Максимальная длина генерируемой последовательности
            temperature: Температура для сэмплирования (>1 = более случайно, <1 = более детерминированно)
            top_k: Оставить только k самых вероятных токенов (nucleus sampling)
            top_p: Оставить минимальное множество токенов с суммарной вероятностью ≥ p
            do_sample: True = сэмплирование, False = greedy (argmax)

        Returns:
        --------
            generated_ids: Сгенерированная последовательность формы (batch, max_length)

        Examples:
        ---------
            >>> # Greedy decoding
            >>> output = model.generate(input_ids, max_length=50, do_sample=False)
            >>>
            >>> # Nucleus sampling (top-p)
            >>> output = model.generate(input_ids, temperature=0.8, top_p=0.9)
            >>>
            >>> # Top-k sampling
            >>> output = model.generate(input_ids, temperature=1.0, top_k=50)
        """
        # NOTE: Этот метод будет реализован на следующем этапе
        # После того, как forward() заработает и пройдут тесты
        raise NotImplementedError(
            "Метод generate() будет реализован после завершения forward(). "
            "Текущий приоритет: базовая модель и тесты."
        )

    def chat(self, prompt: str, max_length: int = 100, **generation_kwargs) -> str:
        """
        Высокоуровневый интерфейс text-to-text.

        Args:
        -----
            prompt: Входной текст от пользователя
            max_length: Максимальная длина ответа
            **generation_kwargs: Параметры для generate() (temperature, top_k, top_p)

        Returns:
        --------
            response: Сгенерированный текст

        Examples:
        ---------
            >>> response = model.chat("Привет! Как дела?", temperature=0.7)
            >>> print(response)
            "Привет! Всё отлично, спасибо! Как у тебя дела?"
        """
        # NOTE: Этот метод будет реализован на последнем этапе
        # Требует интеграции с tokenizer
        raise NotImplementedError(
            "Метод chat() будет реализован после завершения generate() и интеграции tokenizer."
        )
