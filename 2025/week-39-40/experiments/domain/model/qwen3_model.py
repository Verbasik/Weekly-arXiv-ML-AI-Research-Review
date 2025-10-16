"""
Qwen3 MoE Language Model

Полная реализация генеративной языковой модели с MoE архитектурой.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import GPT2Tokenizer

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
        tokenizer: GPT2Tokenizer для encode/decode текста

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

    def __init__(self, config: Qwen3Config, tokenizer: Optional[GPT2Tokenizer] = None):
        super().__init__()
        # TODO: Инициализируйте все компоненты модели
        # Вопрос: Какие основные блоки нужны для полной модели?
        # Подсказка: Используйте config для параметров
        
        # TODO: Инициализация tokenizer (если None, загрузите GPT-2)

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

        self.config = config

        # Инициализация tokenizer для text ↔ token_ids
        # Используем GPT-2 tokenizer (vocab_size=50257), совместимый с config
        if tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            # Важно: добавляем pad_token, т.к. GPT-2 изначально его не имеет
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        # Token Embedding Layer: преобразование token IDs → continuous vectors
        # Создаёт таблицу размера (y = vocab_size, x = hidden_size)
        self.embed_tokens = nn.Embedding(
            num_embeddings = config.vocab_size, 
            embedding_dim  = config.hidden_size
            )

        # Стек из N transformer блоков
        # Каждый блок содержит: RMSNorm → GQA → RMSNorm → SimpleMoELayer
        self.layers = nn.ModuleList([
            MoETransformerBlock(
                hidden_size=config.hidden_size,
                num_query_groups=config.num_key_value_heads,
                num_attention_heads=config.num_attention_heads,
                num_experts=config.num_experts,
                top_k=config.top_k,
                intermediate_size=config.intermediate_size,
                expert_dropout = config.dropout,
                balance_loss_coef=config.balance_loss_coef
            ) for _ in range(config.num_layers)
        ])

        # Финальная нормализация скрытых состояний
        self.norm = RMSNorm(normalized_shape = self.config.hidden_size)

        # Проекция hidden_size → vocab_size для предсказания токенов
        # y = x Aᵀ (без bias для LM head)
        self.lm_head = nn.Linear(
            in_features  = self.config.hidden_size,
            out_features = self.config.vocab_size,
            bias         = False
        )

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """
        Description:
        ---------------
            Инициализация весов модели.

        Стратегия:
        ----------
        - Embeddings: normal distribution N(0, initializer_range)
        - Linear layers: уже инициализированы в sub-модулях
        - LM Head: normal distribution N(0, initializer_range)
        """
        # TODO: Инициализируйте веса эмбеддингов и LM head

        # Вопросы для размышления:
        # - Почему инициализация важна для стабильного обучения?
        # - Как влияет stddev на обучение?
        # - Почему линейные слои не требуют дополнительной инициализации?

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
        Description:
        ---------------
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
        # TODO: Преобразуйте input_ids → embeddings через эмбендинг слой
        # TODO: Инициализируйте total_balance_loss нулевым тензором на device embeddings
        # TODO: Пройдите циклом по self.layers, накапливая balance_loss
        # TODO: Примените финальную нормализацию self.norm
        # TODO: Спроецируйте через self.lm_head в vocab space
        # TODO: Верните (logits, total_balance_loss)

        # Вопросы для размышления:
        # - Почему важно указать device при создании total_balance_loss?
        # - Что возвращает каждый MoE блок?
        # - Чем logits отличаются от вероятностей?
        # pass

        # Преобразование token IDs в embeddings через lookup table
        embeddings = self.embed_tokens(input_ids)
        
        # Инициализация тензора для накопления balance loss из всех MoE блоков
        # Важно: используем device от embeddings для совместимости с GPU/CPU
        total_balance_loss = torch.tensor(
            data=0.0,
            device=embeddings.device
        )
        
        # Проход через все transformer блоки с накоплением balance loss
        # Каждый layer - это экземпляр MoETransformerBlock, который возвращает:
        # 1. Обработанные embeddings (RMSNorm → GQA → RMSNorm → SimpleMoELayer)
        # 2. Balance loss для load balancing экспертов
        for layer in self.layers:
            embeddings, balance_loss = layer(embeddings, attention_mask)
            total_balance_loss += balance_loss

        # Финальная нормализация скрытых состояний перед LM head
        final_norm = self.norm(embeddings)
        
        # Проекция в пространство словаря для предсказания следующего токена
        logits = self.lm_head(final_norm)

        return logits, total_balance_loss

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
        Description:
        ---------------
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
        # TODO: Инициализация переменных для генерации
        # - Скопировать input_ids для безопасного изменения
        # - Инициализировать key-value cache (если используется)
        # - Вычислить начальную длину последовательности
        # - Подготовить attention mask для начальной последовательности
        
        # TODO: Основной цикл автогрессивной генерации
        # - while current_length < max_length:
        #   a. Forward pass: получить logits для последнего токена
        #   b. Извлечь logits только для последней позиции (shape: [batch, vocab_size])
        #   c. Применить temperature scaling: logits = logits / temperature
        #   d. Применить top-k фильтрацию (если задан top_k)
        #   e. Применить top-p (nucleus) фильтрацию (если задан top_p)
        #   f. Вычислить вероятности через softmax
        #   g. Сэмплировать следующий токен (greedy или sampling)
        #   h. Добавить токен к последовательности
        #   i. Обновить attention mask для новой длины
        #   j. Обновить key-value cache (если используется)
        
        # TODO: Обработка критериев остановки
        # - Проверить специальные токены окончания (если есть)
        # - Обрезать до максимальной длины
        # - Вернуть финальную последовательность
        
        # TODO: Вопросы для размышления:
        # - Как эффективно обновлять attention mask при росте последовательности?
        # - Какой формат должен иметь key-value cache для MoE блоков?
        # - Как обрабатывать batch с разными длинами последовательностей?
        # - Как оптимизировать memory usage для длинных последовательностей?
        
        # pass

        # Инициализация переменных для генерации
        generated_ids  = input_ids.clone()
        current_length = input_ids.shape[1]
        # Создаем тензор такой же размерности как input_ids, но заполненный единицами
        attention_mask = torch.ones_like(input_ids)

        while current_length < max_length:
            # a. Forward pass: получить logits для всей последовательности
            logits, _ = self.forward(generated_ids, attention_mask)

            # b. Извлечь logits только для последней позиции (shape: [batch, vocab_size])
            logits = logits[:, -1, :]

            # c. Применить temperature scaling
            # -------------------------------------------------------------
            # Это стандартная формула в LLM!
            # Математика: temperature "сжимает" или "растягивает" logits

            # Temperature > 1: "разогревает" распределение
            # - Более равномерные вероятности
            # - Больше случайности в генерации

            # Temperature < 1: "охлаждает" распределение
            # - Более острые пики вероятностей
            # - Более детерминированная генерация

            # Temperature = 1: без изменений (стандартный softmax)
            probabilities = torch.softmax(logits / temperature, dim=-1)
            # -------------------------------------------------------------

            # d. Применить top-k фильтрацию (если задан)
            # e. Применить top-p фильтрацию (если задан)
            # -------------------------------------------------------------
            # TOP-K алгоритм:
            # 1. Найти k самых вероятных токенов
            # 2. Обнулить ВСЕ остальные вероятности
            # 3. Оставить только top-k токенов

            # TOP-P (nucleus) алгоритм:
            # 1. Отсортировать токены по убыванию вероятности
            # 2. Накапливать вероятности до достижения порога p
            # 3. Обнулить все токены после этого порога

            # Пример:
            # probabilities = [0.4, 0.3, 0.2, 0.1]
            # top_k=2:   [0.4, 0.3, 0.0, 0.0]  # только 2 лучших
            # top_p=0.6: [0.4, 0.3, 0.0, 0.0]  # накопили до 0.7 > 0.6
            if top_k is not None:
                top_k_values, top_k_indices = torch.topk(probabilities, top_k)
                probabilities_filtered = torch.zeros_like(probabilities)
                probabilities_filtered.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
                probabilities = probabilities_filtered

            if top_p is not None:
                # Сортируем вероятности по убыванию
                sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
                # Вычисляем накопленную сумму
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # Находим токены, которые нужно удалить (cumsum > top_p)
                # Сдвигаем на 1 вправо, чтобы сохранить хотя бы первый токен
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                # Обнуляем вероятности для удаляемых токенов
                sorted_probs[sorted_indices_to_remove] = 0.0
                # Возвращаем вероятности в исходный порядок
                probabilities = torch.zeros_like(probabilities)
                probabilities.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
            # -------------------------------------------------------------

            # Ре-нормализуем вероятности после фильтрации
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)

            # g. Сэмплировать следующий токен (greedy или sampling)
            if do_sample:
                # Стохастическое сэмплирование из распределения
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                # Greedy decoding: выбираем токен с максимальной вероятностью
                next_token = torch.argmax(probabilities, dim=-1, keepdim=True)

            # h. Добавить токен к последовательности
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # i. Обновить attention mask для новой длины
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token)],
                dim=-1
            )

            # j. Обновить длину последовательности
            current_length += 1

        # Обработка критериев остановки и возврат результата
        return generated_ids


    def chat(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> str:
        """
        Description:
        ---------------
            Высокоуровневый интерфейс text-to-text.

        Pipeline:
        ---------
        1. Encode: prompt (str) → token_ids (tensor)
        2. Generate: token_ids → generated_ids (автогрессивная генерация)
        3. Decode: generated_ids → response (str)

        Args:
        -----
            prompt: Входной текст от пользователя
            max_length: Максимальная длина сгенерированного текста (в токенах)
            temperature: Температура для сэмплирования (по умолчанию 1.0)
            top_k: Количество токенов для top-k фильтрации (опционально)
            top_p: Порог для nucleus sampling (опционально)
            do_sample: True = стохастическое сэмплирование, False = greedy decoding

        Returns:
        --------
            response: Сгенерированный текст (включая исходный prompt)

        Examples:
        ---------
            >>> # Greedy decoding (детерминированный)
            >>> response = model.chat("Once upon a time", do_sample=False)
            >>> print(response)
            "Once upon a time there was a kingdom..."

            >>> # Nucleus sampling (более креативный)
            >>> response = model.chat("Hello world", temperature=0.8, top_p=0.9)
            >>> print(response)
            "Hello world! How are you doing today?"

            >>> # Top-k sampling
            >>> response = model.chat("The quick brown", temperature=1.0, top_k=40)
            >>> print(response)
            "The quick brown fox jumps over the lazy dog"
        """
        # TODO: Реализуйте три шага: Encode → Generate → Decode
        # TODO: Используйте self.tokenizer для encode/decode
        # TODO: Используйте self.generate() для генерации
        # TODO: Верните сгенерированный текст

        # Вопросы для размышления:
        # - Почему важно использовать тот же tokenizer, что и при обучении?
        # - Как обработать специальные токены (BOS/EOS/PAD) при encode/decode?
        # - Как гарантировать, что сгенерированный текст не превышает max_length?


        # Шаг 1: Encode - преобразование текста в token IDs
        # return_tensors="pt" возвращает PyTorch тензоры
        # add_special_tokens=True добавляет специальные токены (BOS/EOS если есть)
        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        )

        # Перемещаем на то же устройство, где находится модель
        # Проверяем device через параметры модели (например, embed_tokens.weight)
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)

        # Шаг 2: Generate - автогрессивная генерация через self.generate()
        generated_ids = self.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample
        )

        # Шаг 3: Decode - преобразование token IDs обратно в текст
        # skip_special_tokens=True удаляет специальные токены (BOS/EOS/PAD)
        # Берём первую (и единственную) последовательность из batch: generated_ids[0]
        response = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )

        return response
