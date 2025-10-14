"""
OptimizedMoELayer - векторизованная версия MoE для production использования.

В отличие от SimpleMoELayer (учебная версия с циклами), эта реализация использует
batch operations для максимальной производительности на GPU. Ключевая идея:
вместо обработки токенов по одному, группируем все токены каждого эксперта
и обрабатываем батчем.

Speedup: 2-3x по сравнению с SimpleMoELayer при сохранении численной эквивалентности.
"""


# Стандартная библиотека
from typing import Tuple, Optional

# Сторонние библиотеки
import torch
import torch.nn as nn

# Локальные импорты
from experiments.domain.moe.router import MoERouter
from experiments.domain.moe.expert import Expert


class OptimizedMoELayer(nn.Module):
    """
    Description:
    ---------------
        Оптимизированная (векторизованная) реализация MoE Layer для production.

        Эта версия использует batch operations вместо циклов для максимальной
        производительности на GPU. API полностью совместим с SimpleMoELayer.

        Архитектура (3 фазы):
        Input → Router → Phase 1 (Flatten) → Phase 2 (Parallel Process) →
        Phase 3 (Combine) → Residual → Output

        Pipeline:
        1. Router: выбирает top_k экспертов для каждого токена
        2. Phase 1 (Flatten):
           - Expand токены для K выборов: (B,S,H) → (B,S,K,H)
           - Flatten всё в 1D: (B,S,K,H) → (B*S*K, H)
        3. Phase 2 (Parallel Process):
           - Для каждого эксперта: batch обработка всех его токенов
           - Boolean masking: experts_flat == expert_idx
           - Взвешивание: output * routing_weights
        4. Phase 3 (Combine):
           - Reshape: (B*S*K, H) → (B,S,K,H)
           - Sum по оси K: (B,S,K,H) → (B,S,H)
        5. Residual: добавляет входной тензор к выходному

        Для модели 0.6B:
        - num_experts = 8
        - top_k = 2 (каждый токен → 2 эксперта)
        - hidden_size = 512
        - intermediate_size = 2048

    Mathematical Flow:
    ---------------
        x ∈ ℝ^(B×S×H)
            ↓
        Router: (weights, experts_idx, loss) = Router(x)
            weights ∈ ℝ^(B×S×K)      # Веса для K экспертов
            experts_idx ∈ ℤ^(B×S×K)  # Индексы экспертов [0, N)
            ↓
        Flatten: x_flat ∈ ℝ^(B*S*K × H)
            ↓
        For each expert i in parallel:
            mask_i = (experts_idx == i)
            tokens_i = x_flat[mask_i]
            outputs_i = Expert_i(tokens_i) * weights[mask_i]
            ↓
        Combine: reshape → sum(K) → (B×S×H)
            ↓
        output = output + x  # Residual connection
            ↓
        return output, loss

    Args:
    ---------------
        hidden_size: Размерность входа/выхода (должна совпадать с моделью)
        num_experts: Количество экспертов (8 для модели 0.6B)
        top_k: Количество активных экспертов per token (2 для модели 0.6B)
        intermediate_size: Размерность промежуточного слоя экспертов (обычно 4*hidden_size)
        expert_dropout: Dropout для экспертов (default: 0.0)
        capacity_factor: Фактор емкости для Router (default: 1.25)
        balance_loss_coef: Коэффициент для load balancing loss (default: 0.01)

    Returns (from forward):
    ---------------
        output: Тензор формы (batch_size, seq_len, hidden_size)
                Выходные скрытые состояния после MoE обработки
        balance_loss: Скаляр - load balancing loss для обучения

    Example:
    ---------------
        >>> # Создание оптимизированной MoE Layer для модели 0.6B
        >>> moe = OptimizedMoELayer(
        ...     hidden_size=512,
        ...     num_experts=8,
        ...     top_k=2,
        ...     intermediate_size=2048
        ... )
        >>> x = torch.randn(2, 10, 512)  # (batch=2, seq=10, hidden=512)
        >>> output, loss = moe(x, training=True)
        >>> output.shape  # torch.Size([2, 10, 512])
        >>> loss.item()   # Скаляр loss

    Note:
    ---------------
        Эта версия для PRODUCTION использования. Использует векторизованные
        batch операции для максимальной производительности. Численно эквивалентна
        SimpleMoELayer (до точности float32).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        intermediate_size: int = 2048,
        expert_dropout: float = 0.0,
        capacity_factor: float = 1.25,
        balance_loss_coef: float = 0.01
    ):
        super().__init__()

        # TODO(human): Валидация параметров
        # TODO(human): Сохраните параметры как атрибуты класса
        # TODO(human): Создайте self.router - экземпляр MoERouter
        # TODO(human): Создайте self.experts - nn.ModuleList из num_experts экспертов

        # Вопросы для размышления:
        # - Почему используем nn.ModuleList, а не обычный Python list?
        # - Будет ли эта версия API-совместима с SimpleMoELayer?
        # - Какие параметры влияют на memory usage?
        # pass

        assert hidden_size > 0, "hidden_size должен быть > 0"
        assert num_experts > 0, "num_experts должен быть > 0"
        assert top_k > 0 and top_k <= num_experts, "top_k должен быть > 0 и <= num_experts"
        assert intermediate_size > 0, "intermediate_size должен быть > 0"

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.expert_dropout = expert_dropout
        self.capacity_factor = capacity_factor
        self.balance_loss_coef = balance_loss_coef

        self.router = MoERouter(
            hidden_size = hidden_size,
            num_experts = num_experts,
            top_k = top_k,
            capacity_factor = capacity_factor,
            balance_loss_coef = balance_loss_coef
        )

        self.experts = nn.ModuleList([
            Expert(
                hidden_size = hidden_size,
                intermediate_size = intermediate_size,
                dropout = expert_dropout
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Description:
        ---------------
            Применяет векторизованную MoE трансформацию к входным скрытым состояниям.

            Оптимизированная реализация через batch operations:
            1. Router выбирает экспертов для всех токенов
            2. Phase 1 - Flatten: (B,S,H) → (B,S,K,H) → (B*S*K, H)
            3. Phase 2 - Parallel Process: batch обработка каждым экспертом
            4. Phase 3 - Combine: (B*S*K, H) → (B,S,K,H) → sum(K) → (B,S,H)
            5. Residual connection

        Args:
        ---------------
            hidden_states: Входной тензор формы (batch_size, seq_len, hidden_size)
            training: Флаг режима обучения (для balance loss)

        Returns:
        ---------------
            output: Тензор формы (batch_size, seq_len, hidden_size)
                   Выходные скрытые состояния
            balance_loss: Скаляр - load balancing loss

        Shape Transformations:
        ---------------
            hidden_states:     (B, S, H)
                ↓ unsqueeze(2)
            tokens:            (B, S, 1, H)
                ↓ expand(-1, -1, K, -1)
            tokens_expanded:   (B, S, K, H)
                ↓ reshape(-1, H)
            tokens_flat:       (B*S*K, H)
                ↓ expert processing + weighting
            expert_outputs:    (B*S*K, H)
                ↓ reshape(B, S, K, H)
            expert_outputs:    (B, S, K, H)
                ↓ sum(dim=2)
            combined:          (B, S, H)
                ↓ residual
            output:            (B, S, H)
        """
        # TODO(human): Шаг 0 - Router
        #       Получите routing_weights, selected_experts, balance_loss от self.router

        # TODO(human): Шаг 1 - Flatten для batch processing
        #       1.1. Извлеките размерности: hidden_states.shape
        #       1.2. Сохраните self.top_k
        #       1.3. Expand токены для K выборов:
        #            Преобразуйте hidden_states: (B, S, H) → (B, S, 1, H) → (B, S, K, H)
        #       1.4. Flatten всё в 1D:
        #            tokens_flat  = (B*S*K, H)
        #            weights_flat = (B*S*K,)
        #            experts_flat = (B*S*K,)

        # TODO(human): Шаг 2 - Parallel Expert Processing
        #       2.1. Создайте output тензор:
        #            expert_outputs = torch.zeros_like(tokens_flat)  # (B*S*K, H)
        #       2.2. Для каждого эксперта (цикл от 0 до num_experts):
        #            a) Создайте boolean маску: mask = (experts_flat == expert_idx)
        #            b) Проверьте: if mask.sum() > 0 (skip пустых экспертов)
        #            c) Извлеките токены эксперта: expert_tokens = tokens_flat[mask]
        #            d) Обработайте батчем: output = self.experts[expert_idx](expert_tokens)
        #            e) Взвесьте по routing_weights:
        #               expert_weights = weights_flat[mask].unsqueeze(-1)  # (num_tokens, 1)
        #               weighted_output = output * expert_weights          # (num_tokens, H)
        #            f) Запишите обратно: expert_outputs[mask] = weighted_output

        # TODO(human): Шаг 3 - Combine - суммируем K вкладов для каждого токена
        #       3.1. Reshape: expert_outputs = expert_outputs.reshape(B, S, K, H)
        #       3.2. Суммируем по оси K: combined = expert_outputs.sum(dim=2)  # (B, S, H)

        # TODO(human): Шаг 4 - Residual connection
        #       output = combined + hidden_states

        # TODO(human): Шаг 5 - Return
        #       return output, balance_loss

        # Вопросы для размышления:
        # - Почему мы используем unsqueeze(2).expand(), а не repeat()?
        # - Зачем проверять mask.sum() > 0 перед вызовом эксперта?
        # - Как weights_flat[mask].unsqueeze(-1) влияет на broadcasting?
        # - Почему sum(dim=2) корректно объединяет K вкладов?
        # - В чём разница между этой реализацией и SimpleMoELayer в плане памяти?
        # pass

        # ════════════════════════════════════════════════════════════════════
        # Шаг 0: Router - выбираем top_k экспертов для каждого токена
        # ════════════════════════════════════════════════════════════════════
        routing_weights, selected_experts, load_balance_loss = self.router(hidden_states, training)

        # ════════════════════════════════════════════════════════════════════
        # Шаг 1: Flatten - подготовка для batch processing
        # ════════════════════════════════════════════════════════════════════
        # Извлекаем размерности входного тензора
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Сохраняем top_k для удобства
        top_k = self.top_k

        # ────────────────────────────────────────────────────────────────────
        # Шаг 1.1: Expand токены для K выборов (memory-efficient дублирование)
        # ────────────────────────────────────────────────────────────────────
        # Трансформация: (B, S, H) → (B, S, 1, H) → (B, S, K, H)
        # unsqueeze(2): добавляем новую ось размера 1 на позиции 2
        # expand(): "растягиваем" ось 1 → K (БЕЗ копирования в памяти!)
        # Результат: каждый токен виртуально дублирован K раз (для K экспертов)
        # Исходный токен (вектор):
        # hidden_states[b=0, s=0] = [1, 2, 3, 4]  # shape: (H=4,)

        # # После unsqueeze(2).expand(..., K=2, ...):
        # tokens[b=0, s=0] = [
        #     [1, 2, 3, 4],  # k=0 (копия для первого эксперта)
        #     [1, 2, 3, 4]   # k=1 (копия для второго эксперта)
        # ]  # shape: (K=2, H=4) - это матрица!
        tokens = hidden_states.unsqueeze(2).expand(batch_size, seq_len, top_k, hidden_size)

        # ────────────────────────────────────────────────────────────────────
        # Шаг 1.2: Flatten всё в 1D для векторизованной обработки
        # ────────────────────────────────────────────────────────────────────
        # Flatten: (B, S, K, H) → (B*S*K, H)
        # "Разворачиваем" 4D тензор в 2D матрицу (список векторов)

        # До reshape (4D): tokens.shape = (B=2, S=3, K=2, H=4)
        # tokens = [
        #   # Batch 0:
        #   [ [[1,2,3,4], [1,2,3,4]],    # s=0: матрица 2×4
        #     [[5,6,7,8], [5,6,7,8]],    # s=1: матрица 2×4
        #     [[9,10,11,12], [9,10,11,12]] ], # s=2: матрица 2×4
        #   # Batch 1: аналогично...
        # ]

        # После reshape (2D): tokens_flat.shape = (B*S*K=12, H=4)
        # tokens_flat = [
        #   [1, 2, 3, 4],      # индекс 0: (b=0, s=0, k=0)
        #   [1, 2, 3, 4],      # индекс 1: (b=0, s=0, k=1)
        #   [5, 6, 7, 8],      # индекс 2: (b=0, s=1, k=0)
        #   [5, 6, 7, 8],      # индекс 3: (b=0, s=1, k=1)
        #   [9, 10, 11, 12],   # индекс 4: (b=0, s=2, k=0)
        #   [9, 10, 11, 12],   # индекс 5: (b=0, s=2, k=1)
        #   # ... batch 1: индексы 6-11
        # ]
        # ⚠️ Порядок flatten: сначала batch, потом sequence, потом K
        tokens_flat = tokens.reshape(-1, hidden_size)  # (B*S*K, H)

        # Flatten весов: (B, S, K) → (B*S*K,)
        # weights_flat[i] = вес для tokens_flat[i]
        weights_flat = routing_weights.reshape(-1)     # (B*S*K,)

        # Flatten индексов экспертов: (B, S, K) → (B*S*K,)
        # experts_flat[i] = какой эксперт должен обработать tokens_flat[i]
        experts_flat = selected_experts.reshape(-1)    # (B*S*K,)

        # ⚠️ ВАЖНО: После flatten все 3 тензора синхронизированы по индексу:
        #   tokens_flat[i]   - токен для обработки
        #   weights_flat[i]  - вес результата при комбинировании
        #   experts_flat[i]  - индекс эксперта [0, num_experts)

        # ════════════════════════════════════════════════════════════════════
        # Шаг 2: Parallel Expert Processing - batch обработка каждым экспертом
        # ════════════════════════════════════════════════════════════════════
        # Инициализируем выходной тензор нулями (будем заполнять по маскам)
        # Каждый токен в tokens_flat будет обработан ровно 1 раз (по своему эксперту)
        expert_outputs = torch.zeros_like(tokens_flat)  # (B*S*K, H)

        # Цикл по экспертам: каждый обрабатывает свою группу токенов батчем
        # ⚠️ ВАЖНО: Это единственный цикл в оптимизированной версии!
        #   SimpleMoELayer: 3 вложенных цикла (batch × sequence × top_k)
        #   OptimizedMoELayer: 1 цикл (num_experts), внутри - batch operations
        for expert_idx in range(self.num_experts):
            # ────────────────────────────────────────────────────────────────
            # Шаг 2.1: Boolean masking - находим все токены для этого эксперта
            # ────────────────────────────────────────────────────────────────
            # mask = True для позиций, где experts_flat == expert_idx
            # Например, если expert_idx=3, то mask выделит все токены,
            # которые Router назначил третьему эксперту
            mask = (experts_flat == expert_idx)  # (B*S*K,) - boolean тензор

            # Пример mask для expert_idx=0:
            # experts_flat = [0, 2, 0, 5, 0, 1, ...]
            # mask         = [T, F, T, F, T, F, ...]
            # Где T означает "этот токен для эксперта 0"

            # ────────────────────────────────────────────────────────────────
            # Шаг 2.2: Skip пустых экспертов (оптимизация)
            # ────────────────────────────────────────────────────────────────
            # Если Router не назначил ни одного токена этому эксперту, пропускаем
            # Это экономит время на forward pass пустого эксперта
            if mask.sum() > 0:
                # ────────────────────────────────────────────────────────────
                # Шаг 2.3: Извлечение токенов эксперта через boolean indexing
                # ────────────────────────────────────────────────────────────
                # Извлекаем только те токены, для которых mask == True
                # Это создаёт новый тензор (компактный, без пустых мест)
                expert_tokens = tokens_flat[mask]  # (num_selected_tokens, H)

                # Пример:
                # tokens_flat = [
                #   [1, 2, 3, 4],   # индекс 0 (mask=True для expert_idx=0)
                #   [5, 6, 7, 8],   # индекс 1 (mask=False)
                #   [9, 10, 11, 12] # индекс 2 (mask=True для expert_idx=0)
                # ]
                # expert_tokens = [
                #   [1, 2, 3, 4],   # из индекса 0
                #   [9, 10, 11, 12] # из индекса 2
                # ]  # shape: (2, H) - только выбранные токены!

                # ────────────────────────────────────────────────────────────
                # Шаг 2.4: Batch обработка экспертом
                # ────────────────────────────────────────────────────────────
                # Вызываем эксперта ОДИН РАЗ для ВСЕХ его токенов
                # Это ключ к ускорению: вместо N вызовов - 1 вызов с батчем
                output = self.experts[expert_idx](expert_tokens)  # (num_selected_tokens, H)

                # ────────────────────────────────────────────────────────────
                # Шаг 2.5: Взвешивание по routing weights
                # ────────────────────────────────────────────────────────────
                # Извлекаем веса для выбранных токенов (используя ту же маску)
                # unsqueeze(-1): (num_tokens,) → (num_tokens, 1) для broadcasting
                expert_weights = weights_flat[mask].unsqueeze(-1)  # (num_selected_tokens, 1)

                # Broadcasting: (num_tokens, H) * (num_tokens, 1) → (num_tokens, H)
                # Каждая строка output умножается на свой скалярный вес
                # Пример:
                # output = [[1, 2], [3, 4]]       # (2, 2)
                # weights = [[0.7], [0.3]]        # (2, 1)
                # result = [[0.7, 1.4], [0.9, 1.2]] # (2, 2)
                weighted_output = output * expert_weights  # (num_selected_tokens, H)

                # ────────────────────────────────────────────────────────────
                # Шаг 2.6: Запись обратно в исходные позиции
                # ────────────────────────────────────────────────────────────
                # Используем ту же маску для записи результатов обратно
                # Boolean indexing работает и для присваивания!
                expert_outputs[mask] = weighted_output

                # Визуализация процесса заполнения:
                # До обработки: expert_outputs = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
                # После expert_idx=0: expert_outputs = [[1,2,3,4], [0,0,0,0], [9,10,11,12]]
                # После expert_idx=1: expert_outputs = [[1,2,3,4], [5,6,7,8], [9,10,11,12]]

        # ════════════════════════════════════════════════════════════════════
        # Шаг 3: Combine - суммируем K вкладов для каждого токена
        # ════════════════════════════════════════════════════════════════════
        # ────────────────────────────────────────────────────────────────────
        # Шаг 3.1: Reshape обратно в 4D структуру
        # ────────────────────────────────────────────────────────────────────
        # Восстанавливаем исходную структуру: (B*S*K, H) → (B, S, K, H)
        # Это обратная операция к flatten из Шага 1.2
        expert_outputs = expert_outputs.reshape(batch_size, seq_len, top_k, hidden_size)

        # Визуализация reshape:
        # До reshape (2D): expert_outputs.shape = (B*S*K=12, H=4)
        # expert_outputs_flat = [
        #   [1, 2, 3, 4],      # (b=0, s=0, k=0) - вклад эксперта 0
        #   [0.5, 1, 1.5, 2],  # (b=0, s=0, k=1) - вклад эксперта 2
        #   [5, 6, 7, 8],      # (b=0, s=1, k=0)
        #   [2.5, 3, 3.5, 4],  # (b=0, s=1, k=1)
        #   ...
        # ]
        #
        # После reshape (4D): expert_outputs.shape = (B=2, S=3, K=2, H=4)
        # expert_outputs = [
        #   [ # Batch 0
        #     [[1,2,3,4], [0.5,1,1.5,2]],           # s=0: K=2 вклада
        #     [[5,6,7,8], [2.5,3,3.5,4]],           # s=1: K=2 вклада
        #     [[9,10,11,12], [4.5,5,5.5,6]]         # s=2: K=2 вклада
        #   ],
        #   [ # Batch 1: аналогично... ]
        # ]

        # ────────────────────────────────────────────────────────────────────
        # Шаг 3.2: Суммирование по оси K (объединение вкладов экспертов)
        # ────────────────────────────────────────────────────────────────────
        # Суммируем по оси K (dim=2): каждый токен получает сумму от K экспертов
        # (B, S, K, H) → sum(dim=2) → (B, S, H)
        combined = expert_outputs.sum(dim=2)  # (B, S, H)

        # Визуализация суммирования:
        # До sum: expert_outputs[b=0, s=0] = [[1,2,3,4], [0.5,1,1.5,2]]  # (K=2, H=4)
        # После sum: combined[b=0, s=0] = [1.5, 3, 4.5, 6]  # (H=4) - сумма вкладов!
        #
        # Это и есть "взвешенное комбинирование" выходов экспертов.
        # Каждый эксперт внёс свой вклад (умноженный на routing_weight),
        # а мы их суммируем для финального представления токена.

        # ════════════════════════════════════════════════════════════════════
        # Шаг 4: Residual Connection
        # ════════════════════════════════════════════════════════════════════
        # Добавляем исходный вход к выходу MoE слоя
        # Это стандартная практика в Transformer архитектурах для:
        # 1. Стабилизации обучения (градиенты текут напрямую)
        # 2. Сохранения исходной информации (слой может научиться "не делать ничего")
        # 3. Улучшения сходимости (каждый слой учит только дельту)
        output = combined + hidden_states  # (B, S, H) + (B, S, H) → (B, S, H)

        # ════════════════════════════════════════════════════════════════════
        # Шаг 5: Return - возвращаем результат и loss
        # ════════════════════════════════════════════════════════════════════
        # output: финальные скрытые состояния после MoE трансформации
        # load_balance_loss: метрика для обучения (стимулирует равномерное использование экспертов)
        return output, load_balance_loss

