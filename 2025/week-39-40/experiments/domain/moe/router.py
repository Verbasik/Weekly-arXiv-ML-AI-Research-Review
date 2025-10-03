# Стандартная библиотека
import math
from typing import Tuple, Optional

# Сторонние библиотеки
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    """
    Description:
    ---------------
        MoE Router (Mixture-of-Experts Router) для архитектуры Qwen3.

        Роутер решает для каждого токена:
        1. Какие K экспертов из N активировать (Top-K selection)
        2. С какими весами комбинировать их выходы (gating weights)
        3. Как балансировать нагрузку между экспертами (load balancing)

        Архитектура:
        Input Token → Linear Projection → Softmax → Top-K Selection → Gating Weights

        Для этой модели (0.6B): N=8 экспертов, K=2 активных per token
        Для справки, Qwen3-30B использует: N=128 экспертов, K=8 активных per token

    Mathematical Formulation:
    ---------------
        1. Gating scores: g = Softmax(W_g * x)
           где W_g - обучаемая матрица размера (hidden_size, num_experts)

        2. Top-K selection: indices, weights = TopK(g, k=top_k)
           Выбираем K экспертов с наибольшими весами

        3. Renormalization: weights = Softmax(weights)
           Нормализуем веса выбранных экспертов (сумма = 1)

        4. Load balancing loss: L_balance = α * mean(f * P)
           где f - частота выбора эксперта, P - средний вес эксперта

    Args:
    ---------------
        hidden_size: Размерность входного скрытого состояния
        num_experts: Общее количество экспертов (N)
        top_k: Количество активных экспертов per token (K)
        capacity_factor: Фактор емкости для ограничения токенов per expert (default: 1.25)
        balance_loss_coef: Коэффициент для load balancing loss (default: 0.01)

    Returns (from forward):
    ---------------
        routing_weights: Тензор формы (batch_size, seq_len, top_k)
                        Веса для каждого из K выбранных экспертов
        selected_experts: Тензор формы (batch_size, seq_len, top_k) dtype=long
                         Индексы выбранных экспертов [0, num_experts)
        balance_loss: Скаляр - loss для балансировки нагрузки между экспертами

    Example:
    ---------------
        >>> # Для модели 0.6B
        >>> router = MoERouter(hidden_size=512, num_experts=8, top_k=2)
        >>> x = torch.randn(2, 10, 512)  # (batch=2, seq=10, hidden=512)
        >>> weights, experts, loss = router(x)
        >>> weights.shape  # torch.Size([2, 10, 2])
        >>> experts.shape  # torch.Size([2, 10, 2])
        >>> loss.item()    # Скаляр loss
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        balance_loss_coef: float = 0.01
    ):
        super().__init__()

        # TODO: Проверьте валидность параметров (hidden_size, num_experts, top_k)
        # TODO: Убедитесь что top_k <= num_experts
        # TODO: Сохраните все параметры как атрибуты класса
        # TODO: Создайте self.gate - линейный слой для проекции в пространство экспертов
        #       Размеры: (hidden_size) -> (num_experts)
        # TODO: Инициализируйте веса gate небольшими значениями для стабильности

        # Вопросы для размышления:
        # - Почему важно, чтобы top_k был меньше num_experts?
        # - Как capacity_factor влияет на балансировку нагрузки?
        # - Зачем нужна небольшая инициализация весов gate?
        # - Какие альтернативы Softmax можно использовать для gating?
        # pass

        # --- Валидация параметров -------------------------------------------------
        # Используем assert для раннего обнаружения ошибок конфигурации.
        assert isinstance(hidden_size, int) and hidden_size > 0, (
            "hidden_size должен быть положительным целым числом"
        )
        assert isinstance(num_experts, int) and num_experts > 0, (
            "num_experts должен быть положительным целым числом"
        )
        assert isinstance(top_k, int) and top_k > 0, (
            "top_k должен быть положительным целым числом"
        )
        assert top_k <= num_experts, (
            "num_experts должено быть больше или равно top_k"
        )
        assert isinstance(capacity_factor, float) and capacity_factor > 0, (
            "capacity_factor должен быть положительным числом"
        )
        assert isinstance(balance_loss_coef, float) and balance_loss_coef >= 0, (
            "balance_loss_coef должен быть неотрицательным числом"
        )

        # --- Инициализация атрибутов ----------------------------------------------
        # Храним параметры как атрибуты экземпляра, чтобы использовать их
        # при дальнейшей маршрутизации и расчёте регуляризаций.
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.balance_loss_coef = balance_loss_coef

        # Линейный слой-gate предсказывает логиты по экспертам на основе входного скрытого состояния
        # последующий softmax (как правило, в forward) превращает их в вероятности.
        self.gate = nn.Linear(hidden_size, num_experts)

        # Инициализация: небольшой нормальный шум ускоряет сходимость.
        self.gate.weight.data.normal_(0, 0.01)

        # Нулевой сдвиг предотвращает смещение распределения по экспертам на старте обучения
        # проверка на наличие bias — на случай future-refactor.
        if self.gate.bias is not None:
            self.gate.bias.data.zero_()


    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Description:
        ---------------
            Применяет MoE routing к входным скрытым состояниям.

            Процесс:
            1. Проекция входа в пространство экспертов
            2. Вычисление gating scores через Softmax
            3. Top-K selection - выбор K лучших экспертов
            4. Renormalization весов выбранных экспертов
            5. Вычисление load balancing loss (только при training=True)

        Args:
        ---------------
            hidden_states: Входной тензор формы (batch_size, seq_len, hidden_size)
            training: Флаг режима обучения (для load balancing loss)

        Returns:
        ---------------
            routing_weights: Тензор формы (batch_size, seq_len, top_k)
                           Нормализованные веса для выбранных экспертов
            selected_experts: Тензор формы (batch_size, seq_len, top_k)
                            Индексы выбранных экспертов
            balance_loss: Скаляр - load balancing loss (0.0 если training=False)
        """
        # TODO: Получите размерности входного тензора (batch_size, seq_len, hidden_size)
        # TODO: Примените self.gate к hidden_states для получения логитов
        # TODO: Примените Softmax по оси num_experts для получения gating_scores
        #       Это дает распределение вероятностей по всем экспертам
        # TODO: Используйте torch.topk для выбора top_k экспертов
        #       Получите: routing_weights (веса), selected_experts (индексы)
        # TODO: Ре-нормализуйте routing_weights через Softmax
        #       Важно: веса K выбранных экспертов должны суммироваться в 1
        # TODO: Если training=True, вычислите load balancing loss
        #       Используйте вспомогательный метод _compute_balance_loss
        # TODO: Верните (routing_weights, selected_experts, balance_loss)

        # Вопросы для размышления:
        # - Почему нужна ре-нормализация после Top-K selection?
        # - Что произойдет, если все токены выберут одних и тех же экспертов?
        # - Как Top-K selection влияет на вычислительную эффективность?
        # - Почему balance_loss вычисляется только при training=True?
        # pass

        batch_size, seq_len, hidden_size = hidden_states.shape

        # Linear projection (W·x + b)
        # Проекция токенов в пространство экспертов: (B, S, H) → (B, S, N)
        # Логиты для всех N экспертов (например, 8 для модели 0.6B)
        logits = self.gate(hidden_states)

        # Softmax по оси num_experts для получения gating_scores
        # Распределение вероятностей по всем экспертам
        gating_scores = F.softmax(
            input = logits,
            dim = -1
        )
        
        # Top-K selection - выбор K лучших экспертов
        # Получение routing_weights (веса) и selected_experts (индексы)
        routing_weights, selected_experts = torch.topk(
            input = gating_scores,
            k = self.top_k,
            dim = -1
        )

        # Renormalization весов выбранных экспертов
        # Нормализация весов K выбранных экспертов (сумма = 1)
        routing_weights = F.softmax(
            input = routing_weights,
            dim = -1
        )

        if training:
            load_balance_loss = self._compute_balance_loss(
                gating_scores = gating_scores,
                selected_experts = selected_experts
            )
        else:
            load_balance_loss = torch.tensor(0.0, device=gating_scores.device)

        return routing_weights, selected_experts, load_balance_loss


    def _compute_balance_loss(
        self,
        gating_scores: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """
        Description:
        ---------------
            Вычисляет load balancing loss для равномерного распределения нагрузки.

            Цель: Предотвратить ситуацию, когда модель использует только малую часть экспертов.

            Формула: L_balance = α * num_experts * Σ(f_i * P_i)
            где:
            - f_i - fraction of tokens routed to expert i
            - P_i - mean gating score for expert i
            - α - balance_loss_coef

        Args:
        ---------------
            gating_scores: Тензор формы (batch_size, seq_len, num_experts)
                          Softmax scores для всех экспертов
            selected_experts: Тензор формы (batch_size, seq_len, top_k)
                            Индексы выбранных экспертов

        Returns:
        ---------------
            balance_loss: Скаляр тензор - loss для балансировки
        """
        # TODO: Вычислите frequency (f_i) - сколько токенов выбрали каждого эксперта
        #       Подсказка: используйте torch.bincount или создайте one-hot и усредните
        # TODO: Вычислите mean gating probability (P_i) для каждого эксперта
        #       Подсказка: усредните gating_scores по batch и sequence dimensions
        # TODO: Вычислите loss = balance_loss_coef * num_experts * sum(f_i * P_i)
        # TODO: Верните balance_loss

        # Вопросы для размышления:
        # - Почему мы умножаем на num_experts в формуле?
        # - Как этот loss влияет на распределение нагрузки?
        # - Что произойдет, если balance_loss_coef слишком большой?
        # - Какие альтернативные метрики балансировки существуют?
        # pass

        if gating_scores.dim() != 3:
            raise ValueError("gating_scores должен иметь форму (B, S, N).")
        if selected_experts.dim() != 3:
            raise ValueError("selected_experts должен иметь форму (B, S, K).")
        if gating_scores.size(-1) != self.num_experts:
            raise ValueError("Последняя размерность gating_scores должна быть N.")

        # # .view() перестраивает тензор уже выбранных экспертов в новую форму, не изменяя данные.
        # Было: (batch_size, seq_len, top_k) = (2, 10, 8)
        # Стало: (160,) — все индексы в одном массиве, мы получаем один длинный одномерный вектор
        flattened_experts = selected_experts.view(-1)

        # expert_counts[i] = сколько раз эксперт i был выбран
        expert_counts = torch.bincount(
            flattened_experts,
            minlength=self.num_experts  # Гарантируем вектор длины num_experts (например, 8)
        )

        # Общее количество выборов = batch_size * seq_len * top_k
        batch_size, seq_len, top_k = selected_experts.shape
        total_selections = batch_size * seq_len * top_k

        # f_i = (количество раз, когда эксперт i был выбран) / (общее количество выборов)
        f_i = expert_counts.float() / total_selections

        # Зачем вычисляем среднее для gating_scores, когда это уже тензор вероятностей экспертов после softmax?
        # Потому что нам нужно знать среднюю уверенность модели в каждом эксперте по всем токенам. Это отличается от частоты выбора:
        #   - f_i = как часто эксперт попадает в Top-K (0 или 1 для каждого токена)
        #   - P_i = какую среднюю вероятность модель назначает эксперту (до Top-K)
        # Произведение f_i * P_i максимально, когда эксперт и часто выбирается, и модель в нём уверена → это дисбаланс → высокий loss → градиент штрафует.
        P_i = gating_scores.mean(dim=(0, 1))

        balance_loss = self.balance_loss_coef * self.num_experts * (f_i * P_i).sum()

        return balance_loss


    def expert_capacity(self, num_tokens: int) -> int:
        """
        Description:
        ---------------
            Вычисляет максимальную емкость каждого эксперта.

            Capacity = (num_tokens / num_experts) * capacity_factor * top_k

            Это ограничивает количество токенов, которые может обработать один эксперт,
            предотвращая перегрузку отдельных экспертов.

        Args:
        ---------------
            num_tokens: Общее количество токенов (batch_size * seq_len)

        Returns:
        ---------------
            capacity: Максимальное количество токенов per expert
        """
        # TODO: Вычислите базовую capacity = num_tokens / num_experts
        # TODO: Умножьте на capacity_factor для запаса
        # TODO: Умножьте на top_k (каждый токен идет к K экспертам)
        # TODO: Округлите до целого числа (ceil)
        # TODO: Верните capacity

        # Вопросы для размышления:
        # - Зачем нужен capacity_factor > 1.0?
        # - Что делать с токенами, превышающими capacity?
        # - Как capacity влияет на memory footprint?
        # pass

        capacity = math.ceil((num_tokens / self.num_experts) * self.capacity_factor * self.top_k)

        return capacity