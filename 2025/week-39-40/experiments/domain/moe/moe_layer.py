# Стандартная библиотека
from typing import Tuple, Optional

# Сторонние библиотеки
import torch
import torch.nn as nn

# Локальные импорты
from experiments.domain.moe.router import MoERouter
from experiments.domain.moe.expert import Expert


class SimpleMoELayer(nn.Module):
    """
    Description:
    ---------------
        Простая (наивная) реализация MoE Layer для обучения и тестирования.

        Эта версия использует простые циклы вместо оптимизированных тензорных операций.
        Идеально подходит для понимания логики MoE перед переходом к оптимизированной версии.

        Архитектура:
        Input → Router (выбор экспертов) → Dispatch → Experts → Combine → Residual → Output

        Pipeline:
        1. Router: выбирает top_k экспертов для каждого токена
        2. Dispatch: распределяет токены по выбранным экспертам
        3. Process: каждый эксперт обрабатывает свои токены
        4. Combine: собирает результаты с весами от Router
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
        For each token t in (B×S):
            output[t] = Σ(k=1 to K) weights[t,k] * Expert[experts_idx[t,k]](x[t])
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
        >>> # Создание MoE Layer для модели 0.6B
        >>> moe = SimpleMoELayer(
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
        Это ПРОСТАЯ версия для обучения. Использует циклы вместо
        оптимизированных batch операций. Для production используйте
        оптимизированную версию MoELayer.
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

        # TODO: Проверьте валидность параметров
        #       - hidden_size > 0
        #       - num_experts > 0
        #       - top_k > 0 и top_k <= num_experts
        #       - intermediate_size > 0
        # TODO: Сохраните параметры как атрибуты класса
        # TODO: Создайте self.router - экземпляр MoERouter
        #       Параметры: hidden_size, num_experts, top_k, capacity_factor, balance_loss_coef
        # TODO: Создайте self.experts - nn.ModuleList из num_experts экспертов
        #       Каждый эксперт: Expert(hidden_size, intermediate_size, expert_dropout)

        # Вопросы для размышления:
        # - Почему используем nn.ModuleList, а не обычный Python list?
        # - Зачем нужен residual connection в MoE Layer?
        # - Как top_k влияет на вычислительную сложность?
        # - Что произойдет, если эксперт получит 0 токенов?
        pass


    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Description:
        ---------------
            Применяет MoE трансформацию к входным скрытым состояниям.

            Наивная реализация через циклы (простая, но медленная):
            1. Router выбирает экспертов
            2. Для каждого токена:
               - Берём top_k экспертов
               - Обрабатываем токен каждым экспертом
               - Взвешенное суммирование результатов
            3. Residual connection

        Args:
        ---------------
            hidden_states: Входной тензор формы (batch_size, seq_len, hidden_size)
            training: Флаг режима обучения (для balance loss)

        Returns:
        ---------------
            output: Тензор формы (batch_size, seq_len, hidden_size)
                   Выходные скрытые состояния
            balance_loss: Скаляр - load balancing loss
        """
        # TODO: Шаг 1 - Router
        #       Вызовите self.router(hidden_states, training)
        #       Получите: routing_weights, selected_experts, balance_loss

        # TODO: Шаг 2 - Получите размерности
        #       batch_size, seq_len, hidden_size = hidden_states.shape

        # TODO: Шаг 3 - Создайте output тензор
        #       Инициализируйте нулями: torch.zeros_like(hidden_states)

        # TODO: Шаг 4 - Dispatch + Process + Combine (наивный подход)
        #       Для каждого batch b в range(batch_size):
        #           Для каждого sequence s в range(seq_len):
        #               Извлеките токен: token = hidden_states[b, s:s+1, :]  # (1, 1, H)
        #               Создайте token_output = torch.zeros(1, 1, hidden_size)
        #
        #               Для каждого k в range(self.top_k):
        #                   Получите индекс эксперта: expert_idx = selected_experts[b, s, k].item()
        #                   Получите вес: weight = routing_weights[b, s, k].item()
        #
        #                   Обработайте токен экспертом:
        #                       expert_output = self.experts[expert_idx](token)
        #
        #                   Добавьте взвешенный результат:
        #                       token_output += weight * expert_output
        #
        #               Сохраните результат: output[b, s, :] = token_output.squeeze()

        # TODO: Шаг 5 - Residual connection
        #       output = output + hidden_states

        # TODO: Шаг 6 - Return
        #       Верните (output, balance_loss)

        # Вопросы для размышления:
        # - Почему используем token = hidden_states[b, s:s+1, :] с s:s+1, а не s?
        # - Зачем нужен .item() при извлечении expert_idx и weight?
        # - Что произойдет, если убрать residual connection?
        # - Как можно оптимизировать эти циклы?
        pass
