"""
SimpleMoELayer - это учебная версия MoE, которая фокусируется на правильности логики, а не на производительности. Используя простой цикл по токенам, мы избегаем сложных
тензорных операций индексации, делая код понятным и легко отлаживаемым. Это идеальный first step перед оптимизированной версией.
"""


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
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            balance_loss_coef=balance_loss_coef
        )

        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size, expert_dropout)
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
        # TODO: Шаг 1 - Вызовите self.router
        # TODO: Шаг 2 - Получите размерности hidden_states.shape=
        # TODO: Шаг 3 - Создайте output тензор
        # TODO: Шаг 4 - Dispatch + Process + Combine (наивный подход)
        # TODO: Шаг 5 - output = output + hidden_states
        # TODO: Шаг 6 - Верните (output, balance_loss)

        # Вопросы для размышления:
        # - Почему используем token = hidden_states[b, s:s+1, :] с s:s+1, а не s?
        # - Зачем нужен .item() при извлечении expert_idx и weight?
        # - Что произойдет, если убрать residual connection?
        # - Как можно оптимизировать эти циклы?
        # pass

        # Шаг 1 - Router
        # nn.Module.__call__ обёртка: router(...) автоматически вызывает router.forward(...)
        routing_weights, selected_experts, balance_loss = self.router(hidden_states, training)

        # Шаг 2 - Размерности
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Шаг 3 - Output тензор
        output = torch.zeros(batch_size, seq_len, hidden_size, device=hidden_states.device)

        # Шаг 4 - Dispatch + Process + Combine (наивный подход)
        for b in range(batch_size):
            for s in range(seq_len):
                token = hidden_states[b, s:s+1, :]  # (1, 1, H)
                token_output = torch.zeros(1, 1, hidden_size, device=hidden_states.device)

                for k in range(self.top_k):
                    expert_idx = selected_experts[b, s, k].item()
                    weight = routing_weights[b, s, k].item()

                    expert_output = self.experts[expert_idx](token)  # (1, 1, H)

                    token_output += weight * expert_output  # Взвешенное суммирование

                output[b, s, :] = token_output.squeeze()    # Сохранение результата

        # Шаг 5 - Residual connection
        output = output + hidden_states

        # Шаг 6 - Return
        return output, balance_loss


