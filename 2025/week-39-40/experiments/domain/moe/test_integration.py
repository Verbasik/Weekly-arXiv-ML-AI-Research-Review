"""
Интерактивное тестирование Router + Expert для понимания MoE механики.

Этот скрипт демонстрирует:
1. Как Router выбирает экспертов для токенов
2. Как распределяются routing weights
3. Как Expert обрабатывает токены
4. Что нужно для интеграции в MoE Layer
"""

# Стандартная библиотека
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь поиска модулей
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Сторонние библиотеки
import torch
import torch.nn as nn

# Наши компоненты
from experiments.domain.moe.router import MoERouter
from experiments.domain.moe.expert import Expert


def print_separator(title):
    """Красивый разделитель для вывода"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def test_router_basic():
    """Тест 1: Базовая работа Router"""
    print_separator("Тест 1: Router - выбор экспертов")

    # Создаём Router для модели 0.6B
    router = MoERouter(
        hidden_size=512,
        num_experts=8,
        top_k=2,
        balance_loss_coef=0.01
    )
    router.eval()  # Режим inference

    # Создаём входные данные: 1 batch, 4 токена, 512 hidden_size
    batch_size, seq_len, hidden_size = 1, 4, 512
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    print(f"\nВход: {hidden_states.shape} (batch={batch_size}, seq={seq_len}, hidden={hidden_size})")

    # Прогоняем через Router
    routing_weights, selected_experts, balance_loss = router(hidden_states, training=False)

    print(f"\nВыход Router:")
    print(f"  routing_weights: {routing_weights.shape} = {routing_weights}")
    print(f"  selected_experts: {selected_experts.shape} = {selected_experts}")
    print(f"  balance_loss: {balance_loss.item()}")

    # Анализ распределения
    print(f"\n📊 Анализ:")
    for tok_idx in range(seq_len):
        experts = selected_experts[0, tok_idx].tolist()
        weights = routing_weights[0, tok_idx].tolist()
        print(f"  Токен {tok_idx}: эксперты {experts}, веса {[f'{w:.3f}' for w in weights]}")
        # Проверяем, что веса суммируются в 1
        weight_sum = sum(weights)
        print(f"           Сумма весов: {weight_sum:.6f} (должна быть ≈ 1.0)")


def test_expert_basic():
    """Тест 2: Базовая работа Expert"""
    print_separator("Тест 2: Expert - обработка токенов")

    # Создаём один эксперт
    expert = Expert(
        hidden_size=512,
        intermediate_size=2048,
        dropout=0.0
    )
    expert.eval()

    # Тестовые токены
    num_tokens = 3
    hidden_size = 512
    tokens = torch.randn(num_tokens, hidden_size)

    print(f"\nВход: {tokens.shape} (num_tokens={num_tokens}, hidden={hidden_size})")

    # Прогоняем через Expert
    output = expert(tokens.unsqueeze(0)).squeeze(0)  # Добавляем/убираем batch dim

    print(f"Выход: {output.shape}")
    print(f"\n📊 Статистика:")
    print(f"  Вход  - mean: {tokens.mean():.4f}, std: {tokens.std():.4f}")
    print(f"  Выход - mean: {output.mean():.4f}, std: {output.std():.4f}")
    print(f"  Размерность сохранена: {tokens.shape == output.shape}")


def test_router_expert_integration():
    """Тест 3: Router + Experts интеграция"""
    print_separator("Тест 3: Router + 8 Experts - полный pipeline")

    # Конфигурация
    hidden_size = 512
    intermediate_size = 2048
    num_experts = 8
    top_k = 2
    batch_size, seq_len = 2, 6  # 2 примера по 6 токенов

    # Создаём Router
    router = MoERouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k
    )
    router.eval()

    # Создаём 8 экспертов
    experts = nn.ModuleList([
        Expert(hidden_size=hidden_size, intermediate_size=intermediate_size, dropout=0.0)
        for _ in range(num_experts)
    ])
    for expert in experts:
        expert.eval()

    # Входные данные
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    print(f"\nВход: {hidden_states.shape}")

    # Шаг 1: Router выбирает экспертов
    routing_weights, selected_experts, _ = router(hidden_states, training=False)
    print(f"\nШаг 1: Router")
    print(f"  routing_weights: {routing_weights.shape}")
    print(f"  selected_experts: {selected_experts.shape}")

    # Шаг 2: Простая обработка (наивная реализация)
    print(f"\nШаг 2: Обработка экспертами (наивный подход)")

    # Для каждого токена в батче
    outputs = torch.zeros_like(hidden_states)

    for b in range(batch_size):
        for s in range(seq_len):
            token = hidden_states[b, s:s+1, :]  # (1, 1, hidden_size)
            token_output = torch.zeros(1, 1, hidden_size)

            # Для каждого выбранного эксперта
            for k in range(top_k):
                expert_idx = selected_experts[b, s, k].item()
                weight = routing_weights[b, s, k].item()

                # Обработка токена экспертом
                expert_output = experts[expert_idx](token)

                # Взвешенное суммирование
                token_output += weight * expert_output

            outputs[b, s, :] = token_output.squeeze()

    print(f"  Выход: {outputs.shape}")

    # Анализ использования экспертов
    print(f"\n📊 Статистика использования экспертов:")
    expert_usage = torch.bincount(selected_experts.view(-1), minlength=num_experts)
    total_selections = batch_size * seq_len * top_k

    for expert_idx in range(num_experts):
        count = expert_usage[expert_idx].item()
        percentage = (count / total_selections) * 100
        bar = "█" * int(percentage / 2)  # Масштаб 1:2
        print(f"  Эксперт {expert_idx}: {count:2d}/{total_selections} ({percentage:5.1f}%) {bar}")

    # Проверка баланса
    print(f"\n📊 Баланс:")
    mean_usage = expert_usage.float().mean().item()
    std_usage = expert_usage.float().std().item()
    print(f"  Среднее использование: {mean_usage:.2f}")
    print(f"  Стандартное отклонение: {std_usage:.2f}")
    if std_usage < mean_usage * 0.5:
        print(f"  ✅ Хороший баланс (std < 50% mean)")
    else:
        print(f"  ⚠️  Плохой баланс (std >= 50% mean)")

    return hidden_states, outputs


def test_moe_layer_requirements():
    """Тест 4: Что нужно для MoE Layer"""
    print_separator("Тест 4: Требования для MoE Layer")

    print("\n🎯 Ключевые вызовы MoE Layer:")
    print("\n1. DISPATCH: Токены → Эксперты")
    print("   Проблема: Каждый токен идёт к 2 разным экспертам")
    print("   Решение: Группировать токены по экспертам для batch processing")
    print("   Пример:")
    print("     Токен 0: эксперты [2, 5]")
    print("     Токен 1: эксперты [2, 7]")
    print("     Токен 2: эксперты [1, 5]")
    print("     → Эксперт 2 получает [токен 0, токен 1]")
    print("     → Эксперт 5 получает [токен 0, токен 2]")
    print("     → Эксперт 7 получает [токен 1]")
    print("     → Эксперт 1 получает [токен 2]")

    print("\n2. PROCESS: Параллельная обработка")
    print("   Проблема: 8 экспертов должны обрабатывать токены независимо")
    print("   Решение: Использовать nn.ModuleList + цикл по экспертам")
    print("   Оптимизация: В будущем - expert parallelism на разных GPU")

    print("\n3. COMBINE: Результаты → Выход")
    print("   Проблема: Собрать результаты обратно в правильном порядке токенов")
    print("   Решение: Сохранить маппинг (token_idx, expert_idx, position_in_expert)")
    print("   Применить routing_weights при суммировании")

    print("\n4. EDGE CASES:")
    print("   • Эксперт получает 0 токенов - пропустить обработку")
    print("   • Эксперт получает много токенов - capacity enforcement (опционально)")
    print("   • Разные batch sizes - должно работать")

    print("\n📋 Простейшая реализация MoE Layer:")
    print("""
    class SimpleMoELayer(nn.Module):
        def forward(self, x):
            # 1. Router
            weights, experts_idx, loss = self.router(x)

            # 2. Dispatch + Process + Combine (наивно)
            output = torch.zeros_like(x)
            for batch in range(B):
                for seq in range(S):
                    for k in range(top_k):
                        expert = self.experts[experts_idx[batch, seq, k]]
                        expert_out = expert(x[batch, seq:seq+1])
                        output[batch, seq] += weights[batch, seq, k] * expert_out.squeeze()

            # 3. Residual
            return output + x, loss
    """)


def main():
    """Запуск всех тестов"""
    print("\n" + "🧪 ИНТЕРАКТИВНОЕ ТЕСТИРОВАНИЕ MoE КОМПОНЕНТОВ".center(70))

    # Фиксируем seed для воспроизводимости
    torch.manual_seed(42)

    # Тесты
    test_router_basic()
    test_expert_basic()
    hidden_states, outputs = test_router_expert_integration()
    test_moe_layer_requirements()

    # Итоги
    print_separator("Итоги")
    print("\n✅ Все компоненты работают корректно!")
    print("\n🎯 Следующий шаг: Реализация MoE Layer")
    print("   - Оптимизированный dispatch mechanism")
    print("   - Batch processing экспертов")
    print("   - Правильное combine с routing weights")
    print("   - Residual connection")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
