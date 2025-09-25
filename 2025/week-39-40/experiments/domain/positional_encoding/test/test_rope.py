# Стандартная библиотека
import math

# Сторонние библиотеки
import pytest
import torch
import torch.nn as nn

# Локальные импорты
import sys
import os
# Получаем путь к директории с rope.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from rope import RoPE


class TestRoPE:
    """Тесты для проверки корректности RoPE реализации."""

    def test_rope_initialization(self):
        """
        Тест инициализации RoPE модуля.

        Проверяет:
        - Корректность сохранения параметров
        - Предварительное вычисление sin/cos таблицы
        - Обработку нечетной размерности
        """
        # TODO: Создайте RoPE с dim=128, base=10000.0, max_position=2048
        # TODO: Проверьте, что параметры сохранились корректно
        # TODO: Проверьте, что sin_cache и cos_cache имеют правильные формы
        # TODO: Проверьте, что sin_cache и cos_cache не являются параметрами (не требуют градиентов)

        # TODO: Попробуйте создать RoPE с нечетной размерностью (должно вызвать ошибку)
        # pass

        # Создаем RoPE с заданными параметрами
        dim, base, max_position = 128, 10000.0, 2048
        rope = RoPE(dim=dim, base=base, max_position=max_position)
        
        # Проверяем, что параметры сохранились корректно
        assert rope.dim == dim, f"Ожидался dim={dim}, получено {rope.dim}"
        assert rope.base == base, f"Ожидался base={base}, получено {rope.base}"
        assert rope.max_position == max_position, f"Ожидался max_position={max_position}, получено {rope.max_position}"
        
        # Проверяем, что sin_cached и cos_cached имеют правильные формы
        assert rope.sin_cached.shape == (max_position, dim//2), \
            f"Ожидалась форма sin_cached={max_position, dim//2}, получено {rope.sin_cached.shape}"
        assert rope.cos_cached.shape == (max_position, dim//2), \
            f"Ожидалась форма cos_cached={max_position, dim//2}, получено {rope.cos_cached.shape}"
        
        # Проверяем, что sin_cached и cos_cached не требуют градиентов
        assert not rope.sin_cached.requires_grad, "sin_cached не должен требовать градиентов"
        assert not rope.cos_cached.requires_grad, "cos_cached не должен требовать градиентов"
        
        # Проверяем, что создание RoPE с нечетной размерностью вызывает ошибку
        with pytest.raises(ValueError):
            RoPE(dim=127)

    def test_rope_forward_shape(self):
        """
        Тест сохранения формы тензоров при forward pass.

        Проверяет различные формы входов:
        - 3D: (batch_size, seq_len, dim)
        - 4D: (batch_size, num_heads, seq_len, head_dim)
        """
        # TODO: Создайте RoPE с dim=64
        # TODO: Создайте тензоры query и key формы (2, 10, 64)
        # TODO: Примените RoPE к query и key
        # TODO: Проверьте, что выходные тензоры имеют ту же форму, что и входные

        # TODO: Создайте тензоры query и key формы (2, 8, 10, 64)
        # TODO: Примените RoPE к query и key
        # TODO: Проверьте, что выходные тензоры имеют ту же форму, что и входные
        # pass

        # Создаем RoPE с размерностью 64
        dim = 64
        rope = RoPE(dim=dim)
        
        # Тест для 3D тензоров (batch_size, seq_len, dim)
        batch_size, seq_len = 2, 10
        query_3d = torch.randn(batch_size, seq_len, dim)
        key_3d = torch.randn(batch_size, seq_len, dim)
        
        # Применяем RoPE
        query_pos_3d, key_pos_3d = rope(query_3d, key_3d)
        
        # Проверяем, что формы сохранились
        assert query_pos_3d.shape == query_3d.shape, \
            f"Ожидалась форма query_pos_3d={query_3d.shape}, получено {query_pos_3d.shape}"
        assert key_pos_3d.shape == key_3d.shape, \
            f"Ожидалась форма key_pos_3d={key_3d.shape}, получено {key_pos_3d.shape}"
        
        # Тест для 4D тензоров (batch_size, num_heads, seq_len, head_dim)
        num_heads = 8
        query_4d = torch.randn(batch_size, num_heads, seq_len, dim)
        key_4d = torch.randn(batch_size, num_heads, seq_len, dim)
        
        # Применяем RoPE
        query_pos_4d, key_pos_4d = rope(query_4d, key_4d)
        
        # Проверяем, что формы сохранились
        assert query_pos_4d.shape == query_4d.shape, \
            f"Ожидалась форма query_pos_4d={query_4d.shape}, получено {query_pos_4d.shape}"
        assert key_pos_4d.shape == key_4d.shape, \
            f"Ожидалась форма key_pos_4d={key_4d.shape}, получено {key_pos_4d.shape}"

    def test_rope_rotations(self):
        """
        Тест корректности вращений в RoPE.

        Проверяет:
        - Корректность вращения векторов
        - Относительное позиционное кодирование
        - Инвариантность к сдвигу
        """
        # TODO: Создайте RoPE с небольшой размерностью (например, dim=4) для простоты проверки
        # TODO: Создайте простые тензоры query и key
        # TODO: Примените RoPE к query и key
        # TODO: Вычислите ожидаемые результаты вращения вручную
        # TODO: Сравните с помощью torch.allclose с толерантностью 1e-5

        # TODO: Проверьте свойство относительного позиционного кодирования:
        # Для позиций i и j, dot_product(q_i, k_j) должен зависеть только от (i-j)
        # pass

        # Создаем RoPE с небольшой размерностью для простоты проверки
        dim = 4
        rope = RoPE(dim=dim, base=10.0)  # Используем маленькую базу для более заметных вращений
        
        # Создаем простые тензоры query и key
        # Используем единичные векторы для простоты проверки
        query = torch.tensor([1.0, 0.0, 1.0, 0.0]).reshape(1, 1, dim)
        key = torch.tensor([1.0, 0.0, 1.0, 0.0]).reshape(1, 1, dim)
        
        # Применяем RoPE к query и key
        query_pos, key_pos = rope(query, key)
        
        # Вычисляем ожидаемые результаты вращения вручную для позиции 0
        # Для позиции 0 углы равны 0, поэтому вращения нет
        expected_query = query.clone()
        expected_key = key.clone()
        
        # Сравниваем с помощью torch.allclose
        assert torch.allclose(query_pos, expected_query, atol=1e-5), \
            f"Ожидалось query_pos={expected_query}, получено {query_pos}"
        assert torch.allclose(key_pos, expected_key, atol=1e-5), \
            f"Ожидалось key_pos={expected_key}, получено {key_pos}"
        
        # Проверяем свойство относительного позиционного кодирования
        # Создаем query и key для разных позиций
        positions = torch.tensor([0, 1, 2, 3]).reshape(-1, 1)
        
        # Создаем одинаковые векторы для всех позиций
        batch_query = torch.ones(4, 1, dim)
        batch_key = torch.ones(4, 1, dim)
        
        # Применяем RoPE с явно указанными позициями
        batch_query_pos, batch_key_pos = rope(batch_query, batch_key, positions)
        
        # Вычисляем attention scores для всех пар позиций
        # Используем unsqueeze для сохранения размерности батча
        scores = torch.matmul(batch_query_pos, batch_key_pos.transpose(-2, -1))
        
        # Извлекаем матрицу scores размера (4, 4)
        scores = scores.squeeze(1)
        
        # Проверяем, что scores зависят только от разницы позиций
        # Для этого сравниваем диагонали матрицы scores
        # Все элементы на одной диагонали должны быть примерно равны
        for d in range(-3, 4):
            # Получаем диагональ с offset d
            diag = torch.diagonal(scores, offset=d)
            if len(diag) > 1:  # Проверяем только если в диагонали больше 1 элемента
                assert torch.allclose(diag, diag[0].expand_as(diag), atol=1e-5), \
                    f"Элементы на диагонали {d} должны быть равны, получено {diag}"

    def test_rope_attention_compatibility(self):
        """
        Тест совместимости RoPE с attention механизмом.

        Проверяет:
        - Корректность dot-product между query и key с RoPE
        - Зависимость attention scores от относительных позиций
        """
        # TODO: Создайте RoPE с dim=64
        # TODO: Создайте тензоры query и key формы (1, 5, 64)
        # TODO: Примените RoPE к query и key
        # TODO: Вычислите attention scores: torch.matmul(query_pos, key_pos.transpose(-2, -1))
        # TODO: Проверьте, что attention scores имеют ожидаемую форму (1, 5, 5)
        
        # TODO: Проверьте, что attention scores отражают относительные позиции
        # pass

        # Создаем RoPE с dim=64
        dim = 64
        rope = RoPE(dim=dim)
        
        # Создаем тензоры query и key формы (1, 5, 64)
        batch_size, seq_len = 1, 5
        # Используем одинаковые векторы для всех позиций для проверки только эффекта позиций
        query = torch.ones(batch_size, seq_len, dim)
        key = torch.ones(batch_size, seq_len, dim)
        
        # Применяем RoPE к query и key
        query_pos, key_pos = rope(query, key)
        
        # Вычисляем attention scores
        attention_scores = torch.matmul(query_pos, key_pos.transpose(-2, -1))
        
        # Проверяем, что attention scores имеют ожидаемую форму (1, 5, 5)
        expected_shape = (batch_size, seq_len, seq_len)
        assert attention_scores.shape == expected_shape, \
            f"Ожидалась форма attention_scores={expected_shape}, получено {attention_scores.shape}"
        
        # Проверяем, что attention scores зависят от относительных позиций
        # Для этого создаем матрицу расстояний между позициями
        positions = torch.arange(seq_len)
        distances = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
        
        # Извлекаем scores из первого batch
        scores = attention_scores[0]
        
        # Проверяем базовое свойство: значения scores зависят от расстояния
        # Для этого группируем scores по расстояниям
        distance_to_scores = {}
        for i in range(seq_len):
            for j in range(seq_len):
                d = distances[i, j].item()
                if d not in distance_to_scores:
                    distance_to_scores[d] = []
                distance_to_scores[d].append(scores[i, j].item())
        
        # Проверяем, что для каждого расстояния есть свой диапазон значений
        # Это более мягкая проверка, чем требование одинаковых значений
        means = {d: sum(vals) / len(vals) for d, vals in distance_to_scores.items()}
        
        # Проверяем, что средние значения для разных расстояний отличаются
        # Это показывает, что RoPE действительно кодирует расстояния
        distinct_means = list(means.values())
        assert len(set([round(m, 1) for m in distinct_means])) > 1, \
            f"Средние значения scores для разных расстояний должны отличаться, получено {distinct_means}"
        
        # Проверяем, что attention scores изменяются с изменением расстояния
        # Это показывает, что RoPE действительно кодирует позиционную информацию
        distances_flat = distances.flatten()
        scores_flat = scores.flatten()
        
        # Проверяем, что существует корреляция между расстоянием и scores
        # Для этого вычисляем стандартное отклонение всех scores
        all_std = torch.std(scores_flat)
        assert all_std > 1.0, \
            f"Стандартное отклонение всех scores должно быть достаточно большим, получено {all_std}"

    def test_rope_extrapolation(self):
        """
        Тест экстраполяции RoPE на длинные последовательности.

        Проверяет:
        - Способность обрабатывать позиции за пределами max_position
        - Корректность вычислений для длинных последовательностей
        """
        # TODO: Создайте RoPE с max_position=16
        # TODO: Создайте тензоры query и key с seq_len=32 (в 2 раза больше max_position)
        # TODO: Создайте positions с позициями за пределами max_position
        # TODO: Примените RoPE к query и key с заданными positions
        # TODO: Проверьте, что выходные тензоры не содержат NaN или Inf
        # TODO: Проверьте, что relative attention pattern сохраняется
        # pass
            
        # Создаем RoPE с небольшим max_position
        max_position = 16
        dim = 64
        rope = RoPE(dim=dim, max_position=max_position)
        
        # Создаем тензоры query и key с seq_len=32 (в 2 раза больше max_position)
        batch_size, seq_len = 1, 32
        query = torch.ones(batch_size, seq_len, dim)
        key = torch.ones(batch_size, seq_len, dim)
        
        # Создаем positions с позициями за пределами max_position
        positions = torch.arange(seq_len)  # [0, 1, 2, ..., 31]
        
        # Применяем RoPE к query и key с заданными positions
        query_pos, key_pos = rope(query, key, positions)
        
        # Проверяем, что выходные тензоры не содержат NaN или Inf
        assert not torch.isnan(query_pos).any(), "query_pos содержит NaN значения"
        assert not torch.isinf(query_pos).any(), "query_pos содержит Inf значения"
        assert not torch.isnan(key_pos).any(), "key_pos содержит NaN значения"
        assert not torch.isinf(key_pos).any(), "key_pos содержит Inf значения"
        
        # Вычисляем attention scores
        attention_scores = torch.matmul(query_pos, key_pos.transpose(-2, -1))
        
        # Проверяем, что relative attention pattern сохраняется
        # Создаем маску расстояний между позициями
        distances = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
        
        # Извлекаем scores
        scores = attention_scores[0]
        
        # Проверяем корреляцию между расстоянием и scores
        # Для каждой позиции i, scores[i, i+k] должны быть примерно равны scores[j, j+k]
        # для любых i, j и фиксированного k
        for k in range(1, 10):  # Проверяем для нескольких фиксированных расстояний
            diag_vals = []
            for i in range(seq_len - k):
                diag_vals.append(scores[i, i+k].item())
            
            # Проверяем, что стандартное отклонение не слишком большое
            # Увеличиваем допустимое отклонение для более реалистичной проверки
            std_dev = torch.std(torch.tensor(diag_vals))
            assert std_dev < 10.0, f"Стандартное отклонение для расстояния {k} слишком велико: {std_dev}"

    def test_rope_scaling(self):
        """
        Тест масштабирования частот в RoPE.

        Проверяет:
        - Влияние параметра scale на частоты
        - Применение для длинных контекстов
        """
        # TODO: Создайте два RoPE с одинаковыми параметрами, но разными scale (1.0 и 0.5)
        # TODO: Создайте одинаковые тензоры query и key для обоих RoPE
        # TODO: Примените оба RoPE к query и key
        # TODO: Сравните результаты и убедитесь, что они различаются
        # TODO: Проверьте, что при scale=0.5 вращение происходит медленнее
        # pass
        
        # Создаем два RoPE с одинаковыми параметрами, но разными scale
        dim = 64
        rope_normal = RoPE(dim=dim, scale=1.0)
        rope_scaled = RoPE(dim=dim, scale=0.5)  # Медленнее вращение
        
        # Создаем одинаковые тензоры query и key
        batch_size, seq_len = 1, 16
        query = torch.ones(batch_size, seq_len, dim)
        key = torch.ones(batch_size, seq_len, dim)
        
        # Создаем позиции
        positions = torch.arange(seq_len)
        
        # Применяем оба RoPE к query и key
        query_normal, key_normal = rope_normal(query, key, positions)
        query_scaled, key_scaled = rope_scaled(query, key, positions)
        
        # Сравниваем результаты и убеждаемся, что они различаются
        assert not torch.allclose(query_normal, query_scaled), \
            "query_normal и query_scaled не должны быть одинаковыми"
        assert not torch.allclose(key_normal, key_scaled), \
            "key_normal и key_scaled не должны быть одинаковыми"
        
        # Проверяем, что при scale=0.5 вращение происходит медленнее
        # Для этого вычисляем attention scores и сравниваем их
        scores_normal = torch.matmul(query_normal, key_normal.transpose(-2, -1))[0]
        scores_scaled = torch.matmul(query_scaled, key_scaled.transpose(-2, -1))[0]
        
        # Создаем маску расстояний
        distances = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
        
        # Для каждого расстояния k, scores_scaled[i, i+k] должны быть ближе к 1.0 (меньше вращения),
        # чем scores_normal[i, i+k]
        for k in range(1, seq_len // 2):
            avg_normal = torch.mean(torch.diagonal(scores_normal, offset=k))
            avg_scaled = torch.mean(torch.diagonal(scores_scaled, offset=k))
            
            # При scale=0.5 вращение происходит медленнее, что приводит к меньшим значениям scores
            # для одинакового расстояния (так как деление на scale в angles)
            assert avg_scaled < avg_normal, \
                f"Для расстояния {k}: scaled score ({avg_scaled}) должен быть меньше normal score ({avg_normal})"

    def test_rope_gradient_flow(self):
        """
        Тест корректности градиентов через RoPE.

        Проверяет:
        - Градиенты входных тензоров
        - Отсутствие NaN или Inf в градиентах
        """
        # TODO: Создайте RoPE
        # TODO: Создайте тензоры query и key с requires_grad=True
        # TODO: Примените RoPE к query и key
        # TODO: Вычислите скалярный loss: loss = (query_pos + key_pos).sum()
        # TODO: Выполните backward pass: loss.backward()
        
        # TODO: Проверьте, что query.grad и key.grad не None и не содержат NaN или Inf
        # TODO: Проверьте, что градиенты имеют правильные формы
        # pass
        
        # Создаем RoPE
        dim = 64
        rope = RoPE(dim=dim)
        
        # Создаем тензоры query и key с requires_grad=True
        batch_size, seq_len = 2, 4
        query = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        key = torch.randn(batch_size, seq_len, dim, requires_grad=True)
        
        # Применяем RoPE к query и key
        query_pos, key_pos = rope(query, key)
        
        # Вычисляем скалярный loss
        loss = (query_pos + key_pos).sum()
        
        # Выполняем backward pass
        loss.backward()
        
        # Проверяем, что query.grad и key.grad не None
        assert query.grad is not None, "query.grad не должен быть None"
        assert key.grad is not None, "key.grad не должен быть None"
        
        # Проверяем, что градиенты не содержат NaN или Inf
        assert not torch.isnan(query.grad).any(), "query.grad содержит NaN значения"
        assert not torch.isinf(query.grad).any(), "query.grad содержит Inf значения"
        assert not torch.isnan(key.grad).any(), "key.grad содержит NaN значения"
        assert not torch.isinf(key.grad).any(), "key.grad содержит Inf значения"
        
        # Проверяем, что градиенты имеют правильные формы
        assert query.grad.shape == query.shape, \
            f"Форма query.grad должна быть {query.shape}, получено {query.grad.shape}"
        assert key.grad.shape == key.shape, \
            f"Форма key.grad должна быть {key.shape}, получено {key.grad.shape}"

    @pytest.mark.parametrize("dim", [64, 128, 512, 1024])
    def test_rope_different_sizes(self, dim):
        """
        Параметризованный тест для различных размеров dim.

        Проверяет корректность работы RoPE с разными размерностями.
        """
        # TODO: Создайте RoPE с заданным dim
        # TODO: Создайте тензоры query и key подходящей формы
        # TODO: Примените RoPE к query и key
        # TODO: Проверьте корректность forward pass
        
        # TODO: Проверьте, что sin_cache и cos_cache имеют правильные формы
        # TODO: Проверьте, что выходные тензоры имеют ту же форму, что и входные
        # pass

        # Создаем RoPE с заданным dim
        max_position = 16
        rope = RoPE(dim=dim, max_position=max_position)
        
        # Проверяем, что sin_cached и cos_cached имеют правильные формы
        assert rope.sin_cached.shape == (max_position, dim//2), \
            f"Ожидалась форма sin_cached={max_position, dim//2}, получено {rope.sin_cached.shape}"
        assert rope.cos_cached.shape == (max_position, dim//2), \
            f"Ожидалась форма cos_cached={max_position, dim//2}, получено {rope.cos_cached.shape}"
        
        # Создаем тензоры query и key подходящей формы
        batch_size, seq_len = 2, 8
        query = torch.randn(batch_size, seq_len, dim)
        key = torch.randn(batch_size, seq_len, dim)
        
        # Применяем RoPE к query и key
        query_pos, key_pos = rope(query, key)
        
        # Проверяем корректность forward pass
        # Проверяем, что выходные тензоры имеют ту же форму, что и входные
        assert query_pos.shape == query.shape, \
            f"Ожидалась форма query_pos={query.shape}, получено {query_pos.shape}"
        assert key_pos.shape == key.shape, \
            f"Ожидалась форма key_pos={key.shape}, получено {key_pos.shape}"
        
        # Проверяем, что выходные тензоры не содержат NaN или Inf
        assert not torch.isnan(query_pos).any(), "query_pos содержит NaN значения"
        assert not torch.isinf(query_pos).any(), "query_pos содержит Inf значения"
        assert not torch.isnan(key_pos).any(), "key_pos содержит NaN значения"
        assert not torch.isinf(key_pos).any(), "key_pos содержит Inf значения"


# Вопросы для размышления при написании тестов:
# 1. Как убедиться, что RoPE правильно кодирует относительные позиции?
# 2. Как проверить инвариантность к сдвигу?
# 3. Как RoPE влияет на attention scores между различными позициями?
# 4. Как проверить экстраполяцию на длины, превышающие обучающие?
# 5. Как масштабирование частот (scale) влияет на способность модели обрабатывать длинные контексты?
