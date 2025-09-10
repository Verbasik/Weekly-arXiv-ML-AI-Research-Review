#!/usr/bin/env python3
"""
Confidence Algorithm for LLM Analysis

Description:
---------------
    Реализация алгоритма вычисления уверенности токенов по методологии
    DeepConf. Используется общая инфраструктура (shared) для устранения
    дублирования кода.

    Формула: C = -(1/k) * ∑(log2(P_j)) по j=1..k (топ‑k токенов).

Examples:
---------------
    Запуск через CLI (метрики по сгенерированным токенам):
    >>> python confidence_algorithm.py -m analyze -t "Hello" --max-tokens 5
"""

# Стандартная библиотека
import os
import sys
from typing import Any, Dict, Union

# Добавляем путь к shared модулям
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# Сторонние библиотеки
import torch

# Shared infrastructure
from shared.framework.experiment_runner import run_experiment
from shared.infrastructure.base_analyzer import MetricCalculator


class ConfidenceCalculator:
    """
    Description:
    ---------------
        Калькулятор уверенности по DeepConf: усреднённый логарифм вероятностей
        топ‑k альтернатив со знаком минус. Даёт более устойчивую к шуму оценку
        "решительности" модели по сравнению с энтропией по всему словарю.

    Raises:
    ---------------
        ValueError: при некорректных параметрах k или eps (<= 0).
    """
    
    def __init__(self, k: int = 10, eps: float = 1e-8):
        """
        Description:
        ---------------
            Инициализация параметров расчёта уверенности.

        Args:
        ---------------
            k: Количество топ‑токенов для анализа (k > 0).
            eps: Константа для защиты от log(0) (eps > 0).

        Raises:
        ---------------
            ValueError: Если k <= 0 или eps <= 0.

        Examples:
        ---------------
            >>> ConfidenceCalculator(k=5, eps=1e-8)
            <...ConfidenceCalculator>
        """
        # TODO: инициализация и инварианты параметров
        # 1) Проверьте k > 0 и eps > 0 (иначе ValueError)
        # 2) Сохраните значения в self.k и self.eps
        # Вопрос: почему валидировать здесь, а не откладывать до вычисления?
        
        if k <= 0:
            raise ValueError(f"k должно быть положительным числом, получено: {k}")
        if eps <= 0:
            raise ValueError(f"eps должно быть положительным числом, получено: {eps}")
        
        # Вопрос: чем учёт только top-k отличается от энтропии по всему словарю?
            
        self.k = k
        self.eps = eps
    
    def calculate_confidence(
        self,
        probabilities: torch.Tensor,
        k: Union[int, None] = None,
        eps: Union[float, None] = None,
    ) -> torch.Tensor:
        """
        Description:
        ---------------
            Вычисляет уверенность по формуле DeepConf по последней оси
            (словарной). Возвращает тензор без словарной оси.

        Args:
        ---------------
            probabilities: Тензор вероятностей формы (..., vocab_size).
            k: Переопределение количества топ‑токенов. Если None — self.k.
            eps: Переопределение eps. Если None — self.eps.

        Returns:
        ---------------
            torch.Tensor: Тензор уверенности формы (...,).

        Raises:
        ---------------
            TypeError: Если probabilities не torch.Tensor.
            ValueError: Если k > vocab_size или dim == 0.

        Examples:
        ---------------
            >>> import torch
            >>> p = torch.tensor([0.7, 0.2, 0.1])
            >>> ConfidenceCalculator(k=1).calculate_confidence(p)
            tensor(0.5146)
        """
        # TODO: реализуйте расчёт уверенности (DeepConf)
        # 1. Выберите k и eps: аргументы функции или значения self
        # 2. Проверьте входы: probabilities — тензор, dim>0, k ≤ vocab_size
        # 3. Извлеките top-k вероятности: torch.topk(..., k, dim=-1). Нужны values
        # 4. Примените clamp(min=eps), затем вычислите log2
        # 5. Просуммируйте по последней оси и домножьте на -1/k
        # Вопросы: форма результата при батче/последовательности? почему база 2?
        
        # Используем переданные параметры или значения по умолчанию
        k = k if k is not None else self.k
        eps = eps if eps is not None else self.eps
        
        # Инварианты входа: тип тензора, размерность, ограничение k ≤ vocab_size
        
        # Валидация входов
        if not isinstance(probabilities, torch.Tensor):
            raise ValueError("probabilities должен быть torch.Tensor")
        if probabilities.dim() == 0:
            raise ValueError("probabilities не может быть скаляром")
        if k > probabilities.shape[-1]:
            raise ValueError(f"k={k} больше размера словаря {probabilities.shape[-1]}")
        
        # Извлечение top-k по последней оси
        
        # Получаем топ-k вероятностей
        top_k_values, _ = torch.topk(probabilities, k, dim=-1)
        
        # Применение eps до логарифма
        
        # Применяем epsilon для numerical stability
        top_k_values = torch.clamp(top_k_values, min=eps)
        
        # Суммирование по последней оси и знак «минус»
        
        # Вычисляем log2 и уверенность
        log2_top_k_values = torch.log2(top_k_values)
        confidence = -1/k * torch.sum(log2_top_k_values, dim=-1)
        
        # Интерпретация: форма выхода и отличие от энтропии
        
        return confidence


class ConfidenceAlgorithm:
    """
    Description:
    ---------------
        Алгоритм эксперимента, предоставляющий калькулятор уверенности
        DeepConf и конфигурацию по умолчанию для общего раннера.
    """
    
    def __init__(self, k: int = 10):
        # TODO: инициализация алгоритма
        # 1) Создайте ConfidenceCalculator(k=k) и сохраните self.k
        # Вопрос: зачем разделять «алгоритм» и «калькулятор»?
        self.calculator = ConfidenceCalculator(k=k)
        self.k = k
    
    def get_calculator(self) -> MetricCalculator:
        """
        Description:
        ---------------
            Возвращает калькулятор уверенности для использования в
            инфраструктуре.

        Returns:
        ---------------
            MetricCalculator: Экземпляр калькулятора метрики.
        """
        # TODO: как BaseAnalyzer поймёт, какой метод вызывать? (см. hasattr в BaseAnalyzer)
        return self.calculator
    
    def get_metric_name(self) -> str:
        """
        Description:
        ---------------
            Имя метрики, используемое в логах и ключах результата.

        Returns:
        ---------------
            str: Строковый идентификатор метрики.
        """
        # TODO: как имя метрики используется в логах/выводе?
        return "confidence"
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Конфигурация по умолчанию для модели и эксперимента DeepConf.

        Returns:
        ---------------
            Dict[str, Any]: Секции "model" и "experiment".
        """
        # TODO: какие секции нужны раннеру и менеджеру модели?
        # Что влияет именно на метрику confidence (k, eps)?
        return {
            "model": {
                "name": "google/gemma-2b-it",
                "max_length": 512,
                "device": "auto"
            },
            "experiment": {
                "k": self.k,
                "eps": 1e-8
            }
        }


if __name__ == "__main__":
    # TODO: запуск через общий раннер
    # Примеры:
    #   python confidence_algorithm.py -m analyze -t "Hello"
    #   python confidence_algorithm.py -m generate -p "Explain transformers" --max-tokens 5
    
    # Запуск эксперимента через shared framework
    algorithm = ConfidenceAlgorithm(k=10)
    run_experiment(algorithm, "Confidence Analysis")
