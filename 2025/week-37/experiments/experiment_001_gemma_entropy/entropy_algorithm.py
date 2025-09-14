#!/usr/bin/env python3
"""
Entropy Algorithm for LLM Analysis

Description:
---------------
    Реализация алгоритма вычисления энтропии Шеннона для анализа
    неопределённости токенов. Использует общую инфраструктуру (shared) для
    исключения дублирования кода.

    Формула: H = -∑(p_i * log_b(p_i)) по словарю; по умолчанию b=2.

Examples:
---------------
    Запуск через CLI (метрики по сгенерированным токенам):
    >>> python entropy_algorithm.py -m analyze -t "Hello" --max-tokens 5
"""

# Стандартная библиотека
import os
import sys
from typing import Any, Dict

# Добавляем путь к shared модулям (адаптер к общей инфраструктуре)
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

# Сторонние библиотеки
import torch

# Shared infrastructure
from shared.framework.experiment_runner import run_experiment
from shared.infrastructure.base_analyzer import MetricCalculator


class EntropyCalculator:
    """
    Description:
    ---------------
        Калькулятор энтропии Шеннона по всему словарю вероятностей.

        Формула: H = -∑(p_i * log_b(p_i)), где p_i — вероятность i‑го токена,
        b — основание логарифма (по умолчанию 2, то есть биты).

    Raises:
    ---------------
        ValueError: если base <= 0 или eps <= 0 (небезопасные параметры).
    """
    
    def __init__(self, base: float = 2.0, eps: float = 1e-10):
        """
        Description:
        ---------------
            Инициализирует параметры вычисления энтропии.

        Args:
        ---------------
            base: Основание логарифма (2 для битов, e для натов, 10 для
                hartley). Должно быть > 0.
            eps: Малая константа для численной устойчивости при log(0).

        Raises:
        ---------------
            ValueError: Если base <= 0 или eps <= 0.

        Examples:
        ---------------
            >>> EntropyCalculator(base=2.0, eps=1e-10)
            <...EntropyCalculator>
        """
        # TODO: Параметры устойчивости и единиц измерения
        # 1) base — основание логарифма (2, e, 10)
        # 2) eps  — защита от log(0)
        # Вопросы: что будет при eps=0? какие единицы у результата при разных base?
        if base <= 0:
            raise ValueError("Параметр base должен быть > 0")
        if eps <= 0:
            raise ValueError("Параметр eps должен быть > 0")

        self.base = base
        self.eps = eps
    
    def calculate_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Description:
        ---------------
            Вычисляет энтропию по тензору вероятностей по последней оси
            (словарной). Возвращает тензор энтропий без словарной оси.

        Args:
        ---------------
            probabilities: Тензор вероятностей формы (..., vocab_size).

        Returns:
        ---------------
            torch.Tensor: Тензор энтропии формы (...,).

        Raises:
        ---------------
            TypeError: Если вход не является torch.Tensor.

        Examples:
        ---------------
            >>> import torch
            >>> p = torch.tensor([0.5, 0.5])
            >>> EntropyCalculator().calculate_entropy(p)
            tensor(1.)
        """
        # TODO: реализуйте расчет энтропии:
        # 1. Добавьте epsilon к вероятностям или примените clamp для избежания log(0)
        # 2. Вычислите логарифм с нужным основанием (log2/log или через смену основания)
        # 3. Умножьте вероятности на логарифмы
        # 4. Просуммируйте по последнему измерению (dim=-1)
        # 5. Примените отрицательный знак
        # Вопросы: нужно ли нормировать вход? какую ось считать «словарной»? в каких единицах ответ?
        if not isinstance(probabilities, torch.Tensor):
            raise TypeError("'probabilities' должен быть torch.Tensor")
        probs_safe = torch.clamp(probabilities, min=self.eps)
        
        # Подсказка: log_b(x) = ln(x) / ln(b)
        if self.base == 2.0:
            log_probs = torch.log2(probs_safe)
        elif self.base == torch.e:
            log_probs = torch.log(probs_safe)  
        else:
            log_probs = torch.log(probs_safe) / torch.log(torch.tensor(self.base))
        
        # Вопросы: почему нужен «минус»? что даст one-hot и равномерное распределение?
        
        entropy = -torch.sum(probabilities * log_probs, dim=-1)
        return entropy


class EntropyAlgorithm:
    """
    Description:
    ---------------
        Алгоритм эксперимента, предоставляющий калькулятор энтропии и
        конфигурацию по умолчанию для общего раннера.
    """
    
    def __init__(self):
        # TODO: Какой объект инкапсулирует формулу и параметры (base, eps)?
        # Вопрос: где правильнее держать логику метрики — в алгоритме или калькуляторе?
        self.calculator = EntropyCalculator()
    
    def get_calculator(self) -> MetricCalculator:
        """
        Description:
        ---------------
            Возвращает объект калькулятора энтропии для использования в
            инфраструктуре.

        Returns:
        ---------------
            MetricCalculator: Экземпляр, реализующий метод вычисления метрики.

        Examples:
        ---------------
            >>> EntropyAlgorithm().get_calculator()  # doctest: +ELLIPSIS
            <...EntropyCalculator>
        """
        # TODO: Как BaseAnalyzer узнаёт, какой метод вызывать? (см. hasattr в BaseAnalyzer)
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
        # TODO: Какое имя будет ключом в результатах и заголовком в выводе CLI?
        return "entropy"
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Возвращает конфигурацию по умолчанию для модели и эксперимента.

        Returns:
        ---------------
            Dict[str, Any]: Словарь с секциями "model" и "experiment".

        Examples:
        ---------------
            >>> EntropyAlgorithm().get_default_config().keys()
            dict_keys(['model', 'experiment'])
        """
        # TODO: Мини-конфиг по умолчанию
        # Вопросы:
        # - Какие секции обязательны для ModelManager и алгоритма?
        # - Какие параметры повлияют на энтропию (base, eps)?
        # - Когда брать YAML, а когда оставить embedded defaults?
        return {
            "model": {
                "name": "google/gemma-2b-it",
                "max_length": 512,
                "device": "auto"
            },
            "experiment": {
                "base": 2.0,
                "eps": 1e-10
            }
        }


if __name__ == "__main__":
    # TODO: Как запустить через общий CLI?
    # Подумайте:
    # - Какие режимы есть у runner: analyze vs generate?
    # - Какие флаги обязательны в каждом режиме?
    # Пример для проверки: 
    #   python entropy_algorithm.py -m analyze -t "Hello world"
    #   python entropy_algorithm.py -m generate -p "Once upon a time" --max-tokens 5
    
    # Запуск эксперимента через shared framework
    algorithm = EntropyAlgorithm()
    run_experiment(algorithm, "Entropy Analysis")
