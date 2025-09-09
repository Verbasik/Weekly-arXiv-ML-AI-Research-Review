"""
Модуль домейн-слоя для расчёта энтропии Шеннона.

Вы должны реализовать весь класс EntropyCalculator.

Предлагаю вам дропнуть весь код, и оставить только подсказки TODO, что нужно реализовать 🤗
"""

# Сторонние библиотеки
import torch
from typing import Any


class EntropyCalculator:
    """
    Description:
    ---------------
        Доменный сервис для расчёта энтропии Шеннона над распределениями
        вероятностей по словарю токенов.

    Examples:
    ---------------
        >>> import torch
        >>> calc = EntropyCalculator()
        >>> probs = torch.tensor([[0.5, 0.5]])  # (T=1, V=2)
        >>> H = calc.calculate_entropy(probs)
        >>> H.shape
        torch.Size([1])
    """

    def calculate_entropy(self, probabilities: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
        """
        Description:
        ---------------
            Вычисляет энтропию Шеннона по каждой позиции последовательности
            на основе распределения вероятностей по словарю.

        Args:
        ---------------
            probabilities: тензор вероятностей формы (B, T, V) или (T, V)
            epsilon: малое значение для численной стабильности

        Returns:
        ---------------
            torch.Tensor: тензор энтропий формы (B, T) или (T,)

        Raises:
        ---------------
            ValueError: если вероятность содержит отрицательные значения.

        Examples:
        ---------------
            >>> import torch
            >>> calc = EntropyCalculator()
            >>> probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
            >>> H = calc.calculate_entropy(probs)
            >>> H.dim() in (1, 2)
            True
        """
        # TODO: реализуйте расчет энтропии:
        # 1. Добавьте epsilon к probabilities для избежания log(0)
        # 2. Вычислите torch.log2(probabilities + epsilon)
        # 3. Умножьте probabilities на логарифм
        # 4. Просуммируйте по последнему измерению (dim=-1)
        # 5. Примените отрицательный знак
        # Формула: -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=-1)

        # pass

        if torch.any(probabilities < 0):
            raise ValueError("Вероятности не могут быть отрицательными")

        probabilities = probabilities + epsilon
        log_probabilities = torch.log2(probabilities)
        entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
        return entropy

