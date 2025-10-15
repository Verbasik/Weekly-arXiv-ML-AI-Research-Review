from typing import List, Dict, Any


class LMFixedLengthCollator:
    """
    Description:
    ---------------
        Скелет collator'а для фиксированной длины (causal LM).

    Assumptions:
    ---------------
        - Каждый элемент батча имеет поля: input_ids (seq_len,), labels (seq_len,)
        - Паддинг не требуется (длина фиксированная)

    TODO:
    ---------------
        - Сформировать батч: torch.stack для input_ids и labels
        - Вернуть словарь с ключами: "input_ids", "labels"
        - (Опционально) поддержать динамический паддинг в будущем

    Вопросы для размышления:
    ---------------
        - Чем отличается collate_fn от transform внутри датасета?
        - Когда лучше паддить: на уровне датасета или collate?
    """

    def __init__(self):
        # TODO: добавить конфигурационные параметры (например, pad_value) при необходимости
        pass

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Преобразовать список примеров в тензоры батча.

        Returns:
        ---------------
            Dict[str, Any]: {"input_ids": (B, S), "labels": (B, S)}

        TODO:
        ---------------
            - Использовать torch.stack для формирования батча
        """
        # TODO: реализуйте сборку батча через torch.stack([...], dim=0)
        pass
