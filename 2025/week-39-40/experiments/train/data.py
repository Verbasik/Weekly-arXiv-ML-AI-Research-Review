"""
Educational skeletons for Phase 4 data pipeline (WikiText-2).

Format: detailed docstrings + TODO comments + pass (no implementation).
"""
from typing import Dict, List, Any


def get_tokenizer() -> Any:
    """
    Description:
    ---------------
        Вернуть совместимый с GPT‑2 токенизатор с корректной настройкой паддинга.

    Args:
    ---------------
        None

    Returns:
    ---------------
        Any: Инициализированный токенизатор HF (GPT2Tokenizer или совместимый).

    TODO:
    ---------------
        - Загрузить GPT‑2 tokenizer через HF Transformers
        - Установить pad_token = eos_token и padding_side = "right"
        - Решить, нужны ли BOS/EOS при предобучении (обычно False для add_special_tokens)

    Вопросы для размышления:
    ---------------
        - Почему у GPT‑2 нет pad_token по умолчанию?
        - Как padding_side влияет на батчинг в LM?
    """
    # TODO: реализуйте загрузку и настройку токенизатора
    pass


def load_wikitext2_splits():
    """
    Description:
    ---------------
        Загрузить разбиения WikiText‑2 (raw): train/validation/test.

    Returns:
    ---------------
        Tuple[Dataset, Dataset, Dataset]: Три сплита датасета.

    TODO:
    ---------------
        - Использовать datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
        - Вернуть ds["train"], ds["validation"], ds["test"]
        - (Опционально) учесть offline‑кэш

    Вопросы для размышления:
    ---------------
        - Чем raw‑v1 отличается от v1 и почему это важно для предобучения?
    """
    # TODO: реализуйте загрузку датасета
    pass


def tokenize_function(tokenizer: Any, examples: Dict[str, List[str]]):
    """
    Description:
    ---------------
        Преобразует батч сырых текстов в токен-идентификаторы.

    Args:
    ---------------
        tokenizer: Объект токенизатора
        examples: Словарь с ключом "text" и списком строк

    Returns:
    ---------------
        Dict: Результат токенизации (как ожидает datasets.map), минимум input_ids

    TODO:
    ---------------
        - Вызвать tokenizer(examples["text"], add_special_tokens=False)
        - Не возвращать attention_mask (здесь не требуется)
        - Совместимость с batched=True

    Вопросы для размышления:
    ---------------
        - Когда стоит включать truncation на этом шаге, а когда при группировке?
    """
    # TODO: реализуйте токенизацию батча
    pass


def group_texts(examples: Dict[str, List[List[int]]], seq_len: int):
    """
    Description:
    ---------------
        Конкатенирует токены и нарезает на блоки длиной (seq_len+1),
        возвращая пары (input_ids, labels) со сдвигом на 1 токен вперёд.

    Args:
    ---------------
        examples: Словарь с ключом "input_ids" (список списков токенов)
        seq_len: Длина входной последовательности для LM (без следующего токена)

    Returns:
    ---------------
        Dict[str, Any]: {"input_ids": (N, seq_len), "labels": (N, seq_len)}

    TODO:
    ---------------
        - Сконкатенировать списки из examples["input_ids"]
        - Обрезать до кратного (seq_len+1)
        - Преобразовать в тензор размерности (-1, seq_len+1), затем разрезать на inputs/labels
        - (Опционально) реализовать скользящее окно (stride) позднее

    Вопросы для размышления:
    ---------------
        - Почему блоки длины (S+1) удобны для next-token задачи?
        - Чем страйд‑окна лучше простого разбиения без перекрытия?
    """
    # TODO: реализуйте конкатенацию и нарезку
    pass


def to_torch_dataset(hf_dataset, tokenizer, seq_len: int):
    """
    Description:
    ---------------
        Построить конвейер: tokenize → group_texts и вернуть датасет в формате torch.

    Args:
    ---------------
        hf_dataset: Исходный HF датасет (split)
        tokenizer: Токенизатор GPT‑2
        seq_len: Длина блока для LM

    Returns:
    ---------------
        Dataset: HF датасет, у которого columns = ["input_ids", "labels"], set_format("torch")

    TODO:
    ---------------
        - Вызвать .map(tokenize_function, batched=True)
        - Вызвать .map(lambda batch: group_texts(batch, seq_len), batched=True)
        - Оставить только нужные колонки и set_format("torch")

    Вопросы для размышления:
    ---------------
        - Какие колонки нужно удалить/сохранить между трансформациями?
    """
    # TODO: реализуйте конвейер map → map → set_format
    pass
