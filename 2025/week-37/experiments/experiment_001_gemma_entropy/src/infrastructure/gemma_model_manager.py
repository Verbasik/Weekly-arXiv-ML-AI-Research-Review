"""
Модуль для загрузки и настройки модели Gemma-3n-E2B-it (инфраструктура).

Вы должны реализовать весь класс GemmaModelManager.

Предлагаю вам дропнуть весь код, и оставить только подсказки TODO, что нужно реализовать 🤗
"""

# Стандартная библиотека
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Сторонние библиотеки
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from domain.ports import ModelPort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GemmaModelManager(ModelPort):
    """
    Description:
    ---------------
        Класс-обёртка для загрузки конфигурации, инициализации устройства,
        токенизатора и модели Gemma-3n-E2B-it.

    Examples:
    ---------------
        >>> manager = GemmaModelManager("config/experiment.yaml")
        >>> manager.load_model()
        >>> info = manager.get_model_info()
        >>> isinstance(info, dict)
        True
    """

    def __init__(self, config_path: str = "config/experiment.yaml") -> None:
        """
        Description:
        ---------------
            Инициализирует менеджер модели: загружает конфигурацию,
            подготавливает атрибуты модели, токенизатора и устройства.

        Args:
        ---------------
            config_path: путь к конфигурационному файлу.

        Returns:
        ---------------
            None

        Raises:
        ---------------
            FileNotFoundError: если файл конфигурации не найден

        Examples:
        ---------------
            >>> manager = GemmaModelManager("config/experiment.yaml")
            >>> manager.model is None
            True
        """
        # TODO: инициализируйте следующие атрибуты:
        # - self.config (загрузите конфигурацию через _load_config)
        # - self.model = None
        # - self.tokenizer = None
        # - self._device (определите CUDA или CPU)
        # Выведите информацию об устройстве через logger.info

        # pass

        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        self._device = torch.device("cpu")
        logger.info("Device: %s", self._device)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Загружает YAML-конфигурацию эксперимента и возвращает её
            как словарь.

        Args:
        ---------------
            config_path: путь к файлу конфигурации.

        Returns:
        ---------------
            dict: загруженная конфигурация.

        Raises:
        ---------------
            FileNotFoundError: если файл конфигурации не найден

        Examples:
        ---------------
            >>> manager = GemmaModelManager()
            >>> isinstance(manager.config, dict)
            True
        """
        # TODO: реализуйте загрузку YAML конфигурации:
        # 1. Создайте Path объект из config_path
        # 2. Если файл не существует, попробуйте относительный путь от родительской директории
        # 3. Откройте и загрузите YAML файл с помощью yaml.safe_load
        # 4. Верните словарь конфигурации

        # pass

        config_path = Path(config_path)
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / config_path
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_model(self) -> None:
        """
        Description:
        ---------------
            Загружает токенизатор и модель на основе конфигурации, приводит
            модель в режим eval и сохраняет ссылки в атрибуты `self.model`
            и `self.tokenizer`.

        Args:
        ---------------
            Нет параметров. Используется `self.config`.

        Returns:
        ---------------
            None

        Raises:
        ---------------
            KeyError: отсутствуют необходимые ключи в конфигурации
            RuntimeError: ошибка при загрузке токенизатора или модели

        Examples:
        ---------------
            >>> manager = GemmaModelManager("config/experiment.yaml")
            >>> manager.load_model()
            >>> manager.model is not None and manager.tokenizer is not None
            True
        """
        # TODO: реализуйте загрузку модели:
        # 1. Получите имя модели из self.config['model']['name']
        # 2. Загрузите токенизатор через AutoTokenizer.from_pretrained
        #    с параметром trust_remote_code=True
        # 3. Установите pad_token = eos_token если pad_token отсутствует
        # 4. Загрузите модель через AutoModelForCausalLM.from_pretrained
        #    с параметрами: torch_dtype=torch.bfloat16, device_map из конфига, trust_remote_code=True
        # 5. Переведите модель в режим eval()
        # 6. Выведите информацию о загрузке и размере словаря

        # pass

        try:
            model_name = self.config["model"]["name"]
        except KeyError as error:
            raise KeyError("В конфигурации отсутствует ключ 'model.name'") from error

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )
        except Exception as error:
            raise RuntimeError("Не удалось загрузить модель или токенизатор") from error

        model.eval()

        logger.info("Model loaded successfully")
        logger.info("Vocab size: %d", len(tokenizer))
        logger.info("Model type: %s", type(model).__name__)
        logger.info(
            "Model parameters: %d",
            sum(p.numel() for p in model.parameters()),
        )
        logger.info(
            "Trainable parameters: %d",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        self.model = model
        self.tokenizer = tokenizer

    # Реализация портовых методов
    def tokenize(self, text: str, *, max_length: int) -> Dict[str, Any]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    def forward(self, **inputs: Any) -> Any:
        with torch.no_grad():
            return self.model(**inputs)

    def convert_ids_to_tokens(self, ids_tensor: Any) -> Any:
        return self.tokenizer.convert_ids_to_tokens(ids_tensor)

    def decode_token(self, token_id_tensor: Any) -> str:
        return self.tokenizer.decode(token_id_tensor[0], skip_special_tokens=True)

    def decode_sequence(self, token_ids: Any) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def eos_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    def context_length(self) -> int:
        return int(self.config["model"]["context_length"])

    @property
    def device(self) -> Any:
        return self._device

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Description:
        ---------------
            Возвращает сводную информацию о загруженной модели и окружении.

        Returns:
        ---------------
            dict | None: Словарь с полями о модели или None, если модель
            не загружена.

        Examples:
        ---------------
            >>> manager = GemmaModelManager()
            >>> manager.load_model()
            >>> info = manager.get_model_info()
            >>> isinstance(info, dict)
            True
        """
        # TODO: Реализуйте получение информации о модели:
        # Если модель не загружена, верните None
        # Иначе верните словарь с ключами:
        # - model_name: имя из конфигурации
        # - vocab_size: размер словаря токенизатора
        # - device: строковое представление устройства
        # - model_type: тип модели через type(self.model).__name__
        # - parameters: общее количество параметров
        # - trainable_parameters: количество обучаемых параметров

        # pass

        if self.model is None:
            return None
        return {
            "model_name": self.config["model"]["name"],
            "vocab_size": len(self.tokenizer),
            "device": str(self._device),
            "model_type": type(self.model).__name__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }
