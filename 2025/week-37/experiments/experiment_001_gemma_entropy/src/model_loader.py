"""
Модуль для загрузки и настройки модели Gemma-3n-E2B-it.
Вы должены реализовать весь класс GemmaEntropyAnalyzer.
"""

# Стандартная библиотека
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Сторонние библиотеки
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GemmaEntropyAnalyzer:
    """
    Description:
    ---------------
        Класс-обёртка для загрузки модели и токенизатора, вычисления
        вероятностей токенов и расчёта энтропии для заданного текста.

    Examples:
    ---------------
        >>> analyzer = GemmaEntropyAnalyzer("config/experiment.yaml")
        >>> analyzer.load_model()
        >>> info = analyzer.get_model_info()
        >>> isinstance(info, dict)
        True
    """
    
    def __init__(self, config_path: str = "config/experiment.yaml") -> None:
        """
        Description:
        ---------------
            Инициализирует анализатор: загружает конфигурацию, подготавливает
            атрибуты модели, токенизатора и устройства.

        Args:
        ---------------
            config_path: Путь к конфигурационному файлу.

        Returns:
        ---------------
            None

        Raises:
        ---------------
            FileNotFoundError: Если файл конфигурации не найден

        Examples:
        ---------------
            >>> analyzer = GemmaEntropyAnalyzer("config/experiment.yaml")
            >>> analyzer.model is None
            True
        """
        # TODO: Инициализируйте следующие атрибуты:
        # - self.config (загрузите конфигурацию через _load_config)
        # - self.model = None
        # - self.tokenizer = None  
        # - self.device (определите CUDA или CPU)
        # Выведите информацию об устройстве через logger.info

        # pass

        self.config = self._load_config(config_path)
        self.model = None
        self.tokenizer = None
        # Принудительно используем CPU для избежания проблем с MPS на Apple Silicon
        self.device = torch.device("cpu")

        logger.info("Device: %s", self.device)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Загружает YAML-конфигурацию эксперимента и возвращает её
            как словарь.

        Args:
        ---------------
            config_path: Путь к файлу конфигурации.

        Returns:
        ---------------
            dict: Загруженная конфигурация.

        Raises:
        ---------------
            FileNotFoundError: Если файл конфигурации не найден

        Examples:
        ---------------
            >>> analyzer = GemmaEntropyAnalyzer()
            >>> isinstance(analyzer.config, dict)
            True
        """
        # TODO: Реализуйте загрузку YAML конфигурации:
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
            KeyError: Отсутствуют необходимые ключи в конфигурации
            RuntimeError: Ошибка при загрузке токенизатора или модели

        Examples:
        ---------------
            >>> analyzer = GemmaEntropyAnalyzer("config/experiment.yaml")
            >>> analyzer.load_model()
            >>> analyzer.model is not None and analyzer.tokenizer is not None
            True
        """
        # TODO: Реализуйте загрузку модели:
        # 1. Получите имя модели из self.config['model']['name']
        # 2. Загрузите токенизатор через AutoTokenizer.from_pretrained
        #    с параметром trust_remote_code=True
        # 3. Установите pad_token = eos_token если pad_token отсутствует
        # 4. Загрузите модель через AutoModelForCausalLM.from_pretrained
        #    с параметрами: torch_dtype=torch.bfloat16, device_map из конфига, trust_remote_code=True
        # 5. Переведите модель в режим eval()
        # 6. Выведите информацию о загрузке и размере словаря
        
        # pass

        # Почему: Явно валидируем конфигурацию и перехватываем ошибки,
        # чтобы обеспечить предсказуемые и информативные сообщения.
        try:
            model_config = self.config['model']['name']
        except KeyError as error:
            raise KeyError(
                "В конфигурации отсутствует ключ 'model.name'"
            ) from error

        try:
            # Загружаем токенизатор. trust_remote_code=True позволяет
            # использовать пользовательские реализации из репозитория модели.
            tokenizer = AutoTokenizer.from_pretrained(
                model_config,
                trust_remote_code=True,
            )

            # Если pad_token не определён, используем eos_token для корректной
            # работы padding при пакетной обработке.
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Загружаем модель на CPU для стабильной работы
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_config,
                torch_dtype=torch.float32,  # Используем float32 для CPU
                device_map="cpu",
                trust_remote_code=True,
            )
        except Exception as error:  # noqa: BLE001 — здесь важно собрать все ошибки загрузки
            raise RuntimeError("Не удалось загрузить модель или токенизатор") from error

        # Переводим модель в режим оценки для выключения dropout и др.
        model.eval()

        # Диагностическая информация для воспроизводимости и аудита.
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

        # Сохраняем ссылки на загруженные объекты в атрибуты экземпляра.
        self.model = model
        self.tokenizer = tokenizer
 
    def get_token_probabilities(self, text: str) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Токенизирует входной текст, выполняет прямой проход модели и
            возвращает словарь с токенами, их идентификаторами, логитами и
            вероятностями по словарю на каждой позиции.

        Args:
        ---------------
            text: Входной текст для анализа вероятностей токенов

        Returns:
        ---------------
            dict: Словарь с ключами:
                - 'tokens': список строковых токенов
                - 'token_ids': тензор идентификаторов токенов
                - 'probabilities': тензор вероятностей (B, T, V)
                - 'logits': тензор логитов (B, T, V)

        Raises:
        ---------------
            RuntimeError: Если модель или токенизатор не загружены

        Examples:
        ---------------
            >>> analyzer = GemmaEntropyAnalyzer("config/experiment.yaml")
            >>> analyzer.load_model()
            >>> res = analyzer.get_token_probabilities("Привет, мир!")
            >>> set(res.keys()) == {"tokens", "token_ids", "probabilities", "logits"}
            True
        """
        # TODO: Реализуйте получение вероятностей:
        # 1. Проверьте, что модель и токенизатор загружены
        # 2. Токенизируйте текст (return_tensors="pt", padding=True, truncation=True, max_length из конфига)
        # 3. Переместите inputs на устройство
        # 4. Получите outputs модели (используйте torch.no_grad())
        # 5. Извлеките logits из outputs
        # 6. Примените torch.softmax для получения вероятностей
        # 7. Декодируйте токены
        # 8. Верните словарь с ключами: 'tokens', 'token_ids', 'probabilities', 'logits'
        
        # pass

        # Проверяем, что все необходимые компоненты инициализированы.
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Модель или токенизатор не загружены")

        # Токенизация входа. Почему: задаём явные параметры для
        # воспроизводимости и контроля длины последовательности.
        input = self.tokenizer(  # noqa: A002 — имя сохранено по требованию
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['model']['context_length'],
        )

        # Переносим данные на корректное устройство, чтобы избежать ошибок
        # несоответствия устройств при вызове модели.
        input = {key: value.to(self.device) for key, value in input.items()}

        # Прямой проход без вычисления градиентов для эффективности.
        with torch.no_grad():
            outputs = self.model(**input)

        # Извлекаем логиты и получаем вероятности softmax по размерности словаря.
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)

        # Декодируем токены для удобства анализа человеком.
        tokens = self.tokenizer.convert_ids_to_tokens(input["input_ids"][0])
        
        logger.info("Tokens: %s", tokens)
        logger.info("Token IDs: %s", input["input_ids"][0])
        logger.info("Probabilities: %s", probabilities)
        logger.info("Logits: %s", logits)

        return {
            "tokens": tokens,
            "token_ids": input["input_ids"][0],
            "probabilities": probabilities,
            "logits": logits,
        }
    
    def calculate_entropy(
        self,
        probabilities: torch.Tensor,
        epsilon: float = 1e-10,
    ) -> torch.Tensor:
        """
        Description:
        ---------------
            Вычисляет энтропию Шеннона по каждой позиции последовательности
            на основе распределения вероятностей по словарю.

        Args:
        ---------------
            probabilities: Тензор вероятностей формы (B, T, V) или (T, V), где
                B — размер батча (необязательно), T — длина последовательности,
                V — размер словаря. Значения должны быть нормированными по V.
            epsilon: Малое добавочное значение для численной стабильности,
                предотвращающее log(0).

        Returns:
        ---------------
            torch.Tensor: Тензор энтропий формы (B, T) или (T,) в зависимости
            от входной размерности.

        Raises:
        ---------------
            ValueError: Если вероятность содержит отрицательные значения.

        Examples:
        ---------------
            >>> probs = torch.tensor([[0.5, 0.5]])  # (T=1, V=2)
            >>> analyzer = GemmaEntropyAnalyzer()
            >>> H = analyzer.calculate_entropy(probs)
            >>> H.shape
            torch.Size([1])
        """
        # TODO: Реализуйте расчет энтропии:
        # 1. Добавьте epsilon к probabilities для избежания log(0)
        # 2. Вычислите torch.log2(probabilities + epsilon)
        # 3. Умножьте probabilities на логарифм
        # 4. Просуммируйте по последнему измерению (dim=-1)
        # 5. Примените отрицательный знак
        # Формула: -torch.sum(probabilities * torch.log2(probabilities + epsilon), dim=-1)
        
        # pass

        # Почему: Проверяем входные данные на валидность, чтобы рано выявлять
        # ошибки источника данных и получать понятные сообщения.
        if torch.any(probabilities < 0):
            raise ValueError("Вероятности не могут быть отрицательными")

        # Добавляем epsilon для избегания log(0) и улучшения стабильности.
        probabilities = probabilities + epsilon

        # Логарифм по основанию 2 даёт энтропию в битах, что удобно для
        # интерпретации в задачах обработки текста.
        log_probabilities = torch.log2(probabilities)

        # Формула энтропии Шеннона: -sum(p * log2(p)) по размерности словаря.
        entropy = -torch.sum(probabilities * log_probabilities, dim=-1)

        return entropy
    
    def analyze_text_entropy(self, text: str) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Выполняет полный цикл анализа: получает вероятности токенов,
            рассчитывает энтропию и возвращает расширенный результат.

        Args:
        ---------------
            text: Текст для анализа.

        Returns:
        ---------------
            dict: Результаты анализа с добавленным ключом 'entropy'.

        Examples:
        ---------------
            >>> analyzer = GemmaEntropyAnalyzer()
            >>> analyzer.load_model()
            >>> res = analyzer.analyze_text_entropy("Пример текста")
            >>> "entropy" in res
            True
        """
        results = self.get_token_probabilities(text)
        entropy = self.calculate_entropy(results["probabilities"])
        results["entropy"] = entropy

        return results
    
    def generate_with_entropy_analysis(self, prompt: str, max_new_tokens: int = 10) -> Dict[str, Any]:
        """
        Генерация текста с пошаговым анализом энтропии каждого токена
        
        Args:
            prompt: Начальный промпт
            max_new_tokens: Максимальное количество новых токенов
            
        Returns:
            dict: Подробная информация о генерации с энтропией для каждого шага
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Модель или токенизатор не загружены")
            
        # Токенизируем начальный промпт
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config['model']['context_length']
        ).to(self.device)
        
        generated_tokens = []
        generated_text_parts = []
        entropies = []
        probabilities_list = []
        
        current_input_ids = inputs['input_ids'].clone()
        
        # Анализируем энтропию исходного промпта
        with torch.no_grad():
            outputs = self.model(input_ids=current_input_ids)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Энтропия для последнего токена (который будет использован для предсказания)
            last_token_probs = probs[0, -1, :]  # Вероятности для следующего токена
            entropy = self.calculate_entropy(last_token_probs.unsqueeze(0).unsqueeze(0))
            
        # Генерируем токены один за другим
        for step in range(max_new_tokens):
            with torch.no_grad():
                # Получаем логиты для текущей последовательности
                outputs = self.model(input_ids=current_input_ids)
                logits = outputs.logits
                
                # Берем логиты для последнего токена
                next_token_logits = logits[0, -1, :]
                
                # Применяем softmax для получения вероятностей
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                
                # Выбираем следующий токен (жадный поиск)
                next_token_id = torch.argmax(next_token_probs, dim=-1).unsqueeze(0).unsqueeze(0)
                
                # Декодируем токен
                next_token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
                
                # Вычисляем энтропию для этого распределения
                token_entropy = self.calculate_entropy(next_token_probs.unsqueeze(0).unsqueeze(0))
                
                # Сохраняем результаты
                generated_tokens.append(int(next_token_id[0, 0]))
                generated_text_parts.append(next_token_text)
                entropies.append(float(token_entropy[0, 0]))
                probabilities_list.append(float(next_token_probs[next_token_id[0, 0]]))
                
                # Добавляем новый токен к последовательности
                current_input_ids = torch.cat([current_input_ids, next_token_id], dim=-1)
                
                # Остановка если достигли EOS токена
                if next_token_id[0, 0] == self.tokenizer.eos_token_id:
                    break
        
        # Собираем полный сгенерированный текст
        full_generated = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            'prompt': prompt,
            'generated_tokens': generated_tokens,
            'generated_text_parts': generated_text_parts,
            'full_generated_text': full_generated,
            'complete_text': prompt + full_generated,
            'entropies': entropies,
            'probabilities': probabilities_list,
            'generation_steps': len(generated_tokens)
        }
    
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
            >>> analyzer = GemmaEntropyAnalyzer()
            >>> analyzer.load_model()
            >>> info = analyzer.get_model_info()
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
            "device": str(self.device),
            "model_type": type(self.model).__name__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }