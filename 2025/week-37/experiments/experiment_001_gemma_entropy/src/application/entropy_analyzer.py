"""
Модуль прикладного слоя для анализа вероятностей и энтропии.

Вы должны реализовать весь класс GemmaEntropyAnalyzer.

Предлагаю вам дропнуть весь код, и оставить только подсказки TODO, что нужно реализовать 🤗
"""

# Стандартная библиотека
import logging
from typing import Any, Dict, Optional

# Сторонние библиотеки
import torch

# Внутренние модули (DDD)
from domain.ports import ModelPort
from domain.entropy_calculator import EntropyCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GemmaEntropyAnalyzer:
    """
    Description:
    ---------------
        Класс-обёртка прикладного уровня: получает вероятности от порта модели,
        рассчитывает энтропию через доменный сервис и формирует результаты.

    Examples:
    ---------------
        >>> from infrastructure.gemma_model_manager import GemmaModelManager
        >>> manager = GemmaModelManager("config/experiment.yaml")
        >>> manager.load_model()
        >>> analyzer = GemmaEntropyAnalyzer(manager)
        >>> res = analyzer.analyze_text_entropy("Привет, мир!")
        >>> "entropy" in res
        True
    """

    def __init__(self, model_port: ModelPort, calculator: Optional[EntropyCalculator] = None) -> None:
        """
        Description:
        ---------------
            Инициализирует анализатор, сохраняя порт модели и доменный калькулятор.

        Args:
        ---------------
            model_port: реализация ModelPort (инфраструктура)
            calculator: опциональный экземпляр EntropyCalculator

        Returns:
        ---------------
            None
        """
        # TODO: сохраните зависимости в атрибутах экземпляра:
        # - self.model = model_port
        # - self.calculator = calculator or EntropyCalculator()

        # pass

        self.model = model_port
        self.calculator = calculator or EntropyCalculator()

    def get_token_probabilities(self, text: str) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Токенизирует входной текст, выполняет прямой проход модели и
            возвращает словарь с токенами, их идентификаторами, логитами и
            вероятностями по словарю на каждой позиции.

        Args:
        ---------------
            text: входной текст для анализа вероятностей токенов

        Returns:
        ---------------
            dict: словарь с ключами:
                - 'tokens': список строковых токенов
                - 'token_ids': тензор идентификаторов токенов
                - 'probabilities': тензор вероятностей (B, T, V)
                - 'logits': тензор логитов (B, T, V)

        Raises:
        ---------------
            RuntimeError: если модель не загружена

        Examples:
        ---------------
            >>> from infrastructure.gemma_model_manager import GemmaModelManager
            >>> manager = GemmaModelManager("config/experiment.yaml")
            >>> manager.load_model()
            >>> analyzer = GemmaEntropyAnalyzer(manager)
            >>> res = analyzer.get_token_probabilities("Привет, мир!")
            >>> set(res.keys()) == {"tokens", "token_ids", "probabilities", "logits"}
            True
        """
        # TODO: реализуйте получение вероятностей через порт:
        # 1. Токенизируйте текст через self.model.tokenize(..., max_length)
        # 2. Получите outputs = self.model.forward(**inputs)
        # 3. Извлеките logits из outputs
        # 4. Примените torch.softmax для получения вероятностей
        # 5. Декодируйте токены через self.model.convert_ids_to_tokens
        # 6. Верните словарь результата

        # pass

        inputs = self.model.tokenize(text, max_length=self.model.context_length())
        outputs = self.model.forward(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        tokens = self.model.convert_ids_to_tokens(inputs["input_ids"][0])

        logger.info("Tokens: %s", tokens)
        logger.info("Token IDs: %s", inputs["input_ids"][0])
        logger.info("Probabilities: %s", probabilities)
        logger.info("Logits: %s", logits)

        return {
            "tokens": tokens,
            "token_ids": inputs["input_ids"][0],
            "probabilities": probabilities,
            "logits": logits,
        }

    def analyze_text_entropy(self, text: str) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Выполняет полный цикл анализа: получает вероятности токенов,
            рассчитывает энтропию и возвращает расширенный результат.

        Args:
        ---------------
            text: текст для анализа.

        Returns:
        ---------------
            dict: результаты анализа с добавленным ключом 'entropy'.

        Examples:
        ---------------
            >>> from infrastructure.gemma_model_manager import GemmaModelManager
            >>> manager = GemmaModelManager()
            >>> manager.load_model()
            >>> analyzer = GemmaEntropyAnalyzer(manager)
            >>> res = analyzer.analyze_text_entropy("Пример текста")
            >>> "entropy" in res
            True
        """
        # TODO: реализуйте следующую логику
        # 1. Вызвать self.get_token_probabilities(text) для получения результатов
        # 2. Извлечь probabilities из результатов
        # 3. Вызвать self.calculator.calculate_entropy(probabilities)
        # 4. Добавить ключ 'entropy' в словарь результатов и вернуть

        results = self.get_token_probabilities(text)
        entropy = self.calculator.calculate_entropy(results["probabilities"])
        results["entropy"] = entropy
        return results

    def generate_with_entropy_analysis(self, prompt: str, max_new_tokens: int = 10) -> Dict[str, Any]:
        """
        Description:
        ---------------
            Генерация текста с пошаговым анализом энтропии каждого токена

        Args:
        ---------------
            prompt: Начальный промпт
            max_new_tokens: Максимальное количество новых токенов

        Returns:
        ---------------
            dict: Подробная информация о генерации с энтропией для каждого шага
        """
        # TODO: реализуйте следующую логику
        # 1. Токенизировать промпт через порт
        # 2. Инициализировать списки результатов
        # 3. Выполнить анализ энтропии для исходного промпта
        # 4. Цикл по шагам генерации (жадный выбор argmax):
        #    - forward, извлечь logits[-1]
        #    - softmax → вероятности
        #    - выбрать next_token_id
        #    - декодировать токен в текст
        #    - рассчитать энтропию через self.calculator
        #    - сохранить результаты и расширить input_ids
        #    - остановка по EOS
        # 5. Собрать полный текст и вернуть словарь

        inputs = self.model.tokenize(
            prompt,
            max_length=self.model.context_length(),
        )

        generated_tokens = []
        generated_text_parts = []
        entropies = []
        probabilities_list = []

        current_input_ids = inputs["input_ids"].clone()

        # Анализируем энтропию исходного промпта
        with torch.no_grad():
            outputs = self.model.forward(input_ids=current_input_ids)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            last_token_probs = probs[0, -1, :]
            _ = self.calculator.calculate_entropy(last_token_probs.unsqueeze(0).unsqueeze(0))

        # Генерируем токены один за другим
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model.forward(input_ids=current_input_ids)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.argmax(next_token_probs, dim=-1).unsqueeze(0).unsqueeze(0)

                next_token_text = self.model.decode_token(next_token_id)
                token_entropy = self.calculator.calculate_entropy(next_token_probs.unsqueeze(0).unsqueeze(0))

                generated_tokens.append(int(next_token_id[0, 0]))
                generated_text_parts.append(next_token_text)
                entropies.append(float(token_entropy[0, 0]))
                probabilities_list.append(float(next_token_probs[next_token_id[0, 0]]))

                current_input_ids = torch.cat([current_input_ids, next_token_id], dim=-1)

                if next_token_id[0, 0] == self.model.eos_token_id():
                    break

        full_generated = self.model.decode_sequence(generated_tokens) if generated_tokens else ""

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
