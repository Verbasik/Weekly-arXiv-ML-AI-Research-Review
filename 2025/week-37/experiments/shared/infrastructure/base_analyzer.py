"""
Shared Infrastructure: Base Analyzer

Общий базовый класс для всех анализаторов экспериментов.
Устраняет дублирование кода между EntropyAnalyzer и ConfidenceAnalyzer.
"""

# Стандартная библиотека
import logging
from typing import Any, Dict, Protocol, runtime_checkable

# Сторонние библиотеки
import torch

# Общая инфраструктура
from .ports import ModelPort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@runtime_checkable
class MetricCalculator(Protocol):
    """Протокол для всех калькуляторов метрик (entropy, confidence, etc.)"""
    
    def calculate(self, probabilities: torch.Tensor, **kwargs) -> torch.Tensor:
        """Вычисляет метрику по тензору вероятностей"""
        ...


class BaseAnalyzer:
    """
    Базовый анализатор для экспериментов с LLM метриками.
    
    Содержит общую логику для:
    - Получения вероятностей токенов от модели
    - Координации с доменными калькуляторами 
    - Генерации с пошаговым анализом
    """

    def __init__(self, model_port: ModelPort, calculator: MetricCalculator, metric_name: str = "metric"):
        """
        Args:
            model_port: реализация ModelPort (инфраструктура)
            calculator: калькулятор метрики (domain layer)
            metric_name: название метрики для логирования и результатов
        """
        self.model = model_port
        self.calculator = calculator
        self.metric_name = metric_name

    def get_token_probabilities(self, text: str) -> Dict[str, Any]:
        """
        Получает вероятности токенов через ModelPort.
        
        Returns:
            dict: словарь с токенами, ID, вероятностями и логитами
        """
        inputs = self.model.tokenize(text, max_length=self.model.context_length())
        outputs = self.model.forward(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        tokens = self.model.convert_ids_to_tokens(inputs["input_ids"][0])

        logger.info("Tokens: %s", tokens)
        logger.info("Token IDs: %s", inputs["input_ids"][0])
        logger.info("Probabilities shape: %s", probabilities.shape)

        return {
            "tokens": tokens,
            "token_ids": inputs["input_ids"][0],
            "probabilities": probabilities,
            "logits": logits,
        }

    def analyze_text(self, text: str, **calc_kwargs) -> Dict[str, Any]:
        """
        Выполняет анализ текста с заданной метрикой.
        
        Args:
            text: текст для анализа
            **calc_kwargs: дополнительные параметры для калькулятора
            
        Returns:
            dict: результаты с добавленной метрикой
        """
        results = self.get_token_probabilities(text)
        
        # Вызываем калькулятор с правильным методом
        if hasattr(self.calculator, 'calculate_entropy'):
            metric_value = self.calculator.calculate_entropy(results["probabilities"], **calc_kwargs)
        elif hasattr(self.calculator, 'calculate_confidence'):
            metric_value = self.calculator.calculate_confidence(results["probabilities"], **calc_kwargs)
        else:
            # Fallback для общего протокола
            metric_value = self.calculator.calculate(results["probabilities"], **calc_kwargs)
            
        results[self.metric_name] = metric_value
        return results

    def generate_with_analysis(self, prompt: str, max_new_tokens: int = 10, **calc_kwargs) -> Dict[str, Any]:
        """
        Генерация с пошаговым анализом метрики.
        
        Args:
            prompt: начальный промпт
            max_new_tokens: максимум новых токенов
            **calc_kwargs: параметры для калькулятора
            
        Returns:
            dict: детальная информация о генерации с метриками
        """
        inputs = self.model.tokenize(prompt, max_length=self.model.context_length())
        
        generated_tokens = []
        generated_text_parts = []
        metric_values = []
        probabilities_list = []
        
        current_input_ids = inputs["input_ids"].clone()

        # Генерируем токены один за другим
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model.forward(input_ids=current_input_ids)
                logits = outputs.logits
                next_token_logits = logits[0, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.argmax(next_token_probs, dim=-1).unsqueeze(0).unsqueeze(0)

                next_token_text = self.model.decode_token(next_token_id)
                
                # Вычисляем метрику для текущего шага
                if hasattr(self.calculator, 'calculate_entropy'):
                    token_metric = self.calculator.calculate_entropy(
                        next_token_probs.unsqueeze(0).unsqueeze(0), **calc_kwargs)
                elif hasattr(self.calculator, 'calculate_confidence'):
                    token_metric = self.calculator.calculate_confidence(
                        next_token_probs.unsqueeze(0).unsqueeze(0), **calc_kwargs)
                else:
                    token_metric = self.calculator.calculate(
                        next_token_probs.unsqueeze(0).unsqueeze(0), **calc_kwargs)

                generated_tokens.append(int(next_token_id[0, 0]))
                generated_text_parts.append(next_token_text)
                metric_values.append(float(token_metric[0, 0]))
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
            f'{self.metric_name}s': metric_values,  # entropies/confidences
            'probabilities': probabilities_list,
            'generation_steps': len(generated_tokens)
        }