#!/usr/bin/env python3
"""
Ollama Client for SGR Mathematical Agent
HTTP клиент для интеграции с локальной Ollama моделью
"""

import json
import requests
import time
from typing import Type, TypeVar, Dict, Any, Optional, List
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel

console = Console()

T = TypeVar('T', bound=BaseModel)

class OllamaClient:
    """HTTP клиент для взаимодействия с Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3n:e2b"):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        self.tags_url = f"{self.base_url}/api/tags"
    
    def is_available(self) -> bool:
        """Проверка доступности Ollama сервера"""
        try:
            response = requests.get(self.tags_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            console.print(f"[red]❌ Ollama недоступен: {e}[/red]")
            return False
    
    def list_models(self) -> List[str]:
        """Получение списка доступных моделей"""
        try:
            response = requests.get(self.tags_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except Exception as e:
            console.print(f"[red]❌ Ошибка получения моделей: {e}[/red]")
            return []
    
    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, 
                     temperature: float = 0.1, max_tokens: Optional[int] = None, 
                     stream_output: bool = False) -> str:
        """Генерация текста через Ollama API с опциональным streaming"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream_output,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or 4000,
                "stop": []
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            console.print(f"[dim]🔄 Запрос к Ollama ({self.model}){'[STREAM]' if stream_output else ''}...[/dim]")
            
            response = requests.post(
                self.generate_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=600,  # 10 минут на генерацию
                stream=stream_output
            )
            
            if response.status_code == 200:
                if stream_output:
                    return self._handle_streaming_response(response)
                else:
                    result = response.json()
                    generated_text = result.get("response", "")
                    
                    # Логируем статистику
                    if "eval_count" in result:
                        tokens = result.get("eval_count", 0)
                        duration = result.get("eval_duration", 0) / 1_000_000  # ns to ms
                        speed = tokens / (duration / 1000) if duration > 0 else 0
                        console.print(f"[dim]📊 Токены: {tokens}, Скорость: {speed:.1f} tok/sec[/dim]")
                    
                    return generated_text
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                console.print(f"[red]❌ Ollama API error: {error_msg}[/red]")
                raise Exception(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "Timeout при обращении к Ollama"
            console.print(f"[red]❌ {error_msg}[/red]")
            raise Exception(error_msg)
        except Exception as e:
            console.print(f"[red]❌ Ошибка Ollama: {e}[/red]")
            raise
    
    def _handle_streaming_response(self, response) -> str:
        """Обработка streaming ответа с real-time выводом"""
        import time
        from rich.live import Live
        from rich.text import Text
        
        full_text = ""
        display_text = Text()
        
        console.print("\n[yellow]🔥 REAL-TIME ГЕНЕРАЦИЯ:[/yellow]")
        console.print("─" * 80)
        
        with Live(display_text, refresh_per_second=4, console=console) as live:
            start_time = time.time()
            
            try:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk_data = json.loads(line.decode('utf-8'))
                            
                            if 'response' in chunk_data:
                                chunk_text = chunk_data['response']
                                full_text += chunk_text
                                
                                # Обновляем отображение
                                display_text = Text(full_text)
                                display_text.stylize("green")
                                live.update(display_text)
                                
                                # Если генерация завершена
                                if chunk_data.get('done', False):
                                    # Финальная статистика
                                    duration = time.time() - start_time
                                    tokens = len(full_text.split())
                                    speed = tokens / duration if duration > 0 else 0
                                    
                                    console.print(f"\n[dim]📊 Завершено: {tokens} токенов за {duration:.1f}с, {speed:.1f} tok/sec[/dim]")
                                    break
                                    
                        except json.JSONDecodeError:
                            continue
                            
            except Exception as e:
                console.print(f"\n[red]❌ Ошибка streaming: {e}[/red]")
        
        console.print("─" * 80)
        return full_text
    
    def generate_structured(self, schema: Type[T], prompt: str, 
                          system_prompt: Optional[str] = None,
                          max_retries: int = 3, stream_output: bool = True) -> T:
        """
        Генерация структурированного ответа с валидацией через Pydantic схему
        
        ВАЖНО: Ollama не поддерживает нативную JSON Schema валидацию,
        поэтому используется post-processing подход с повторными попытками
        """
        
        # ИСПРАВЛЕННЫЙ системный промпт - БЕЗ JSON Schema!
        json_system_prompt = f"""
{system_prompt or ''}

КРИТИЧЕСКИ ВАЖНО: Отвечайте ТОЛЬКО валидным JSON объектом с нужными полями.

НЕ ГЕНЕРИРУЙТЕ JSON SCHEMA! Генерируйте JSON ДАННЫЕ!

ТРЕБОВАНИЯ К ФОРМАТУ:
1. ТОЛЬКО JSON объект - никакого текста до или после
2. Все строки в двойных кавычках  
3. Все обязательные поля заполнены
4. Правильные типы данных (строки, числа, массивы, булево)

ПРАВИЛЬНЫЕ ПРИМЕРЫ JSON ДАННЫХ:
{{
  "tool": "analyze_problem",
  "reasoning": "Анализирую линейное уравнение",
  "problem_domain": "algebra"
}}

{{
  "problem_type": "linear equation", 
  "difficulty": "easy",
  "approach": "isolation method",
  "estimated_steps": 2
}}

НЕПРАВИЛЬНО (НЕ ДЕЛАЙТЕ ТАК):
- Не генерируйте "properties", "required", "type", "description" 
- Не добавляйте markdown ````json блоки
- Не добавляйте объяснения до/после JSON

НЕ ДОБАВЛЯЙТЕ НИКАКОГО ТЕКСТА ДО ИЛИ ПОСЛЕ JSON!
""".strip()
        
        for attempt in range(max_retries):
            try:
                console.print(f"[dim]🎯 Попытка генерации структурированного ответа #{attempt + 1}[/dim]")
                
                raw_response = self.generate_text(
                    prompt=prompt,
                    system_prompt=json_system_prompt,
                    temperature=0.1,  # Низкая температура для структурированности
                    stream_output=stream_output
                )
                
                # Попытка очистки ответа от лишнего текста
                cleaned_response = self._extract_json(raw_response)
                
                # Отладочная информация
                console.print(f"[dim]📝 Raw response length: {len(raw_response)}[/dim]")
                console.print(f"[dim]📝 Cleaned JSON length: {len(cleaned_response)}[/dim]")
                
                # Парсинг JSON
                try:
                    parsed_data = json.loads(cleaned_response)
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]⚠️ JSON parse error (attempt {attempt + 1}): {e}[/yellow]")
                    console.print(f"[dim]Raw: {raw_response[:200]}...[/dim]")
                    continue
                
                # Валидация через Pydantic
                try:
                    validated_obj = schema(**parsed_data)
                    console.print(f"[green]✅ Структурированный ответ валидирован[/green]")
                    return validated_obj
                    
                except ValidationError as e:
                    console.print(f"[yellow]⚠️ Validation error (attempt {attempt + 1}):[/yellow]")
                    error_count = len(e.errors())
                    console.print(f"[dim]  Всего ошибок: {error_count}[/dim]")
                    
                    # Показываем только первые 5 ошибок для краткости
                    for i, error in enumerate(e.errors()[:5]):
                        console.print(f"  - {error['loc']}: {error['msg']}")
                    if error_count > 5:
                        console.print(f"  ... и еще {error_count - 5} ошибок")
                    
                    # Попытка коррекции на каждой итерации, не только на последней
                    console.print("[yellow]🔧 Попытка автокоррекции...[/yellow]")
                    corrected = self._attempt_correction(parsed_data, schema, e)
                    if corrected:
                        console.print("[green]✅ Автокоррекция успешна![/green]")
                        return corrected
                    else:
                        console.print("[dim]❌ Автокоррекция не удалась, повторяем генерацию[/dim]")
                    continue
                    
            except Exception as e:
                console.print(f"[red]❌ Неожиданная ошибка (attempt {attempt + 1}): {e}[/red]")
                if attempt == max_retries - 1:
                    raise
                continue
        
        # Если все попытки неудачны
        error_msg = f"Не удалось получить валидный структурированный ответ за {max_retries} попыток"
        console.print(f"[red]❌ {error_msg}[/red]")
        raise Exception(error_msg)
    
    def _extract_json(self, text: str) -> str:
        """Извлечение JSON из текста с markdown и другим мусором"""
        import re
        
        text = text.strip()
        
        # Паттерны для очистки markdown
        markdown_patterns = [
            r'```json\s*',       # ```json в начале
            r'```\s*$',          # ``` в конце
            r'^```\s*',          # ``` в начале строки
            r'`{3,}\w*\s*',      # любые тройные бэктики с языком
        ]
        
        # Удаляем markdown разметку
        for pattern in markdown_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        
        text = text.strip()
        
        # Пытаемся найти JSON через регулярные выражения
        json_patterns = [
            # Полный JSON объект от { до }
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
            # JSON объект с возможной вложенностью
            r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})',
            # Простое извлечение от первой { до последней }
            r'(\{.*\})',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                candidate = match.group(1).strip()
                # Проверим, что это похоже на валидный JSON
                if candidate.count('{') == candidate.count('}') and candidate.count('"') % 2 == 0:
                    return candidate
        
        # Fallback: старый метод поиска скобок
        start_idx = text.find('{')
        if start_idx == -1:
            return text  # Нет JSON скобок
        
        # Ищем соответствующую закрывающую скобку
        brace_count = 0
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[start_idx:i+1]
        
        # Если не нашли закрывающую скобку, возвращаем от первой скобки до конца
        return text[start_idx:]
    
    def _attempt_correction(self, data: Dict[str, Any], schema: Type[T], 
                          validation_error: ValidationError) -> Optional[T]:
        """Улучшенная автоматическая коррекция валидационных ошибок"""
        try:
            corrected_data = data.copy()
            
            # Словарь значений по умолчанию для разных типов схем
            default_values = {
                'tool': self._guess_tool_from_schema(schema),
                'reasoning': "Автоматически сгенерированное обоснование",
                'current_situation': "Анализ текущей ситуации",
                'problem_understanding': "good",
                'solution_progress': "analysis_done",
                'task_completed': False,
                'remaining_steps': ["Продолжить анализ"],
                'reasoning_chain': ["Шаг 1", "Шаг 2"],
                'verification_attempts': 0,
                'improvement_attempts': 0,
                'chosen_approach': "стандартный подход",
                'solution_steps_plan': ["Шаг 1", "Шаг 2", "Шаг 3"],
                'expected_techniques': ["базовые методы"],
                'problem_domain': "algebra",
                'problem_type': "уравнение",
                'key_concepts': ["математика", "решение"],
                'difficulty_assessment': "medium",
                'suggested_approaches': ["метод 1", "метод 2"],
                'solution_summary': "Краткое решение задачи",
                'detailed_solution': "Подробное пошаговое решение",
                'mathematical_rigor': "complete",
                'confidence': "high",
                'final_answer': "Финальный ответ",
                'solution_quality': "good",
                'completed_steps': ["анализ", "решение"]
            }
            
            # Специальная обработка для сложных объектов в данных
            self._fix_complex_fields(corrected_data)
            
            for error in validation_error.errors():
                field_name = error['loc'][0] if error['loc'] else None
                error_type = error['type']
                
                if field_name and error_type == 'missing':
                    # Используем умные значения по умолчанию
                    if field_name in default_values:
                        corrected_data[field_name] = default_values[field_name]
                    else:
                        # Fallback для неизвестных полей
                        field_info = schema.model_fields.get(field_name)
                        if field_info:
                            if hasattr(field_info.annotation, '__origin__'):
                                if field_info.annotation.__origin__ is list:
                                    corrected_data[field_name] = ["значение"]
                                elif field_info.annotation.__origin__ is dict:
                                    corrected_data[field_name] = {}
                            else:
                                if field_info.annotation == str:
                                    corrected_data[field_name] = "автозаполнено"
                                elif field_info.annotation == bool:
                                    corrected_data[field_name] = False
                                elif field_info.annotation == int:
                                    corrected_data[field_name] = 0
                
                elif field_name == 'tool' and error_type == 'literal_error':
                    # Исправление неправильных значений tool
                    corrected_data['tool'] = self._guess_tool_from_schema(schema)
            
            # Специальные исправления для Union типов
            if hasattr(schema, '__name__') and 'NextStep' in schema.__name__:
                corrected_data = self._fix_nextstep_union(corrected_data, schema)
            
            return schema(**corrected_data)
            
        except Exception as e:
            console.print(f"[dim]🔧 Коррекция не удалась: {e}[/dim]")
            return None
    
    def _guess_tool_from_schema(self, schema: Type[T]) -> str:
        """Угадываем значение поля tool на основе имени схемы"""
        schema_name = schema.__name__.lower()
        
        # Для MathSolutionNextStep - определяем по контексту
        if 'nextstep' in schema_name or 'mathsolution' in schema_name:
            return 'generate_solution'  # Переходим к решению после анализа
        elif 'analysis' in schema_name:
            return 'analyze_problem'
        elif 'strategy' in schema_name:
            return 'choose_strategy'
        elif 'solution' in schema_name:
            return 'generate_solution'
        elif 'verification' in schema_name:
            return 'verify_solution'
        elif 'improvement' in schema_name:
            return 'improve_solution'
        elif 'completion' in schema_name:
            return 'complete_task'
        else:
            return 'generate_solution'  # По умолчанию переходим к решению
    
    def _fix_nextstep_union(self, data: Dict[str, Any], schema: Type[T]) -> Dict[str, Any]:
        """Улучшенное исправление для NextStep схем с Union типами"""
        if 'function' not in data or not isinstance(data['function'], dict):
            # Создаем функцию по умолчанию на основе контекста
            data['function'] = self._create_default_function(data)
            return data
            
        function_data = data['function']
        
        # Определяем какой Union тип нужен на основе контекста
        target_union_type = self._detect_union_type(data, function_data)
        
        # Создаем правильную структуру для выбранного типа
        if target_union_type == 'analyze_problem':
            data['function'] = {
                'tool': 'analyze_problem',
                'reasoning': function_data.get('reasoning', 'Анализ математической задачи'),
                'problem_domain': function_data.get('problem_domain', 'algebra'),
                'problem_type': function_data.get('problem_type', 'уравнение'),
                'key_concepts': function_data.get('key_concepts', ['математика', 'решение']),
                'difficulty_assessment': function_data.get('difficulty_assessment', 'medium'),
                'suggested_approaches': function_data.get('suggested_approaches', ['стандартный метод', 'альтернативный подход'])
            }
        elif target_union_type == 'choose_strategy':
            data['function'] = {
                'tool': 'choose_strategy',
                'reasoning': function_data.get('reasoning', 'Выбор стратегии решения'),
                'chosen_approach': function_data.get('chosen_approach', 'стандартный подход'),
                'solution_steps_plan': function_data.get('solution_steps_plan', ['шаг 1', 'шаг 2', 'шаг 3']),
                'expected_techniques': function_data.get('expected_techniques', ['базовые методы'])
            }
        elif target_union_type == 'generate_solution':
            data['function'] = {
                'tool': 'generate_solution',
                'reasoning': function_data.get('reasoning', 'Генерация решения задачи'),
                'solution_summary': function_data.get('solution_summary', 'Краткое решение'),
                'detailed_solution': function_data.get('detailed_solution', 'Подробное решение с шагами'),
                'key_insights': function_data.get('key_insights', ['ключевая идея']),
                'mathematical_rigor': function_data.get('mathematical_rigor', 'complete'),
                'confidence': function_data.get('confidence', 'high')
            }
        elif target_union_type == 'complete_task':
            data['function'] = {
                'tool': 'complete_task',
                'reasoning': function_data.get('reasoning', 'Задача решена'),
                'final_answer': function_data.get('final_answer', 'Финальный ответ'),
                'solution_quality': function_data.get('solution_quality', 'good'),
                'completed_steps': function_data.get('completed_steps', ['анализ', 'решение'])
            }
        else:
            # По умолчанию - анализ
            data['function'] = self._create_default_function(data)
            
        return data
    
    def _fix_complex_fields(self, data: Dict[str, Any]) -> None:
        """Исправление сложных полей, которые модель генерирует неправильно"""
        
        # Исправляем remaining_steps - должен быть List[str], а не List[dict]
        if 'remaining_steps' in data and isinstance(data['remaining_steps'], list):
            steps = data['remaining_steps']
            if steps and isinstance(steps[0], dict):
                # Преобразуем dict в строки
                string_steps = []
                for step in steps:
                    if isinstance(step, dict) and 'description' in step:
                        string_steps.append(step['description'])
                    elif isinstance(step, dict):
                        # Берем первое строковое значение
                        for value in step.values():
                            if isinstance(value, str):
                                string_steps.append(value)
                                break
                    else:
                        string_steps.append(str(step))
                data['remaining_steps'] = string_steps[:5]  # Максимум 5 шагов
        
        # Исправляем problem_domain - убираем недопустимые значения
        if 'problem_domain' in data:
            domain = data['problem_domain']
            valid_domains = ['algebra', 'geometry', 'calculus', 'number_theory', 'analysis', 'combinatorics']
            if domain not in valid_domains:
                data['problem_domain'] = 'algebra'  # По умолчанию
    
    def _detect_union_type(self, context_data: Dict[str, Any], function_data: Dict[str, Any]) -> str:
        """Определение нужного Union типа на основе контекста"""
        
        # Проверяем явные указания в данных
        if 'tool' in function_data:
            tool = function_data['tool']
            if tool in ['analyze_problem', 'choose_strategy', 'generate_solution', 
                       'verify_solution', 'improve_solution', 'complete_task']:
                return tool
        
        # Анализируем прогресс задачи
        progress = context_data.get('solution_progress', 'not_started')
        task_completed = context_data.get('task_completed', False)
        
        if task_completed:
            return 'complete_task'
        elif progress == 'not_started':
            return 'analyze_problem'
        elif progress in ['analysis_done', 'analyzing']:
            return 'choose_strategy' 
        elif progress in ['planning', 'solving_in_progress']:
            return 'generate_solution'
        elif progress == 'solution_complete':
            return 'complete_task'
        else:
            return 'analyze_problem'  # По умолчанию
    
    def _create_default_function(self, context_data: Dict[str, Any]) -> Dict[str, str]:
        """Создание функции по умолчанию"""
        return {
            'tool': 'analyze_problem',
            'reasoning': 'Начинаю анализ математической задачи',
            'problem_domain': 'algebra',
            'problem_type': 'уравнение',
            'key_concepts': ['математика', 'решение'],
            'difficulty_assessment': 'medium',
            'suggested_approaches': ['стандартный метод', 'альтернативный подход']
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Проверка состояния Ollama и доступных моделей"""
        result = {
            "available": False,
            "models": [],
            "current_model": self.model,
            "base_url": self.base_url
        }
        
        try:
            # Проверяем доступность
            result["available"] = self.is_available()
            
            if result["available"]:
                result["models"] = self.list_models()
                
                # Проверяем, доступна ли выбранная модель
                if self.model not in result["models"]:
                    console.print(f"[yellow]⚠️ Модель {self.model} не найдена среди доступных[/yellow]")
                    console.print(f"[dim]Доступные модели: {', '.join(result['models'])}[/dim]")
                
        except Exception as e:
            console.print(f"[red]❌ Ошибка health check: {e}[/red]")
        
        return result

# =============================================================================
# УТИЛИТЫ ДЛЯ ТЕСТИРОВАНИЯ
# =============================================================================

def test_ollama_connection():
    """Тестирование подключения к Ollama"""
    console.print(Panel("🧪 Тестирование Ollama интеграции", title="Test"))
    
    client = OllamaClient()
    
    # Health check
    health = client.health_check()
    console.print(f"Status: {'✅ OK' if health['available'] else '❌ Unavailable'}")
    console.print(f"URL: {health['base_url']}")
    console.print(f"Model: {health['current_model']}")
    console.print(f"Available models: {', '.join(health['models'])}")
    
    if not health["available"]:
        console.print("[red]❌ Ollama недоступен. Запустите Docker контейнер:[/red]")
        console.print("[dim]docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama[/dim]")
        console.print("[dim]docker exec -it ollama ollama run gemma2:7b[/dim]")
        return False
    
    # Тестовая генерация
    try:
        console.print("\n🔄 Тестовая генерация...")
        response = client.generate_text("Решите уравнение: x^2 = 4")
        console.print(f"✅ Ответ получен: {len(response)} символов")
        console.print(f"Превью: {response[:100]}...")
        return True
    except Exception as e:
        console.print(f"[red]❌ Ошибка генерации: {e}[/red]")
        return False

if __name__ == "__main__":
    test_ollama_connection()