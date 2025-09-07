#!/usr/bin/env python3
"""
SGR Mathematical Agent - Schema-Guided Reasoning для решения математических задач
Рефакторинг agent.py под SGR методологию с использованием Ollama Docker

🧠 Принципы SGR для математических задач:
📊 Структурированные схемы рассуждения  
🔍 Итеративная верификация и улучшение
🏗️ Модульная архитектура с dispatch pattern
🚀 Локальная LLM через Ollama Docker
"""

import json
import os
import yaml
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# Импорт наших модулей
from math_sgr_schemas import (
    MathSolutionNextStep, ProblemAnalysis, SolutionStrategy, 
    MathematicalSolution, SolutionVerification, SolutionImprovement,
    TaskCompletion, get_problem_system_prompt, create_math_context
)
from ollama_client import OllamaClient

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================

def load_config():
    """Загрузка конфигурации из файла и переменных окружения"""
    config = {
        'ollama_base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
        'ollama_model': os.getenv('OLLAMA_MODEL', 'gemma3n:e2b'),
        'temperature': float(os.getenv('TEMPERATURE', '0.1')),
        'max_tokens': int(os.getenv('MAX_TOKENS', '4000')),
        'max_execution_steps': int(os.getenv('MAX_EXECUTION_STEPS', '15')),
        'max_verification_attempts': int(os.getenv('MAX_VERIFICATION_ATTEMPTS', '3')),
        'max_improvement_attempts': int(os.getenv('MAX_IMPROVEMENT_ATTEMPTS', '5')),
        'reports_directory': os.getenv('REPORTS_DIRECTORY', 'reports'),
        'memory_directory': os.getenv('MEMORY_DIRECTORY', 'memory'),
    }
    
    # Попытка загрузить из config.yaml
    config_file = Path('config.yaml')
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            if yaml_config and 'sgr_math_agent' in yaml_config:
                sgr_config = yaml_config['sgr_math_agent']
                config.update({k: v for k, v in sgr_config.items() if v is not None})
                
        except Exception as e:
            print(f"Предупреждение: Не удалось загрузить config.yaml: {e}")
    
    return config

CONFIG = load_config()

# =============================================================================
# ГЛОБАЛЬНЫЕ КОМПОНЕНТЫ
# =============================================================================

console = Console()
ollama_client = OllamaClient(
    base_url=CONFIG['ollama_base_url'],
    model=CONFIG['ollama_model']
)

# Контекст выполнения
CONTEXT = create_math_context()

# =============================================================================
# СИСТЕМА ЛОГИРОВАНИЯ И ПАМЯТИ
# =============================================================================

def setup_logging(log_file: Optional[str] = None):
    """Настройка системы логирования"""
    if log_file:
        # Создаем директорию если не существует
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Настраиваем логирование в файл
        import logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )
        console.print(f"[dim]📝 Логирование в файл: {log_file}[/dim]")

def save_memory(memory_file: str, problem_statement: str, 
               current_iteration: int = 0, max_runs: int = 1):
    """Сохранение состояния агента в память"""
    memory = {
        "problem_statement": problem_statement,
        "current_iteration": current_iteration,
        "max_runs": max_runs,
        "context": CONTEXT,
        "config": CONFIG,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        memory_path = Path(memory_file)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(memory_file, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False, default=str)
        
        console.print(f"[dim]💾 Память сохранена: {memory_file}[/dim]")
        return True
    except Exception as e:
        console.print(f"[red]❌ Ошибка сохранения памяти: {e}[/red]")
        return False

def load_memory(memory_file: str) -> Optional[Dict[str, Any]]:
    """Загрузка состояния агента из памяти"""
    try:
        with open(memory_file, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        
        # Восстановление контекста
        if "context" in memory:
            CONTEXT.update(memory["context"])
        
        console.print(f"[dim]📂 Память загружена: {memory_file}[/dim]")
        return memory
    except Exception as e:
        console.print(f"[red]❌ Ошибка загрузки памяти: {e}[/red]")
        return None

# =============================================================================
# DISPATCH СИСТЕМА - ВЫПОЛНЕНИЕ SGR КОМАНД
# =============================================================================


def dispatch(cmd: Any, context: Dict[str, Any]) -> Any:
    """Выполнение SGR команд через dispatch pattern"""
    
    if isinstance(cmd, ProblemAnalysis):
        return handle_problem_analysis(cmd, context)
    elif isinstance(cmd, SolutionStrategy):
        return handle_solution_strategy(cmd, context)
    elif isinstance(cmd, MathematicalSolution):
        return handle_mathematical_solution(cmd, context)
    elif isinstance(cmd, SolutionVerification):
        return handle_solution_verification(cmd, context)
    elif isinstance(cmd, SolutionImprovement):
        return handle_solution_improvement(cmd, context)
    elif isinstance(cmd, TaskCompletion):
        return handle_task_completion(cmd, context)
    else:
        return f"Неизвестная команда: {type(cmd)}"

def handle_problem_analysis(cmd: ProblemAnalysis, context: Dict[str, Any]) -> Dict[str, Any]:
    """Обработка анализа математической задачи"""
    console.print(f"🔍 [bold cyan]АНАЛИЗ ЗАДАЧИ[/bold cyan]")
    console.print(f"📊 Область: [green]{cmd.problem_domain}[/green]")
    console.print(f"🎯 Тип: {cmd.problem_type}")
    console.print(f"💡 Ключевые концепции: {', '.join(cmd.key_concepts)}")
    console.print(f"📈 Сложность: [yellow]{cmd.difficulty_assessment}[/yellow]")
    
    # Сохранение в контекст
    analysis_result = {
        "tool": "analyze_problem",
        "domain": cmd.problem_domain,
        "type": cmd.problem_type,
        "concepts": cmd.key_concepts,
        "difficulty": cmd.difficulty_assessment,
        "approaches": cmd.suggested_approaches,
        "reasoning": cmd.reasoning,
        "timestamp": datetime.now().isoformat()
    }
    
    context["analysis"] = analysis_result
    console.print(f"💭 Обоснование: {cmd.reasoning[:100]}...")
    
    return analysis_result

def handle_solution_strategy(cmd: SolutionStrategy, context: Dict[str, Any]) -> Dict[str, Any]:
    """Обработка выбора стратегии решения"""
    console.print(f"🎯 [bold magenta]ВЫБОР СТРАТЕГИИ[/bold magenta]")
    console.print(f"🔧 Подход: [green]{cmd.chosen_approach}[/green]")
    console.print(f"📋 Шагов планируется: {len(cmd.solution_steps_plan)}")
    
    for i, step in enumerate(cmd.solution_steps_plan, 1):
        console.print(f"   {i}. {step}")
    
    if cmd.required_lemmas:
        console.print(f"📚 Требуемые леммы: {', '.join(cmd.required_lemmas)}")
    
    if cmd.case_analysis_needed:
        console.print(f"🔀 [yellow]Потребуется разбор по случаям[/yellow]")
    
    # Сохранение в контекст
    strategy_result = {
        "tool": "choose_strategy",
        "approach": cmd.chosen_approach,
        "steps_plan": cmd.solution_steps_plan,
        "lemmas": cmd.required_lemmas,
        "case_analysis": cmd.case_analysis_needed,
        "techniques": cmd.expected_techniques,
        "reasoning": cmd.reasoning,
        "timestamp": datetime.now().isoformat()
    }
    
    context["strategy"] = strategy_result
    return strategy_result

def handle_mathematical_solution(cmd: MathematicalSolution, context: Dict[str, Any]) -> Dict[str, Any]:
    """Обработка генерации математического решения"""
    console.print(f"📝 [bold blue]МАТЕМАТИЧЕСКОЕ РЕШЕНИЕ[/bold blue]")
    console.print(f"📋 Резюме: {cmd.solution_summary}")
    console.print(f"📊 Строгость: [green]{cmd.mathematical_rigor}[/green]")
    console.print(f"🎯 Уверенность: [cyan]{cmd.confidence}[/cyan]")
    
    # Отображение ключевых инсайтов
    if cmd.key_insights:
        console.print(f"💡 [bold yellow]Ключевые инсайты:[/bold yellow]")
        for insight in cmd.key_insights:
            console.print(f"   • {insight}")
    
    # Сохранение решения в файл
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(CONFIG['reports_directory'])
    report_dir.mkdir(exist_ok=True)
    
    solution_file = report_dir / f"solution_{timestamp}.md"
    with open(solution_file, 'w', encoding='utf-8') as f:
        f.write(f"# Математическое решение\n\n")
        f.write(f"*Создано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(f"## Резюме\n{cmd.solution_summary}\n\n")
        f.write(f"## Подробное решение\n{cmd.detailed_solution}\n\n")
        f.write(f"## Анализ\n")
        f.write(f"- **Строгость**: {cmd.mathematical_rigor}\n")
        f.write(f"- **Уверенность**: {cmd.confidence}\n")
        if cmd.key_insights:
            f.write(f"- **Ключевые инсайты**: {', '.join(cmd.key_insights)}\n")
    
    console.print(f"💾 Решение сохранено: [dim]{solution_file}[/dim]")
    
    # Сохранение в контекст
    solution_result = {
        "tool": "generate_solution",
        "summary": cmd.solution_summary,
        "detailed_solution": cmd.detailed_solution,
        "insights": cmd.key_insights,
        "rigor": cmd.mathematical_rigor,
        "confidence": cmd.confidence,
        "reasoning": cmd.reasoning,
        "file_path": str(solution_file),
        "timestamp": datetime.now().isoformat()
    }
    
    context["solution"] = solution_result
    context["solution_history"] = context.get("solution_history", [])
    context["solution_history"].append(solution_result)
    
    return solution_result

def handle_solution_verification(cmd: SolutionVerification, context: Dict[str, Any]) -> Dict[str, Any]:
    """Обработка верификации решения"""
    console.print(f"🔍 [bold yellow]ВЕРИФИКАЦИЯ РЕШЕНИЯ[/bold yellow]")
    console.print(f"🔧 Метод: [cyan]{cmd.verification_approach}[/cyan]")
    console.print(f"✅ Результат: [green]{cmd.verification_result}[/green]")
    
    if cmd.identified_issues:
        console.print(f"⚠️ [bold red]Обнаруженные проблемы:[/bold red]")
        for i, issue in enumerate(cmd.identified_issues, 1):
            severity = ""
            if cmd.issue_severity and i <= len(cmd.issue_severity):
                sev = cmd.issue_severity[i-1]
                severity = f" [{sev}]"
            console.print(f"   {i}. {issue}{severity}")
    
    if cmd.suggestions:
        console.print(f"💡 [bold blue]Предложения:[/bold blue]")
        for suggestion in cmd.suggestions:
            console.print(f"   • {suggestion}")
    
    # Обновление счетчика верификаций
    context["verification_count"] = context.get("verification_count", 0) + 1
    
    # Сохранение в контекст
    verification_result = {
        "tool": "verify_solution",
        "approach": cmd.verification_approach,
        "result": cmd.verification_result,
        "issues": cmd.identified_issues,
        "severity": cmd.issue_severity,
        "suggestions": cmd.suggestions,
        "reasoning": cmd.reasoning,
        "attempt_number": context["verification_count"],
        "timestamp": datetime.now().isoformat()
    }
    
    context["verification"] = verification_result
    return verification_result

def handle_solution_improvement(cmd: SolutionImprovement, context: Dict[str, Any]) -> Dict[str, Any]:
    """Обработка улучшения решения"""
    console.print(f"🔧 [bold green]УЛУЧШЕНИЕ РЕШЕНИЯ[/bold green]")
    console.print(f"📝 Стратегия: {cmd.improvement_strategy}")
    
    console.print(f"🎯 [bold red]Проблемы для исправления:[/bold red]")
    for issue in cmd.issues_to_address:
        console.print(f"   • {issue}")
    
    console.print(f"📈 [bold blue]Ожидаемые улучшения:[/bold blue]")
    for improvement in cmd.expected_improvements:
        console.print(f"   • {improvement}")
    
    # Обновление счетчика улучшений
    context["improvement_count"] = context.get("improvement_count", 0) + 1
    
    # Сохранение в контекст
    improvement_result = {
        "tool": "improve_solution",
        "strategy": cmd.improvement_strategy,
        "issues": cmd.issues_to_address,
        "expected_improvements": cmd.expected_improvements,
        "reasoning": cmd.reasoning,
        "attempt_number": context["improvement_count"],
        "timestamp": datetime.now().isoformat()
    }
    
    context["improvements"] = context.get("improvements", [])
    context["improvements"].append(improvement_result)
    
    return improvement_result

def handle_task_completion(cmd: TaskCompletion, context: Dict[str, Any]) -> Dict[str, Any]:
    """Обработка завершения задачи"""
    console.print(f"🎉 [bold green]ЗАДАЧА ЗАВЕРШЕНА[/bold green]")
    console.print(f"📋 Финальный ответ: [yellow]{cmd.final_answer}[/yellow]")
    console.print(f"📊 Качество решения: [cyan]{cmd.solution_quality}[/cyan]")
    
    console.print(f"✅ [bold blue]Выполненные этапы:[/bold blue]")
    for step in cmd.completed_steps:
        console.print(f"   • {step}")
    
    # Статистика сессии
    verification_count = context.get("verification_count", 0)
    improvement_count = context.get("improvement_count", 0)
    
    table = Table(title="Статистика решения")
    table.add_column("Метрика", style="cyan")
    table.add_column("Значение", style="yellow")
    
    table.add_row("Попыток верификации", str(verification_count))
    table.add_row("Попыток улучшения", str(improvement_count))
    table.add_row("Качество решения", cmd.solution_quality)
    
    console.print(table)
    
    # Сохранение в контекст
    completion_result = {
        "tool": "complete_task",
        "final_answer": cmd.final_answer,
        "quality": cmd.solution_quality,
        "completed_steps": cmd.completed_steps,
        "reasoning": cmd.reasoning,
        "statistics": {
            "verification_attempts": verification_count,
            "improvement_attempts": improvement_count,
        },
        "timestamp": datetime.now().isoformat()
    }
    
    context["completion"] = completion_result
    return completion_result

# =============================================================================
# ОСНОВНОЙ ДВИЖОК SGR
# =============================================================================

def execute_sgr_math_task(problem_statement: str) -> str:
    """Выполнение математической задачи с использованием SGR"""
    
    console.print(Panel(problem_statement, title="📐 Математическая задача", title_align="left"))
    
    # Проверка доступности Ollama
    if not ollama_client.is_available():
        console.print("[red]❌ Ollama недоступен. Запустите Docker контейнер.[/red]")
        console.print("[dim]docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama[/dim]")
        console.print("[dim]docker exec -it ollama ollama run gemma2:7b[/dim]")
        return "OLLAMA_UNAVAILABLE"
    
    # Инициализация контекста
    CONTEXT.clear()
    CONTEXT.update(create_math_context())
    CONTEXT["problem_text"] = problem_statement
    CONTEXT["start_time"] = datetime.now()
    
    # Системный промпт для математических задач
    system_prompt = get_problem_system_prompt()
    
    console.print(f"[bold green]🚀 SGR МАТЕМАТИЧЕСКИЙ АГЕНТ ЗАПУЩЕН[/bold green]")
    console.print(f"[dim]🤖 Модель: {CONFIG['ollama_model']}[/dim]")
    console.print(f"[dim]🔗 Ollama URL: {CONFIG['ollama_base_url']}[/dim]")
    console.print(f"[dim]🔥 Температура: {CONFIG['temperature']}[/dim]")
    
    # Основной цикл SGR
    for i in range(CONFIG['max_execution_steps']):
        step_id = f"step_{i+1}"
        console.print(f"\n🧠 [bold]{step_id}[/bold]: Планирование следующего действия...")
        
        # Подготовка контекстного сообщения
        context_msg = f"ТЕКУЩИЙ КОНТЕКСТ ЗАДАЧИ:\n"
        context_msg += f"Задача: {problem_statement}\n"
        context_msg += f"Прогресс: Шаг {i+1}/{CONFIG['max_execution_steps']}\n"
        
        # Добавление информации о предыдущих шагах
        if CONTEXT.get("analysis"):
            context_msg += f"✅ Анализ выполнен: {CONTEXT['analysis']['domain']} / {CONTEXT['analysis']['type']}\n"
        if CONTEXT.get("strategy"):
            context_msg += f"✅ Стратегия выбрана: {CONTEXT['strategy']['approach']}\n"
        if CONTEXT.get("solution"):
            context_msg += f"✅ Решение создано: {CONTEXT['solution']['rigor']} уровень\n"
        if CONTEXT.get("verification"):
            context_msg += f"🔍 Верификация: {CONTEXT['verification']['result']}\n"
        
        context_msg += f"Попытки верификации: {CONTEXT.get('verification_count', 0)}/{CONFIG['max_verification_attempts']}\n"
        context_msg += f"Попытки улучшения: {CONTEXT.get('improvement_count', 0)}/{CONFIG['max_improvement_attempts']}\n"
        
        try:
            # Генерация следующего шага через SGR схему
            next_step = ollama_client.generate_structured(
                schema=MathSolutionNextStep,
                prompt=f"Определите следующий шаг для решения математической задачи:\n\n{context_msg}",
                system_prompt=system_prompt,
                max_retries=3
            )
            
            # Отладочная информация
            console.print(f"[dim]💭 Рассуждение: {next_step.reasoning_chain}[/dim]")
            console.print(f"[dim]📊 Ситуация: {next_step.current_situation[:100]}...[/dim]")
            console.print(f"[dim]🎯 Понимание задачи: {next_step.problem_understanding}[/dim]")
            console.print(f"[dim]📈 Прогресс: {next_step.solution_progress}[/dim]")
            console.print(f"[dim]🔧 Инструмент: {next_step.function.tool}[/dim]")
            
        except Exception as e:
            console.print(f"[red]❌ Ошибка генерации SGR шага: {e}[/red]")
            break
        
        # Проверка завершения задачи
        if next_step.task_completed or isinstance(next_step.function, TaskCompletion):
            console.print(f"[bold green]✅ Задача помечена как завершенная[/bold green]")
            dispatch(next_step.function, CONTEXT)
            break
        
        # Проверка лимитов циклов
        if (next_step.verification_attempts >= CONFIG['max_verification_attempts'] or
            next_step.improvement_attempts >= CONFIG['max_improvement_attempts']):
            console.print(f"[yellow]⚠️ Достигнуты лимиты попыток, принудительное завершение[/yellow]")
            # Создаем принудительное завершение
            forced_completion = TaskCompletion(
                tool="complete_task",
                reasoning="Достигнуты максимальные лимиты попыток улучшения",
                final_answer="Решение завершено по лимитам итераций",
                solution_quality="satisfactory",
                completed_steps=["Анализ", "Решение", "Попытки улучшения"]
            )
            dispatch(forced_completion, CONTEXT)
            break
        
        # Отображение текущего шага
        current_step = next_step.remaining_steps[0] if next_step.remaining_steps else "Завершение"
        console.print(f"[blue]🎯 Текущий шаг: {current_step}[/blue]")
        
        # Выполнение команды
        result = dispatch(next_step.function, CONTEXT)
        
        console.print(f"[dim]✅ Результат: {str(result)[:100]}...[/dim]")
    
    return "COMPLETED"

# =============================================================================
# ОСНОВНОЙ ИНТЕРФЕЙС
# =============================================================================

def main():
    """Главная функция приложения"""
    parser = argparse.ArgumentParser(description='SGR Mathematical Agent - Решение математических задач')
    parser.add_argument('problem_file', nargs='?', help='Путь к файлу с математической задачей')
    parser.add_argument('--log', '-l', type=str, help='Путь к файлу логов')
    parser.add_argument('--memory', '-mem', type=str, help='Путь к файлу памяти для сохранения состояния')
    parser.add_argument('--resume', '-r', action='store_true', help='Восстановить из файла памяти')
    parser.add_argument('--test-ollama', action='store_true', help='Тестировать подключение к Ollama')
    parser.add_argument('--interactive', '-i', action='store_true', help='Интерактивный режим')
    
    args = parser.parse_args()
    
    # Настройка логирования
    if args.log:
        setup_logging(args.log)
    
    # Тестирование Ollama
    if args.test_ollama:
        from ollama_client import test_ollama_connection
        test_ollama_connection()
        return
    
    console.print(Panel("🧠 SGR Mathematical Agent", subtitle="Schema-Guided Reasoning для решения математических задач"))
    
    # Восстановление из памяти
    if args.resume and args.memory:
        memory = load_memory(args.memory)
        if memory:
            console.print(f"[green]📂 Состояние восстановлено из {args.memory}[/green]")
    
    # Интерактивный режим
    if args.interactive:
        console.print("\n[bold cyan]🎯 Интерактивный режим[/bold cyan]")
        console.print("Введите математическую задачу или 'quit' для выхода")
        
        while True:
            try:
                problem = input("\n📐 Задача: ").strip()
                if problem.lower() in ['quit', 'exit', 'q']:
                    console.print("👋 До свидания!")
                    break
                
                if not problem:
                    console.print("❌ Пустая задача, попробуйте еще раз")
                    continue
                
                result = execute_sgr_math_task(problem)
                
                # Сохранение памяти после каждой задачи
                if args.memory:
                    save_memory(args.memory, problem)
                    
            except KeyboardInterrupt:
                console.print("\n👋 Прервано пользователем")
                break
            except Exception as e:
                console.print(f"[red]❌ Ошибка: {e}[/red]")
                continue
    
    # Режим с файлом
    elif args.problem_file:
        try:
            problem_path = Path(args.problem_file)
            if not problem_path.exists():
                console.print(f"[red]❌ Файл не найден: {args.problem_file}[/red]")
                return
            
            with open(problem_path, 'r', encoding='utf-8') as f:
                problem_statement = f.read().strip()
            
            console.print(f"[dim]📂 Загружена задача из: {args.problem_file}[/dim]")
            
            result = execute_sgr_math_task(problem_statement)
            
            # Сохранение памяти
            if args.memory:
                save_memory(args.memory, problem_statement)
                
        except Exception as e:
            console.print(f"[red]❌ Ошибка чтения файла: {e}[/red]")
    
    else:
        console.print("[yellow]💡 Используйте --interactive для интерактивного режима или укажите файл с задачей[/yellow]")
        console.print("\nПример:")
        console.print("  python sgr_math_agent.py --interactive")
        console.print("  python sgr_math_agent.py problems/problem1.txt")
        console.print("  python sgr_math_agent.py --test-ollama")

if __name__ == "__main__":
    main()