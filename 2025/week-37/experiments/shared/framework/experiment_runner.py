"""
Shared Framework: Universal Experiment Runner

Универсальный CLI framework для запуска экспериментов с LLM метриками.
Устраняет дублирование CLI логики между экспериментами.
"""

# Стандартная библиотека
import argparse
import sys
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Protocol, Type, runtime_checkable

# Красивый вывод (при наличии rich)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.traceback import install as rich_traceback_install
    from rich.logging import RichHandler
    from rich import box
    RICH_AVAILABLE = True
except Exception:  # pragma: no cover
    RICH_AVAILABLE = False

# Сторонние библиотеки
import yaml

# Общая инфраструктура 
from shared.infrastructure.model_manager import GemmaModelManager
from shared.infrastructure.base_analyzer import BaseAnalyzer, MetricCalculator

# Фильтрация нерелевантных предупреждений и настройка логов/tracebacks
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

if RICH_AVAILABLE:
    rich_traceback_install(show_locals=False)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    console = Console()
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(asctime)s | %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console = None

logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


@runtime_checkable
class ExperimentAlgorithm(Protocol):
    """Протокол для алгоритмов экспериментов"""
    
    def get_calculator(self) -> MetricCalculator:
        """Возвращает калькулятор метрики"""
        ...
    
    def get_metric_name(self) -> str:
        """Возвращает название метрики"""
        ...
    
    def get_default_config(self) -> Dict[str, Any]:
        """Возвращает конфигурацию по умолчанию"""
        ...


class ExperimentRunner:
    """
    Универсальный runner для экспериментов.
    
    Координирует:
    - Загрузку конфигурации и модели
    - Создание анализатора с нужным калькулятором  
    - Выполнение анализа текста или генерации
    - Вывод результатов
    """

    def __init__(self, algorithm: ExperimentAlgorithm):
        self.algorithm = algorithm
        self.config = None
        self.model_manager = None
        self.analyzer = None

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Загружает конфигурацию из YAML файла"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info("Конфигурация загружена из %s", config_path)
            return config
        except FileNotFoundError:
            logger.warning("Файл конфигурации не найден: %s. Использую defaults.", config_path)
            return self.algorithm.get_default_config()
        except yaml.YAMLError as e:
            logger.error("Ошибка парсинга YAML: %s", e)
            sys.exit(1)

    def setup_model(self, config: Dict[str, Any]) -> None:
        """Настраивает и загружает модель"""
        self.model_manager = GemmaModelManager(config)
        if RICH_AVAILABLE and console is not None:
            with console.status("[bold cyan]Загрузка модели...[/]", spinner="dots"):
                self.model_manager.load_model()
        else:
            logger.info("Загрузка модели...")
            self.model_manager.load_model()
        logger.info("Модель загружена: %s", self.model_manager.get_model_info())

    def setup_analyzer(self) -> None:
        """Создаёт анализатор с алгоритмом эксперимента"""
        calculator = self.algorithm.get_calculator()
        metric_name = self.algorithm.get_metric_name()
        self.analyzer = BaseAnalyzer(
            model_port=self.model_manager,
            calculator=calculator,
            metric_name=metric_name
        )
        logger.info("Анализатор создан для метрики: %s", metric_name)

    def run_text_analysis(self, text: str) -> Dict[str, Any]:
        """Запускает анализ текста"""
        logger.info("Анализ текста: '%s'", text[:50] + "..." if len(text) > 50 else text)
        results = self.analyzer.analyze_text(text)
        
        # Красивый вывод результатов
        title = "Результаты анализа (вход)"
        metric_name = self.algorithm.get_metric_name().capitalize()
        if RICH_AVAILABLE and console is not None:
            console.rule(f"[bold green]{title}")
            tokens = results['tokens']
            metric = results[self.algorithm.get_metric_name()]
            table = Table(box=box.SIMPLE, show_lines=False)
            table.add_column("Поле", style="cyan", no_wrap=True)
            table.add_column("Значение", style="white")
            table.add_row("Текст", text)
            table.add_row("Токены", str(tokens))
            table.add_row(metric_name, str(metric))
            console.print(table)
        else:
            print(f"\n=== {title} ===")
            print(f"Текст: {text}")
            print(f"Токены: {results['tokens']}")
            print(f"{metric_name}: {results[self.algorithm.get_metric_name()]}")
        
        return results

    def run_generation_analysis(self, prompt: str, max_tokens: int = 10) -> Dict[str, Any]:
        """Запускает генерацию с анализом"""
        logger.info("Генерация с анализом для промпта: '%s'", prompt)
        results = self.analyzer.generate_with_analysis(prompt, max_new_tokens=max_tokens)
        
        # Красивый вывод результатов генерации (метрики на сгенерированных токенах)
        title = "Результаты генерации (метрики на выходе)"
        metric_key = f"{self.algorithm.get_metric_name()}s"
        steps = results['generation_steps']
        tokens_txt = results.get('generated_text_parts', [])
        probs = results.get('probabilities', [])
        metrics = results.get(metric_key, [])

        if RICH_AVAILABLE and console is not None:
            console.rule(f"[bold green]{title}")
            header = Table(box=box.SIMPLE)
            header.add_column("Поле", style="cyan", no_wrap=True)
            header.add_column("Значение", style="white")
            header.add_row("Промпт", prompt)
            header.add_row("Сгенерировано", results['full_generated_text'])
            header.add_row("Шагов генерации", str(steps))
            console.print(header)

            if steps:
                table = Table(box=box.SIMPLE_HEAVY)
                table.add_column("Шаг", justify="right", style="magenta")
                table.add_column("Токен", style="white")
                table.add_column("Prob", justify="right", style="yellow")
                table.add_column(self.algorithm.get_metric_name().capitalize(), justify="right", style="green")
                for i in range(steps):
                    tok = tokens_txt[i] if i < len(tokens_txt) else ""
                    pr = probs[i] if i < len(probs) else float('nan')
                    mv = metrics[i] if i < len(metrics) else float('nan')
                    table.add_row(f"{i+1}", str(tok), f"{pr:.4f}", f"{mv:.4f}")
                console.print(table)
            else:
                console.print(Panel.fit("Нет сгенерированных шагов (возможен ранний EOS)", style="red"))
        else:
            print(f"\n=== {title} ===")
            print(f"Промпт: {prompt}")
            print(f"Сгенерировано: {results['full_generated_text']}")
            print(f"Шагов генерации: {steps}")
            if steps:
                print("\nШаг | Токен | Prob | Метрика")
                print("----+-------+------+--------")
                for i in range(steps):
                    tok = tokens_txt[i] if i < len(tokens_txt) else ""
                    pr = probs[i] if i < len(probs) else float('nan')
                    mv = metrics[i] if i < len(metrics) else float('nan')
                    print(f"{i+1:>3} | {tok} | {pr:.4f} | {mv:.4f}")
            else:
                print("Нет сгенерированных шагов (возможно, ранний EOS).")
        
        return results

    def run(self, args: argparse.Namespace) -> None:
        """Главная функция запуска эксперимента"""
        # Загружаем конфигурацию
        self.config = self.load_config(args.config)
        
        # Настраиваем модель и анализатор
        self.setup_model(self.config)
        self.setup_analyzer()
        
        # Выполняем анализ
        if args.mode == 'analyze':
            if not args.text:
                logger.error("Для режима 'analyze' требуется параметр --text")
                sys.exit(1)
            # По умолчанию считаем метрики на сгенерированных токенах
            if getattr(args, 'on', 'generated') == 'input':
                self.run_text_analysis(args.text)
            else:
                self.run_generation_analysis(args.text, max_tokens=args.max_tokens)
            
        elif args.mode == 'generate':
            if not args.prompt:
                logger.error("Для режима 'generate' требуется параметр --prompt")
                sys.exit(1)
            self.run_generation_analysis(args.prompt, max_tokens=args.max_tokens)
            
        else:
            logger.error("Неизвестный режим: %s", args.mode)
            sys.exit(1)

    @staticmethod
    def create_cli_parser(experiment_name: str) -> argparse.ArgumentParser:
        """Создаёт универсальный CLI parser"""
        parser = argparse.ArgumentParser(
            description=f'Эксперимент {experiment_name} с анализом LLM метрик'
        )
        
        parser.add_argument(
            '--config', '-c',
            type=str,
            default='shared/config/experiment.yaml',
            help='Путь к файлу конфигурации (default: config/experiment.yaml)'
        )
        
        parser.add_argument(
            '--mode', '-m',
            choices=['analyze', 'generate'],
            required=True,
            help='Режим работы: analyze (анализ текста) или generate (генерация с анализом)'
        )
        
        parser.add_argument(
            '--text', '-t',
            type=str,
            help='Текст для анализа (режим analyze)'
        )
        
        parser.add_argument(
            '--prompt', '-p',
            type=str,
            help='Промпт для генерации (режим generate)'
        )
        
        parser.add_argument(
            '--max-tokens',
            type=int,
            default=10,
            help='Максимум токенов для генерации (default: 10)'
        )

        parser.add_argument(
            '--on',
            choices=['generated', 'input'],
            default='generated',
            help='Где считать метрики в analyze: на сгенерированных токенах (default) или на входе'
        )
        
        return parser


def run_experiment(algorithm: ExperimentAlgorithm, experiment_name: str) -> None:
    """
    Универсальная функция запуска эксперимента.
    
    Использование:
        if __name__ == "__main__":
            from my_algorithm import MyAlgorithm
            run_experiment(MyAlgorithm(), "My Experiment")
    """
    parser = ExperimentRunner.create_cli_parser(experiment_name)
    args = parser.parse_args()
    
    runner = ExperimentRunner(algorithm)
    runner.run(args)
