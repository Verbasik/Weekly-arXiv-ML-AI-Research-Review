# Эксперимент 001: Анализ энтропии токенов модели Gemma-3n-E2B-it

## Цель эксперимента
Исследование неопределенности модели Google Gemma-3n-E2B-it через анализ энтропии токенов по формуле:

$H_i = -\sum_{j} P_i(j) \log P_i(j)$

где:
- **$H_i$** — энтропия распределения вероятностей токенов на позиции $i$
- **$P_i(j)$** — вероятность $j$-го токена из словаря в позиции $i$

## Описание модели
- **Модель**: Google Gemma-3n-E2B-it
- **Размер**: 6B параметров (эффективно работает как 2B модель)
- **Возможности**: Мультимодальная (текст, изображения, аудио, видео)
- **Контекст**: До 32K токенов

## Гипотеза
Энтропия токенов будет варьироваться в зависимости от:
1. Сложности контекста (простые vs сложные предложения)
2. Предсказуемости продолжения (устойчивые фразы vs творческие тексты)
3. Позиции в последовательности (начало vs середина vs конец)

## Структура проекта (DDD)
```
experiment_001_gemma_entropy/
├── config/                         # Конфигурация (experiment.yaml)
├── src/
│   ├── domain/                     # Доменный слой (чистая логика)
│   │   ├── __init__.py
│   │   ├── entropy_calculator.py   # EntropyCalculator (формула и расчёты)
│   │   └── ports.py                # ModelPort (контракт доступа к модели)
│   ├── infrastructure/             # Инфраструктурный слой (адаптеры)
│   │   ├── __init__.py
│   │   └── gemma_model_manager.py  # GemmaModelManager (реализация ModelPort)
│   ├── application/                # Прикладной слой (use-cases)
│   │   ├── __init__.py
│   │   └── entropy_analyzer.py     # GemmaEntropyAnalyzer (оркестрация)
│   └── entropy_experiment.py       # CLI-раннер эксперимента
└── README.md                       # Этот файл
```

## Методология
1. Загрузка предобученной модели Gemma-3n-E2B-it
2. Подготовка набора тестовых промптов различной сложности
3. Получение распределений вероятностей для каждого токена
4. Вычисление энтропии по формуле
5. Анализ закономерностей и интерпретация результатов

## Ожидаемые результаты
- Количественная оценка уверенности модели для разных типов текста
- Выявление паттернов в распределении энтропии
- Понимание поведения модели на различных типах задач

## Технические требования
- Python 3.8+
- PyTorch
- HuggingFace Transformers
- CUDA-совместимая GPU (рекомендуется 12GB+ VRAM)
- NumPy, matplotlib для анализа и визуализации

По умолчанию код принудительно использует CPU для стабильности. Для включения GPU/MPS понадобятся правки в инфраструктурном адаптере (`infrastructure/gemma_model_manager.py`).

## Установка и запуск
1) Подготовка окружения
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

2) Запуск эксперимента (красивый вывод с генерацией и энтропией)
```
python src/entropy_experiment.py
```

## Конфигурация
Файл: `config/experiment.yaml`

Минимально необходимые ключи:
```
model:
  name: google/gemma-3-2b-it
  context_length: 512
```

- `model.name`: имя модели в Hugging Face Hub
- `model.context_length`: максимальная длина контекста при токенизации

## Использование как библиотеки (API)
Пример анализа готового текста:
```
from infrastructure.gemma_model_manager import GemmaModelManager
from application.entropy_analyzer import GemmaEntropyAnalyzer

manager = GemmaModelManager("config/experiment.yaml")
manager.load_model()

analyzer = GemmaEntropyAnalyzer(manager)
res = analyzer.analyze_text_entropy("Привет, мир!")
print(res["entropy"])  # тензор энтропий по позициям
```

Пример генерации с поэтапным анализом энтропии:
```
results = analyzer.generate_with_entropy_analysis("Теорема Пифагора гласит, что", max_new_tokens=8)
print(results["full_generated_text"], results["entropies"])
```