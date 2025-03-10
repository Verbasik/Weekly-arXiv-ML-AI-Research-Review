# Законы масштабирования дистилляции 

**Рекомендация для читателей:**  
Прежде чем погрузиться в детали, советую ознакомиться с двумя отличными статьями инженера из Яндекса ([статья 1](https://habr.com/ru/companies/yandex/articles/801119/), [статья 2](https://habr.com/ru/companies/yandex/articles/878230/)). В них отличны объясняются принципы дистилляции, её применение в промышленных задачах и ключевые практические аспекты. Это идеальный старт для тех, кто только начинает знакомиться с темой.  

**Однако**, если вы, как и я, стремитесь к *глубокому пониманию* — этого может оказаться недостаточно. В данном обзоре мы пойдём дальше:  
1. **Математическая формализация**: Разберём более глубако уравнения, лежащие в основе дистилляции, включая функцию потерь с температурным параметром, оптимизацию распределений и законы масштабирования из работы Apple.  
2. **Примеры кода**: Покажем, как реализовать дистилляцию на практике — от простых моделей на PyTorch до тонкой настройки гиперпараметров.  
3. **Нюансы исследований**: Ответим на вопросы, оставшиеся за рамками вводных материалов. Например, почему «слишком умный учитель» вредит ученику и как математически обосновать оптимальное соотношение их размеров.  

**Для кого это?**  
Если вы хотите не просто использовать дистилляцию «из коробки», а *понимать, как и почему она работает* — этот разбор для вас. Мы заглянем «под капот» методов, чтобы вы могли осознанно применять их в своих проектах.  

## Knowledge Distillation:

**Knowledge Distillation (Дистилляция знаний)** — это метод обучения моделей-студентов (обычно меньшего размера и менее сложных) путем передачи "знаний" от предварительно обученной модели-учителя (обычно большей и более сложной).  Основная идея заключается в том, что модель-учитель, обладающая большей емкостью и обученная на большом объеме данных, может передать не только свои "жесткие" предсказания (например, класс объекта), но и более богатую информацию о распределении вероятностей классов, которую модель-студент может использовать для более эффективного обучения.

### **Teacher и Student модели:**

В парадигме Knowledge Distillation участвуют две основные модели:

*   **Teacher (Учитель):**  Это большая, предварительно обученная модель, которая считается "экспертом" в решении определенной задачи. Учитель уже достиг высокой точности и обладает "знаниями", которые мы хотим передать студенту. Математически учитель представляется как функция $p(y|x)$, которая для входных данных $x$ выдает распределение вероятностей $p$ по классам $y$.

*   **Student (Студент):** Это меньшая, более простая модель, которую мы хотим обучить. Цель студента — научиться имитировать поведение учителя, чтобы достичь сравнимой производительности, но при этом быть более эффективной с точки зрения вычислительных ресурсов, памяти или времени инференса. Студент представляется как функция $q_{\theta}(y|x)$, где $\theta$ — параметры модели, которые мы оптимизируем в процессе обучения.

**Функция потерь (Loss Function) в Knowledge Distillation:**

Общая цель Knowledge Distillation — минимизировать разницу между предсказаниями учителя и студента. Это формализуется через функцию потерь $L$, которая зависит от предсказаний учителя $p(y|x)$ и студента $q_{\theta}(y|x)$.  Процесс обучения заключается в поиске оптимальных параметров $\theta$ для студента, которые минимизируют эту функцию потерь:

$$L(p(y|x), q_{\theta}(y|x)) \rightarrow \min_{\theta}$$

Это общее выражение, и конкретный вид функции потерь и способ дистилляции определяют различные подходы. Рассмотрим два основных подхода: hard-label и soft-label дистилляцию.

## I. Hard-label Distillation: Дистилляция с использованием "жестких" меток

**Концепция:**

Hard-label distillation — это наиболее простой и интуитивно понятный подход. В этом методе учитель используется для генерации "жестких" меток (hard labels) для обучающей выборки.  "Жесткая" метка — это просто класс с наивысшей вероятностью, предсказанный учителем для каждого входного примера.  Затем студент обучается на этих сгенерированных метках, как если бы это были истинные метки из размеченного датасета.  По сути, мы используем учителя для создания синтетического датасета, на котором обучаем студента стандартным образом.

**Hard-label Distillation для GPT моделей: объяснение на пальцах**

Представьте, что у нас есть две модели:

*   **Учитель (Teacher):** Большая, мощная GPT модель, например, GPT-3 или что-то подобное. Она обладает огромным количеством знаний о языке и мире, и способна генерировать очень качественный и связный текст.
*   **Студент (Student):** Маленькая, более компактная GPT модель, например, уменьшенная версия GPT или Transformer меньшего размера. Она менее ресурсоемкая, но изначально уступает учителю в качестве генерации текста.

Наша цель - "научить" маленькую модель-студента генерировать текст так же хорошо, как и большая модель-учитель, используя метод Hard-label Distillation.

**Шаги Hard-label Distillation в этом контексте:**

1.  **Генерация "жестких" меток учителем (Большой GPT):**

    *   Мы берем большой набор текстовых данных (например, обучающую выборку, на которой изначально обучался учитель, или просто большой корпус текстов).
    *   Для каждого фрагмента текста (или запроса) из этого набора, мы просим большую модель-учителя сгенерировать текст.  В контексте GPT, это означает, что мы подаем учителю входной текст (например, начало предложения или запрос) и просим его сгенерировать продолжение.
    *   Учитель генерирует последовательность токенов, которые он считает наиболее вероятными для продолжения данного текста.  Эти сгенерированные последовательности токенов и являются нашими "жесткими" метками.

    **Пример:**

    *   **Входной текст (запрос):** "Столица Франции - это"
    *   **Учитель (Большая GPT) генерирует:** "Париж." (токены: "Па", "ри", "ж", ".")
    *   **"Жесткая" метка:** Последовательность токенов: ("Па", "ри", "ж", ".")

    Мы повторяем этот процесс для большого количества различных входных текстов, получая набор пар: (исходный входной текст, "жесткая" метка - последовательность токенов, сгенерированная учителем).

2.  **Обучение студента (Маленький GPT) на "жестких" метках:**

    *   Теперь у нас есть синтетический датасет, состоящий из пар (исходный входной текст, "жесткая" метка).  Мы будем использовать этот датасет для обучения маленькой модели-студента.
    *   Мы обучаем студента предсказывать "жесткие" метки, сгенерированные учителем, используя стандартную задачу языкового моделирования.  Это означает, что для каждого входного текста мы хотим, чтобы студент генерировал последовательность токенов, максимально похожую на "жесткую" метку, сгенерированную учителем.
    *   В процессе обучения мы используем функцию потерь кросс-энтропии.  Мы сравниваем распределение вероятностей токенов, предсказанное студентом, с "жесткой" меткой (которая по сути является распределением, где вероятность "правильного" токена равна 1, а всех остальных - 0).  Мы стремимся минимизировать эту кросс-энтропию, заставляя студента "подражать" учителю в предсказании токенов.

    В нашем примере, если студент на вход "Столица Франции - это" предсказывает, например, "Лондон", то функция потерь будет высокой, так как "жесткая" метка учителя была "Париж".  В процессе обучения студент будет корректировать свои параметры, чтобы в будущем для аналогичных запросов предсказывать "Париж" или что-то очень похожее на предсказание учителя.

**Почему маленькая модель может предсказывать те же токены, что и большая?**

*   **Передача знаний через "жесткие" метки:**  Хотя Hard-label Distillation и теряет часть информации из распределения вероятностей учителя, она все равно эффективно передает **ключевые знания** о том, какие токены являются наиболее вероятными в определенных контекстах.  Большая модель, будучи хорошо обученной, "знает", какие продолжения текста являются грамматически правильными, семантически уместными и стилистически подходящими.  Генерируя "жесткие" метки, она как бы "подсказывает" маленькой модели, какие именно токены нужно предсказывать.
*   **Фокус на наиболее важной информации:**  "Жесткие" метки концентрируются на наиболее вероятных токенах.  В языковом моделировании часто бывает так, что для многих контекстов есть один или несколько доминирующих "правильных" продолжений.  Hard-label Distillation помогает маленькой модели быстро освоить эти наиболее важные закономерности, игнорируя менее значимые детали, которые могут быть избыточными для достижения хорошего качества генерации.
*   **Упрощение задачи обучения:**  Обучение на "жестких" метках превращает дистилляцию в стандартную задачу обучения с учителем.  Это упрощает процесс обучения и позволяет использовать хорошо известные методы и оптимизаторы.  Маленькой модели не нужно пытаться воспроизвести все тонкости распределения вероятностей учителя, ей достаточно научиться предсказывать наиболее вероятные токены, что является более простой задачей.

**Важно отметить ограничения Hard-label Distillation:**

*   **Потеря "мягкой" информации:** Как и указано в тексте, Hard-label Distillation теряет информацию о вероятностях других классов и "мягких" отношениях между классами.  В контексте языковых моделей это означает, что студент может не улавливать все нюансы стиля, семантики и разнообразия, которые присутствуют в распределении вероятностей учителя.  Например, учитель может знать, что "Париж" является самым вероятным ответом на "Столица Франции - это", но также понимать, что "Рим" или "Берлин" являются менее вероятными, но все же допустимыми ответами в определенных контекстах.  Hard-label Distillation фокусируется только на "Париже", игнорируя эту "мягкую" информацию.
*   **Потенциальное ухудшение разнообразия:**  Из-за фокусировки на "жестких" метках, студент может стать менее разнообразным в своих генерациях, чем учитель.  Он может слишком точно копировать наиболее вероятные ответы учителя, упуская возможность генерировать альтернативные, но все еще качественные варианты.

**Математическая формализация:**

1.  **Генерация "жестких" меток учителем:**  Для каждого примера $x^{(n)}$ из обучающей выборки, учитель $p(y|x)$ предсказывает распределение вероятностей классов. "Жесткая" метка $y^{(n)}$ выбирается как класс с максимальной вероятностью, предсказанной учителем.  В контексте языков моделей, где $y$ представляет собой последовательность токенов, учитель генерирует последовательность "жестких" меток $y^{(1)}, \ldots y^{(N)}$ для $N$ примеров.  Здесь $y^{(n)} = (y_1^{(n)}, \ldots, y_{T_n}^{(n)})$ представляет собой последовательность токенов длиной $T_n$.

    $$y^{(1)}, \ldots y^{(N)} \sim p(y|x)$$
    В более простом варианте, для классификации,  $y^{(n)} = \arg\max_{y} p(y|x^{(n)})$. В случае последовательностей, учитель может генерировать целые последовательности наиболее вероятных токенов.

2.  **Обучение студента на "жестких" метках:** Студент $q_{\theta}(y|x)$ обучается максимизировать логарифмическую вероятность "жестких" меток, сгенерированных учителем. Это стандартная задача обучения с учителем, где целевыми метками являются $y^{(1)}, \ldots y^{(N)}$.  Функция потерь, которую мы минимизируем (или эквивалентно, максимизируем отрицательную потерю), представляет собой ожидание логарифмической вероятности "жестких" меток под распределением $p(y|x)$ учителя.

    $$\mathbb{E}_{p(y|x)} [\log q_{\theta}(y|x)] \rightarrow \max_{\theta}$$

    В практической реализации, это ожидание аппроксимируется эмпирическим средним по обучающей выборке. Для последовательностей текста, функция потерь выглядит следующим образом:

    $$\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \log q_{\theta}(y_t^{(n)}|y_{<t}^{(n)})$$

    Здесь:
    *   $N$ — количество примеров в обучающей выборке.
    *   $T_n$ — длина последовательности для $n$-го примера.
    *   $y_t^{(n)}$ — $t$-й токен в последовательности "жестких" меток для $n$-го примера, сгенерированных учителем.
    *   $y_{<t}^{(n)} = (y_1^{(n)}, \ldots, y_{t-1}^{(n)})$ — префикс последовательности до $t$-го токена.
    *   $q_{\theta}(y_t^{(n)}|y_{<t}^{(n)})$ — вероятность предсказания студентом $t$-го токена $y_t^{(n)}$ при условии предыдущих токенов $y_{<t}^{(n)}$, параметризованная $\theta$.

    Эта функция потерь представляет собой **кросс-энтропию** между распределением "жестких" меток, сгенерированных учителем, и предсказаниями студента.  Мы стремимся максимизировать эту величину, что эквивалентно минимизации отрицательной логарифмической правдоподобности или кросс-энтропии.

**Преимущества и недостатки Hard-label Distillation:**

*   **Преимущества:** Простота реализации и понимания. Можно использовать стандартные методы обучения с учителем.
*   **Недостатки:**  Потеря информации, содержащейся в распределении вероятностей учителя. "Жесткие" метки содержат только информацию о наиболее вероятном классе, игнорируя вероятности других классов и "мягкие" отношения между классами, которые учитель "знает".  Это может ограничить эффективность передачи знаний.

## **Реализация Hard-label Distillation на основе Open R1**

Ниже представлена реализация Hard-label Distillation с использованием подхода, применяемого в проекте Open R1. Процесс разделен на два этапа: генерация данных учителем и обучение ученика.

```
@misc{openr1,
    title = {Open R1: A fully open reproduction of DeepSeek-R1},
    url = {https://github.com/huggingface/open-r1},
    author = {Hugging Face},
    month = {January},
    year = {2025}
}
```

### **Этап 1: Генерация "жестких" меток большой моделью (учителем)**

```python
import argparse
from datasets import load_dataset
from typing import Optional, Dict, Any

from distilabel.pipeline import Pipeline
from distilabel.models import vLLM
from distilabel.steps.tasks import TextGeneration

def build_hard_label_pipeline(
    teacher_model: str,
    base_url: str = "http://localhost:8000/v1",
    prompt_column: Optional[str] = None,
    prompt_template: str = "{{ instruction }}",
    temperature: float = 0.0,
    max_new_tokens: int = 4096,
    input_batch_size: int = 32,
) -> Pipeline:
    """
    Description:
    ---------------
        Создает конвейер для генерации "жестких" меток с использованием модели-учителя.

    Args:
    ---------------
        teacher_model: Идентификатор модели-учителя
        base_url: URL сервера vLLM
        prompt_column: Имя колонки в датасете, содержащей входные тексты
        prompt_template: Шаблон для форматирования промптов
        temperature: Температура для генерации (0.0 для "жестких" меток)
        max_new_tokens: Максимальное количество генерируемых токенов
        input_batch_size: Размер батча для входных данных

    Returns:
    ---------------
        Настроенный конвейер Distilabel

    Raises:
    ---------------
        Exception: В случае ошибки настройки конвейера

    Examples:
    ---------------
        >>> pipeline = build_hard_label_pipeline("deepseek-ai/DeepSeek-R1")
        >>> pipeline.run(dataset)
    """
    # Настраиваем параметры генерации с temperature=0 для получения детерминированных ответов
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "do_sample": False,          # Отключаем семплирование для получения "жестких" меток
    }

    with Pipeline(
        name="hard-label-distillation",
        description="Конвейер для генерации 'жестких' меток с использованием модели-учителя",
    ) as pipeline:
        # Настраиваем модель-учителя через vLLM
        teacher = vLLM(
            model=teacher_model,
            tokenizer=teacher_model,
            extra_kwargs={
                "tensor_parallel_size": 1,               # Можно увеличить для больших моделей
                "max_model_len": max_new_tokens + 2048,  # Добавляем запас для контекста
            },
            generation_kwargs=generation_kwargs,
        )

        # Настраиваем шаг генерации текста
        text_generation = TextGeneration(
            llm=teacher,
            template=prompt_template,
            num_generations=1,           # Для "жестких" меток нам нужна только одна генерация
            input_mappings={"instruction": prompt_column} if prompt_column is not None else {},
            input_batch_size=input_batch_size,
        )

    return pipeline

def generate_hard_labels(
    dataset_name: str,
    dataset_split: str = "train",
    teacher_model: str = "deepseek-ai/DeepSeek-R1",
    output_dataset: str = "my-username/hard-label-distill-dataset",
    prompt_column: str = "problem",
    prompt_template: str = "You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}: {{ instruction }}",
    max_examples: Optional[int] = None,
    private: bool = False,
) -> Any:
    """
    Description:
    ---------------
        Генерирует "жесткие" метки с использованием модели-учителя и сохраняет результаты как набор данных на HuggingFace Hub.

    Args:
    ---------------
        dataset_name: Имя исходного датасета
        dataset_split: Имя сплита датасета
        teacher_model: Модель-учитель для генерации "жестких" меток
        output_dataset: Имя выходного датасета на HuggingFace Hub
        prompt_column: Имя колонки, содержащей входные данные
        prompt_template: Шаблон для форматирования промптов
        max_examples: Максимальное количество примеров для обработки
        private: Приватный ли выходной датасет

    Returns:
    ---------------
        Датасет с "жесткими" метками

    Raises:
    ---------------
        Exception: В случае ошибки генерации меток

    Examples:
    ---------------
        >>> hard_label_dataset = generate_hard_labels("my-dataset", "train")
        >>> hard_label_dataset.push_to_hub("my-username/hard-label-dataset")
    """
    # Загружаем исходный датасет
    print(f"Загрузка датасета '{dataset_name}' (сплит: {dataset_split})...")
    dataset = load_dataset(dataset_name, split=dataset_split)

    # Ограничиваем количество примеров, если указано
    if max_examples is not None and max_examples < len(dataset):
        dataset = dataset.select(range(max_examples))

    print(f"Создание конвейера для генерации 'жестких' меток с использованием {teacher_model}...")
    pipeline = build_hard_label_pipeline(
        teacher_model=teacher_model,
        prompt_column=prompt_column,
        prompt_template=prompt_template,
    )

    print(f"Запуск конвейера для генерации 'жестких' меток на {len(dataset)} примерах...")
    # Генерируем "жесткие" метки
    hard_label_dataset = pipeline.run(dataset=dataset)

    # Сохраняем результаты на HuggingFace Hub
    if output_dataset:
        print(f"Сохранение результатов в '{output_dataset}'...")
        hard_label_dataset.push_to_hub(output_dataset, private=private)
        print(f"Датасет с 'жесткими' метками успешно сохранен в '{output_dataset}'.")

    return hard_label_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация 'жестких' меток с использованием модели-учителя")
    parser.add_argument("--dataset", type=str, required=True, help="Имя исходного датасета")
    parser.add_argument("--split", type=str, default="train", help="Сплит датасета")
    parser.add_argument("--teacher-model", type=str, default="deepseek-ai/DeepSeek-R1", help="Модель-учитель")
    parser.add_argument("--output-dataset", type=str, required=True, help="Имя выходного датасета")
    parser.add_argument("--prompt-column", type=str, default="problem", help="Колонка с входными данными")
    parser.add_argument("--prompt-template", type=str,
                       default="You will be given a problem. Please reason step by step, and put your final answer within \\boxed{}: {{ instruction }}",
                       help="Шаблон для форматирования промптов")
    parser.add_argument("--max-examples", type=int, default=None, help="Максимальное количество примеров")
    parser.add_argument("--private", action="store_true", help="Сделать выходной датасет приватным")

    args = parser.parse_args()

    generate_hard_labels(
        dataset_name=args.dataset,
        dataset_split=args.split,
        teacher_model=args.teacher_model,
        output_dataset=args.output_dataset,
        prompt_column=args.prompt_column,
        prompt_template=args.prompt_template,
        max_examples=args.max_examples,
        private=args.private,
    )
```

### **Этап 2: Обучение модели-ученика на "жестких" метках**

```python
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint

from trl import SFTTrainer, ModelConfig, TrlParser, get_peft_config
from open_r1.configs import SFTConfig
from open_r1.utils.wandb_logging import init_wandb_training

logger = logging.getLogger(__name__)

@dataclass
class HardLabelDistillConfig(SFTConfig):
    """Конфигурация для обучения ученика с использованием Hard-label Distillation."""

    dataset_name: str = field(
        default=None, metadata={"help": "Датасет с 'жесткими' метками, сгенерированными учителем"}
    )
    input_column: str = field(
        default="problem", metadata={"help": "Колонка с входными данными"}
    )
    target_column: str = field(
        default="generation_0", metadata={"help": "Колонка с выходными данными (жесткими метками) учителя"}
    )
    max_seq_length: int = field(
        default=2048, metadata={"help": "Максимальная длина последовательности"}
    )

def train_student_model(config: HardLabelDistillConfig, model_args: ModelConfig) -> None:
    """
    Description:
    ---------------
    Обучает модель-ученика на 'жестких' метках, сгенерированных учителем.

    Args:
    ---------------
        config: Конфигурация обучения
        model_args: Конфигурация модели

    Returns:
    ---------------
        None

    Raises:
    ---------------
        Exception: В случае ошибки обучения модели

    Examples:
    ---------------
        >>> train_student_model(config, model_args)
    """
    # Настраиваем логирование
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = config.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # Устанавливаем сид для воспроизводимости
    set_seed(config.seed)

    # Проверяем наличие последнего чекпоинта
    last_checkpoint: Optional[str] = None
    if os.path.isdir(config.output_dir):
        last_checkpoint = get_last_checkpoint(config.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Найден чекпоинт, продолжаем обучение с {last_checkpoint}")

    # Инициализируем Weights & Biases, если нужно
    if "wandb" in config.report_to:
        init_wandb_training(config)

    # Загружаем датасет с 'жесткими' метками
    logger.info(f"Загрузка датасета с 'жесткими' метками: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name)

    # Подготавливаем входные данные и метки для обучения
    def prepare_dataset(examples: Dict[str, Any]) -> Dict[str, Any]:
        """Форматирует данные для обучения с учителем."""
        return {
            "input_ids": examples[config.input_column],
            "labels": examples[config.target_column],
        }

    # Трансформируем датасет
    dataset = dataset.map(prepare_dataset, batched=True)

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Настраиваем chat_template, если указан
    if config.chat_template is not None:
        tokenizer.chat_template = config.chat_template

    # Настраиваем параметры модели
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs: Dict[str, Any] = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if config.gradient_checkpointing else True,
    )
    config.model_init_kwargs = model_kwargs

    # Создаем SFT тренер
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset and config.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Запускаем обучение
    logger.info("Начало обучения модели-ученика...")
    checkpoint: Optional[str] = None
    if config.resume_from_checkpoint is not None:
        checkpoint = config.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Сохраняем модель
    logger.info(f"Сохранение модели в {config.output_dir}")
    trainer.save_model(config.output_dir)

    # Создаем карточку модели и загружаем на HuggingFace Hub, если нужно
    kwargs: Dict[str, Any] = {
        "dataset_name": config.dataset_name,
        "tags": ["hard-label-distillation", "open-r1"],
    }

    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Восстанавливаем кэш для быстрого инференса
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(config.output_dir)

    # Оцениваем модель, если нужно
    if config.do_eval and "validation" in dataset:
        logger.info("Оценка модели...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Загружаем модель на HuggingFace Hub, если нужно
    if config.push_to_hub:
        logger.info("Загрузка модели на HuggingFace Hub...")
        trainer.push_to_hub(**kwargs)

if __name__ == "__main__":
    # Создаем парсер аргументов
    parser = TrlParser((HardLabelDistillConfig, ModelConfig))
    config, model_args = parser.parse_args_and_config()

    # Запускаем обучение
    train_student_model(config, model_args)
```

### **Пример использования**

```python
# Этап 1: Генерация "жестких" меток с использованием модели-учителя
python hard_label_distill.py \
  --dataset AI-MO/NuminaMath-TIR \
  --teacher-model deepseek-ai/DeepSeek-R1 \
  --output-dataset username/hard-label-math-dataset \
  --prompt-column problem

# Этап 2: Обучение модели-ученика на сгенерированных "жестких" метках
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml train_student.py \
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_name username/hard-label-math-dataset \
  --input_column problem \
  --target_column generation_0 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 2 \
  --packing \
  --max_seq_length 4096 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --bf16 \
  --output_dir models/Qwen2.5-1.5B-Hard-Label-Distill
```


## II. Soft-label Distillation: Дистилляция с использованием "мягких" меток

**Концепция:**

Soft-label distillation, в отличие от hard-label, использует не только "жесткие" метки, но и **распределение вероятностей**, предсказанное учителем, в качестве "мягких" меток (soft labels).  "Мягкие" метки содержат больше информации, чем "жесткие", поскольку они отражают уверенность учителя в различных классах и отношения между ними.  Студент обучается имитировать это распределение вероятностей учителя, а не только "жесткие" классы.

**Математическая формализация:**

В soft-label distillation, функция потерь модифицируется, чтобы учитывать "мягкие" метки, предсказанные учителем.  В примере текста функция потерь для soft-label дистилляции представлена следующим образом:

$$\frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T_n} \sum_{v \in V} p(y_t^{(n)} = v|y_{<t}^{(n)}) \log q_{\theta}(y_t^{(n)} = v|y_{<t}^{(n)})$$

Здесь:
*   $V$ — словарь (множество всех возможных токенов).
*   $p(y_t^{(n)} = v|y_{<t}^{(n)})$ — условная вероятность того, что $t$-й токен в $n$-й последовательности равен токену $v \in V$, предсказанная учителем, при условии предыдущих токенов $y_{<t}^{(n)}$. Это и есть "мягкая" метка от учителя.
*   $q_{\theta}(y_t^{(n)} = v|y_{<t}^{(n)})$ — условная вероятность того, что $t$-й токен в $n$-й последовательности равен токену $v \in V$, предсказанная студентом, при условии предыдущих токенов $y_{<t}^{(n)}$.

Эта функция потерь также представляет собой **кросс-энтропию**, но на этот раз между **распределением вероятностей учителя** $p(y_t^{(n)} = v|y_{<t}^{(n)})$ и **распределением вероятностей студента** $q_{\theta}(y_t^{(n)} = v|y_{<t}^{(n)})$ для каждого токена $t$ и примера $n$.  Мы минимизируем эту кросс-энтропию, чтобы заставить распределение вероятностей студента максимально приблизиться к распределению вероятностей учителя.

**Дополнительные детали и улучшения в Soft-label Distillation:**

*   **Temperature Scaling:**  Часто в soft-label distillation используется техника **temperature scaling**.  Перед вычислением "мягких" меток, логиты (не нормализованные вероятности) учителя делятся на "температуру" $T > 1$.  Это делает распределение вероятностей учителя "мягче" и более равномерным, что может помочь студенту лучше улавливать отношения между классами.  Аналогично, температуру можно применять и к логитам студента.

    $$p_{\text{soft}}(y|x) = \text{softmax}(\frac{\text{logits}_p(x)}{T})$$
    $$q_{\text{soft}}(y|x) = \text{softmax}(\frac{\text{logits}_q(x)}{T})$$

    Затем функция потерь строится на основе этих "температурных" распределений.

*   **Комбинированная функция потерь:**  Часто soft-label distillation комбинируют с обычной hard-label функцией потерь (кросс-энтропией с истинными метками, если они доступны).  Это позволяет студенту учиться как у учителя (через soft-labels), так и непосредственно из данных (через hard-labels).  Комбинированная функция потерь может выглядеть так:

    $$L_{\text{combined}} = \alpha L_{\text{soft}} + (1-\alpha) L_{\text{hard}}$$

    где $L_{\text{soft}}$ — функция потерь soft-label distillation (например, кросс-энтропия между распределениями учителя и студента), $L_{\text{hard}}$ — стандартная функция потерь hard-label (например, кросс-энтропия между предсказаниями студента и истинными метками), а $\alpha$ — гиперпараметр, контролирующий баланс между двумя типами потерь.

**Преимущества и недостатки Soft-label Distillation:**

*   **Преимущества:**  Более эффективная передача знаний, так как используются "мягкие" метки, содержащие больше информации.  Студент может лучше имитировать поведение учителя и часто достигает лучшей производительности, чем при hard-label distillation.
*   **Недостатки:**  Может быть немного сложнее в реализации, чем hard-label distillation, особенно при использовании temperature scaling и комбинированных функций потерь.

**Заключение:**

Knowledge Distillation является мощным методом для обучения эффективных и компактных моделей, используя знания, полученные от более сложных моделей-учителей.  Выбор между hard-label и soft-label distillation зависит от конкретной задачи и желаемого уровня производительности. Soft-label distillation, как правило, обеспечивает лучшую передачу знаний за счет использования более информативных "мягких" меток, но hard-label distillation проще в реализации и может быть достаточным в некоторых случаях.  Оба подхода позволяют эффективно использовать предварительно обученные модели для улучшения обучения более простых моделей.