#!/usr/bin/env python3
"""
SGR Schemas for Mathematical Problem Solving
Специализированные схемы для решения математических задач IMO уровня
"""

from typing import List, Union, Literal, Optional, Dict, Any
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from pydantic import BaseModel, Field
from annotated_types import MinLen, MaxLen

# =============================================================================
# БАЗОВЫЕ СХЕМЫ МАТЕМАТИЧЕСКОГО АНАЛИЗА
# =============================================================================

class ProblemAnalysis(BaseModel):
    """Анализ математической задачи и определение типа"""
    tool: Literal["analyze_problem"]
    reasoning: str = Field(description="Обоснование анализа задачи")
    
    problem_domain: Literal[
        "algebra", "geometry", "combinatorics", "number_theory", 
        "analysis", "discrete_math", "mixed"
    ] = Field(description="Основная область математики")
    
    problem_type: str = Field(description="Конкретный тип задачи (например: polynomial equations, triangle geometry)")
    
    key_concepts: Annotated[List[str], MinLen(2), MaxLen(6)] = Field(
        description="Ключевые математические концепции для решения"
    )
    
    difficulty_assessment: Literal["low", "medium", "high", "very_high"] = Field(
        description="Оценка сложности задачи"
    )
    
    suggested_approaches: Annotated[List[str], MinLen(2), MaxLen(4)] = Field(
        description="Возможные подходы к решению"
    )

class SolutionStrategy(BaseModel):
    """Выбор стратегии решения на основе анализа"""
    tool: Literal["choose_strategy"] 
    reasoning: str = Field(description="Обоснование выбора стратегии")
    
    chosen_approach: str = Field(description="Выбранный подход к решению")
    
    solution_steps_plan: Annotated[List[str], MinLen(3), MaxLen(8)] = Field(
        description="Планируемые шаги решения"
    )
    
    required_lemmas: Optional[List[str]] = Field(
        default=None, description="Необходимые леммы или теоремы"
    )
    
    case_analysis_needed: bool = Field(
        default=False, description="Требуется ли разбор по случаям"
    )
    
    expected_techniques: Annotated[List[str], MinLen(1), MaxLen(5)] = Field(
        description="Ожидаемые математические техники"
    )

class MathematicalSolution(BaseModel):
    """Структурированное математическое решение"""
    tool: Literal["generate_solution"]
    reasoning: str = Field(description="Обоснование подхода к решению")
    
    # Структурированное решение по шагам
    solution_summary: str = Field(
        description="Краткое резюме решения и финального ответа"
    )
    
    detailed_solution: str = Field(
        description="""
        Подробное пошаговое решение с использованием TeX для математики.
        
        ОБЯЗАТЕЛЬНЫЕ ТРЕБОВАНИЯ:
        - Каждая математическая переменная, выражение в TeX: $x$, $f(x) = x^2 + 1$
        - Четкая логическая структура с обоснованием каждого шага
        - Строгие математические доказательства без пропусков
        - Структура: 1) Анализ, 2) Основное решение, 3) Проверка, 4) Ответ
        
        Пример структуры:
        **Анализ задачи:**
        [анализ условий и постановки]
        
        **Решение:**
        Шаг 1: [первый шаг с обоснованием]
        Шаг 2: [второй шаг с обоснованием]  
        ...
        
        **Проверка:**
        [верификация полученного результата]
        
        **Ответ:** [финальный ответ]
        """
    )
    
    key_insights: Annotated[List[str], MinLen(1), MaxLen(4)] = Field(
        description="Ключевые инсайты или идеи решения"
    )
    
    mathematical_rigor: Literal["complete", "mostly_complete", "partial", "incomplete"] = Field(
        description="Уровень математической строгости решения"
    )
    
    confidence: Literal["very_high", "high", "medium", "low"] = Field(
        description="Уверенность в корректности решения"
    )

class SolutionVerification(BaseModel):
    """Верификация математического решения"""
    tool: Literal["verify_solution"]
    reasoning: str = Field(description="Обоснование подхода к верификации")
    
    verification_approach: Literal[
        "step_by_step_check", "alternative_method", "substitution_check", 
        "boundary_analysis", "proof_by_contradiction"
    ] = Field(description="Метод верификации")
    
    identified_issues: List[str] = Field(
        default=[], description="Обнаруженные проблемы или ошибки"
    )
    
    verification_result: Literal["correct", "incorrect", "partially_correct", "unclear"] = Field(
        description="Результат верификации"
    )
    
    issue_severity: Optional[List[Literal["critical", "major", "minor", "stylistic"]]] = Field(
        default=None, description="Серьезность обнаруженных проблем"
    )
    
    suggestions: List[str] = Field(
        default=[], description="Предложения по улучшению решения"
    )

class SolutionImprovement(BaseModel):
    """Улучшение решения на основе верификации"""
    tool: Literal["improve_solution"]
    reasoning: str = Field(description="Обоснование необходимости улучшения")
    
    issues_to_address: Annotated[List[str], MinLen(1), MaxLen(5)] = Field(
        description="Проблемы для исправления"
    )
    
    improvement_strategy: str = Field(description="Стратегия улучшения решения")
    
    expected_improvements: Annotated[List[str], MinLen(1), MaxLen(4)] = Field(
        description="Ожидаемые улучшения"
    )

class TaskCompletion(BaseModel):
    """Завершение решения задачи"""
    tool: Literal["complete_task"]
    reasoning: str = Field(description="Обоснование завершения работы")
    
    final_answer: str = Field(description="Финальный ответ на задачу")
    
    solution_quality: Literal["excellent", "good", "satisfactory", "poor"] = Field(
        description="Итоговая оценка качества решения"
    )
    
    completed_steps: Annotated[List[str], MinLen(1), MaxLen(8)] = Field(
        description="Выполненные этапы решения"
    )

# =============================================================================
# ГЛАВНАЯ SGR СХЕМА - УПРАВЛЕНИЕ ПРОЦЕССОМ РЕШЕНИЯ
# =============================================================================

class MathSolutionNextStep(BaseModel):
    """Центральная SGR схема для управления процессом решения математических задач"""
    
    # Цепочка рассуждений для устойчивости модели
    reasoning_chain: Annotated[List[str], MinLen(2), MaxLen(5)] = Field(
        description="Пошаговый процесс рассуждения к решению"
    )
    
    # Оценка текущей ситуации
    current_situation: str = Field(
        description="Анализ текущего состояния процесса решения"
    )
    
    problem_understanding: Literal["unclear", "partial", "good", "complete"] = Field(
        description="Уровень понимания задачи"
    )
    
    # Прогресс решения
    solution_progress: Literal["not_started", "analysis_done", "strategy_chosen", 
                              "solving_in_progress", "solution_complete", "verified"] = Field(
        description="Текущий прогресс в решении"
    )
    
    verification_attempts: int = Field(
        default=0, description="Количество попыток верификации (МАКС 3)"
    )
    
    improvement_attempts: int = Field(
        default=0, description="Количество попыток улучшения (МАКС 5)"  
    )
    
    # Планирование следующих шагов
    remaining_steps: Annotated[List[str], MinLen(1), MaxLen(4)] = Field(
        description="Оставшиеся шаги для завершения задачи"
    )
    
    task_completed: bool = Field(
        description="Завершена ли задача решения?"
    )
    
    # Маршрутизация инструментов с приоритетом анализа
    function: Union[
        ProblemAnalysis,        # ПЕРВЫЙ: понимание задачи
        SolutionStrategy,       # ВТОРОЙ: выбор стратегии  
        MathematicalSolution,   # ТРЕТИЙ: генерация решения
        SolutionVerification,   # ЧЕТВЕРТЫЙ: проверка решения
        SolutionImprovement,    # ПЯТЫЙ: улучшение при ошибках
        TaskCompletion          # ШЕСТОЙ: завершение задачи
    ] = Field(description="""
    ПРИОРИТЕТЫ ПРИНЯТИЯ РЕШЕНИЙ:
    
    1. Если понимание задачи unclear/partial → ProblemAnalysis
    2. Если анализ есть но нет стратегии → SolutionStrategy  
    3. Если стратегия есть но нет решения → MathematicalSolution
    4. Если есть решение но не проверено → SolutionVerification
    5. Если verification показал ошибки И improvement_attempts < 5 → SolutionImprovement
    6. Если все готово ИЛИ превышены лимиты → TaskCompletion
    
    АНТИ-ЦИКЛИЧЕСКИЕ ПРАВИЛА:
    - Максимум 3 верификации на решение
    - Максимум 5 попыток улучшения
    - Если лимиты превышены → принудительное завершение
    
    МАТЕМАТИЧЕСКАЯ СПЕЦИФИКА:
    - Приоритет строгости над скоростью
    - Обязательная верификация перед завершением  
    - TeX форматирование для всех математических выражений
    """)

# =============================================================================
# УТИЛИТЫ ДЛЯ РАБОТЫ СО СХЕМАМИ
# =============================================================================

def get_problem_system_prompt() -> str:
    """Системный промпт для математического решения с SGR"""
    return """
Оформи аккуратно следующий текст

<system>
  <role>Ты — эксперт-математик уровня Международной математической олимпиады (IMO).</role>

  <language>Язык вывода: русский. Все математические выражения оформляй в TeX (например, $x$, $f(x)=x^2$, \(\forall,\exists\)).</language>

  <rules>
    <must>
      <item>Строгость: каждое утверждение должно быть доказано или обосновано.</item>
      <item>TeX: все формулы — в TeX; используй \( \cdot \) и \[ \cdot \] при необходимости.</item>
      <item>Структура: соблюдай фиксированный формат разделов из &lt;output_format&gt;.</item>
      <item>SGR-процесс: analyze_problem → choose_strategy → generate_solution → verify_solution → improve_solution → complete_task (выполняй скрытно, без вывода внутренних рассуждений).</item>
      <item>Верификация: перед финалом прогони чек-лист «Verification Rubric (10/10)».</item>
      <item>Честность: если полное строгое доказательство недоступно, представь частичное строгое решение с явной маркировкой недостающих шагов.</item>
    </must>
    <must_not>
      <item>Не выводи внутренние рассуждения, брейнсторм, попытки и мета-анализ («подумал/попробуем/кажется»).</item>
      <item>Не используй дисклеймеры вида «как ИИ-модель», «я могу ошибаться».</item>
      <item>Не нарушай структуру разделов и не смешивай их.</item>
    </must_not>
  </rules>

  <sgr_process visibility="internal">
    <step id="1">analyze_problem — глубокий анализ формулировки и классификация задачи.</step>
    <step id="2">choose_strategy — выбор оптимальных методов/лемм.</step>
    <step id="3">generate_solution — полное строгое решение в TeX.</step>
    <step id="4">verify_solution — критическая проверка по рубрике 10/10.</step>
    <step id="5">improve_solution — исправление и усиление слабых мест при несоответствии.</step>
    <step id="6">complete_task — финализация и явный ответ.</step>
  </sgr_process>

  <output_format>
    <section name="json_header" required="true" description="Одна строка JSON со статусом">
      <![CDATA[
```json
{"problem_class": "<алгебра|геометрия|числа|комбинаторика|смешанная>", "primary_methods": ["<метод1>", "<метод2>"], "difficulty_estimate_imo": "<1–7>", "verification_status": "<pending|passed|partial>"}
```
</section>

<section name="analysis" required="true">
  Краткая формальная классификация, ключевые определения и существенные граничные/исключительные случаи (без мета-текста).
</section>

<section name="strategy" required="true">
  Сжатое формальное описание применяемых приёмов (леммы, инварианты, неравенства, разложения и т. п.).
</section>

<section name="solution" required="true">
  Полное строгое доказательство/вывод с TeX; при разборе случаев — подзаголовки «Случай A», «Случай B»; обязательно укажи граничные условия.
</section>

<section name="verification" required="true">
  Чек-лист «Verification Rubric (10/10)»: для каждого пункта поставь 1/0 и одну строку формального пояснения.
</section>

<section name="final_answer" required="true">
  Короткая и явная формулировка результата (число/формула/утверждение).
</section>
```

\</output\_format>

\<verification\_rubric threshold="10/10"> <criterion id="1">Термины корректны и определены (1/0).</criterion> <criterion id="2">Каждое преобразование обосновано (1/0).</criterion> <criterion id="3">Разобраны все необходимые случаи (1/0).</criterion> <criterion id="4">Проверены граничные/исключительные случаи (1/0).</criterion> <criterion id="5">Отсутствуют скрытые предположения (1/0).</criterion> <criterion id="6">Нет логических скачков; шаги связаны (1/0).</criterion> <criterion id="7">Формулы в TeX синтаксически корректны (1/0).</criterion> <criterion id="8">Итог согласован со всеми условиями (1/0).</criterion> <criterion id="9">При частичном решении явно отмечены пробелы (1/0).</criterion> <criterion id="10">Финальный ответ явно выделен (1/0).</criterion>
\</verification\_rubric>

\<incomplete\_proof\_policy>
Если доказательство не завершено: <rule>В разделе \<verification> пункты несоответствия пометь 0 с пояснением.</rule> <rule>В \<solution> чётко укажи, каких лемм/шагов не хватает.</rule> <rule>В \<final\_answer> укажи «частично решено» и что требуется для завершения.</rule>
\<json\_header\_update>Установи "verification\_status": "partial".\</json\_header\_update>
\</incomplete\_proof\_policy>

  <markers>
    <allowed>
      <example>«Используем неравенство Чебышёва для упорядоченных последовательностей …»</example>
      <example>«Инвариант: сумма чётна, так как …»</example>
    </allowed>
    <forbidden>
      <example>«Давайте попробуем…», «Похоже, это не сработает», «Я думаю, что…»</example>
    </forbidden>
  </markers>

\<pre\_final\_checks> <step>Заполни раздел \<verification> по всем 10 пунктам.</step> <step>Если сумма < 10: выполни скрытно improve\_solution и обнови \<solution> и \<verification>.</step> <step>В JSON-шапке установи "verification\_status": "passed" при 10/10, иначе "partial".</step>
\</pre\_final\_checks>

\<completion\_criteria> <item>Доказательство строгое и полное (или корректно обозначено как частичное).</item> <item>Все утверждения обоснованы, граничные случаи проверены.</item> <item>Финальный ответ сформулирован явно.</item> <item>Верификация выполнена по рубрике 10/10 и отражена во выводе.</item>
\</completion\_criteria> </system>
""".strip()

def create_math_context() -> Dict[str, Any]:
    """Создание контекста для математического SGR агента"""
    return {
        "problem_text": "",
        "analysis": None,
        "strategy": None, 
        "solution": None,
        "verification": None,
        "improvements": [],
        "verification_count": 0,
        "improvement_count": 0,
        "start_time": None,
        "solution_history": []
    }