# Tech Context: SGR Research Project

## Технологический стек

### Core Technologies

#### Language Models & APIs
- **Primary**: OpenAI GPT-4o-mini via OpenAI API
  - Поддержка Structured Outputs для schema validation
  - Configurable base_url для custom deployments
  - Temperature: 0.4 для баланса creativity/consistency
- **Alternative LLM Support**: 
  - Локальные модели через Ollama, vLLM
  - Поддержка constrained decoding через xgrammar, Outlines

#### Schema Validation & Type Safety
```python
# Core dependencies
pydantic = "^2.x"           # Схемы и валидация данных
annotated-types = "^0.7.0"  # Дополнительные type annotations
typing_extensions = "*"     # Python 3.8+ compatibility
```

#### Search & Information Retrieval  
- **Tavily API**: Веб-поиск с credibility scoring
  - Max 15 results per query
  - Content summarization и fact extraction
  - Автоматическое управление цитированием

#### User Interface & Output
```python
rich = "^13.x"  # Красивый console output с панелями и прогресс-барами
```

### Development Environment

#### Configuration Management
- **YAML Configuration**: `config.yaml` для централизованных настроек
- **Environment Variables**: Fallback для sensitive data
- **Multi-environment Support**: Dev/staging/production конфигурации

#### File Structure
```
├── sgr-deep-research.py     # Main application
├── config.yaml              # Configuration
├── reports/                 # Generated research reports
│   └── YYYYMMDD_HHMMSS_*.md
└── memory/                  # Memory Bank
    ├── memory-bank/         # Core project knowledge
    └── rules/               # Learning patterns
```

## Архитектурные принципы

### Schema-Guided Design Patterns

#### 1. Cascade Pattern
```python
class NextStep(BaseModel):
    reasoning_steps: List[str]
    current_situation: str
    function: Union[Step1, Step2, Step3]
```
- **Использование**: Линейные процессы с предопределенной последовательностью
- **Преимущества**: Предсказуемость, простота отладки

#### 2. Routing Pattern  
```python
class DecisionPoint(BaseModel):
    analysis: str
    decision_criteria: List[str] 
    chosen_path: Union[PathA, PathB, PathC]
```
- **Использование**: Ветвление логики на основе условий
- **Преимущества**: Явные критерии выбора

#### 3. Cycle Pattern
```python
class IterativeStep(BaseModel):
    iteration_count: int
    convergence_check: bool
    next_action: Union[Continue, Adapt, Complete]
```
- **Использование**: Итеративные процессы с адаптацией
- **Применение в проекте**: Адаптивное планирование исследования

### Anti-Cycling Mechanisms

#### Проблема бесконечных циклов
- **Risk**: LLM могут "застревать" в повторяющихся действиях
- **Solution**: Explicit counters в схемах

```python
class NextStep(BaseModel):
    searches_done: int = Field(description="MAX 3-4 searches")
    enough_data: bool = Field(description="Stop condition") 
    clarification_used: bool  # Prevent multiple clarifications
```

#### State Management
```python
CONTEXT = {
    "plan": None,
    "searches": [],
    "sources": {},
    "citation_counter": 0,
    "clarification_used": False
}
```

## Integration Points

### OpenAI Structured Outputs Integration
```python
completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    response_format=NextStep,  # Pydantic model
    messages=conversation_log,
    temperature=0.4
)
```

#### Benefits
- **Guaranteed JSON compliance**: No parsing errors
- **Type safety**: Automatic validation
- **Schema evolution**: Версионирование схем

### Search Integration Architecture
```python
def dispatch(cmd: WebSearch) -> SearchResult:
    # 1. Execute Tavily search
    response = tavily.search(query=cmd.query)
    
    # 2. Auto-generate citations
    for result in response['results']:
        citation_num = add_citation(result['url'])
    
    # 3. Format for LLM consumption
    return structured_result
```

## Data Flow & Processing

### Information Pipeline
```mermaid
User Request → Clarification → Plan → Search → Adapt → Report → Completion
```

#### 1. Input Processing
- Language detection из original request
- Schema validation всех входных параметров
- Context preservation между шагами

#### 2. Search & Citation Management  
- Автоматическое присвоение citation numbers [1], [2], [3]
- Deduplication источников по URL
- Persistent storage в session context

#### 3. Report Generation
- Inline citation integration в content  
- Language consistency enforcement
- Multi-format output support (.md files)

### Memory Management

#### In-Memory Context (Session-scoped)
```python
CONTEXT = {
    "plan": {...},           # Current research plan
    "searches": [...],       # Search history
    "sources": {},          # URL → citation mapping
    "citation_counter": 0   # Auto-increment counter
}
```

#### Persistent Memory (Memory Bank)
- **Project knowledge**: Структурированные .md файлы  
- **Learning patterns**: memory/rules/memory-bank.mdc
- **Report archive**: Timestamped files в reports/

## Performance Considerations

### Token Optimization
- **Context management**: Только релевантная информация в prompts
- **Search result truncation**: Max 300 chars per result content
- **Schema efficiency**: Минимальные field descriptions

### Rate Limiting & Cost Management
```python
CONFIG = {
    'max_tokens': 8000,           # Prevent excessive costs
    'max_search_results': 10,     # Limit Tavily usage  
    'max_execution_steps': 6      # Prevent infinite loops
}
```

### Error Handling & Resilience
```python
try:
    completion = client.beta.chat.completions.parse(...)
except Exception as e:
    print(f"LLM request error: {e}")
    # Fallback or retry logic
```

## Security & Compliance

### API Key Management
- **Environment variables**: Preferred method
- **Config file encryption**: For team deployments  
- **No hardcoding**: Secrets never in source code

### Data Privacy
- **Local processing**: Sensitive data не покидает infrastructure
- **Configurable endpoints**: Support для private LLM deployments
- **Audit trails**: Full conversation logs для compliance

### Input Validation  
- **Pydantic validation**: Всех user inputs и LLM outputs
- **Schema constraints**: MinLen, MaxLen, Literal types
- **Sanitization**: Safe filename generation для reports

## Scalability & Extension Points

### Multi-Agent Extension
```python
class AgentCoordinator(BaseModel):
    active_agents: List[str]
    agent_states: Dict[str, Any]
    coordination_protocol: CoordinationSchema
```

### Plugin Architecture
- **Tool plugins**: Новые search providers
- **Schema extensions**: Domain-specific patterns  
- **Output formats**: JSON, XML, custom formats

### Deployment Options
- **Standalone script**: Current implementation
- **Web service**: FastAPI wrapper
- **Container deployment**: Docker для production
- **Serverless**: AWS Lambda/Azure Functions

**Дата создания**: 2025-09-05  
**Tech Stack Version**: Python 3.9+, Pydantic 2.x, OpenAI API v1