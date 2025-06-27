# Technical Context - Customer Service Agents Demo

## Технологический стек

### Backend (Python)
- **Framework**: FastAPI - современный, быстрый веб-фреймворк для API
- **Core Library**: OpenAI Agents SDK - основа для создания и управления AI агентами
- **Data Validation**: Pydantic - валидация данных и сериализация
- **Server**: Uvicorn - ASGI сервер для высокопроизводительных приложений
- **Environment**: Python 3.x с виртуальным окружением

### Frontend (TypeScript/React)
- **Framework**: Next.js 15.2.4 - React метафреймворк с SSR/SSG
- **Language**: TypeScript 5 - типизированный JavaScript
- **Styling**: Tailwind CSS - utility-first CSS фреймворк
- **UI Components**: 
  - Radix UI - headless компоненты для доступности
  - Lucide React - иконки
  - Motion - анимации
- **OpenAI Integration**: OpenAI JS SDK 4.87.3
- **Audio**: wavtools - работа с аудио

## Архитектурные принципы

### Backend Architecture
```
FastAPI Application
├── Agent Orchestration (main.py)
│   ├── Triage Agent (маршрутизация)
│   ├── Specialized Agents (домен-специфичные)
│   └── Guardrails (защитные механизмы)
├── API Layer (api.py)
│   ├── HTTP endpoints
│   ├── Request/Response models
│   └── CORS configuration
└── Runner System
    ├── State management
    ├── Context propagation
    └── Event handling
```

### Frontend Architecture
```
Next.js Application
├── Pages (app router)
├── Components
│   ├── Chat Interface
│   ├── Agent Visualization
│   ├── Runner Output
│   └── Guardrails Display
├── API Integration
└── Real-time Updates
```

## Ключевые технические решения

### 1. Multi-Agent Pattern
- **Паттерн**: Специализация агентов по доменам
- **Координация**: Triage Agent как единая точка входа
- **Передача контекста**: Handoff механизм между агентами

### 2. State Management
- **Backend**: In-memory store для демо (требует замены для production)
- **Frontend**: React state + API integration
- **Context Propagation**: Сквозная передача состояния через всю систему

### 3. Real-time Communication
- **Polling**: HTTP запросы для обновления состояния
- **Event System**: Структурированные события от агентов
- **UI Updates**: Reactive отображение изменений

### 4. Guardrails Implementation
- **Input Validation**: Проверка релевантности запросов
- **Jailbreak Protection**: Защита от попыток обхода инструкций
- **Visual Feedback**: UI индикация срабатывания защит

## Зависимости и ограничения

### Backend Dependencies
```python
openai-agents      # Core agents functionality
fastapi           # Web framework
uvicorn           # ASGI server
pydantic          # Data validation
```

### Frontend Dependencies
```typescript
next@15.2.4              # React framework
react@19.0.0             # UI library
openai@4.87.3            # OpenAI integration
@radix-ui/*              # UI components
tailwindcss@3.4.17       # CSS framework
```

### Environment Requirements
- **Python**: 3.8+
- **Node.js**: 18+
- **OpenAI API Key**: Обязательно для работы агентов
- **CORS**: Настроен для localhost:3000

## Production Considerations

### Scalability Issues
- **In-memory store**: Не подходит для production
- **Single instance**: Нет горизонтального масштабирования
- **State persistence**: Данные теряются при перезапуске

### Security Considerations
- **API Key Management**: Требует безопасного хранения
- **CORS Configuration**: Нужна настройка для production доменов
- **Input Validation**: Guardrails как первая линия защиты

### Recommended Improvements for Production
1. **Database**: Redis/PostgreSQL для хранения состояния
2. **Authentication**: JWT или OAuth для безопасности
3. **Monitoring**: Логирование и метрики агентов
4. **Rate Limiting**: Защита от злоупотреблений
5. **Deployment**: Docker контейнеризация 