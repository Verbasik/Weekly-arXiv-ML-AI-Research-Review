# System Patterns - Системные паттерны

## Архитектурные паттерны

### Domain-Driven Design (DDD)
Основной архитектурный подход проекта основан на принципах DDD:

**Слои архитектуры:**
1. **Presentation Layer** (`web/presentation/`):
   - UI компоненты (WeekCard, ModalWindow)
   - Мобильные улучшения
   - Обработка пользовательского ввода

2. **Application Layer** (`web/application/`):
   - ResearchController - координация между слоями
   - Orchestration логика
   - Use cases

3. **Domain Layer** (`web/domain/`):
   - Entities (Week, Year) - бизнес-сущности
   - Services (ResearchService) - доменные сервисы
   - Repositories (абстракции)

4. **Infrastructure Layer** (`web/infrastructure/`):
   - GitHubDataSource - реализация источника данных
   - ErrorHandler - системные сервисы
   - External integrations

### Repository Pattern
**Описание**: Абстракция доступа к данным
**Реализация**: `ResearchRepository` предоставляет интерфейс для работы с данными
**Преимущества**: 
- Тестируемость
- Изоляция от источников данных
- Единообразный API

```javascript
class ResearchRepository {
    async getAllYears() { /* ... */ }
    async getWeek(yearNumber, weekId) { /* ... */ }
    async searchWeeks(query) { /* ... */ }
}
```

### Service Layer Pattern
**Описание**: Инкапсуляция бизнес-логики
**Реализация**: `ResearchService` содержит логику работы с исследованиями
**Ответственности**:
- Валидация данных
- Бизнес-правила
- Координация между repositories

## Паттерны проектирования

### Factory Pattern
**Применение**: Создание компонентов UI
**Реализация**: Классы WeekCard, ModalWindow создают DOM элементы

```javascript
class WeekCard {
    createElement() {
        // Factory method для создания DOM элемента
        return this._createCardElement();
    }
}
```

### Observer Pattern
**Применение**: Обработка событий DOM, network status
**Реализация**: EventListener для UI взаимодействий

### Strategy Pattern
**Применение**: Обработка различных типов ошибок
**Реализация**: ErrorHandler.classifyError() выбирает стратегию обработки

```javascript
const ErrorHandler = {
    classifyError: (error, response) => {
        // Стратегия определяется типом ошибки
        if (!navigator.onLine) return { type: 'offline' };
        if (response?.status === 404) return { type: 'not_found' };
        // ...
    }
};
```

### Template Method Pattern
**Применение**: Загрузка и отображение контента
**Реализация**: Базовая структура в ModalWindow.open()

### Command Pattern
**Применение**: Обработка пользовательских действий
**Реализация**: Button handlers, retry mechanisms

## Данные и состояние

### State Management Pattern
**Подход**: Локальное состояние в компонентах
**Реализация**: 
- Controller держит состояние приложения
- Компоненты управляют своим локальным состоянием
- URL как источник истины для navigation

### Caching Pattern
**Применение**: Кэширование загруженных данных
**Реализация**: In-memory кэш в компонентах
**Стратегия**: Simple caching без TTL

## Error Handling Patterns

### Retry Pattern
**Реализация**: `fetchWithRetry` с экспоненциальным backoff
**Конфигурация**:
```javascript
const RETRY_CONFIG = {
    maxRetries: 3,
    baseDelay: 1000,
    backoffFactor: 2
};
```

### Circuit Breaker Pattern
**Статус**: Частично реализован
**Применение**: Определение offline состояния
**Логика**: Fallback на cached контент при недоступности сети

### Graceful Degradation
**Реализация**: 
- Offline indicators
- Fallback UI states
- Error boundaries

## UI/UX Patterns

### Progressive Enhancement
**Подход**: Базовая функциональность без JavaScript + enhanced experience
**Реализация**: Статический контент + динамические улучшения

### Responsive Design Pattern
**Стратегия**: Mobile-first approach
**Breakpoints**:
- Mobile: < 768px
- Tablet: 768px - 1024px  
- Desktop: > 1024px

### Modal Pattern
**Реализация**: ModalWindow класс
**Особенности**:
- URL synchronization
- Keyboard navigation
- Mobile gestures

### Loading States Pattern
**Применение**: Индикаторы загрузки для асинхронных операций
**Типы**:
- Skeleton loading
- Spinner loading
- Progressive loading

## Performance Patterns

### Lazy Loading
**Применение**: Контент загружается по требованию
**Реализация**: Modal content загрузка при открытии

### Asset Optimization
**Стратегия**: 
- CDN для внешних библиотек
- Optimized images
- CSS/JS minification (production)

### Caching Strategy
**Browser Caching**: Статические ресурсы
**Memory Caching**: Загруженные данные
**No Server Caching**: Статический сайт

## Security Patterns

### Input Sanitization
**Применение**: Markdown content processing
**Библиотека**: Marked.js с базовой санитизацией

### XSS Prevention
**Стратегия**: Использование textContent вместо innerHTML где возможно
**CSP**: Content Security Policy headers

## Integration Patterns

### API Integration
**GitHub API**: Интеграция с GitHub для получения данных
**Pattern**: Simple HTTP client с retry logic

### External Dependencies
**CDN Strategy**: Внешние библиотеки через CDN
**Fallback**: Local fallbacks не реализованы (технический долг)

## Testing Patterns (Планируемые)

### Unit Testing Pattern
**Статус**: Не реализовано
**План**: Jest/Vitest для unit тестов

### Integration Testing Pattern  
**Статус**: Не реализовано
**План**: Testing Library для UI тестов

### E2E Testing Pattern
**Статус**: Не реализовано
**План**: Playwright для end-to-end тестов

## Anti-patterns (Избегаем)

### God Object
**Избегаем**: Концентрацию всей логики в одном классе
**Решение**: Разделение ответственности между слоями

### Tight Coupling
**Избегаем**: Жесткие зависимости между модулями
**Решение**: Dependency injection, интерфейсы

### Magic Numbers/Strings
**Избегаем**: Хардкод значений
**Решение**: Конфигурационные константы

## Архитектурные решения

### Почему DDD?
- Четкое разделение бизнес-логики и технических деталей
- Масштабируемость
- Тестируемость
- Понятная структура проекта

### Почему статический сайт?
- Простота развертывания
- Высокая производительность
- Низкие затраты на поддержку
- GitHub Pages integration

### Почему без фреймворка?
- Контроль над кодом
- Минимальные зависимости
- Образовательная ценность
- Производительность 