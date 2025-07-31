# Настройка и запуск Google ADK Git Workflow Agent

## Предварительные требования

### Системные требования
- Python 3.9+
- Node.js 18+
- Git
- Google AI API ключ

### Установка зависимостей

1. **Python зависимости**
```bash
cd Google-Agent-Development-Kit
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# или
.venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

2. **Node.js зависимости**
```bash
cd ui
npm install
```

## Настройка API ключа

### Получение Google AI API ключа
1. Перейдите на [Google AI Studio](https://aistudio.google.com/)
2. Создайте новый API ключ
3. Скопируйте ключ

### Установка переменных окружения
```bash
export GOOGLE_API_KEY=your_google_ai_api_key
export GOOGLE_GENAI_USE_VERTEXAI=FALSE
```

Или создайте файл `.env` в корне проекта:
```bash
GOOGLE_API_KEY=your_google_ai_api_key
GOOGLE_GENAI_USE_VERTEXAI=FALSE
```

## Запуск приложения

### 1. Запуск backend (Python API)
```bash
cd src
python main.py server
```

API будет доступен по адресу: http://localhost:8000

### 2. Запуск frontend (Next.js)
```bash
cd ui
npm run dev
```

Веб-интерфейс будет доступен по адресу: http://localhost:3000

### 3. Альтернативный запуск через ADK CLI
```bash
cd src
adk api_server
```

## Использование

### Через веб-интерфейс
1. Откройте http://localhost:3000
2. Введите путь к Git репозиторию
3. Нажмите "Анализировать изменения"
4. Просмотрите предложенный коммит
5. Нажмите "Авто-коммит и пуш"

### Через CLI
```bash
# Анализ изменений
python main.py /path/to/project analyze

# Создание коммита с кастомным сообщением
python main.py /path/to/project commit "feat: add new feature"

# Автоматический коммит и пуш
python main.py /path/to/project auto-commit
```

### Через API
```bash
# Анализ изменений
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"project_path": "/path/to/project"}'

# Автоматический коммит
curl -X POST http://localhost:8000/api/auto-commit \
  -H "Content-Type: application/json" \
  -d '{"project_path": "/path/to/project"}'

# Получение статуса проекта
curl http://localhost:8000/api/status/%2Fpath%2Fto%2Fproject
```

## Структура проекта

```
Google-Agent-Development-Kit/
├── src/
│   ├── git_workflow_agent/
│   │   ├── __init__.py
│   │   ├── agent.py          # Основной агент
│   │   └── config.py         # Конфигурация
│   ├── utils/
│   │   ├── file_utils.py     # Работа с файлами
│   │   └── git_utils.py      # Git операции
│   └── main.py               # API сервер
├── ui/                       # Next.js frontend
├── requirements.txt          # Python зависимости
└── README.md                # Документация
```

## Конфигурация

### Настройки агента
Отредактируйте `src/git_workflow_agent/config.py`:

```python
AGENT_CONFIG = {
    "model": "gemini-2.0-flash",  # Модель Gemini
    "name": "git_workflow_agent",
    "instruction": "Твои инструкции для агента..."
}
```

### Настройки Git
```python
GIT_CONFIG = {
    "max_file_size": 1024 * 1024,  # Максимальный размер файла
    "supported_extensions": [".py", ".js", ".ts", ...],
    "ignore_patterns": ["__pycache__", "node_modules", ...]
}
```

### Настройки веб-интерфейса
```python
WEB_CONFIG = {
    "host": "localhost",
    "port": 3000,
    "cors_origins": ["http://localhost:3000"]
}
```

## Устранение неполадок

### Ошибка "Не установлена переменная GOOGLE_API_KEY"
```bash
export GOOGLE_API_KEY=your_api_key
```

### Ошибка "Директория не является Git репозиторием"
```bash
cd /path/to/project
git init
git remote add origin <repository_url>
```

### Ошибка MCP соединения
```bash
# Установите npx
npm install -g npx

# Проверьте доступность MCP сервера
npx -y @modelcontextprotocol/server-filesystem --help
```

### Ошибка портов
```bash
# Проверьте занятые порты
lsof -i :8000
lsof -i :3000

# Убейте процессы если нужно
kill -9 <PID>
```

## Разработка

### Добавление новых функций
1. Создайте новый модуль в `src/`
2. Добавьте инструменты в агента
3. Обновите API endpoints
4. Обновите веб-интерфейс
5. Добавьте тесты

### Тестирование
```bash
# Python тесты
cd src
pytest tests/

# Frontend тесты
cd ui
npm test
```

### Логирование
Логи сохраняются в `logs/git_workflow_agent.log`

## Поддержка

- **Issues**: Создайте issue в репозитории
- **Документация**: См. README.md
- **API**: См. http://localhost:8000/docs (Swagger UI) 