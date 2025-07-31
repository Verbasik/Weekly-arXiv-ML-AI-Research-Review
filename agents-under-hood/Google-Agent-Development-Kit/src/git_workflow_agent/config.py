"""
Конфигурация Git Workflow Agent

Настройки агента для автоматического анализа изменений и создания коммитов.
"""

import os
from typing import Dict, Any

# Основная конфигурация агента
AGENT_CONFIG: Dict[str, Any] = {
    "model": os.getenv("ADK_AGENT_MODEL", "gemini-2.0-flash"),
    "name": os.getenv("ADK_AGENT_NAME", "git_workflow_agent"),
    "description": "Автоматический анализ изменений в коде и создание коммитов",
    "instruction": """
    Ты - эксперт по анализу изменений в коде и созданию качественных коммитов.
    
    Твоя основная задача:
    1. Анализировать изменения в файлах проекта
    2. Создавать информативные и структурированные сообщения коммитов
    3. Следовать лучшим практикам Git и Conventional Commits
    4. Обеспечивать качественную документацию изменений
    
    Принципы работы:
    - Всегда анализируй контекст изменений
    - Используй четкие и описательные сообщения
    - Следуй формату Conventional Commits (type(scope): description)
    - Указывай влияние изменений на функциональность
    - Предлагай краткое и длинное описание коммита
    
    Типы коммитов:
    - feat: новая функциональность
    - fix: исправление ошибок
    - docs: изменения в документации
    - style: форматирование кода
    - refactor: рефакторинг кода
    - test: добавление тестов
    - chore: обновление зависимостей, конфигурации
    
    Структура сообщения:
    ```
    type(scope): краткое описание
    
    Подробное описание изменений:
    - Что изменилось
    - Почему изменилось
    - Какое влияние на систему
    
    Breaking changes: (если есть)
    - Описание несовместимых изменений
    ```
    """,
    "examples": [
        {
            "input": "Добавлена новая функция для валидации email",
            "output": "feat(validation): add email validation function\n\nAdded comprehensive email validation with regex patterns and domain checking.\n- Supports multiple email formats\n- Includes domain validation\n- Returns detailed error messages"
        },
        {
            "input": "Исправлена ошибка в обработке JSON данных",
            "output": "fix(json): resolve JSON parsing error in data processor\n\nFixed issue where malformed JSON caused application crash.\n- Added proper error handling\n- Improved error messages\n- Added validation for JSON structure"
        },
        {
            "input": "Обновлена документация API",
            "output": "docs(api): update API documentation with new endpoints\n\nUpdated API documentation to include recently added endpoints.\n- Added authentication examples\n- Included error response formats\n- Updated code samples"
        }
    ]
}

# Конфигурация Git
GIT_CONFIG = {
    "commit_template": """
{type}({scope}): {description}

{detailed_description}

{breaking_changes}
    """,
    "max_file_size": 1024 * 1024,  # 1MB
    "supported_extensions": [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".go",
        ".rs", ".php", ".rb", ".swift", ".kt", ".scala", ".clj", ".hs",
        ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"
    ],
    "ignore_patterns": [
        "__pycache__", "node_modules", ".git", ".env", ".DS_Store",
        "*.log", "*.tmp", "*.cache", "dist", "build", "coverage"
    ]
}

# Конфигурация MCP
MCP_CONFIG = {
    "server_command": "npx",
    "server_args": ["-y", "@modelcontextprotocol/server-filesystem"],
    "tool_filter": ["read_file", "list_directory", "write_file"],
    "connection_timeout": 30,
    "max_retries": 3
}

# Конфигурация анализа
ANALYSIS_CONFIG = {
    "max_files_per_analysis": 50,
    "max_changes_per_file": 100,
    "context_lines": 5,
    "analysis_timeout": 60,
    "cache_duration": 300  # 5 минут
}

# Конфигурация веб-интерфейса
WEB_CONFIG = {
    "host": os.getenv("WEB_HOST", "localhost"),
    "port": int(os.getenv("WEB_PORT", "3000")),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "cors_origins": ["http://localhost:3000", "http://localhost:8000"],
    "rate_limit": {
        "requests_per_minute": 60,
        "burst_size": 10
    }
}

# Конфигурация логирования
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    "file": "logs/git_workflow_agent.log",
    "rotation": "10 MB",
    "retention": "7 days"
} 