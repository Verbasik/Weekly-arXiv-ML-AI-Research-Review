# 🚀 Быстрый старт Google ADK Git Workflow Agent

## 1. Настройка окружения

### Автоматическая настройка (рекомендуется)
```bash
./setup.sh
```

### Ручная настройка
```bash
# Создание виртуального окружения
python3 -m venv .venv

# Активация
source .venv/bin/activate  # macOS/Linux
# или
.venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt

# Установка UI зависимостей
cd ui && npm install && cd ..
```

## 2. Настройка API ключа

### Получение Google AI API ключа
1. Перейдите на [Google AI Studio](https://aistudio.google.com/)
2. Создайте новый API ключ
3. Скопируйте ключ

### Настройка .env файла
Отредактируйте файл `.env`:
```bash
# Замените на ваш реальный API ключ
GOOGLE_API_KEY=your_actual_google_ai_api_key_here
```

## 3. Запуск приложения

### Запуск backend
```bash
# Активируйте виртуальное окружение
source .venv/bin/activate

# Запустите API сервер
cd src
python main.py server
```

API будет доступен по адресу: http://localhost:8000

### Запуск frontend (в новом терминале)
```bash
cd ui
npm run dev
```

Веб-интерфейс будет доступен по адресу: http://localhost:3000

## 4. Использование

1. Откройте http://localhost:3000
2. Введите путь к Git репозиторию
3. Нажмите "Анализировать изменения"
4. Просмотрите предложенный коммит
5. Нажмите "Авто-коммит и пуш"

## 5. CLI использование

```bash
# Анализ изменений
python main.py /path/to/project analyze

# Автоматический коммит
python main.py /path/to/project auto-commit
```

## 🔧 Устранение неполадок

### Ошибка "Не установлена переменная GOOGLE_API_KEY"
- Проверьте файл `.env`
- Убедитесь, что API ключ указан правильно

### Ошибка "Директория не является Git репозиторием"
```bash
cd /path/to/project
git init
git remote add origin <repository_url>
```

### Ошибка портов
```bash
# Проверьте занятые порты
lsof -i :8000
lsof -i :3000
```

## 📚 Дополнительная документация

- [Подробная настройка](SETUP.md)
- [API документация](http://localhost:8000/docs)
- [Основная документация](README.md) 