#!/bin/bash

# Google ADK Git Workflow Agent - Setup Script
echo "🚀 Настройка Google ADK Git Workflow Agent..."

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.9+"
    exit 1
fi

# Проверяем версию Python
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Требуется Python 3.9+, найдена версия $python_version"
    exit 1
fi

echo "✅ Python $python_version найден"

# Создаем виртуальное окружение если его нет
if [ ! -d ".venv" ]; then
    echo "📦 Создание виртуального окружения..."
    python3 -m venv .venv
    echo "✅ Виртуальное окружение создано"
else
    echo "✅ Виртуальное окружение уже существует"
fi

# Активируем виртуальное окружение
echo "🔧 Активация виртуального окружения..."
source .venv/bin/activate

# Обновляем pip
echo "⬆️ Обновление pip..."
pip install --upgrade pip

# Устанавливаем зависимости
echo "📚 Установка Python зависимостей..."
pip install -r requirements.txt

# Проверяем наличие .env файла
if [ ! -f ".env" ]; then
    echo "⚠️ Файл .env не найден. Создайте его с вашим Google AI API ключом"
    echo "Пример содержимого .env:"
    echo "GOOGLE_API_KEY=your_google_ai_api_key_here"
    echo "GOOGLE_GENAI_USE_VERTEXAI=FALSE"
else
    echo "✅ Файл .env найден"
fi

# Проверяем наличие Node.js
if ! command -v node &> /dev/null; then
    echo "⚠️ Node.js не найден. Установите Node.js 18+ для веб-интерфейса"
else
    echo "✅ Node.js найден"
    
    # Устанавливаем зависимости для UI
    if [ -d "ui" ]; then
        echo "📚 Установка Node.js зависимостей..."
        cd ui
        npm install
        cd ..
        echo "✅ Node.js зависимости установлены"
    fi
fi

# Создаем директорию для логов
mkdir -p logs

echo ""
echo "🎉 Настройка завершена!"
echo ""
echo "📋 Следующие шаги:"
echo "1. Отредактируйте файл .env и добавьте ваш Google AI API ключ"
echo "2. Активируйте виртуальное окружение: source .venv/bin/activate"
echo "3. Запустите backend: cd src && python main.py server"
echo "4. Запустите frontend: cd ui && npm run dev"
echo ""
echo "📖 Подробные инструкции: SETUP.md" 