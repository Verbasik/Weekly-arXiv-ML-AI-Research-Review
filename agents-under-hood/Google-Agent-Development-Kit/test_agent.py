#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы Google ADK агентов
"""

import asyncio
import os
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from loguru import logger

async def test_basic_agent():
    """Тестирует базовую работу агента"""
    print("🤖 Тестирование базового агента...")
    
    try:
        # Создаем простого агента
        agent = Agent(
            model="gemini-2.0-flash",
            name="test_agent",
            instruction="Ты - полезный ассистент. Отвечай кратко и по делу."
        )
        
        # Создаем session service
        session_service = InMemorySessionService()
        
        # Создаем runner
        runner = Runner(
            app_name="test_app",
            agent=agent,
            session_service=session_service
        )
        
        # Создаем сессию
        session = await session_service.create_session(
            state={}, 
            app_name="test_app", 
            user_id="test_user"
        )
        
        # Создаем сообщение
        message = Content(
            role='user', 
            parts=[Part(text="Привет! Как дела?")]
        )
        
        print(f"📝 Запрос: Привет! Как дела?")
        
        # Запускаем агента
        events = runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=message
        )
        
        result = ""
        async for event in events:
            if hasattr(event, 'content') and event.content:
                for part in event.content.parts:
                    if part.text:
                        result += part.text
        
        print(f"🤖 Ответ: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

async def test_git_analysis_agent():
    """Тестирует агента для анализа Git изменений"""
    print("\n🔍 Тестирование агента анализа Git...")
    
    try:
        # Создаем агента для анализа Git
        agent = Agent(
            model="gemini-2.0-flash",
            name="git_analyzer",
            instruction="""
            Ты - эксперт по анализу изменений в коде и созданию качественных коммитов.
            
            Твоя задача:
            1. Анализировать изменения в файлах
            2. Создавать информативные сообщения коммитов
            3. Следовать формату Conventional Commits
            
            Типы коммитов:
            - feat: новая функциональность
            - fix: исправление ошибок
            - docs: изменения в документации
            - style: форматирование кода
            - refactor: рефакторинг
            - test: добавление тестов
            - chore: обновление зависимостей
            """
        )
        
        # Создаем session service
        session_service = InMemorySessionService()
        
        # Создаем runner
        runner = Runner(
            app_name="git_analysis_app",
            agent=agent,
            session_service=session_service
        )
        
        # Создаем сессию
        session = await session_service.create_session(
            state={}, 
            app_name="git_analysis_app", 
            user_id="test_user"
        )
        
        # Тестируем анализ изменений
        test_changes = """
        Изменения в проекте:
        - Добавлен новый файл: src/utils/helper.py
        - Изменен файл: src/main.py (добавлена функция validate_input)
        - Обновлен файл: requirements.txt (добавлена зависимость requests)
        
        Создай качественное сообщение коммита в формате Conventional Commits.
        """
        
        # Создаем сообщение
        message = Content(
            role='user', 
            parts=[Part(text=test_changes)]
        )
        
        print(f"📝 Запрос: {test_changes}")
        
        # Запускаем агента
        events = runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=message
        )
        
        result = ""
        async for event in events:
            if hasattr(event, 'content') and event.content:
                for part in event.content.parts:
                    if part.text:
                        result += part.text
        
        print(f"🤖 Анализ: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

async def test_code_review_agent():
    """Тестирует агента для code review"""
    print("\n🔍 Тестирование агента code review...")
    
    try:
        # Создаем агента для code review
        agent = Agent(
            model="gemini-2.0-flash",
            name="code_reviewer",
            instruction="""
            Ты - опытный code reviewer. Анализируй код и давай конструктивную обратную связь.
            
            Проверяй:
            1. Качество кода и читаемость
            2. Потенциальные ошибки и баги
            3. Производительность
            4. Безопасность
            5. Соответствие лучшим практикам
            
            Давай конкретные рекомендации по улучшению.
            """
        )
        
        # Создаем session service
        session_service = InMemorySessionService()
        
        # Создаем runner
        runner = Runner(
            app_name="code_review_app",
            agent=agent,
            session_service=session_service
        )
        
        # Создаем сессию
        session = await session_service.create_session(
            state={}, 
            app_name="code_review_app", 
            user_id="test_user"
        )
        
        # Тестируем code review
        test_code = """
        def calculate_fibonacci(n):
            if n <= 1:
                return n
            return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
        
        Проанализируй этот код и дай рекомендации по улучшению.
        """
        
        # Создаем сообщение
        message = Content(
            role='user', 
            parts=[Part(text=test_code)]
        )
        
        print(f"📝 Код для анализа: {test_code}")
        
        # Запускаем агента
        events = runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=message
        )
        
        result = ""
        async for event in events:
            if hasattr(event, 'content') and event.content:
                for part in event.content.parts:
                    if part.text:
                        result += part.text
        
        print(f"🤖 Code Review: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

async def main():
    """Основная функция тестирования"""
    print("🚀 Запуск тестирования Google ADK агентов...")
    print("=" * 50)
    
    # Проверяем API ключ
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Не найден GOOGLE_API_KEY в переменных окружения")
        return
    
    print(f"✅ API ключ найден: {api_key[:10]}...")
    
    # Запускаем тесты
    tests = [
        test_basic_agent,
        test_git_analysis_agent,
        test_code_review_agent
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Ошибка в тесте {test.__name__}: {e}")
            results.append(False)
    
    # Выводим результаты
    print("\n" + "=" * 50)
    print("📊 Результаты тестирования:")
    
    test_names = ["Базовый агент", "Git анализ", "Code Review"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ УСПЕХ" if result else "❌ ОШИБКА"
        print(f"{i+1}. {name}: {status}")
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n🎯 Итого: {success_count}/{total_count} тестов прошли успешно")
    
    if success_count == total_count:
        print("🎉 Все тесты прошли успешно! Google ADK агенты работают корректно.")
    else:
        print("⚠️ Некоторые тесты не прошли. Проверьте настройки и логи.")

if __name__ == "__main__":
    asyncio.run(main()) 