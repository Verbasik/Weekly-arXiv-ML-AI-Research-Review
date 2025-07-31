#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправлений ошибок
"""

import asyncio
import os
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

from utils.file_utils import ChangeDetector
from utils.git_utils import GitManager
from git_workflow_agent.config import GIT_CONFIG

async def test_git_status_fix():
    """Тестирует исправление ошибки в get_changed_files"""
    print("🔧 Тестирование исправления ошибки в get_changed_files...")
    
    try:
        # Создаем GitManager для текущей директории
        current_dir = Path.cwd()
        git_manager = GitManager(str(current_dir))
        
        # Получаем статус Git
        status = git_manager.get_status()
        print(f"✅ Статус Git получен: {status['is_repo']}")
        
        # Создаем ChangeDetector
        change_detector = ChangeDetector(GIT_CONFIG)
        
        # Тестируем get_changed_files
        changes = change_detector.get_changed_files(str(current_dir), git_manager)
        print(f"✅ Изменения получены: {len(changes)} файлов")
        
        for change in changes:
            print(f"  - {change.file_path} ({change.change_type})")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

async def test_date_parsing_fix():
    """Тестирует исправление парсинга дат"""
    print("\n📅 Тестирование исправления парсинга дат...")
    
    try:
        # Создаем GitManager для текущей директории
        current_dir = Path.cwd()
        git_manager = GitManager(str(current_dir))
        
        # Получаем последние коммиты
        commits = git_manager.get_recent_commits(5)
        print(f"✅ Коммиты получены: {len(commits)}")
        
        for commit in commits:
            print(f"  - {commit.hash[:8]}: {commit.message[:50]}... ({commit.date})")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

async def test_analysis_summary():
    """Тестирует создание сводки анализа"""
    print("\n📊 Тестирование создания сводки анализа...")
    
    try:
        # Создаем ChangeDetector
        change_detector = ChangeDetector(GIT_CONFIG)
        
        # Создаем тестовые изменения
        from utils.file_utils import FileChange
        from datetime import datetime
        
        test_changes = [
            FileChange(
                file_path="test1.py",
                change_type="modified",
                language="Python",
                file_size=1024
            ),
            FileChange(
                file_path="test2.js",
                change_type="added",
                language="JavaScript",
                file_size=512
            ),
            FileChange(
                file_path="test3.txt",
                change_type="deleted",
                language="Text",
                file_size=256
            )
        ]
        
        # Создаем сводку
        summary = change_detector.create_analysis_summary(test_changes)
        print(f"✅ Сводка создана:")
        print(f"  - Всего изменений: {summary['total_changes']}")
        print(f"  - По типам: {summary['by_type']}")
        print(f"  - По языкам: {summary['by_language']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

async def main():
    """Основная функция тестирования"""
    print("🚀 Тестирование исправлений ошибок...")
    print("=" * 50)
    
    # Проверяем API ключ
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️ Не найден GOOGLE_API_KEY в переменных окружения")
    else:
        print(f"✅ API ключ найден: {api_key[:10]}...")
    
    # Запускаем тесты
    tests = [
        test_git_status_fix,
        test_date_parsing_fix,
        test_analysis_summary
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
    print("📊 Результаты тестирования исправлений:")
    
    test_names = ["Git Status Fix", "Date Parsing Fix", "Analysis Summary"]
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ УСПЕХ" if result else "❌ ОШИБКА"
        print(f"{i+1}. {name}: {status}")
    
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n🎯 Итого: {success_count}/{total_count} тестов прошли успешно")
    
    if success_count == total_count:
        print("🎉 Все исправления работают корректно!")
    else:
        print("⚠️ Некоторые исправления требуют доработки.")

if __name__ == "__main__":
    asyncio.run(main()) 