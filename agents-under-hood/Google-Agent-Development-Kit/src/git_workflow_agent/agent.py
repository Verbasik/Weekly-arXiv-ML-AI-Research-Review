"""
Git Workflow Agent

Основной агент для автоматического анализа изменений и создания коммитов с использованием Google ADK.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService

from loguru import logger

from .config import AGENT_CONFIG, MCP_CONFIG, GIT_CONFIG
from utils.file_utils import ChangeDetector, FileAnalyzer
from utils.git_utils import GitManager, CommitBuilder


class GitWorkflowAgent:
    """Агент для автоматического Git workflow"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.git_manager = GitManager(str(self.project_path))
        self.commit_builder = CommitBuilder(self.git_manager)
        self.change_detector = ChangeDetector(GIT_CONFIG)
        self.file_analyzer = FileAnalyzer(GIT_CONFIG)
        
        # Инициализация ADK компонентов
        self.session_service = InMemorySessionService()
        self.artifact_service = InMemoryArtifactService()
        self.mcp_toolset = None
        self.agent = None
        self.runner = None
        
    async def initialize(self):
        """Инициализирует агента"""
        try:
            # Временно отключаем ADK для исправления ошибок
            # TODO: Восстановить ADK интеграцию после исправления API
            self.agent = None
            self.runner = None
            
            logger.info(f"Git Workflow Agent инициализирован (упрощенный режим) для проекта: {self.project_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при инициализации агента: {e}")
            raise
    
    async def analyze_project_changes(self) -> Dict[str, Any]:
        """Анализирует изменения в проекте"""
        try:
            # Получаем статус Git
            git_status = self.git_manager.get_status()
            
            if not git_status["is_repo"]:
                return {
                    "error": "Директория не является Git репозиторием",
                    "status": git_status
                }
            
            # Получаем измененные файлы
            changes = self.change_detector.get_changed_files(
                str(self.project_path), 
                self.git_manager
            )
            
            if not changes:
                return {
                    "message": "Нет изменений для анализа",
                    "status": git_status,
                    "changes": []
                }
            
            # Подготавливаем данные для анализа
            analysis_data = self._prepare_analysis_data(changes, git_status)
            
            # Создаем запрос к агенту
            prompt = f"""
            Проанализируй следующие изменения в проекте и создай качественное сообщение коммита:
            
            Статус Git:
            - Ветка: {git_status.get('branch', 'N/A')}
            - Файлов в индексе: {len(git_status.get('staged_files', []))}
            - Измененных файлов: {len(git_status.get('modified_files', []))}
            - Новых файлов: {len(git_status.get('untracked_files', []))}
            
            Изменения:
            {analysis_data}
            
            Создай сообщение коммита в формате Conventional Commits с подробным описанием изменений.
            """
            
            # Запускаем анализ через агента
            # Упрощенный подход - используем fallback анализ
            result = f"Анализ изменений: {len(changes)} файлов изменено"
            if changes:
                file_names = [c.file_path for c in changes[:3]]
                result += f". Файлы: {', '.join(file_names)}"
                if len(changes) > 3:
                    result += f" и еще {len(changes) - 3} файлов"
                
                # Добавляем информацию о типах изменений
                change_types = {}
                languages = {}
                for change in changes:
                    change_types[change.change_type] = change_types.get(change.change_type, 0) + 1
                    if change.language:
                        languages[change.language] = languages.get(change.language, 0) + 1
                
                if change_types:
                    result += f". Типы изменений: {', '.join([f'{k}({v})' for k, v in change_types.items()])}"
                if languages:
                    result += f". Языки: {', '.join([f'{k}({v})' for k, v in languages.items()])}"
            
            # TODO: Восстановить ADK интеграцию после исправления API
            logger.info("Используется упрощенный анализ (ADK временно отключен)")
            
            # Создаем результат анализа
            summary = self.change_detector.create_analysis_summary(changes)
            
            return {
                "status": git_status,
                "changes": [self._change_to_dict(change) for change in changes],
                "analysis": result,
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ошибка при анализе изменений: {e}")
            return {"error": str(e)}
    
    def _prepare_analysis_data(self, changes: List[Any], git_status: Dict[str, Any]) -> str:
        """Подготавливает данные для анализа"""
        analysis_lines = []
        
        for change in changes:
            if hasattr(change, 'file_path'):
                file_info = f"Файл: {change.file_path}"
                if hasattr(change, 'language') and change.language:
                    file_info += f" (Язык: {change.language})"
                if hasattr(change, 'change_type'):
                    file_info += f" (Тип: {change.change_type})"
                
                analysis_lines.append(file_info)
                
                # Добавляем diff если есть
                if hasattr(change, 'diff') and change.diff:
                    analysis_lines.append("Изменения:")
                    analysis_lines.append(change.diff)
                    analysis_lines.append("---")
        
        return "\n".join(analysis_lines)
    
    def _change_to_dict(self, change: Any) -> Dict[str, Any]:
        """Конвертирует объект изменения в словарь"""
        return {
            "file_path": getattr(change, 'file_path', ''),
            "change_type": getattr(change, 'change_type', ''),
            "language": getattr(change, 'language', ''),
            "file_size": getattr(change, 'file_size', 0),
            "last_modified": getattr(change, 'last_modified', '').isoformat() if getattr(change, 'last_modified', None) else None
        }
    
    async def create_commit(self, message: str, auto_push: bool = False) -> Dict[str, Any]:
        """Создает коммит с указанным сообщением"""
        try:
            # Получаем изменения
            changes = self.change_detector.get_changed_files(
                str(self.project_path), 
                self.git_manager
            )
            
            if not changes:
                return {"error": "Нет изменений для коммита"}
            
            # Создаем коммит
            success = self.commit_builder.create_commit(changes, message)
            
            if not success:
                return {"error": "Не удалось создать коммит"}
            
            # Пушим если нужно
            if auto_push:
                push_success = self.git_manager.push()
                if not push_success:
                    return {
                        "warning": "Коммит создан, но не удалось отправить изменения",
                        "commit_created": True
                    }
            
            return {
                "success": True,
                "message": "Коммит создан успешно",
                "commit_message": message,
                "pushed": auto_push
            }
            
        except Exception as e:
            logger.error(f"Ошибка при создании коммита: {e}")
            return {"error": str(e)}
    
    async def auto_commit_and_push(self) -> Dict[str, Any]:
        """Автоматически анализирует изменения, создает коммит и пушит"""
        try:
            # Анализируем изменения
            analysis_result = await self.analyze_project_changes()
            
            if "error" in analysis_result:
                return analysis_result
            
            if not analysis_result.get("changes"):
                return {"message": "Нет изменений для коммита"}
            
            # Получаем предложенное сообщение коммита
            commit_message = analysis_result.get("analysis", "")
            
            # Создаем коммит и пушим
            result = await self.create_commit(commit_message, auto_push=True)
            
            if result.get("success"):
                result["analysis"] = analysis_result
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при автоматическом коммите: {e}")
            return {"error": str(e)}
    
    async def get_project_status(self) -> Dict[str, Any]:
        """Получает текущий статус проекта"""
        try:
            git_status = self.git_manager.get_status()
            recent_commits = self.git_manager.get_recent_commits(5)
            
            return {
                "project_path": str(self.project_path),
                "git_status": git_status,
                "recent_commits": [
                    {
                        "hash": commit.hash,
                        "author": commit.author,
                        "date": commit.date.isoformat(),
                        "message": commit.message
                    }
                    for commit in recent_commits
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ошибка при получении статуса проекта: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Очищает ресурсы агента"""
        try:
            logger.info("Git Workflow Agent очищен")
        except Exception as e:
            logger.error(f"Ошибка при очистке агента: {e}")


# Функция для создания экземпляра агента
async def create_git_workflow_agent(project_path: str) -> GitWorkflowAgent:
    """Создает и инициализирует Git Workflow Agent"""
    agent = GitWorkflowAgent(project_path)
    await agent.initialize()
    return agent 