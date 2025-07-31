"""
Утилиты для работы с Git

Функции для управления Git репозиторием, создания коммитов и анализа изменений.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

from loguru import logger


@dataclass
class GitStatus:
    """Статус Git репозитория"""
    is_repo: bool
    branch: Optional[str] = None
    remote: Optional[str] = None
    ahead: int = 0
    behind: int = 0
    staged_files: List[str] = None
    modified_files: List[str] = None
    untracked_files: List[str] = None


@dataclass
class CommitInfo:
    """Информация о коммите"""
    hash: str
    author: str
    date: datetime
    message: str
    files_changed: List[str]


class GitManager:
    """Менеджер для работы с Git репозиторием"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self._validate_repo()
    
    def _validate_repo(self) -> bool:
        """Проверяет, является ли директория Git репозиторием"""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            logger.warning(f"Директория {self.repo_path} не является Git репозиторием")
            return False
        return True
    
    def _run_git_command(self, command: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
        """Выполняет Git команду"""
        try:
            result = subprocess.run(
                ["git"] + command,
                cwd=cwd or str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout при выполнении Git команды: {' '.join(command)}")
            return -1, "", "Command timeout"
        except Exception as e:
            logger.error(f"Ошибка при выполнении Git команды: {e}")
            return -1, "", str(e)
    
    def get_status(self) -> Dict[str, Any]:
        """Получает статус Git репозитория"""
        status = {
            "is_repo": self._validate_repo(),
            "branch": None,
            "remote": None,
            "ahead": 0,
            "behind": 0,
            "staged_files": [],
            "modified_files": [],
            "untracked_files": []
        }
        
        if not status["is_repo"]:
            return status
        
        # Получаем текущую ветку
        code, stdout, stderr = self._run_git_command(["branch", "--show-current"])
        if code == 0:
            status["branch"] = stdout.strip()
        
        # Получаем удаленный репозиторий
        code, stdout, stderr = self._run_git_command(["remote", "get-url", "origin"])
        if code == 0:
            status["remote"] = stdout.strip()
        
        # Получаем статус файлов
        code, stdout, stderr = self._run_git_command(["status", "--porcelain"])
        if code == 0:
            for line in stdout.strip().split('\n'):
                if line:
                    status_code = line[:2]
                    file_path = line[3:]
                    
                    if status_code.startswith('A'):  # Added
                        status["staged_files"].append(file_path)
                    elif status_code.startswith('M'):  # Modified
                        if status_code == 'M ':  # Modified, not staged
                            status["modified_files"].append(file_path)
                        else:  # Modified, staged
                            status["staged_files"].append(file_path)
                    elif status_code.startswith('??'):  # Untracked
                        status["untracked_files"].append(file_path)
        
        # Получаем информацию о ahead/behind
        if status["branch"] and status["remote"]:
            code, stdout, stderr = self._run_git_command([
                "rev-list", "--count", "--left-right", 
                f"origin/{status['branch']}...{status['branch']}"
            ])
            if code == 0:
                parts = stdout.strip().split('\t')
                if len(parts) == 2:
                    status["behind"] = int(parts[0])
                    status["ahead"] = int(parts[1])
        
        return status
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """Получает содержимое файла из Git"""
        try:
            # Проверяем, что это файл, а не директория
            full_path = self.repo_path / file_path
            if full_path.exists() and full_path.is_file():
                # Сначала пробуем получить из Git
                code, stdout, stderr = self._run_git_command(["show", f"HEAD:{file_path}"])
                if code == 0:
                    return stdout
                else:
                    # Файл может быть новым или не отслеживаться
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            return f.read()
                    except UnicodeDecodeError:
                        # Пробуем другие кодировки
                        try:
                            with open(full_path, 'r', encoding='latin-1') as f:
                                return f.read()
                        except Exception:
                            logger.warning(f"Не удалось прочитать файл {file_path} (бинарный файл?)")
                            return None
            else:
                logger.warning(f"Путь {file_path} не является файлом или не существует")
                return None
        except Exception as e:
            logger.error(f"Ошибка при получении содержимого файла {file_path}: {e}")
        
        return None
    
    def add_file(self, file_path: str) -> bool:
        """Добавляет файл в индекс Git"""
        code, stdout, stderr = self._run_git_command(["add", file_path])
        if code == 0:
            logger.info(f"Файл {file_path} добавлен в индекс")
            return True
        else:
            logger.error(f"Ошибка при добавлении файла {file_path}: {stderr}")
            return False
    
    def add_all_files(self) -> bool:
        """Добавляет все измененные файлы в индекс"""
        code, stdout, stderr = self._run_git_command(["add", "."])
        if code == 0:
            logger.info("Все файлы добавлены в индекс")
            return True
        else:
            logger.error(f"Ошибка при добавлении файлов: {stderr}")
            return False
    
    def commit(self, message: str, author: Optional[str] = None) -> bool:
        """Создает коммит с указанным сообщением"""
        command = ["commit", "-m", message]
        
        if author:
            command.extend(["--author", author])
        
        code, stdout, stderr = self._run_git_command(command)
        if code == 0:
            logger.info(f"Коммит создан: {message}")
            return True
        else:
            logger.error(f"Ошибка при создании коммита: {stderr}")
            return False
    
    def push(self, remote: str = "origin", branch: Optional[str] = None) -> bool:
        """Пушит изменения в удаленный репозиторий"""
        if not branch:
            status = self.get_status()
            branch = status.get("branch", "main")
        
        code, stdout, stderr = self._run_git_command(["push", remote, branch])
        if code == 0:
            logger.info(f"Изменения отправлены в {remote}/{branch}")
            return True
        else:
            logger.error(f"Ошибка при отправке изменений: {stderr}")
            return False
    
    def get_recent_commits(self, count: int = 10) -> List[CommitInfo]:
        """Получает список последних коммитов"""
        code, stdout, stderr = self._run_git_command([
            "log", f"-{count}", "--pretty=format:%H|%an|%ad|%s", "--date=iso"
        ])
        
        commits = []
        if code == 0:
            for line in stdout.strip().split('\n'):
                if line:
                    parts = line.split('|', 3)
                    if len(parts) == 4:
                        commit_hash, author, date_str, message = parts
                        try:
                            # Обрабатываем различные форматы дат от Git
                            if '+' in date_str:
                                # Формат: 2025-07-31 09:15:19 +0300
                                date_str_clean = date_str.split('+')[0].strip()
                                date = datetime.fromisoformat(date_str_clean)
                            else:
                                # Стандартный ISO формат
                                date = datetime.fromisoformat(date_str)
                            
                            commits.append(CommitInfo(
                                hash=commit_hash,
                                author=author,
                                date=date,
                                message=message,
                                files_changed=[]
                            ))
                        except ValueError:
                            logger.warning(f"Не удалось распарсить дату: {date_str}")
                            # Используем текущую дату как fallback
                            commits.append(CommitInfo(
                                hash=commit_hash,
                                author=author,
                                date=datetime.now(),
                                message=message,
                                files_changed=[]
                            ))
        
        return commits
    
    def get_commit_files(self, commit_hash: str) -> List[str]:
        """Получает список файлов, измененных в коммите"""
        code, stdout, stderr = self._run_git_command([
            "show", "--name-only", "--pretty=format:", commit_hash
        ])
        
        if code == 0:
            return [line.strip() for line in stdout.strip().split('\n') if line.strip()]
        return []


class CommitBuilder:
    """Строитель коммитов с AI-анализом"""
    
    def __init__(self, git_manager: GitManager):
        self.git_manager = git_manager
    
    def analyze_changes_for_commit(self, changes: List[Any]) -> Dict[str, Any]:
        """Анализирует изменения для создания коммита"""
        analysis = {
            "type": "feat",  # По умолчанию
            "scope": "general",
            "description": "",
            "detailed_description": "",
            "breaking_changes": "",
            "files_affected": len(changes),
            "languages": set(),
            "change_types": set()
        }
        
        # Анализируем изменения
        for change in changes:
            if hasattr(change, 'language') and change.language:
                analysis["languages"].add(change.language)
            
            if hasattr(change, 'change_type'):
                analysis["change_types"].add(change.change_type)
        
        # Определяем тип коммита
        if "deleted" in analysis["change_types"]:
            analysis["type"] = "refactor"
        elif len(analysis["languages"]) == 1 and "Python" in analysis["languages"]:
            analysis["scope"] = "python"
        elif len(analysis["languages"]) == 1 and "JavaScript" in analysis["languages"]:
            analysis["scope"] = "javascript"
        
        # Создаем описание
        if len(changes) == 1:
            change = changes[0]
            if hasattr(change, 'file_path'):
                filename = Path(change.file_path).name
                analysis["description"] = f"update {filename}"
        else:
            analysis["description"] = f"update {len(changes)} files"
        
        return analysis
    
    def build_commit_message(self, analysis: Dict[str, Any], 
                           custom_message: Optional[str] = None) -> str:
        """Строит сообщение коммита"""
        if custom_message:
            return custom_message
        
        # Базовая структура
        message_parts = [f"{analysis['type']}({analysis['scope']}): {analysis['description']}"]
        
        if analysis["detailed_description"]:
            message_parts.append("")
            message_parts.append(analysis["detailed_description"])
        
        if analysis["breaking_changes"]:
            message_parts.append("")
            message_parts.append("BREAKING CHANGE:")
            message_parts.append(analysis["breaking_changes"])
        
        return "\n".join(message_parts)
    
    def create_commit(self, changes: List[Any], message: Optional[str] = None,
                     author: Optional[str] = None) -> bool:
        """Создает коммит с анализом изменений"""
        try:
            # Добавляем все файлы
            if not self.git_manager.add_all_files():
                return False
            
            # Анализируем изменения
            analysis = self.analyze_changes_for_commit(changes)
            
            # Строим сообщение коммита
            commit_message = self.build_commit_message(analysis, message)
            
            # Создаем коммит
            return self.git_manager.commit(commit_message, author)
            
        except Exception as e:
            logger.error(f"Ошибка при создании коммита: {e}")
            return False
    
    def create_and_push_commit(self, changes: List[Any], message: Optional[str] = None,
                              author: Optional[str] = None) -> bool:
        """Создает коммит и пушит изменения"""
        if self.create_commit(changes, message, author):
            return self.git_manager.push()
        return False 