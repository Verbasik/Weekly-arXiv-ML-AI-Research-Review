"""
Утилиты для работы с файлами и анализа изменений

Функции для сканирования файлов, определения изменений и анализа контента.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import difflib

from loguru import logger


@dataclass
class FileChange:
    """Информация об изменении файла"""
    file_path: str
    change_type: str  # 'added', 'modified', 'deleted'
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    diff: Optional[str] = None
    file_size: int = 0
    last_modified: Optional[datetime] = None
    language: Optional[str] = None


@dataclass
class AnalysisResult:
    """Результат анализа изменений"""
    project_path: str
    changes: List[FileChange]
    summary: Dict[str, Any]
    timestamp: datetime
    total_files: int
    changed_files: int


class FileAnalyzer:
    """Анализатор файлов и изменений"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_extensions = config.get("supported_extensions", [])
        self.ignore_patterns = config.get("ignore_patterns", [])
        self.max_file_size = config.get("max_file_size", 1024 * 1024)
        
    def get_file_language(self, file_path: str) -> Optional[str]:
        """Определяет язык программирования по расширению файла"""
        extension_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'React JSX',
            '.tsx': 'React TSX',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.clj': 'Clojure',
            '.hs': 'Haskell',
            '.md': 'Markdown',
            '.txt': 'Text',
            '.json': 'JSON',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.toml': 'TOML',
            '.ini': 'INI',
            '.cfg': 'Config'
        }
        
        ext = Path(file_path).suffix.lower()
        return extension_map.get(ext)
    
    def should_ignore_file(self, file_path: str) -> bool:
        """Проверяет, нужно ли игнорировать файл"""
        path = Path(file_path)
        
        # Проверка по паттернам игнорирования
        for pattern in self.ignore_patterns:
            if pattern in path.parts:
                return True
            if path.match(pattern):
                return True
        
        # Проверка размера файла
        try:
            if path.stat().st_size > self.max_file_size:
                logger.warning(f"Файл {file_path} слишком большой, пропускаем")
                return True
        except OSError:
            logger.warning(f"Не удалось получить размер файла {file_path}")
            return True
        
        return False
    
    def is_supported_file(self, file_path: str) -> bool:
        """Проверяет, поддерживается ли тип файла"""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions
    
    def read_file_content(self, file_path: str) -> Optional[str]:
        """Читает содержимое файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Не удалось прочитать файл {file_path}: {e}")
                return None
        except Exception as e:
            logger.error(f"Ошибка при чтении файла {file_path}: {e}")
            return None
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Получает информацию о файле"""
        try:
            stat = Path(file_path).stat()
            return {
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "language": self.get_file_language(file_path),
                "exists": True
            }
        except OSError as e:
            logger.error(f"Ошибка при получении информации о файле {file_path}: {e}")
            return {"exists": False}


class ChangeDetector:
    """Детектор изменений в файлах"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.context_lines = config.get("context_lines", 5)
        
    def calculate_file_hash(self, content: str) -> str:
        """Вычисляет хеш содержимого файла"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def generate_diff(self, old_content: str, new_content: str) -> str:
        """Генерирует diff между двумя версиями файла"""
        try:
            diff = difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                lineterm='',
                n=self.context_lines
            )
            return ''.join(diff)
        except Exception as e:
            logger.error(f"Ошибка при генерации diff: {e}")
            return ""
    
    def analyze_changes(self, file_path: str, old_content: Optional[str], 
                       new_content: Optional[str]) -> FileChange:
        """Анализирует изменения в файле"""
        analyzer = FileAnalyzer(self.config)
        file_info = analyzer.get_file_info(file_path)
        
        if old_content is None and new_content is not None:
            change_type = "added"
        elif old_content is not None and new_content is None:
            change_type = "deleted"
        else:
            change_type = "modified"
        
        diff = None
        if old_content and new_content:
            diff = self.generate_diff(old_content, new_content)
        
        return FileChange(
            file_path=file_path,
            change_type=change_type,
            old_content=old_content,
            new_content=new_content,
            diff=diff,
            file_size=file_info.get("size", 0),
            last_modified=file_info.get("modified"),
            language=file_info.get("language")
        )
    
    def scan_directory(self, directory_path: str) -> List[str]:
        """Сканирует директорию и возвращает список файлов"""
        analyzer = FileAnalyzer(self.config)
        files = []
        
        try:
            for root, dirs, filenames in os.walk(directory_path):
                # Фильтруем директории для игнорирования
                dirs[:] = [d for d in dirs if not analyzer.should_ignore_file(os.path.join(root, d))]
                
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    
                    if analyzer.should_ignore_file(file_path):
                        continue
                    
                    if analyzer.is_supported_file(file_path):
                        files.append(file_path)
                        
        except Exception as e:
            logger.error(f"Ошибка при сканировании директории {directory_path}: {e}")
        
        return files
    
    def get_changed_files(self, project_path: str, git_manager) -> List[FileChange]:
        """Получает список измененных файлов через Git"""
        try:
            # Получаем статус Git
            status = git_manager.get_status()
            changes = []
            
            if not status.get("is_repo", False):
                logger.warning(f"Директория {project_path} не является Git репозиторием")
                return changes
            
            # Обрабатываем staged файлы
            for file_path in status.get("staged_files", []):
                content = git_manager.get_file_content(file_path)
                change = FileChange(
                    file_path=file_path,
                    change_type="staged",
                    new_content=content,
                    language=FileAnalyzer(self.config).get_file_language(file_path)
                )
                changes.append(change)
            
            # Обрабатываем modified файлы
            for file_path in status.get("modified_files", []):
                content = git_manager.get_file_content(file_path)
                change = FileChange(
                    file_path=file_path,
                    change_type="modified",
                    new_content=content,
                    language=FileAnalyzer(self.config).get_file_language(file_path)
                )
                changes.append(change)
            
            # Обрабатываем untracked файлы
            for file_path in status.get("untracked_files", []):
                content = git_manager.get_file_content(file_path)
                change = FileChange(
                    file_path=file_path,
                    change_type="added",
                    new_content=content,
                    language=FileAnalyzer(self.config).get_file_language(file_path)
                )
                changes.append(change)
            
            return changes
            
        except Exception as e:
            logger.error(f"Ошибка при получении измененных файлов: {e}")
            return []
    
    def create_analysis_summary(self, changes: List[FileChange]) -> Dict[str, Any]:
        """Создает сводку анализа изменений"""
        summary = {
            "total_changes": len(changes),
            "by_type": {},
            "by_language": {},
            "largest_changes": [],
            "recent_changes": []
        }
        
        # Группировка по типу изменений
        for change in changes:
            change_type = change.change_type
            summary["by_type"][change_type] = summary["by_type"].get(change_type, 0) + 1
            
            if change.language:
                summary["by_language"][change.language] = summary["by_language"].get(change.language, 0) + 1
        
        # Самые большие изменения
        sorted_by_size = sorted(changes, key=lambda x: x.file_size, reverse=True)
        summary["largest_changes"] = [
            {"file": c.file_path, "size": c.file_size} 
            for c in sorted_by_size[:5]
        ]
        
        # Последние изменения
        sorted_by_time = sorted(
            [c for c in changes if c.last_modified], 
            key=lambda x: x.last_modified, 
            reverse=True
        )
        summary["recent_changes"] = [
            {"file": c.file_path, "modified": c.last_modified.isoformat()}
            for c in sorted_by_time[:5]
        ]
        
        return summary 