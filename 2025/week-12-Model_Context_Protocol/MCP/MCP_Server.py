#!/usr/bin/env python3
"""
MCP Git Local Server - Model Context Protocol Server для работы с локальными Git репозиториями

Эта версия позволяет:
1. Просматривать git status локальных репозиториев
2. Просматривать git diff (что было изменено)
3. Создавать коммиты с помощью LLM для автоматической генерации сообщений
4. Выполнять git push
"""

# Стандартные библиотеки
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# Сторонние библиотеки
import git
from mcp.server.fastmcp import FastMCP

# Используем этот модуль место FastMCP, для того что бы развернуть полноценный сервер
# from fastmcp_http.server import FastMCPHttpServer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация FastMCP сервера
mcp = FastMCP("git-local")

class LocalGitRepoManager:
    """Менеджер для работы с локальными Git репозиториями"""

    def __init__(self):
        """Инициализация менеджера репозиториев"""
        self.known_repos: Dict[str, Dict[str, Any]] = {}  # path -> {repo, path}

    def register_repo(self, repo_path: str) -> None:
        """
        Description:
        ---------------
        Регистрирует локальный репозиторий в менеджере

        Args:
        ---------------
            repo_path: Путь к локальному репозиторию

        Raises:
        ---------------
            ValueError: Если путь не существует, не является директорией или не является Git репозиторием
        """
        try:
            repo_path = os.path.abspath(repo_path)

            # Проверяем, что путь существует
            if not os.path.exists(repo_path):
                raise ValueError(f"Путь не существует: {repo_path}")

            # Проверяем, что это директория
            if not os.path.isdir(repo_path):
                raise ValueError(f"Указанный путь не является директорией: {repo_path}")

            # Пытаемся открыть репозиторий
            try:
                repo = git.Repo(repo_path)
            except git.InvalidGitRepositoryError:
                raise ValueError(f"Указанный путь не является Git репозиторием: {repo_path}")
            except git.NoSuchPathError:
                raise ValueError(f"Путь не существует (NoSuchPathError): {repo_path}")

            # Проверяем работоспособность репозитория
            try:
                # Простая команда для проверки
                repo.git.rev_parse("--git-dir")
            except git.GitCommandError as e:
                raise ValueError(f"Репозиторий поврежден или заблокирован: {e}")

            self.known_repos[repo_path] = {
                "repo": repo,
                "path": repo_path
            }
            logger.info(f"Репозиторий зарегистрирован: {repo_path}")

        except Exception as e:
            logger.error(f"Ошибка при регистрации репозитория {repo_path}: {e}")
            raise ValueError(f"Ошибка при регистрации репозитория: {str(e)}")

    def get_repo(self, repo_path: str) -> git.Repo:
        """
        Description:
        ---------------
        Получает объект репозитория по пути

        Args:
        ---------------
            repo_path: Путь к локальному репозиторию

        Returns:
        ---------------
            Объект репозитория

        Raises:
        ---------------
            ValueError: Если репозиторий поврежден или заблокирован
        """
        repo_path = os.path.abspath(repo_path)

        if repo_path not in self.known_repos:
            # Если репозиторий еще не зарегистрирован, пробуем его зарегистрировать
            self.register_repo(repo_path)

        # Проверяем работоспособность репозитория
        repo = self.known_repos[repo_path]["repo"]
        try:
            # Простая команда для проверки
            repo.git.rev_parse("--git-dir")
        except git.GitCommandError as e:
            # Если возникла ошибка, пробуем переоткрыть репозиторий
            logger.warning(f"Ошибка при проверке репозитория, пробуем переоткрыть: {e}")
            try:
                repo = git.Repo(repo_path)
                self.known_repos[repo_path]["repo"] = repo
            except Exception as inner_e:
                logger.error(f"Ошибка при переоткрытии репозитория: {inner_e}")
                raise ValueError(f"Репозиторий поврежден или заблокирован: {str(e)}\nПереоткрытие не помогло: {str(inner_e)}")

        return repo

    def list_repositories(self) -> List[Dict[str, str]]:
        """
        Description:
        ---------------
        Возвращает список зарегистрированных репозиториев

        Returns:
        ---------------
            Список словарей с путями к репозиториям
        """
        return [
            {"path": path}
            for path in self.known_repos.keys()
        ]

# Создаем глобальный экземпляр менеджера репозиториев
repo_manager = LocalGitRepoManager()

@mcp.tool()
async def list_repositories() -> str:
    """
    Description:
    ---------------
        Возвращает список зарегистрированных локальных Git репозиториев.

    Returns:
    ---------------
        str: Строка с информацией о зарегистрированных репозиториях или сообщение об их отсутствии.

    Raises:
    ---------------
        None

    Examples:
    ---------------
        >>> await list_repositories()
        'Зарегистрированные репозитории: ...'
    """
    logger.info("Запрос списка репозиториев")

    # Получаем список репозиториев
    repos = repo_manager.list_repositories()

    # Проверяем наличие репозиториев
    if not repos:
        return "Нет зарегистрированных репозиториев. Используйте register_repository для добавления репозитория."

    # Формируем результат
    result = "Зарегистрированные репозитории:\n"
    for repo in repos:
        result += f"Путь: {repo['path']}\n"
        result += "-" * 50 + "\n"

    return result

@mcp.tool()
async def register_repository(repo_path: str) -> str:
    """
    Description:
    ---------------
        Регистрирует локальный Git репозиторий для работы.

    Args:
    ---------------
        repo_path (str): Полный путь к локальному Git репозиторию

    Returns:
    ---------------
        str: Сообщение об успешной регистрации или ошибке

    Raises:
    ---------------
        Exception: Ошибка при регистрации репозитория

    Examples:
    ---------------
        >>> await register_repository("/path/to/repo")
        'Репозиторий успешно зарегистрирован: /path/to/repo'
    """
    logger.info(f"Регистрация репозитория: {repo_path}")

    try:
        # Регистрация репозитория
        repo_manager.register_repo(repo_path)
        return f"Репозиторий успешно зарегистрирован: {repo_path}"
    except Exception as e:
        # Логирование ошибки
        logger.error(f"Ошибка при регистрации репозитория: {e}")
        return f"Ошибка при регистрации репозитория: {str(e)}"

@mcp.tool()
async def git_status(repo_path: str) -> str:
    """
    Description:
    ---------------
        Показывает статус рабочего дерева Git.

    Args:
    ---------------
        repo_path (str): Путь к локальному Git репозиторию

    Returns:
    ---------------
        str: Статус рабочего дерева или сообщение об ошибке

    Raises:
    ---------------
        Exception: Ошибка при получении статуса

    Examples:
    ---------------
        >>> await git_status("/path/to/repo")
        'Статус рабочего дерева...'
    """
    logger.info(f"Получение статуса для репозитория {repo_path}")

    try:
        # Получение репозитория
        repo = repo_manager.get_repo(repo_path)
        return repo.git.status()
    except Exception as e:
        # Логирование ошибки
        logger.error(f"Ошибка в git_status: {e}")
        return f"Ошибка при получении статуса: {str(e)}"

@mcp.tool()
async def git_diff(repo_path: str, file_path: Optional[str] = None) -> str:
    """
    Description:
    ---------------
        Показывает изменения в рабочем дереве Git.

    Args:
    ---------------
        repo_path (str): Путь к локальному Git репозиторию
        file_path (Optional[str]): Опциональный путь к файлу (если не указан, показывает все изменения)

    Returns:
    ---------------
        str: Изменения в рабочем дереве или сообщение об ошибке

    Raises:
    ---------------
        Exception: Ошибка при получении diff

    Examples:
    ---------------
        >>> await git_diff("/path/to/repo")
        'Изменения: ...'
    """
    logger.info(f"Получение diff для репозитория {repo_path}, файл: {file_path or 'все'}")

    try:
        repo = repo_manager.get_repo(repo_path)

        if file_path:
            # Полный путь к файлу
            abs_file_path = os.path.join(repo_path, file_path)
            # Относительный путь к файлу (относительно репозитория)
            rel_file_path = os.path.relpath(abs_file_path, repo_path)

            # Проверяем, существует ли файл
            if not os.path.exists(abs_file_path):
                return f"Файл не существует: {file_path}"

            diff_output = repo.git.diff(rel_file_path)
        else:
            diff_output = repo.git.diff()

        if not diff_output:
            return "Нет изменений"

        return f"Изменения:\n\n```diff\n{diff_output}\n```"
    except Exception as e:
        logger.error(f"Ошибка в git_diff: {e}")
        return f"Ошибка при получении diff: {str(e)}"

@mcp.tool()
async def git_add(repo_path: str, file_path: Optional[str] = None) -> str:
    """
    Description:
    ---------------
        Добавляет файлы в индекс Git.

    Args:
    ---------------
        repo_path (str): Путь к локальному Git репозиторию
        file_path (Optional[str]): Путь к файлу или шаблон для добавления (если не указан, добавляет все изменения)

    Returns:
    ---------------
        str: Сообщение о добавлении файлов в индекс или об ошибке

    Raises:
    ---------------
        Exception: Ошибка при добавлении файлов в индекс

    Examples:
    ---------------
        >>> await git_add("/path/to/repo")
        'Все изменения добавлены в индекс'
    """
    logger.info(f"Добавление файлов в индекс для репозитория {repo_path}, файл: {file_path or 'все'}")

    try:
        repo = repo_manager.get_repo(repo_path)

        if file_path:
            # Добавляем конкретный файл или по шаблону
            repo.git.add(file_path)
            return f"Файлы, соответствующие '{file_path}', добавлены в индекс"
        else:
            # Добавляем все изменения
            repo.git.add("--all")
            return "Все изменения добавлены в индекс"
    except Exception as e:
        logger.error(f"Ошибка в git_add: {e}")
        return f"Ошибка при добавлении файлов в индекс: {str(e)}"

@mcp.tool()
async def git_commit(
    repo_path: str,
    message_type: str,
    task_key: Optional[str] = None,
    snapshot: bool = False,
    description: str = "",
    add_all: bool = True
) -> str:
    """
    Description:
    ---------------
        Создает коммит в репозитории с форматированным сообщением.

    Args:
    ---------------
        repo_path (str): Путь к локальному Git репозиторию
        message_type (str): Тип сообщения (feat, fix, docs, style, refactor, test, chore, perf, build, ci)
        task_key (Optional[str]): Ключ задачи (например, AIAGENTS-3)
        snapshot (bool): Флаг для запуска snapshot билда
        description (str): Описание изменений
        add_all (bool): Флаг для автоматического добавления всех файлов в индекс

    Returns:
    ---------------
        str: Сообщение об успешном создании коммита или об ошибке

    Raises:
    ---------------
        Exception: Ошибка при создании коммита

    Examples:
    ---------------
        >>> await git_commit("/path/to/repo", "feat", "AIAGENTS-3", description="Initial commit")
        'Коммит успешно создан: ...'
    """
    logger.info(f"Создание коммита в репозитории {repo_path}")

    try:
        repo = repo_manager.get_repo(repo_path)

        # Проверяем, есть ли изменения
        if not repo.is_dirty() and not repo.untracked_files:
            return "Нет изменений для коммита"

        # Проверяем, есть ли изменения в индексе, только если не нужно добавлять все
        if not add_all and not any(diff.a_path for diff in repo.index.diff("HEAD")):
            return ("Нет проиндексированных изменений для коммита. "
                    "Используйте git_add для добавления файлов в индекс или установите add_all=True.")

        # Проверяем корректность типа сообщения
        valid_types = ['feat', 'fix', 'docs', 'style', 'refactor',
                       'test', 'chore', 'perf', 'build', 'ci']
        if message_type not in valid_types:
            types_str = ", ".join(valid_types)
            return f"Некорректный тип сообщения. Допустимые типы: {types_str}"

        # Формируем сообщение коммита
        commit_message = f"{message_type}({task_key}): " if task_key else f"{message_type}: "
        commit_message += description

        # Добавляем метку snapshot если нужно
        if snapshot:
            commit_message += " (snapshot)"

        # Индексируем все изменения если нужно
        if add_all:
            repo.git.add("--all")

        # Создаем коммит
        commit = repo.index.commit(commit_message)

        return f"Коммит успешно создан: {commit.hexsha}\nСообщение: {commit_message}"
    except Exception as e:
        logger.error(f"Ошибка в git_commit: {e}")
        return f"Ошибка при создании коммита: {str(e)}"

@mcp.tool()
async def git_push(repo_path: str, remote: str = "origin", branch: Optional[str] = None, force: bool = False) -> str:
    """
    Description:
    ---------------
        Отправляет изменения в удаленный репозиторий.

    Args:
    ---------------
        repo_path (str): Путь к локальному Git репозиторию
        remote (str): Имя удаленного репозитория (по умолчанию origin)
        branch (Optional[str]): Ветка для отправки (если не указана, используется текущая)
        force (bool): Флаг для принудительного push (--force)

    Returns:
    ---------------
        str: Сообщение об успешном выполнении push или об ошибке

    Raises:
    ---------------
        Exception: Ошибка при выполнении push

    Examples:
    ---------------
        >>> await git_push("/path/to/repo")
        'Push успешно выполнен в origin/main'
    """
    logger.info(f"Push для репозитория {repo_path} в {remote}")

    try:
        repo = repo_manager.get_repo(repo_path)

        # Если ветка не указана, используем текущую
        if not branch:
            try:
                branch = repo.active_branch.name
            except TypeError:
                return "Не удалось определить текущую ветку. Возможно, HEAD detached. Укажите ветку явно."

        # Проверяем, существует ли удаленный репозиторий
        remotes = [r.name for r in repo.remotes]
        if remote not in remotes:
            return f"Удаленный репозиторий '{remote}' не найден. Доступные: {', '.join(remotes)}"

        # Проверяем, есть ли неотправленные коммиты
        remote_ref = f"{remote}/{branch}"
        try:
            commits_behind = list(repo.iter_commits(f"{branch}..{remote_ref}"))
            commits_ahead = list(repo.iter_commits(f"{remote_ref}..{branch}"))

            if not commits_ahead and not force:
                return f"Нет изменений для отправки в {remote}/{branch}"

            if commits_behind and not force:
                return (f"Локальная ветка отстает от удаленной на {len(commits_behind)} "
                        "коммит(ов). Выполните git pull или используйте force=True.")
        except git.GitCommandError:
            # Возможно, ветка не существует на удаленном сервере
            pass

        # Выполняем push
        push_args = ["--force"] if force else []
        push_info = repo.git.push(remote, branch, *push_args)

        return f"Push успешно выполнен в {remote}/{branch}\n{push_info}"
    except Exception as e:
        logger.error(f"Ошибка в git_push: {e}")
        return f"Ошибка при выполнении push: {str(e)}"

def main() -> None:
    """
    Description:
    ---------------
        Основная точка входа
    """
    parser = argparse.ArgumentParser(description="MCP Git Local Server")
    parser.add_argument("--debug", action="store_true", help="Включить подробное логирование")
    args = parser.parse_args()

    # Установка уровня логирования
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    try:
        # Запускаем сервер в режиме stdio, что соответствует интерфейсу MCP
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        logger.info("Сервер остановлен пользователем")
    except Exception as e:
        logger.exception(f"Ошибка при работе сервера: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()