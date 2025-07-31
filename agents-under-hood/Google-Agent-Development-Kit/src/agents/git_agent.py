import subprocess
import os
from typing import Tuple

class GitAgent:
    """
    Агент, отвечающий за взаимодействие с Git.
    Инкапсулирует все вызовы командной строки git.
    """
    def __init__(self, repo_path: str):
        """
        Инициализирует агента с путем к репозиторию.
        
        :param repo_path: Абсолютный путь к локальному Git-репозиторию.
        """
        if not os.path.isdir(os.path.join(repo_path, '.git')):
            raise ValueError("Указанный путь не является Git-репозиторием.")
        self.repo_path = repo_path

    def _run_command(self, command: list[str]) -> Tuple[bool, str, str]:
        """
        Вспомогательный метод для выполнения команд в директории репозитория.
        
        :param command: Команда для выполнения в виде списка.
        :return: Кортеж (успех, stdout, stderr).
        """
        try:
            process = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return True, process.stdout, process.stderr
        except FileNotFoundError:
            return False, "", "Команда 'git' не найдена. Убедитесь, что Git установлен и доступен в PATH."
        except subprocess.CalledProcessError as e:
            return False, e.stdout, e.stderr

    def add_all(self) -> Tuple[bool, str]:
        """
        Выполняет 'git add .' для индексации всех изменений.
        
        :return: Кортеж (успех, сообщение).
        """
        success, _, stderr = self._run_command(['git', 'add', '.'])
        if not success:
            return False, f"Ошибка при выполнении git add: {stderr}"
        return True, "Все файлы успешно добавлены в индекс."

    def get_staged_diff(self) -> Tuple[bool, str, str]:
        """
        Получает изменения в проиндексированных файлах ('git diff --staged').
        
        :return: Кортеж (успех, diff, сообщение об ошибке).
        """
        success, stdout, stderr = self._run_command(['git', 'diff', '--staged'])
        if not success:
            return False, "", f"Ошибка при получении diff: {stderr}"
        return True, stdout, ""

    def commit(self, message: str) -> Tuple[bool, str]:
        """
        Выполняет 'git commit' с заданным сообщением.
        
        :param message: Сообщение коммита.
        :return: Кортеж (успех, сообщение).
        """
        success, _, stderr = self._run_command(['git', 'commit', '-m', message])
        if not success:
            return False, f"Ошибка при выполнении git commit: {stderr}"
        return True, "Коммит успешно создан."

    def push(self) -> Tuple[bool, str]:
        """
        Выполняет 'git push'.
        
        :return: Кортеж (успех, сообщение).
        """
        success, _, stderr = self._run_command(['git', 'push'])
        if not success:
            # stderr от git push очень информативен, его стоит показать пользователю
            return False, f"Ошибка при выполнении git push:\n{stderr}"
        return True, "Изменения успешно отправлены в удаленный репозиторий."
