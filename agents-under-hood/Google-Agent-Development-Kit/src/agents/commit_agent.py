import google.generativeai as genai
from typing import Tuple

class CommitAgent:
    """
    Агент, отвечающий за генерацию сообщения коммита с помощью Gemini API.
    """
    def __init__(self, api_key: str):
        """
        Инициализирует агента и настраивает доступ к API.
        
        :param api_key: Ключ для доступа к Google Gemini API.
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')

    def _create_prompt(self, diff: str) -> str:
        """
        Создает детализированный промпт для модели.
        """
        prompt = f"""
Ты — опытный разработчик и эксперт по ведению систем контроля версий. Твоя задача — написать идеальное сообщение для коммита на основе предоставленного `git diff`.

**Требования к сообщению:**
1.  **Формат**: Строго следуй формату Conventional Commits.
    - Первая строка: `<type>: <subject>` (например, `feat: добавлена аутентификация пользователей`).
    - `type` должен быть одним из: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.
    - `subject` должен быть кратким, но емким, в нижнем регистре, без точки в конце.
    - После заголовка ОБЯЗАТЕЛЬНО должна быть пустая строка.
2.  **Тело коммита (body)**:
    - Подробно опиши изменения.
    - Объясни **ЧТО** было изменено и, что более важно, **ПОЧЕМУ** это было сделано.
    - Если изменений много, используй маркированные списки.
3.  **Язык**: Русский.

**Вот `git diff` для анализа:**
```diff
{diff}
```

Напиши только сообщение коммита и ничего больше.
"""
        return prompt.strip()

    def generate_commit_message(self, diff: str) -> Tuple[bool, str]:
        """
        Генерирует сообщение коммита на основе diff.
        
        :param diff: Строка с результатом `git diff --staged`.
        :return: Кортеж (успех, сгенерированное сообщение или текст ошибки).
        """
        if not diff.strip():
            return False, "Diff пуст, генерация коммита отменена."

        prompt = self._create_prompt(diff)
        
        try:
            response = self.model.generate_content(prompt)
            commit_message = response.text
            return True, commit_message.strip()
        except Exception as e:
            # Обработка широкого спектра возможных ошибок API
            return False, f"Ошибка при вызове Gemini API: {e}"
