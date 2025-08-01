# -*- coding: utf-8 -*-

"""
Главный модуль CLI-приложения AI Git Assistant.

Этот модуль отвечает за обработку команд из командной строки,
инициализацию AI-агента, управление потоком выполнения и взаимодействие
с пользователем.
"""

# Стандартные библиотеки
import asyncio
import os
import uuid
from functools import wraps
from typing import Callable, Any

# Сторонние библиотеки
import click
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

# Локальные модули
from .agents.assistant_agent import (
    create_git_assistant_agent,
    get_git_diff,
    create_commit,
    push_to_remote
)


def coro(f: Callable) -> Callable:
    """
    Description:
    ---------------
        Декоратор для запуска асинхронных функций `click`
        в синхронном контексте.

    Args:
    ---------------
        f (Callable): Асинхронная функция для декорирования.

    Returns:
    ---------------
        Callable: Синхронная обертка для асинхронной функции.
    """
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Запускает асинхронную функцию и дожидается ее выполнения.
        return asyncio.run(f(*args, **kwargs))
    return wrapper


# Загружаем переменные окружения из файла .env.
# Это позволяет безопасно хранить API-ключи вне кода.
load_dotenv()


@click.command()
@click.option(
    '--yes', '-y',
    is_flag=True,
    help='Пропустить интерактивное подтверждение перед коммитом и push.'
)
@coro
async def cli(yes: bool) -> None:
    """
    Description:
    ---------------
        Запускает полный цикл анализа репозитория, генерации коммита
        и отправки изменений для ТЕКУЩЕЙ директории.

    Args:
    ---------------
        yes (bool): Флаг для пропуска интерактивного подтверждения.

    Returns:
    ---------------
        None
    """
    # Автоматически определяем путь к репозиторию как текущую рабочую директорию
    path = os.getcwd()
    click.echo(f"Запуск анализа для репозитория: {path}")

    # --- Шаг 0: Проверка наличия API-ключа ---
    # Критически важно убедиться, что ключ API доступен.
    if not os.getenv("GEMINI_API_KEY"):
        click.echo(click.style(
            "Ошибка: Переменная окружения GEMINI_API_KEY не найдена.",
            fg='red'
        ))
        click.echo(
            "Пожалуйста, создайте файл .env и добавьте в него строку "
            "GEMINI_API_KEY=ваш_ключ"
        )
        return
    click.echo("API-ключ успешно загружен.")

    try:
        # --- Шаг 1: Получение Git Diff ---
        click.echo("Получение git diff...")
        success, diff_content = get_git_diff(repo_path=path)

        if not success:
            # Если инструмент вернул ошибку, выводим ее и завершаем работу.
            click.echo(click.style(diff_content, fg='red'))
            return

        if not diff_content.strip():
            # Если нет изменений для коммита, сообщаем об этом.
            click.echo(click.style(
                "Нет подготовленных изменений для коммита.", fg='yellow'
            ))
            return

        click.echo("Git diff успешно получен.")

        # --- Шаг 2: Генерация сообщения коммита с помощью ADK Agent ---
        click.echo("Инициализация агента Git Assistant...")
        assistant_agent = create_git_assistant_agent()
        click.echo("Генерация сообщения для коммита...")

        # Runner — это ключевой компонент ADK, который управляет
        # жизненным циклом агента и выполнением его задач.
        runner = Runner(
            agent=assistant_agent,
            app_name="AI-Git-Assistant",
            session_service=InMemorySessionService()
        )

        # Создаем уникальный ID для сессии.
        session_id = str(uuid.uuid4())
        user_id = "local-user"  # Идентификатор пользователя
        await runner.session_service.create_session(
            app_name="AI-Git-Assistant",
            user_id=user_id,
            session_id=session_id
        )

        # Оборачиваем diff в специальный объект `Content` для передачи агенту.
        new_message = Content(role='user', parts=[Part(text=diff_content)])

        # Асинхронно запускаем агента и обрабатываем поток событий.
        commit_message = ""
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=new_message
        ):
            # Нас интересует только финальный ответ от агента.
            if event.is_final_response() and event.content:
                commit_message = event.content.parts[0].text.strip()

        if not commit_message:
            click.echo(click.style(
                "Ошибка: Агент не сгенерировал итоговое сообщение коммита.",
                fg='red'
            ))
            return

        click.echo(click.style("Сгенерированное сообщение:", fg='green'))
        click.echo("------------------------------------")
        click.echo(commit_message)
        click.echo("------------------------------------")

        # --- Шаг 3: Подтверждение и выполнение ---
        # Если не был передан флаг `--yes`, запрашиваем подтверждение.
        if not yes and not click.confirm(
            click.style('Выполнить коммит и push с этим сообщением?',
                        fg='yellow'),
            default=True
        ):
            click.echo("Операция отменена пользователем.")
            return

        # Выполняем коммит с помощью нашего инструмента.
        success, commit_result = create_commit(
            repo_path=path, message=commit_message
        )
        if not success:
            click.echo(click.style(f"Ошибка: {commit_result}", fg='red'))
            return
        click.echo(click.style(commit_result, fg='green'))

        # Отправляем изменения на удаленный сервер.
        click.echo("Отправка на удаленный репозиторий...")
        success, push_result = push_to_remote(repo_path=path)
        if not success:
            click.echo(click.style(f"Ошибка: {push_result}", fg='red'))
            return
        click.echo(click.style(push_result, fg='green'))

        click.echo(click.style(
            "\nПроцесс успешно завершен!", fg='cyan', bold=True
        ))

    except Exception as e:
        # Глобальный обработчик для непредвиденных ошибок.
        click.echo(click.style(
            f"Произошла непредвиденная ошибка: {e}", fg='red'
        ))
        return


if __name__ == '__main__':
    cli()
