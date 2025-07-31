import click
import os
from dotenv import load_dotenv
from agents.git_agent import GitAgent
from agents.commit_agent import CommitAgent

# Загружаем переменные окружения из .env файла
load_dotenv()

@click.group()
def cli():
    """
    AI Git Assistant: Автоматизирует Git-операции с помощью Gemini.
    """
    pass

@cli.command()
@click.argument('path', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option('--yes', '-y', is_flag=True, help='Пропустить интерактивное подтверждение и выполнить коммит и push.')
def run(path, yes):
    """
    Запустить процесс анализа, генерации коммита и push для указанного репозитория.
    
    PATH: Путь к локальному Git-репозиторию.
    """
    click.echo(f"Запуск анализа для репозитория: {path}")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        click.echo(click.style("Ошибка: Переменная окружения GEMINI_API_KEY не найдена.", fg='red'))
        click.echo("Пожалуйста, создайте файл .env и добавьте в него GEMINI_API_KEY=ваш_ключ")
        return
    click.echo("Ключ API успешно загружен.")

    try:
        # --- Этап 1: Работа с Git ---
        git_agent = GitAgent(repo_path=path)
        click.echo("Git-агент инициализирован.")

        add_success, add_message = git_agent.add_all()
        if not add_success:
            click.echo(click.style(f"Ошибка: {add_message}", fg='red'))
            return
        click.echo(add_message)

        diff_success, diff_content, diff_error = git_agent.get_staged_diff()
        if not diff_success:
            click.echo(click.style(f"Ошибка: {diff_error}", fg='red'))
            return
        
        if not diff_content.strip():
            click.echo(click.style("Нет проиндексированных изменений для коммита.", fg='yellow'))
            return
        click.echo("Изменения для коммита успешно получены.")

        # --- Этап 2: Генерация коммита ---
        click.echo("Запуск Commit-генератора...")
        commit_agent = CommitAgent(api_key=api_key)
        
        gen_success, message = commit_agent.generate_commit_message(diff_content)
        if not gen_success:
            click.echo(click.style(f"Ошибка генерации коммита: {message}", fg='red'))
            return
        
        click.echo(click.style("Сгенерированное сообщение коммита:", fg='green'))
        click.echo("------------------------------------")
        click.echo(message)
        click.echo("------------------------------------")

        # --- Этап 3: Подтверждение и выполнение ---
        if not yes and not click.confirm(click.style('Выполнить коммит и push с этим сообщением?', fg='yellow'), default=True):
            click.echo("Операция отменена пользователем.")
            return
            
        # Коммит
        commit_success, commit_message = git_agent.commit(message)
        if not commit_success:
            click.echo(click.style(f"Ошибка: {commit_message}", fg='red'))
            return
        click.echo(click.style(commit_message, fg='green'))

        # Push
        click.echo("Отправка в удаленный репозиторий...")
        push_success, push_message = git_agent.push()
        if not push_success:
            click.echo(click.style(f"Ошибка: {push_message}", fg='red'))
            return
        click.echo(click.style(push_message, fg='green'))
        
        click.echo(click.style("\nПроцесс успешно завершен!", fg='cyan', bold=True))

    except ValueError as e:
        click.echo(click.style(f"Ошибка инициализации: {e}", fg='red'))
        return
    except Exception as e:
        click.echo(click.style(f"Произошла непредвиденная ошибка: {e}", fg='red'))
        return


if __name__ == '__main__':
    cli()
