import click
import os
import uuid
import asyncio
from functools import wraps
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from agents.assistant_agent import (
    create_git_assistant_agent,
    get_git_diff,
    create_commit,
    push_to_remote
)

def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

# Load environment variables from .env file
load_dotenv()

@click.group()
def cli():
    """AI Git Assistant: Automates Git operations using Gemini and ADK."""
    pass

@cli.command()
@click.argument('path', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option('--yes', '-y', is_flag=True, help='Skip interactive confirmation and execute commit and push.')
@coro
async def run(path, yes):
    """
    Run the process of analysis, commit generation, and push for the specified repository.
    
    PATH: The path to the local Git repository.
    """
    click.echo(f"Running analysis for repository: {path}")

    if not os.getenv("GEMINI_API_KEY"):
        click.echo(click.style("Error: GEMINI_API_KEY environment variable not found.", fg='red'))
        click.echo("Please create a .env file and add GEMINI_API_KEY=your_key")
        return
    click.echo("API key loaded successfully.")

    try:
        # --- Step 1: Get Git Diff using the tool ---
        click.echo("Getting git diff...")
        success, diff_content = get_git_diff(repo_path=path)

        if not success:
            click.echo(click.style(diff_content, fg='red'))
            return
        
        if not diff_content.strip():
            click.echo(click.style("No staged changes to commit.", fg='yellow'))
            return
            
        click.echo("Successfully retrieved git diff.")

        # --- Step 2: Generate Commit Message using the ADK Agent ---
        click.echo("Initializing Git Assistant Agent...")
        assistant_agent = create_git_assistant_agent()
        click.echo("Generating commit message...")
        
        runner = Runner(
            agent=assistant_agent,
            app_name="AI-Git-Assistant",
            session_service=InMemorySessionService()
        )
        
        # Create a session to get a session_id
        session_id = str(uuid.uuid4())
        user_id = "local-user"
        await runner.session_service.create_session(
            app_name="AI-Git-Assistant",
            user_id=user_id,
            session_id=session_id
        )

        # Wrap the diff content in a Content object
        new_message = Content(role='user', parts=[Part(text=diff_content)])

        # Call the runner asynchronously and process the event stream
        commit_message = ""
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=new_message):
            if event.is_final_response() and event.content and event.content.parts:
                commit_message = event.content.parts[0].text.strip()
        
        if not commit_message:
            click.echo(click.style("Error: Agent did not produce a final commit message.", fg='red'))
            return

        click.echo(click.style("Generated Commit Message:", fg='green'))
        click.echo("------------------------------------")
        click.echo(commit_message)
        click.echo("------------------------------------")

        # --- Step 3: Confirmation and Execution ---
        if not yes and not click.confirm(click.style('Execute commit and push with this message?', fg='yellow'), default=True):
            click.echo("Operation cancelled by user.")
            return
        
        # Commit using the tool
        success, commit_result = create_commit(repo_path=path, message=commit_message)
        if not success:
            click.echo(click.style(f"Error: {commit_result}", fg='red'))
            return
        click.echo(click.style(commit_result, fg='green'))

        # Push using the tool
        click.echo("Pushing to remote repository...")
        success, push_result = push_to_remote(repo_path=path)
        if not success:
            click.echo(click.style(f"Error: {push_result}", fg='red'))
            return
        click.echo(click.style(push_result, fg='green'))
        
        click.echo(click.style("\nProcess completed successfully!", fg='cyan', bold=True))

    except Exception as e:
        click.echo(click.style(f"An unexpected error occurred: {e}", fg='red'))
        return

if __name__ == '__main__':
    cli()
