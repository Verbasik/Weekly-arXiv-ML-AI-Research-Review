import subprocess
from google.adk.agents import Agent
from typing import Tuple

def get_git_diff(repo_path: str) -> Tuple[bool, str]:
    """
    Adds all changes to the staging area and returns the staged git diff.
    Args:
        repo_path: The absolute path to the git repository.
    Returns:
        A tuple containing a success boolean and the diff or an error message.
    """
    try:
        subprocess.run(
            ["git", "add", "."],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        diff_result = subprocess.run(
            ["git", "diff", "--staged"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        return True, diff_result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"An error occurred: {e.stderr}"
    except FileNotFoundError:
        return False, "Error: 'git' command not found. Is Git installed and in your PATH?"

def create_commit(repo_path: str, message: str) -> Tuple[bool, str]:
    """
    Creates a git commit with the given message.
    Args:
        repo_path: The absolute path to the git repository.
        message: The commit message.
    Returns:
        A tuple containing a success boolean and a status message.
    """
    try:
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        return True, "Commit created successfully."
    except subprocess.CalledProcessError as e:
        return False, f"An error occurred during commit: {e.stderr}"

def push_to_remote(repo_path: str) -> Tuple[bool, str]:
    """
    Pushes the commits to the remote repository.
    Args:
        repo_path: The absolute path to the git repository.
    Returns:
        A tuple containing a success boolean and a status message.
    """
    try:
        subprocess.run(
            ["git", "push"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True
        )
        return True, "Pushed to remote successfully."
    except subprocess.CalledProcessError as e:
        return False, f"An error occurred during push: {e.stderr}"

def create_git_assistant_agent() -> Agent:
    """Creates the Git Assistant Agent."""
    return Agent(
        name="GitAssistant",
        model="gemini-1.5-flash-latest",
        instruction=(
            "You are an expert AI assistant specializing in writing git commit messages."
            "Your task is to analyze a provided 'git diff' and generate a concise, "
            "informative commit message that follows the Conventional Commits specification."
            "The user will provide the diff, and you must return ONLY the commit message."
            "Do not add any extra text, explanations, or markdown formatting."
        ),
        description="An AI assistant for Git operations.",
        tools=[get_git_diff, create_commit, push_to_remote]
    )