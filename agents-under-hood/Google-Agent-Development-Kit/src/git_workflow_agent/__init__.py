"""
Google ADK Git Workflow Agent

Автоматический анализ изменений в коде и создание коммитов с использованием Google Agent Development Kit.
"""

from .agent import GitWorkflowAgent
from .config import AGENT_CONFIG

__version__ = "1.0.0"
__author__ = "Agents-Under-Hood Team"

__all__ = ["GitWorkflowAgent", "AGENT_CONFIG"] 