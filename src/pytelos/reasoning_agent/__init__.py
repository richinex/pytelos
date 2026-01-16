"""Reasoning agent module with multi-step processing and tool calling.

Implements full agent.txt patterns:
- Multi-step processing loop
- Dynamic tool calling
- ReAct reasoning (thought/action/observation)
"""

from .connection_pool import ConnectionPool
from .data_structures import (
    NextStepDecision,
    Task,
    TaskResult,
    TaskStep,
    TaskStepResult,
    ToolCall,
    ToolCallResult,
)
from .reasoning_agent import ReasoningAgent
from .tools import AnalyzeCodeTool, BaseTool, ReadFileTool, SearchCodebaseTool

__all__ = [
    "Task",
    "TaskStep",
    "TaskStepResult",
    "NextStepDecision",
    "TaskResult",
    "ToolCall",
    "ToolCallResult",
    "ReasoningAgent",
    "PyergonReasoningAgent",
    "ConnectionPool",
    "BaseTool",
    "SearchCodebaseTool",
    "ReadFileTool",
    "AnalyzeCodeTool",
]


def __getattr__(name):
    """Lazy import for optional dependencies."""
    if name == "PyergonReasoningAgent":
        try:
            from .pyergon_reasoning_agent import PyergonReasoningAgent
            return PyergonReasoningAgent
        except ImportError as e:
            raise ImportError(
                "PyergonReasoningAgent requires 'pyergon' package. "
                "Install it from pyergon directory: cd pyergon && uv pip install -e ."
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
