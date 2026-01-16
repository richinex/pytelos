"""Terminal UI module for pytelos.

Provides a Textual-based TUI for ReasoningAgent interaction.

Module structure (Parnas principle - each module hides a design decision):
- models.py: Data structures (message representation)
- widgets.py: Custom widgets (input history, metrics display, log rendering)
- styles.py: CSS styling (layout decisions)
- themes.py: Color palettes and theme configuration
- screens.py: Modal dialogs (confirmation screens)
- callbacks.py: Agent integration (how TUI receives updates)
- app.py: Application orchestration (user interaction flow)
"""

from .app import ReasoningTextualApp, run_textual_tui
from .callbacks import TUICallback
from .config import LogLevel
from .models import ChatMessage
from .widgets import ChatHistoryWidget, ChatInputBar, ExecutionLog, MetricsPanel

__all__ = [
    "ChatHistoryWidget",
    "ChatInputBar",
    "ChatMessage",
    "ExecutionLog",
    "LogLevel",
    "MetricsPanel",
    "ReasoningTextualApp",
    "TUICallback",
    "run_textual_tui",
]
