"""Data models for the TUI.

Hides the internal representation of chat messages and other data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChatMessage:
    """A chat message in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
