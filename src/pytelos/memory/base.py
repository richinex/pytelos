"""Abstract base class for conversation memory backends.

This module defines the interface for conversation memory storage.
The abstraction hides:
- Storage format (JSON, SQLite, etc.)
- Persistence mechanism (file, database, in-memory)
- Connection management
"""

from abc import ABC, abstractmethod

from .models import ConversationState, TaskRecord


class ConversationMemory(ABC):
    """Abstract conversation memory backend.

    Provides a unified interface for storing and retrieving
    conversation state across different storage backends.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Initialize the memory backend."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Close the memory backend gracefully."""

    @abstractmethod
    async def get_state(self, session_id: str | None = None) -> ConversationState:
        """Retrieve conversation state."""

    @abstractmethod
    async def save_state(self, state: ConversationState) -> None:
        """Persist conversation state."""

    @abstractmethod
    async def add_task_record(
        self,
        task: str,
        answer: str,
        files_read: list[str] | None = None,
        session_id: str | None = None
    ) -> TaskRecord:
        """Add a task record to the conversation history."""

    @abstractmethod
    async def get_recent_tasks(
        self,
        limit: int = 5,
        session_id: str | None = None
    ) -> list[TaskRecord]:
        """Get recent task records."""

    @abstractmethod
    async def clear_history(self, session_id: str | None = None) -> None:
        """Clear conversation history for a session."""

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Get the backend type identifier."""
