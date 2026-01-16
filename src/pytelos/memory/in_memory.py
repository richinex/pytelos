"""In-memory conversation memory backend.

Simple dict-based storage for session-only memory.
Data is lost when the application exits.
"""

from uuid import uuid4

from .base import ConversationMemory
from .models import ConversationState, TaskRecord


class InMemoryConversationMemory(ConversationMemory):
    """In-memory conversation memory (session-only).

    Data is stored in memory and lost when the app exits.
    Suitable for single-session use or testing.
    """

    def __init__(self, default_session_id: str | None = None):
        self._default_session_id = default_session_id or str(uuid4())
        self._states: dict[str, ConversationState] = {}

    async def connect(self) -> None:
        """Initialize memory (no-op for in-memory)."""
        pass

    async def disconnect(self) -> None:
        """Close memory (no-op for in-memory)."""
        pass

    async def get_state(self, session_id: str | None = None) -> ConversationState:
        """Get or create conversation state."""
        sid = session_id or self._default_session_id
        if sid not in self._states:
            self._states[sid] = ConversationState(session_id=sid)
        return self._states[sid]

    async def save_state(self, state: ConversationState) -> None:
        """Save state (just updates dict)."""
        self._states[state.session_id] = state

    async def add_task_record(
        self,
        task: str,
        answer: str,
        files_read: list[str] | None = None,
        session_id: str | None = None
    ) -> TaskRecord:
        """Add a task record."""
        state = await self.get_state(session_id)
        task_id = len(state.history) + 1

        record = TaskRecord(
            task_id=task_id,
            task=task,
            answer=answer,
            files_read=files_read or [],
        )
        state.add_task(record)
        return record

    async def get_recent_tasks(
        self,
        limit: int = 5,
        session_id: str | None = None
    ) -> list[TaskRecord]:
        """Get recent tasks."""
        state = await self.get_state(session_id)
        return state.get_recent_tasks(limit)

    async def clear_history(self, session_id: str | None = None) -> None:
        """Clear history."""
        sid = session_id or self._default_session_id
        if sid in self._states:
            self._states[sid] = ConversationState(session_id=sid)

    @property
    def backend_type(self) -> str:
        return "memory"
