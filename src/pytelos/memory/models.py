"""Data models for conversation memory.

These models define the structure of conversation state and task records,
independent of the storage backend used.
"""

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskRecord(BaseModel):
    """Record of a single task execution.

    Captures the essential information about a completed task
    for follow-up queries and context retrieval.
    """

    task_id: int = Field(description="Sequential task identifier within session")
    task: str = Field(description="The user's task/prompt")
    answer: str = Field(description="The final answer provided")
    files_read: list[str] = Field(default_factory=list, description="Files read during task")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


class ConversationState(BaseModel):
    """Complete conversation state for a session.

    This is the structured representation of conversation memory
    that can be serialized and stored in any backend.
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    history: list[TaskRecord] = Field(default_factory=list)
    files_read: list[str] = Field(default_factory=list, description="All files read across tasks")
    last_task: str | None = Field(default=None, description="Most recent task")
    last_answer: str | None = Field(default=None, description="Most recent answer")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_task(self, record: TaskRecord) -> None:
        """Add a task record to the conversation history.

        Args:
            record: The task record to add
        """
        self.history.append(record)
        self.files_read = list(set(self.files_read + record.files_read))
        self.last_task = record.task[:200] if len(record.task) > 200 else record.task
        self.last_answer = record.answer[:1000] if len(record.answer) > 1000 else record.answer
        self.updated_at = datetime.utcnow()

    def get_recent_tasks(self, limit: int = 5) -> list[TaskRecord]:
        """Get the most recent task records.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent task records, newest first
        """
        return list(reversed(self.history[-limit:]))

    def to_context_string(self, limit: int = 5, task_limit: int = 200, answer_limit: int = 500) -> str:
        """Convert to a string suitable for LLM context injection.

        Args:
            limit: Maximum number of recent tasks to include
            task_limit: Character limit for task text
            answer_limit: Character limit for answer text

        Returns:
            String representation of recent conversation history
        """
        if not self.history:
            return ""

        lines = []
        for i, record in enumerate(self.history[-limit:], 1):
            task_text = record.task[:task_limit] + "..." if len(record.task) > task_limit else record.task
            answer_text = record.answer[:answer_limit] + "..." if len(record.answer) > answer_limit else record.answer
            lines.append(f"[Task {i}] User: {task_text}")
            lines.append(f"[Task {i}] Assistant: {answer_text}")
            lines.append("")  # Empty line between tasks
        return "\n".join(lines).strip()

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
