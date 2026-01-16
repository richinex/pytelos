"""Data structures for agent module."""

import uuid
from typing import Any

from pydantic import BaseModel, Field


class Task(BaseModel):
    """Represents a task with an instruction.

    Attributes:
        id_: Unique identifier for the task
        instruction: The instruction or question for the agent
        context: Optional additional context
    """

    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str = Field(description="The task instruction or question")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context (e.g., filters, settings)"
    )

    def __str__(self) -> str:
        """String representation of Task."""
        return self.instruction


class TaskResult(BaseModel):
    """Result of task execution.

    Attributes:
        task_id: ID of the task that was executed
        content: The result content (answer, response)
        sources: List of sources used to generate the response
        metadata: Additional metadata (processing time, etc.)
    """

    task_id: str
    content: str
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Sources used for the response"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    def __str__(self) -> str:
        """String representation of TaskResult."""
        return self.content


class RetrievedChunk(BaseModel):
    """Represents a retrieved document chunk.

    Attributes:
        content: The full chunk content
        source: Source file path
        lines: Line range in the source file
        score: Relevance score
        metadata: Additional metadata
    """

    content: str
    source: str
    lines: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class UsageSummary(BaseModel):
    """Summary of LLM token usage across all calls.

    Attributes:
        total_calls: Total number of LLM API calls
        total_input_tokens: Total input tokens across all calls
        total_output_tokens: Total output tokens across all calls
        model_breakdown: Token usage broken down by model name
    """

    total_calls: int = Field(default=0, description="Total API calls")
    total_input_tokens: int = Field(default=0, description="Total input tokens")
    total_output_tokens: int = Field(default=0, description="Total output tokens")
    model_breakdown: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Usage breakdown by model"
    )

    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> None:
        """Add usage statistics for a model call.

        Args:
            model: Model name
            input_tokens: Input tokens used
            output_tokens: Output tokens used
        """
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        if model not in self.model_breakdown:
            self.model_breakdown[model] = {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }

        self.model_breakdown[model]["calls"] += 1
        self.model_breakdown[model]["input_tokens"] += input_tokens
        self.model_breakdown[model]["output_tokens"] += output_tokens
