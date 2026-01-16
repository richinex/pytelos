"""Data structures for reasoning agent with multi-step execution.

Implements TaskStep, TaskStepResult, and NextStepDecision
following the patterns from agent.txt.
"""

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class Task(BaseModel):
    """Represents a task with an instruction.

    Attributes:
        id_: Unique identifier for the task
        instruction: The instruction for the task
    """

    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str = Field(description="The task instruction")

    def __str__(self) -> str:
        """String representation of Task."""
        return self.instruction


class TaskStep(BaseModel):
    """Represents a step within a task.

    Following agent.txt line 289-302.

    Attributes:
        id_: Unique identifier for this step
        task_id: ID of the parent task
        instruction: The instruction for this step
    """

    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = Field(description="ID of the parent task")
    instruction: str = Field(description="The instruction for this step in the task")

    def __str__(self) -> str:
        """String representation of TaskStep."""
        return self.instruction


class TaskStepResult(BaseModel):
    """The result of a task step execution.

    Following agent.txt line 305-313.

    Attributes:
        task_step_id: ID of the step that was executed
        content: The result content
    """

    task_step_id: str
    content: str

    def __str__(self) -> str:
        """String representation of TaskStepResult."""
        return self.content


class NextStepDecision(BaseModel):
    """Decision about the next step in task execution.

    Following agent.txt line 316-325.

    Attributes:
        kind: Whether this is the next step or final result
        content: Either the next step instruction or final result
    """

    kind: Literal["next_step", "final_result"] = Field(
        description="Type of decision: next_step or final_result"
    )
    content: str = Field(
        description="Content for the next step or the final result"
    )


class TaskResult(BaseModel):
    """The result of the overall task execution.

    Attributes:
        task_id: ID of the task that was executed
        content: The final result content
        execution_trace: Complete execution trace
        metadata: Additional metadata
    """

    task_id: str
    content: str
    execution_trace: str = Field(default="", description="Complete execution history")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    def __str__(self) -> str:
        """String representation of TaskResult."""
        return self.content


class ToolCall(BaseModel):
    """Represents a tool call request.

    Attributes:
        id_: Unique identifier for this tool call
        tool_name: Name of the tool to call
        arguments: Arguments for the tool call
    """

    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    """Result of executing a tool call.

    Attributes:
        tool_call_id: ID of the tool call that was executed
        content: The result content
        error: Whether an error occurred
    """

    tool_call_id: str
    content: str
    error: bool = False
