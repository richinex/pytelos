"""Factory for creating conversation memory backends."""

from typing import Any

from .base import ConversationMemory


def create_conversation_memory(
    backend: str = "memory",
    **kwargs: Any
) -> ConversationMemory:
    """Create a conversation memory backend.

    Args:
        backend: Backend type ("memory" or "sqlite")
        **kwargs: Backend-specific configuration

    Returns:
        ConversationMemory instance

    Raises:
        ValueError: If backend type is not supported
    """
    if backend == "memory":
        from .in_memory import InMemoryConversationMemory
        return InMemoryConversationMemory(**kwargs)

    elif backend == "sqlite":
        from .sqlite import SQLiteConversationMemory
        return SQLiteConversationMemory(**kwargs)

    raise ValueError(
        f"Unsupported memory backend: {backend}. "
        f"Supported backends: memory, sqlite"
    )
