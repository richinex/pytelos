from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StreamingResponse:
    """Wrapper for streaming LLM responses that captures usage info.

    Acts as an async iterator for text chunks while storing token usage
    that becomes available at the end of the stream.

    Usage:
        stream = await provider.chat_completion_stream(messages)
        async for chunk in stream:
            print(chunk, end="")
        # After iteration, usage is available
        print(stream.usage)  # {"prompt_tokens": 100, "completion_tokens": 50, ...}
    """

    def __init__(self, async_iter: AsyncIterator[str]):
        """Initialize with an async iterator of text chunks.

        Args:
            async_iter: Async iterator yielding text chunks
        """
        self._iter = async_iter
        self._usage: dict[str, Any] | None = None

    @property
    def usage(self) -> dict[str, Any] | None:
        """Get token usage info (available after iteration completes)."""
        return self._usage

    def set_usage(self, usage: dict[str, Any]) -> None:
        """Set token usage info (called by provider at end of stream)."""
        self._usage = usage

    def __aiter__(self) -> "StreamingResponse":
        """Return self as async iterator."""
        return self

    async def __anext__(self) -> str:
        """Get next chunk from the underlying iterator."""
        return await self._iter.__anext__()


class ChatMessage(BaseModel):
    """Represents a chat message in a conversation."""

    model_config = ConfigDict(frozen=True)

    role: str = Field(description="Role of the message sender: 'user', 'assistant', or 'system'")
    content: str = Field(description="Content of the message")


class LLMResponse(BaseModel):
    """Response from an LLM provider."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(description="Generated text content")
    model: str = Field(description="Model that generated the response")
    usage: dict[str, int] | None = Field(
        default=None,
        description="Token usage information"
    )
