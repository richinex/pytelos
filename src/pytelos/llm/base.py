from abc import ABC, abstractmethod
from typing import Any

from .models import ChatMessage, LLMResponse, StreamingResponse


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    This module hides the design decision of which LLM provider to use.
    Implementations must handle provider-specific details like:
    - API client setup and authentication
    - Request/response format conversion
    - Error handling and retries
    - Rate limiting

    Supports async context manager protocol for proper resource cleanup:
        async with provider:
            response = await provider.chat_completion(messages)
        # Automatically cleaned up
    """

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a chat completion.

        Args:
            messages: List of chat messages forming the conversation history
            model: Model to use (None uses provider's default)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse containing generated content and metadata

        Raises:
            Exception: Provider-specific errors during generation
        """
        pass

    @abstractmethod
    async def chat_completion_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> StreamingResponse:
        """Generate a streaming chat completion.

        Args:
            messages: List of chat messages forming the conversation history
            model: Model to use (None uses provider's default)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters

        Returns:
            StreamingResponse that yields text chunks and captures usage info.
            After iteration, access usage via stream_response.usage

        Raises:
            Exception: Provider-specific errors during generation
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections or resources."""
        pass

    async def __aenter__(self) -> "LLMProvider":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with automatic cleanup.

        Note: Suppresses "Event loop is closed" errors during cleanup.
        This is a known harmless race condition in httpx/anyio cleanup:
        https://github.com/encode/httpx/issues/914
        """
        try:
            await self.close()
        except RuntimeError as e:
            # Suppress harmless cleanup errors from httpx/anyio
            if "Event loop is closed" not in str(e):
                raise
