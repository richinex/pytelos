"""Anthropic Claude LLM provider implementation.

Uses the official Anthropic Python SDK for async chat completions.
Reference: https://github.com/anthropics/anthropic-sdk-python
"""

from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from ..base import LLMProvider
from ..models import ChatMessage, LLMResponse, StreamingResponse


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM provider implementation.

    Hidden design decisions:
    - Anthropic API client initialization
    - Message format conversion (system message handling)
    - Error handling and retry logic
    - Authentication mechanism
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        base_url: str | None = None,
        **client_kwargs: Any
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Default model to use (default: claude-sonnet-4-20250514)
            base_url: Optional custom API base URL
            **client_kwargs: Additional kwargs for AsyncAnthropic client
        """
        self._model = model
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            **client_kwargs
        )

    @property
    def model(self) -> str:
        """Get the default model name."""
        return self._model

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a chat completion using Anthropic Claude.

        Args:
            messages: Conversation history
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (default: 4096)
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            LLMResponse with generated content
        """
        model_to_use = model or self._model

        # Extract system message and convert to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Build request params
        request_params: dict[str, Any] = {
            "model": model_to_use,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,  # Anthropic requires max_tokens
            **kwargs
        }

        if system_message:
            request_params["system"] = system_message

        response = await self._client.messages.create(**request_params)

        # Extract usage
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }

        # Extract content (handle multiple content blocks)
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            usage=usage
        )

    async def chat_completion_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> StreamingResponse:
        """Generate a streaming chat completion using Anthropic Claude.

        Args:
            messages: Conversation history
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (default: 4096)
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            StreamingResponse that yields text chunks and captures usage info
        """
        model_to_use = model or self._model

        # Extract system message and convert to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        # Build request params
        request_params: dict[str, Any] = {
            "model": model_to_use,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            **kwargs
        }

        if system_message:
            request_params["system"] = system_message

        response = StreamingResponse(self._stream_generator(request_params))
        self._current_stream_response = response
        return response

    async def _stream_generator(
        self,
        request_params: dict[str, Any],
    ) -> AsyncIterator[str]:
        """Internal generator that yields text and captures usage from events."""
        input_tokens = 0
        output_tokens = 0

        async with self._client.messages.stream(**request_params) as stream:
            async for event in stream:
                # message_start contains input_tokens
                if hasattr(event, "type") and event.type == "message_start":
                    if hasattr(event, "message") and hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens
                # message_delta contains output_tokens (cumulative)
                elif hasattr(event, "type") and event.type == "message_delta":
                    if hasattr(event, "usage") and hasattr(event.usage, "output_tokens"):
                        output_tokens = event.usage.output_tokens
                # content_block_delta contains text
                elif (
                    hasattr(event, "type")
                    and event.type == "content_block_delta"
                    and hasattr(event, "delta")
                    and hasattr(event.delta, "text")
                ):
                    yield event.delta.text

            # Set usage after streaming completes
            self._current_stream_response.set_usage({
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            })

    async def close(self) -> None:
        """Close the Anthropic client."""
        await self._client.close()
