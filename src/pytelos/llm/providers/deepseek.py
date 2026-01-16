from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from ..base import LLMProvider
from ..models import ChatMessage, LLMResponse, StreamingResponse


class DeepSeekProvider(LLMProvider):
    """DeepSeek LLM provider implementation using OpenAI-compatible API.

    Hidden design decisions:
    - DeepSeek API client initialization (via OpenAI SDK)
    - Message format conversion
    - Error handling and retry logic
    - Authentication mechanism
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        **client_kwargs: Any
    ):
        """Initialize DeepSeek provider.

        Args:
            api_key: DeepSeek API key
            model: Default model to use ('deepseek-chat' or 'deepseek-reasoner')
            base_url: DeepSeek API base URL (default: https://api.deepseek.com)
            **client_kwargs: Additional kwargs for AsyncOpenAI client
        """
        self._model = model
        self._client = AsyncOpenAI(
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
        """Generate a chat completion using DeepSeek.

        Args:
            messages: Conversation history
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional DeepSeek-specific parameters

        Returns:
            LLMResponse with generated content
        """
        model_to_use = model or self._model

        deepseek_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        completion = await self._client.chat.completions.create(
            model=model_to_use,
            messages=deepseek_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        usage = None
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }

        return LLMResponse(
            content=completion.choices[0].message.content or "",
            model=completion.model,
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
        """Generate a streaming chat completion using DeepSeek.

        Args:
            messages: Conversation history
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional DeepSeek-specific parameters

        Returns:
            StreamingResponse that yields text chunks and captures usage info
        """
        model_to_use = model or self._model
        deepseek_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        # Create streaming response wrapper
        response = StreamingResponse(self._stream_generator(
            model_to_use, deepseek_messages, temperature, max_tokens, **kwargs
        ))
        # Store reference so generator can set usage
        self._current_stream_response = response
        return response

    async def _stream_generator(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Internal generator that yields chunks and captures usage."""
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )

        async for chunk in stream:
            # Check for usage in the final chunk
            if chunk.usage is not None:
                self._current_stream_response.set_usage({
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                })
            # Yield content if present
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def close(self) -> None:
        """Close the DeepSeek client.

        Note: Uses the OpenAI SDK's async context manager for proper cleanup.
        See: https://github.com/openai/openai-python#async-usage
        """
        await self._client.close()
