from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from ..base import LLMProvider
from ..models import ChatMessage, LLMResponse, StreamingResponse

# Models that require the Responses API instead of Chat Completions
RESPONSES_API_MODELS = {
    "gpt-5.1-codex",
    "gpt-5.1-codex-max",
    "gpt-5.2-codex",
}


def _is_responses_api_model(model: str) -> bool:
    """Check if a model requires the Responses API."""
    return model in RESPONSES_API_MODELS or "codex" in model.lower()


def _messages_to_responses_format(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Convert chat messages to Responses API input format.

    The Responses API takes an array of messages with roles:
    - 'developer' (same as 'system') for instructions
    - 'user' for user messages
    - 'assistant' for previous model responses

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    responses_messages = []

    for msg in messages:
        if msg.role == "system":
            # System messages become developer messages in Responses API
            responses_messages.append({
                "role": "developer",
                "content": msg.content
            })
        elif msg.role in ("user", "assistant"):
            responses_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        else:
            # Default to user role for unknown roles
            responses_messages.append({
                "role": "user",
                "content": msg.content
            })

    return responses_messages


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation.

    Hidden design decisions:
    - OpenAI API client initialization
    - Message format conversion
    - Error handling and retry logic
    - Authentication mechanism
    - API routing (Chat Completions vs Responses API)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        organization: str | None = None,
        **client_kwargs: Any
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Default model to use (supports both Chat and Responses API models)
            base_url: Optional custom API base URL
            organization: Optional organization ID
            **client_kwargs: Additional kwargs for AsyncOpenAI client
        """
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
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
        """Generate a chat completion using OpenAI.

        Automatically routes to Responses API for codex models,
        otherwise uses Chat Completions API.

        Args:
            messages: Conversation history
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            LLMResponse with generated content
        """
        model_to_use = model or self._model

        # Route to Responses API for codex models
        if _is_responses_api_model(model_to_use):
            return await self._responses_api_completion(
                messages=messages,
                model=model_to_use,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        # Standard Chat Completions API
        openai_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Build request params, only including max_tokens if set
        request_params: dict[str, Any] = {
            "model": model_to_use,
            "messages": openai_messages,
            "temperature": temperature,
            **kwargs
        }
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens

        completion = await self._client.chat.completions.create(**request_params)

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

    async def _responses_api_completion(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate completion using OpenAI Responses API.

        Used for codex models that only support Responses API.

        Args:
            messages: Conversation history (converted to Responses API format)
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        # Convert messages to Responses API format (array with roles)
        responses_input = _messages_to_responses_format(messages)

        # Build request params
        request_params: dict[str, Any] = {
            "model": model,
            "input": responses_input,
        }

        # Add optional parameters if provided
        # Note: temperature only works with reasoning.effort="none"
        if max_tokens is not None:
            request_params["max_output_tokens"] = max_tokens

        # Pass through any additional kwargs
        request_params.update(kwargs)

        response = await self._client.responses.create(**request_params)

        # Extract usage if available
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, "input_tokens", 0),
                "completion_tokens": getattr(response.usage, "output_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0)
            }

        return LLMResponse(
            content=response.output_text or "",
            model=model,
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
        """Generate a streaming chat completion using OpenAI.

        Automatically routes to Responses API for codex models,
        otherwise uses Chat Completions API.

        Args:
            messages: Conversation history
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            StreamingResponse that yields text chunks and captures usage info
        """
        model_to_use = model or self._model

        # Route to Responses API for codex models
        if _is_responses_api_model(model_to_use):
            response = StreamingResponse(self._responses_api_stream(
                messages=messages,
                model=model_to_use,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            ))
            self._current_stream_response = response
            return response

        # Standard Chat Completions API
        openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        response = StreamingResponse(self._chat_stream_generator(
            model_to_use, openai_messages, temperature, max_tokens, **kwargs
        ))
        self._current_stream_response = response
        return response

    async def _chat_stream_generator(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Internal generator for Chat Completions streaming with usage capture."""
        request_params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            **kwargs,
        }
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens

        stream = await self._client.chat.completions.create(**request_params)

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

    async def _responses_api_stream(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Generate streaming completion using OpenAI Responses API.

        Used for codex models that only support Responses API.

        Args:
            messages: Conversation history (converted to Responses API format)
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Text chunks as they are generated
        """
        # Convert messages to Responses API format (array with roles)
        responses_input = _messages_to_responses_format(messages)

        # Build request params
        request_params: dict[str, Any] = {
            "model": model,
            "input": responses_input,
            "stream": True,
        }

        # Add optional parameters if provided
        # Note: temperature only works with reasoning.effort="none"
        if max_tokens is not None:
            request_params["max_output_tokens"] = max_tokens

        # Pass through any additional kwargs
        request_params.update(kwargs)

        stream = await self._client.responses.create(**request_params)

        async for event in stream:
            # Responses API streams events with different types
            if hasattr(event, "type"):
                if event.type == "response.output_text.delta":
                    if hasattr(event, "delta") and event.delta:
                        yield event.delta
                elif (
                    event.type == "response.content_part.delta"
                    and hasattr(event, "delta")
                    and hasattr(event.delta, "text")
                ):
                    yield event.delta.text

    async def close(self) -> None:
        """Close the OpenAI client.

        Note: Uses the OpenAI SDK's async context manager for proper cleanup.
        See: https://github.com/openai/openai-python#async-usage
        """
        await self._client.close()
