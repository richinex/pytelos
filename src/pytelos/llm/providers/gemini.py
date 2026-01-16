"""Google Gemini LLM provider implementation.

Uses the official Google GenAI SDK for async chat completions.
Reference: https://github.com/googleapis/python-genai

Note: Gemini can return empty responses due to safety filtering or service issues.
This implementation includes retry logic and relaxed safety settings.
"""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types

from ..base import LLMProvider
from ..models import ChatMessage, LLMResponse, StreamingResponse

# Default safety settings - relaxed to avoid blocking code-related content
DEFAULT_SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
]


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation.

    Hidden design decisions:
    - Google GenAI client initialization
    - Message format conversion
    - Retry logic for empty responses (known Gemini issue)
    - Relaxed safety settings to avoid blocking code content
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        max_retries: int = 3,
        **client_kwargs: Any
    ):
        """Initialize Gemini provider.

        Args:
            api_key: Google AI API key
            model: Default model (gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash)
            max_retries: Max retries for empty responses (default 3)
            **client_kwargs: Additional kwargs for Client
        """
        self._model = model
        self._max_retries = max_retries
        self._client = genai.Client(api_key=api_key, **client_kwargs)

    @property
    def model(self) -> str:
        """Get the default model name."""
        return self._model

    def _convert_messages(self, messages: list[ChatMessage]) -> tuple[str | None, list[types.Content]]:
        """Convert ChatMessage list to Gemini format.

        Args:
            messages: List of chat messages

        Returns:
            Tuple of (system_instruction, contents)
        """
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            elif msg.role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=msg.content)]
                ))
            elif msg.role == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=msg.content)]
                ))

        return system_instruction, contents

    def _extract_content(self, response) -> str:
        """Extract text content from Gemini response, handling empty responses.

        Args:
            response: Gemini GenerateContentResponse

        Returns:
            Text content or empty string
        """
        # Check if response has valid candidates with content
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                # Join all text parts
                texts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
                if texts:
                    return "".join(texts)

        # Fallback to response.text (may raise or return None)
        try:
            return response.text or ""
        except (ValueError, AttributeError):
            return ""

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a chat completion using Google Gemini.

        Includes retry logic for empty responses (known Gemini service issue).

        Args:
            messages: Conversation history
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Gemini-specific parameters

        Returns:
            LLMResponse with generated content
        """
        model_to_use = model or self._model
        system_instruction, contents = self._convert_messages(messages)

        # Build generation config with safety settings
        # Use tool_config with mode=NONE to disable automatic function detection
        # This prevents UNEXPECTED_TOOL_CALL when prompts contain function-like syntax
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE")
        )
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            safety_settings=DEFAULT_SAFETY_SETTINGS,
            tool_config=tool_config,
            **kwargs
        )
        if max_tokens is not None:
            config.max_output_tokens = max_tokens

        # Retry loop for empty responses (known Gemini issue)
        content = ""
        usage = None

        for attempt in range(self._max_retries):
            response = await self._client.aio.models.generate_content(
                model=model_to_use,
                contents=contents,
                config=config
            )

            # Extract usage
            if response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                    "total_tokens": response.usage_metadata.total_token_count or 0
                }

            # Extract content
            content = self._extract_content(response)

            if content:
                break  # Got valid content, exit retry loop

            # Empty response - wait briefly before retry
            if attempt < self._max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))

        return LLMResponse(
            content=content,
            model=model_to_use,
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
        """Generate a streaming chat completion using Google Gemini.

        Args:
            messages: Conversation history
            model: Model to use (overrides default)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional Gemini-specific parameters

        Returns:
            StreamingResponse that yields text chunks and captures usage info
        """
        model_to_use = model or self._model
        system_instruction, contents = self._convert_messages(messages)

        # Build generation config with safety settings
        # Use tool_config with mode=NONE to disable automatic function detection
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE")
        )
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            safety_settings=DEFAULT_SAFETY_SETTINGS,
            tool_config=tool_config,
            **kwargs
        )
        if max_tokens is not None:
            config.max_output_tokens = max_tokens

        response = StreamingResponse(self._stream_generator(model_to_use, contents, config))
        self._current_stream_response = response
        return response

    async def _stream_generator(
        self,
        model: str,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> AsyncIterator[str]:
        """Internal generator that yields text and captures usage from chunks."""
        usage = None

        stream = await self._client.aio.models.generate_content_stream(
            model=model, contents=contents, config=config
        )
        async for chunk in stream:
            # Capture usage_metadata from chunks (available in final chunk)
            if chunk.usage_metadata:
                usage = {
                    "prompt_tokens": chunk.usage_metadata.prompt_token_count or 0,
                    "completion_tokens": chunk.usage_metadata.candidates_token_count or 0,
                    "total_tokens": chunk.usage_metadata.total_token_count or 0,
                }

            # Extract text from chunk, handling empty chunks
            text = self._extract_content(chunk)
            if text:
                yield text

        # Set usage after streaming completes
        if usage:
            self._current_stream_response.set_usage(usage)

    async def close(self) -> None:
        """Close the Gemini client.

        Note: The Google GenAI client doesn't require explicit closing,
        but we implement this for interface consistency.
        """
        pass
