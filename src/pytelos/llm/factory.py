from typing import Any

from .base import LLMProvider
from .providers import AnthropicProvider, DeepSeekProvider, GeminiProvider, OpenAIProvider


def create_llm_provider(provider: str, **config: Any) -> LLMProvider:
    """Create an LLM provider instance.

    This factory function hides the instantiation logic for different providers.

    Args:
        provider: Provider type ('openai', 'deepseek', 'anthropic', 'gemini')
        **config: Provider-specific configuration
            For OpenAI:
                - api_key: str (required)
                - model: str (default: 'gpt-4o')
                - base_url: str | None
                - organization: str | None
            For DeepSeek:
                - api_key: str (required)
                - model: str (default: 'deepseek-chat')
                - base_url: str (default: 'https://api.deepseek.com')
            For Anthropic (Claude):
                - api_key: str (required)
                - model: str (default: 'claude-sonnet-4-20250514')
                - base_url: str | None
            For Gemini:
                - api_key: str (required)
                - model: str (default: 'gemini-2.5-flash')

    Returns:
        Initialized LLM provider instance

    Raises:
        ValueError: If provider type is not supported
        TypeError: If required configuration is missing

    Examples:
        >>> provider = create_llm_provider(
        ...     "deepseek",
        ...     api_key="sk-...",
        ...     model="deepseek-chat"
        ... )

        >>> provider = create_llm_provider(
        ...     "anthropic",
        ...     api_key="sk-ant-...",
        ...     model="claude-sonnet-4-20250514"
        ... )

        >>> provider = create_llm_provider(
        ...     "gemini",
        ...     api_key="...",
        ...     model="gemini-2.5-flash"
        ... )
    """
    provider_lower = provider.lower()

    if provider_lower == "deepseek":
        if "api_key" not in config:
            raise TypeError("DeepSeek provider requires 'api_key' in config")
        return DeepSeekProvider(**config)

    if provider_lower == "openai":
        if "api_key" not in config:
            raise TypeError("OpenAI provider requires 'api_key' in config")
        return OpenAIProvider(**config)

    if provider_lower in ("anthropic", "claude"):
        if "api_key" not in config:
            raise TypeError("Anthropic provider requires 'api_key' in config")
        return AnthropicProvider(**config)

    if provider_lower == "gemini":
        if "api_key" not in config:
            raise TypeError("Gemini provider requires 'api_key' in config")
        return GeminiProvider(**config)

    raise ValueError(
        f"Unsupported provider: {provider}. "
        f"Supported providers: 'openai', 'deepseek', 'anthropic', 'gemini'"
    )
