from typing import Any

from .base import EmbeddingProvider
from .providers import OpenAIEmbeddingProvider


def create_embedding_provider(provider: str, **config: Any) -> EmbeddingProvider:
    """Create an embedding provider instance.

    This factory function hides the instantiation logic for different providers.

    Args:
        provider: Provider type ('openai')
        **config: Provider-specific configuration
            For OpenAI:
                - api_key: str (required)
                - model: str (default: 'text-embedding-3-small')
                - base_url: str | None
                - organization: str | None

    Returns:
        Initialized embedding provider instance

    Raises:
        ValueError: If provider type is not supported
        TypeError: If required configuration is missing

    Examples:
        >>> provider = create_embedding_provider(
        ...     "openai",
        ...     api_key="sk-...",
        ...     model="text-embedding-3-small"
        ... )
    """
    provider_lower = provider.lower()

    if provider_lower == "openai":
        if "api_key" not in config:
            raise TypeError("OpenAI provider requires 'api_key' in config")
        return OpenAIEmbeddingProvider(**config)

    raise ValueError(
        f"Unsupported provider: {provider}. "
        f"Supported providers: 'openai'"
    )
