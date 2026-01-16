from typing import Any

from ..embedding import EmbeddingProvider
from ..llm import LLMProvider
from ..storage import StorageBackend
from .base import SearchEngine
from .engine import DefaultSearchEngine


def create_search_engine(
    storage: StorageBackend,
    embedder: EmbeddingProvider,
    llm: LLMProvider | None = None,
    **config: Any
) -> SearchEngine:
    """Create a search engine instance.

    This factory function hides the instantiation logic for search engines.

    Args:
        storage: Storage backend instance
        embedder: Embedding provider instance
        llm: Optional LLM provider for query expansion/re-ranking
        **config: Additional configuration
            - alpha: float (default: 0.5)
              Weight for hybrid search:
              * 0.0 = keyword/BM25 only
              * 1.0 = vector/semantic only
              * 0.5 = balanced (default)

    Returns:
        Initialized search engine instance

    Examples:
        >>> engine = create_search_engine(
        ...     storage=storage_backend,
        ...     embedder=embedding_provider,
        ...     llm=llm_provider,
        ...     alpha=0.7  # Favor vector search
        ... )
    """
    return DefaultSearchEngine(
        storage=storage,
        embedder=embedder,
        llm=llm,
        alpha=config.get("alpha", 0.5)
    )
