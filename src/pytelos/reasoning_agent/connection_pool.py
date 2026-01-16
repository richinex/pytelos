"""Connection pooling for distributed reasoning flows.

This module provides singleton connection pools for LLM providers, storage backends,
and embedding providers to eliminate repeated connection overhead in Pyergon flows.

Following Parnas principles, this module hides:
- Connection lifecycle management
- Pool initialization and cleanup
- Thread-safety for concurrent access
"""

import asyncio
from typing import Any

from ..embedding import EmbeddingProvider, create_embedding_provider
from ..llm import LLMProvider, create_llm_provider
from ..storage import StorageBackend, create_storage_backend


class ConnectionPool:
    """Singleton connection pool for shared resources across Pyergon flows.

    This eliminates the overhead of creating/destroying connections for every
    flow invocation. Connections are created once and reused across all flows.

    Thread-safe for concurrent access from multiple workers.
    """

    _lock = asyncio.Lock()
    _llm_pool: dict[str, LLMProvider] = {}
    _storage_pool: dict[str, StorageBackend] = {}
    _embedder_pool: dict[str, EmbeddingProvider] = {}

    @classmethod
    async def get_llm(
        cls,
        config: dict[str, Any]
    ) -> LLMProvider:
        """Get or create LLM provider from pool.

        Args:
            config: LLM configuration with "provider" and "kwargs"

        Returns:
            Cached or newly created LLM provider
        """
        async with cls._lock:
            # Create pool key from config
            provider = config.get("provider", "unknown")
            kwargs = config.get("kwargs", {})
            model = kwargs.get("model", "default")
            key = f"{provider}:{model}"

            if key not in cls._llm_pool:
                cls._llm_pool[key] = create_llm_provider(
                    provider,
                    **kwargs
                )

            return cls._llm_pool[key]

    @classmethod
    async def get_storage(
        cls,
        config: dict[str, Any]
    ) -> StorageBackend:
        """Get or create storage backend from pool.

        Args:
            config: Storage configuration with "backend" and "kwargs"

        Returns:
            Cached or newly created storage backend
        """
        async with cls._lock:
            # Create pool key from config
            backend = config.get("backend", "postgres")
            kwargs = config.get("kwargs", {})
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", 5433)
            database = kwargs.get("database", "pytelos")
            key = f"{backend}:{host}:{port}:{database}"

            if key not in cls._storage_pool:
                storage = create_storage_backend(backend, **kwargs)
                await storage.connect()
                cls._storage_pool[key] = storage

            return cls._storage_pool[key]

    @classmethod
    async def get_embedder(
        cls,
        config: dict[str, Any]
    ) -> EmbeddingProvider:
        """Get or create embedding provider from pool.

        Args:
            config: Embedder configuration with "provider" and "kwargs"

        Returns:
            Cached or newly created embedding provider
        """
        async with cls._lock:
            # Create pool key from config
            provider = config.get("provider", "openai")
            kwargs = config.get("kwargs", {})
            model = kwargs.get("model", "text-embedding-3-small")
            key = f"{provider}:{model}"

            if key not in cls._embedder_pool:
                cls._embedder_pool[key] = create_embedding_provider(
                    provider,
                    **kwargs
                )

            return cls._embedder_pool[key]

    @classmethod
    async def close_all(cls) -> None:
        """Close all pooled connections.

        Called during shutdown to cleanup resources properly.
        """
        async with cls._lock:
            # Close all LLMs
            for llm in cls._llm_pool.values():
                try:
                    await llm.close()
                except Exception:
                    pass
            cls._llm_pool.clear()

            # Close all storage backends
            for storage in cls._storage_pool.values():
                try:
                    await storage.disconnect()
                except Exception:
                    pass
            cls._storage_pool.clear()

            # Close all embedders
            for embedder in cls._embedder_pool.values():
                try:
                    await embedder.close()
                except Exception:
                    pass
            cls._embedder_pool.clear()

    @classmethod
    async def warmup(
        cls,
        llm_config: dict[str, Any],
        storage_config: dict[str, Any] | None = None,
        embedder_config: dict[str, Any] | None = None
    ) -> None:
        """Pre-warm connections in parallel.

        This initializes connections while workers are starting,
        reducing latency for first flow invocation.

        Args:
            llm_config: LLM configuration
            storage_config: Optional storage configuration
            embedder_config: Optional embedder configuration
        """
        tasks = [cls.get_llm(llm_config)]

        if storage_config:
            tasks.append(cls.get_storage(storage_config))

        if embedder_config:
            tasks.append(cls.get_embedder(embedder_config))

        await asyncio.gather(*tasks)
