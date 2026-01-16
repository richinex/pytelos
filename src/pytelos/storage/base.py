"""Abstract base class for storage backends."""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .models import CodeChunk, IndexConfig, SearchResult, StorageStats


class StorageBackend(ABC):
    """
    Abstract storage backend.

    Hides all database implementation details including:
    - Connection management
    - Query construction
    - Index creation and configuration
    - Transaction handling
    """

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish database connection.

        Raises:
            ConnectionError: If connection fails
        """

    @abstractmethod
    async def disconnect(self) -> None:
        """Close database connection gracefully."""

    @abstractmethod
    async def initialize_schema(self, config: IndexConfig) -> None:
        """
        Initialize database schema and indexes.

        Args:
            config: Index configuration

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If schema creation fails
        """

    @abstractmethod
    async def store_chunk(
        self,
        chunk: CodeChunk,
        embedding: NDArray[np.float32]
    ) -> str:
        """
        Store code chunk with its embedding.

        Args:
            chunk: Code chunk to store
            embedding: Vector embedding for the chunk

        Returns:
            Chunk ID

        Raises:
            ValueError: If embedding dimension doesn't match
            RuntimeError: If storage fails
        """

    @abstractmethod
    async def store_chunks_batch(
        self,
        chunks: list[CodeChunk],
        embeddings: list[NDArray[np.float32]]
    ) -> list[str]:
        """
        Store multiple chunks in a batch (more efficient).

        Args:
            chunks: List of code chunks
            embeddings: Corresponding embeddings

        Returns:
            List of chunk IDs

        Raises:
            ValueError: If lengths don't match or embeddings invalid
            RuntimeError: If batch storage fails
        """

    @abstractmethod
    async def get_chunk(self, chunk_id: str) -> CodeChunk | None:
        """
        Retrieve chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            CodeChunk if found, None otherwise
        """

    @abstractmethod
    async def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete chunk by ID.

        Args:
            chunk_id: Chunk identifier

        Returns:
            True if deleted, False if not found
        """

    @abstractmethod
    async def delete_by_file(self, file_path: str) -> int:
        """
        Delete all chunks from a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Number of chunks deleted
        """

    @abstractmethod
    async def vector_search(
        self,
        query_embedding: NDArray[np.float32],
        limit: int = 10,
        file_filter: str | None = None,
        language_filter: str | None = None
    ) -> list[SearchResult]:
        """
        Perform vector similarity search.

        Args:
            query_embedding: Query vector
            limit: Maximum results to return
            file_filter: Optional file path filter
            language_filter: Optional language filter

        Returns:
            Ranked search results

        Raises:
            ValueError: If embedding dimension doesn't match
        """

    @abstractmethod
    async def bm25_search(
        self,
        query: str,
        limit: int = 10,
        file_filter: str | None = None,
        language_filter: str | None = None
    ) -> list[SearchResult]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query text
            limit: Maximum results to return
            file_filter: Optional file path filter
            language_filter: Optional language filter

        Returns:
            Ranked search results
        """

    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        query_embedding: NDArray[np.float32],
        limit: int = 10,
        alpha: float = 0.5,
        file_filter: str | None = None,
        language_filter: str | None = None
    ) -> list[SearchResult]:
        """
        Perform hybrid search using Reciprocal Rank Fusion.

        Args:
            query: Search query text
            query_embedding: Query vector
            limit: Maximum results to return
            alpha: Weight for vector vs BM25 (0.0 = all BM25, 1.0 = all vector)
            file_filter: Optional file path filter
            language_filter: Optional language filter

        Returns:
            Ranked search results combining both methods

        Raises:
            ValueError: If alpha not in [0, 1] or embedding dimension invalid
        """

    @abstractmethod
    async def get_stats(self) -> StorageStats:
        """
        Get storage statistics.

        Returns:
            Storage statistics
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if storage backend is healthy.

        Returns:
            True if healthy, False otherwise
        """

    @abstractmethod
    async def drop_schema(self) -> None:
        """
        Drop all database schema (tables, indexes, views).

        WARNING: This destroys all data!

        Raises:
            RuntimeError: If drop operation fails
        """

    @abstractmethod
    async def clear_all(self) -> int:
        """
        Delete all chunks from the database.

        WARNING: This destroys all indexed data!

        Returns:
            Number of chunks deleted

        Raises:
            RuntimeError: If clear operation fails
        """


class StorageFactory(Protocol):
    """Factory protocol for creating storage backends."""

    def __call__(self, **config) -> StorageBackend:
        """Create storage backend instance."""
        ...
