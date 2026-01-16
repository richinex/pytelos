"""
Pytelos: A highly modularized codebase indexer with semantic search and RAG capabilities.

This package follows Parnas's information hiding principles,
where each module hides a specific design decision.
"""

__version__ = "0.1.0"

# Storage module is complete and ready to use
from .storage import (
    CodeChunk,
    IndexConfig,
    SearchResult,
    StorageBackend,
    StorageStats,
    create_storage_backend,
)

__all__ = [
    "CodeChunk",
    "IndexConfig",
    "SearchResult",
    "StorageBackend",
    "StorageStats",
    "create_storage_backend",
]
