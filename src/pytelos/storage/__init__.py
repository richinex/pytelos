"""Storage abstraction layer for pytelos."""

from .base import StorageBackend
from .factory import create_storage_backend
from .models import CodeChunk, IndexConfig, SearchResult, StorageStats

__all__ = [
    "StorageBackend",
    "create_storage_backend",
    "CodeChunk",
    "IndexConfig",
    "SearchResult",
    "StorageStats",
]
