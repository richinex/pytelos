"""Factory for creating storage backends."""

from typing import Any

from .base import StorageBackend
from .postgres import PostgresBackend


def create_storage_backend(backend: str = "postgres", **config: Any) -> StorageBackend:
    """
    Create a storage backend instance.

    This factory function hides the implementation details of which
    backend is being used, following Parnas's information hiding principle.

    Args:
        backend: Backend type ("postgres" currently supported)
        **config: Backend-specific configuration

    Returns:
        Initialized storage backend

    Raises:
        ValueError: If backend type is not supported

    Example:
        >>> storage = create_storage_backend(
        ...     backend="postgres",
        ...     host="localhost",
        ...     port=5433,
        ...     database="pytelos",
        ...     user="pytelos",
        ...     password="pytelos_dev"
        ... )
        >>> await storage.connect()
    """
    if backend == "postgres":
        return PostgresBackend(**config)

    raise ValueError(
        f"Unsupported storage backend: {backend}. "
        f"Supported backends: postgres"
    )
