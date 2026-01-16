from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    This module hides the design decision of which embedding provider to use.
    Implementations must handle provider-specific details like:
    - API client setup and authentication
    - Batch processing strategy
    - Normalization methods
    - Caching mechanisms
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this provider's model.

        Returns:
            Embedding dimension (e.g., 1536 for text-embedding-3-small)
        """
        pass

    @abstractmethod
    async def embed_text(self, text: str, **kwargs: Any) -> NDArray[np.float32]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed
            **kwargs: Provider-specific parameters

        Returns:
            Numpy array of shape (dimension,) with float32 dtype

        Raises:
            Exception: Provider-specific errors during embedding generation
        """
        pass

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
        **kwargs: Any
    ) -> list[NDArray[np.float32]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            **kwargs: Provider-specific parameters

        Returns:
            List of numpy arrays, each of shape (dimension,) with float32 dtype

        Raises:
            Exception: Provider-specific errors during embedding generation
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections or resources."""
        pass
