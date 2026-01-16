from typing import Any

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI

from ..base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider implementation.

    Hidden design decisions:
    - OpenAI API client initialization
    - Batch processing strategy
    - Response format handling
    - Authentication mechanism
    """

    # Dimensions for different OpenAI embedding models
    _MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        organization: str | None = None,
        **client_kwargs: Any
    ):
        """Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key
            model: Embedding model to use (default: text-embedding-3-small)
            base_url: Optional custom API base URL
            organization: Optional organization ID
            **client_kwargs: Additional kwargs for AsyncOpenAI client
        """
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            **client_kwargs
        )

        if model not in self._MODEL_DIMENSIONS:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Supported models: {list(self._MODEL_DIMENSIONS.keys())}"
            )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension for the configured model.

        Returns:
            Embedding dimension
        """
        return self._MODEL_DIMENSIONS[self._model]

    async def embed_text(self, text: str, **kwargs: Any) -> NDArray[np.float32]:
        """Generate embedding for a single text using OpenAI.

        Args:
            text: Text to embed
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            Numpy array of shape (dimension,) with float32 dtype
        """
        response = await self._client.embeddings.create(
            input=text,
            model=self._model,
            **kwargs
        )

        return np.array(response.data[0].embedding, dtype=np.float32)

    async def embed_batch(
        self,
        texts: list[str],
        **kwargs: Any
    ) -> list[NDArray[np.float32]]:
        """Generate embeddings for multiple texts using OpenAI.

        Args:
            texts: List of texts to embed
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            List of numpy arrays, each of shape (dimension,) with float32 dtype
        """
        if not texts:
            return []

        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
            **kwargs
        )

        return [
            np.array(item.embedding, dtype=np.float32)
            for item in response.data
        ]

    async def close(self) -> None:
        """Close the OpenAI client."""
        await self._client.close()
