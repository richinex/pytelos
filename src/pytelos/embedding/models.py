import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field


class EmbeddingResponse(BaseModel):
    """Response from an embedding provider."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding: list[float] = Field(description="The embedding vector")
    model: str = Field(description="Model that generated the embedding")
    usage: dict[str, int] | None = Field(
        default=None,
        description="Token usage information"
    )

    def to_numpy(self) -> NDArray[np.float32]:
        """Convert embedding to numpy array.

        Returns:
            Numpy array of shape (dimension,) with float32 dtype
        """
        return np.array(self.embedding, dtype=np.float32)
