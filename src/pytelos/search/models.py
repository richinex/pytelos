from enum import Enum

from pydantic import BaseModel, Field

from ..storage.models import SearchResult as StorageSearchResult


class SearchMode(str, Enum):
    """Search mode for query execution."""

    VECTOR = "vector"      # Semantic/vector search only
    KEYWORD = "keyword"    # BM25/keyword search only
    HYBRID = "hybrid"      # Combined vector + keyword search


class RerankStrategy(str, Enum):
    """Strategy for re-ranking search results."""

    NONE = "none"                    # No re-ranking
    RECIPROCAL_RANK_FUSION = "rrf"   # Reciprocal Rank Fusion (used in hybrid search)


class SearchQuery(BaseModel):
    """A search query with metadata."""

    query: str = Field(description="The search query text")
    mode: SearchMode = Field(
        default=SearchMode.HYBRID,
        description="Search mode to use"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results"
    )
    filters: dict[str, str] | None = Field(
        default=None,
        description="Optional filters (e.g., language, file_path)"
    )
    rerank: RerankStrategy = Field(
        default=RerankStrategy.NONE,
        description="Re-ranking strategy"
    )


class SearchResponse(BaseModel):
    """Search response with results and metadata."""

    query: SearchQuery = Field(description="Original query")
    results: list[StorageSearchResult] = Field(description="Search results")
    total_results: int = Field(description="Total number of results")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    expanded_query: str | None = Field(
        default=None,
        description="Query after expansion/preprocessing"
    )
