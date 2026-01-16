"""Data models for storage layer."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


class CodeChunk(BaseModel):
    """Represents a chunk of code with metadata."""

    model_config = ConfigDict(ser_json_timedelta="iso8601")

    id: str = Field(default_factory=lambda: str(uuid4()))
    file_path: str = Field(description="Path to the source file")
    chunk_text: str = Field(description="The actual code content")
    start_line: int = Field(ge=1, description="Starting line number")
    end_line: int = Field(ge=1, description="Ending line number")
    language: str = Field(description="Programming language")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (function name, class name, etc.)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("end_line")
    @classmethod
    def validate_line_numbers(cls, v: int, info) -> int:
        """Ensure end_line >= start_line."""
        if "start_line" in info.data and v < info.data["start_line"]:
            raise ValueError("end_line must be >= start_line")
        return v

    @field_serializer("created_at")
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format."""
        return value.isoformat()


class SearchResult(BaseModel):
    """Represents a search result with score and rank."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk: CodeChunk
    score: float = Field(ge=0.0, description="Relevance score")
    rank: int = Field(ge=1, description="Result rank")
    search_type: str = Field(description="Type of search: vector, bm25, or hybrid")


class StorageStats(BaseModel):
    """Storage backend statistics."""

    total_chunks: int = Field(ge=0)
    total_files: int = Field(ge=0)
    languages: dict[str, int] = Field(default_factory=dict)
    total_size_bytes: int = Field(ge=0)
    last_indexed: datetime | None = None

    @field_serializer("last_indexed")
    def serialize_datetime(self, value: datetime | None) -> str | None:
        """Serialize datetime to ISO format."""
        return value.isoformat() if value else None


class IndexConfig(BaseModel):
    """Configuration for vector and BM25 indexes."""

    # Vector index config (HNSW or IVFFlat)
    vector_index_type: str = Field(
        default="hnsw",
        description="Vector index type: hnsw or ivfflat"
    )
    hnsw_m: int = Field(default=16, ge=4, le=64, description="HNSW m parameter")
    hnsw_ef_construction: int = Field(
        default=64,
        ge=8,
        le=512,
        description="HNSW ef_construction"
    )
    ivfflat_lists: int = Field(
        default=100,
        ge=1,
        description="IVFFlat number of lists"
    )

    # BM25 config
    bm25_k1: float = Field(
        default=1.2,
        ge=0.1,
        le=10.0,
        description="BM25 term frequency saturation"
    )
    bm25_b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 length normalization"
    )
    text_config: str = Field(
        default="english",
        description="PostgreSQL text search configuration"
    )

    @field_validator("vector_index_type")
    @classmethod
    def validate_index_type(cls, v: str) -> str:
        """Validate vector index type."""
        if v not in ("hnsw", "ivfflat"):
            raise ValueError("vector_index_type must be 'hnsw' or 'ivfflat'")
        return v
