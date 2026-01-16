from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class ChunkingStrategy(str, Enum):
    """Strategy for chunking code."""

    BY_FUNCTION = "by_function"  # Chunk by function/class definitions
    BY_LINES = "by_lines"        # Fixed-size line chunks with overlap
    SEMANTIC = "semantic"        # Semantic chunking based on code structure


class CodeChunkMetadata(BaseModel):
    """Metadata for a code chunk."""

    language: str = Field(description="Programming language")
    file_path: str = Field(description="Relative path to source file")
    start_line: int = Field(description="Starting line number (1-indexed)")
    end_line: int = Field(description="Ending line number (inclusive)")
    function_name: str | None = Field(
        default=None,
        description="Name of function/method if chunk is a function"
    )
    class_name: str | None = Field(
        default=None,
        description="Name of class if chunk is inside a class"
    )
    docstring: str | None = Field(
        default=None,
        description="Docstring if present"
    )
    imports: list[str] = Field(
        default_factory=list,
        description="Import statements in this chunk"
    )
    complexity: int | None = Field(
        default=None,
        description="Cyclomatic complexity estimate"
    )


class ParsedChunk(BaseModel):
    """A parsed code chunk ready for indexing."""

    content: str = Field(description="The actual code content")
    metadata: CodeChunkMetadata = Field(description="Chunk metadata")
    chunk_id: str | None = Field(
        default=None,
        description="Unique identifier (generated during storage)"
    )

    @property
    def display_name(self) -> str:
        """Get a display name for this chunk."""
        if self.metadata.function_name:
            if self.metadata.class_name:
                return f"{self.metadata.class_name}.{self.metadata.function_name}"
            return self.metadata.function_name
        return f"{Path(self.metadata.file_path).name}:{self.metadata.start_line}"


class IndexingResult(BaseModel):
    """Result of indexing a file or directory."""

    total_files: int = Field(description="Total files processed")
    total_chunks: int = Field(description="Total chunks created")
    failed_files: list[str] = Field(
        default_factory=list,
        description="Files that failed to process"
    )
    processing_time_seconds: float = Field(description="Total processing time")
