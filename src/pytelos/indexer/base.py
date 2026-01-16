"""Base parser abstractions for all file types.

This module defines the fundamental parser interfaces that all parsers
must implement, following Parnas's information hiding principles.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .models import ChunkingStrategy, ParsedChunk


class Chunk(BaseModel):
    """Universal chunk representation for any file type.

    This is the base chunk type that can represent code, documents,
    data files, etc. Subclasses add type-specific metadata.
    """

    content: str = Field(description="The actual content of the chunk")
    file_path: str = Field(description="Path to the source file")
    start_offset: int = Field(description="Character offset where chunk starts")
    end_offset: int = Field(description="Character offset where chunk ends")
    chunk_type: str = Field(description="Type of chunk: code, document, data, etc.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible metadata specific to chunk type"
    )
    chunk_id: str | None = Field(
        default=None,
        description="Unique identifier (generated during storage)"
    )

    @property
    def size(self) -> int:
        """Get chunk size in characters."""
        return len(self.content)

    @property
    def display_name(self) -> str:
        """Get a display name for this chunk."""
        return f"{Path(self.file_path).name}:{self.start_offset}-{self.end_offset}"


class FileParser(ABC):
    """Abstract base class for all file parsers.

    This interface hides the parsing implementation details and provides
    a uniform way to parse any file type into chunks.

    Hidden design decisions:
    - Parsing library choice
    - File reading and encoding
    - Chunking algorithms
    - Metadata extraction
    """

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file.

        Args:
            file_path: Path to the file

        Returns:
            True if this parser supports the file type
        """

    @abstractmethod
    def parse_file(
        self,
        file_path: Path,
        chunk_size: int = 1000,
        **options: Any
    ) -> list[Chunk]:
        """Parse a file into chunks.

        Args:
            file_path: Path to the file to parse
            chunk_size: Target chunk size in characters
            **options: Parser-specific options

        Returns:
            List of chunks

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """

    @abstractmethod
    def get_file_type(self) -> str:
        """Get the file type this parser handles.

        Returns:
            File type identifier (e.g., 'python', 'pdf', 'markdown')
        """

    def get_supported_extensions(self) -> set[str]:
        """Get file extensions supported by this parser.

        Returns:
            Set of file extensions (e.g., {'.py', '.pyi'})
        """
        return set()


class CodeParser(FileParser):
    """Abstract base class for code parsers.

    Inherits from FileParser and adds code-specific functionality.

    This module hides the design decision of which parsing library to use
    and how to extract code structure.

    Hidden design decisions:
    - AST parsing library choice (tree-sitter, ast module, etc.)
    - Language-specific parsing rules
    - Metadata extraction logic
    - Chunking algorithm implementation
    """

    @abstractmethod
    def parse_file(
        self,
        file_path: Path,
        chunk_size: int = 1000,
        strategy: ChunkingStrategy = ChunkingStrategy.BY_FUNCTION,
        overlap: int = 200,
        **options: Any
    ) -> list[ParsedChunk]:
        """Parse a code file into chunks.

        Args:
            file_path: Path to the file to parse
            chunk_size: Maximum chunk size in characters (for BY_LINES strategy)
            strategy: Chunking strategy to use
            overlap: Number of overlapping characters (for BY_LINES strategy)
            **options: Additional parser-specific options

        Returns:
            List of parsed chunks

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """

    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """Check if this parser supports a given language.

        Args:
            language: Language identifier (e.g., 'python', 'javascript')

        Returns:
            True if language is supported
        """

    @abstractmethod
    def detect_language(self, file_path: Path) -> str | None:
        """Detect the programming language of a file.

        Args:
            file_path: Path to the file

        Returns:
            Language identifier or None if cannot be detected
        """

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the file.

        Implementation uses detect_language for code files.

        Args:
            file_path: Path to the file

        Returns:
            True if this parser can handle the file
        """
        return self.detect_language(file_path) is not None
