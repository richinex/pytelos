"""Base classes for document parsers.

This module provides the base class for parsing non-code documents
like PDFs, Markdown, plain text, YAML, Terraform, etc.

Hidden design decisions:
- Document parsing library choice
- Text extraction methods
- Section/page detection
- Metadata extraction
"""

from abc import abstractmethod
from typing import Any

from pydantic import Field

from ..base import Chunk, FileParser


class DocumentChunk(Chunk):
    """Document-specific chunk with structured metadata.

    Extends the base Chunk with document-specific fields like
    page numbers, sections, headings, etc.
    """

    chunk_type: str = Field(default="document", description="Type is always document")
    page_number: int | None = Field(
        default=None,
        description="Page number if applicable (PDFs)"
    )
    section_title: str | None = Field(
        default=None,
        description="Section or heading title"
    )
    heading_level: int | None = Field(
        default=None,
        description="Heading level (1-6 for markdown)"
    )
    chunk_index: int = Field(
        description="Sequential index within document"
    )

    @property
    def display_name(self) -> str:
        """Get a display name for this chunk."""
        from pathlib import Path

        name = Path(self.file_path).name

        if self.section_title:
            return f"{name} - {self.section_title}"
        elif self.page_number:
            return f"{name} - Page {self.page_number}"
        else:
            return f"{name} - Chunk {self.chunk_index}"


class DocumentParser(FileParser):
    """Abstract base class for document parsers.

    Provides common functionality for parsing text documents
    like PDFs, Markdown, plain text, YAML, Terraform, etc.

    Hidden design decisions:
    - Document parsing library (pypdf, python-markdown, etc.)
    - Text extraction and cleaning
    - Structure preservation (pages, sections)
    """

    @abstractmethod
    def extract_metadata(self, file_path) -> dict[str, Any]:
        """Extract document-level metadata.

        Args:
            file_path: Path to the document

        Returns:
            Dictionary with metadata (title, author, date, etc.)
        """

    @abstractmethod
    def parse_file(
        self,
        file_path,
        chunk_size: int = 1000,
        preserve_structure: bool = True,
        **options: Any
    ) -> list[DocumentChunk]:
        """Parse document into chunks.

        Args:
            file_path: Path to the document
            chunk_size: Target chunk size in characters
            preserve_structure: If True, preserve pages/sections
            **options: Parser-specific options

        Returns:
            List of document chunks
        """
