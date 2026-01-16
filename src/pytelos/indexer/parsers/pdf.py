"""PDF parser implementation using pypdf.

Hidden design decisions:
- Using pypdf library for PDF extraction
- Page-based vs size-based chunking strategy
- Text cleaning and normalization
- Metadata extraction from PDF properties
"""

from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .base import DocumentChunk, DocumentParser


class PDFParser(DocumentParser):
    """Parser for PDF files.

    Extracts text from PDFs and chunks them either by page
    or by character count with overlap.
    """

    def __init__(self):
        """Initialize the PDF parser."""
        self._supported_extensions = {".pdf"}

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the file.

        Args:
            file_path: Path to the file

        Returns:
            True if file has .pdf extension
        """
        return file_path.suffix.lower() in self._supported_extensions

    def get_file_type(self) -> str:
        """Get the file type this parser handles.

        Returns:
            'pdf'
        """
        return "pdf"

    def get_supported_extensions(self) -> set[str]:
        """Get file extensions supported by this parser.

        Returns:
            Set containing '.pdf'
        """
        return self._supported_extensions

    def extract_metadata(self, file_path: Path) -> dict[str, Any]:
        """Extract PDF metadata.

        Args:
            file_path: Path to the PDF

        Returns:
            Dictionary with PDF metadata (title, author, etc.)
        """
        try:
            reader = PdfReader(file_path)
            metadata = reader.metadata or {}

            return {
                "title": metadata.get("/Title", "") or file_path.stem,
                "author": metadata.get("/Author", ""),
                "subject": metadata.get("/Subject", ""),
                "creator": metadata.get("/Creator", ""),
                "producer": metadata.get("/Producer", ""),
                "creation_date": str(metadata.get("/CreationDate", "")),
                "total_pages": len(reader.pages),
                "file_type": "pdf"
            }
        except Exception as e:
            # If metadata extraction fails, return basic info
            return {
                "title": file_path.stem,
                "error": str(e),
                "file_type": "pdf"
            }

    def parse_file(
        self,
        file_path: Path,
        chunk_size: int = 1000,
        preserve_structure: bool = True,
        **options: Any
    ) -> list[DocumentChunk]:
        """Parse PDF into chunks.

        Args:
            file_path: Path to the PDF file
            chunk_size: Target chunk size in characters
            preserve_structure: If True, chunk by pages; otherwise by size
            **options: Additional options

        Returns:
            List of document chunks

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid PDF
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.can_parse(file_path):
            raise ValueError(f"Not a PDF file: {file_path}")

        try:
            reader = PdfReader(file_path)
        except Exception as e:
            raise ValueError(f"Invalid PDF file: {e}") from e

        metadata = self.extract_metadata(file_path)
        chunks = []

        if preserve_structure:
            # One chunk per page
            chunks = self._chunk_by_pages(file_path, reader, metadata)
        else:
            # Chunk by character count with overlap
            chunks = self._chunk_by_size(file_path, reader, metadata, chunk_size)

        return chunks

    def _chunk_by_pages(
        self,
        file_path: Path,
        reader: PdfReader,
        metadata: dict[str, Any]
    ) -> list[DocumentChunk]:
        """Create one chunk per page.

        Args:
            file_path: Path to the PDF
            reader: PDF reader instance
            metadata: Document metadata

        Returns:
            List of page-based chunks
        """
        chunks = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()

                # Skip empty pages
                if not text.strip():
                    continue

                chunk = DocumentChunk(
                    content=text,
                    file_path=str(file_path),
                    start_offset=page_num,  # Use page number as offset
                    end_offset=page_num,
                    page_number=page_num,
                    chunk_index=len(chunks),
                    metadata=metadata
                )
                chunks.append(chunk)
            except Exception as e:
                # Skip pages that fail to extract
                print(f"Warning: Failed to extract page {page_num}: {e}")
                continue

        return chunks

    def _chunk_by_size(
        self,
        file_path: Path,
        reader: PdfReader,
        metadata: dict[str, Any],
        chunk_size: int
    ) -> list[DocumentChunk]:
        """Chunk by character count with overlap.

        Args:
            file_path: Path to the PDF
            reader: PDF reader instance
            metadata: Document metadata
            chunk_size: Target chunk size

        Returns:
            List of size-based chunks
        """
        # Extract all text and track page boundaries
        all_text = ""
        page_boundaries = []

        for page in reader.pages:
            try:
                page_start = len(all_text)
                page_text = page.extract_text()
                all_text += page_text + "\n\n"  # Add spacing between pages
                page_boundaries.append((page_start, len(all_text)))
            except Exception:
                # Skip pages that fail
                continue

        # Chunk by size with 10% overlap
        overlap = int(chunk_size * 0.1)
        chunks = []
        start = 0

        while start < len(all_text):
            end = min(start + chunk_size, len(all_text))
            chunk_text = all_text[start:end]

            # Find which page this chunk belongs to
            page_num = None
            for idx, (page_start, page_end) in enumerate(page_boundaries, start=1):
                if start >= page_start and start < page_end:
                    page_num = idx
                    break

            chunk = DocumentChunk(
                content=chunk_text,
                file_path=str(file_path),
                start_offset=max(1, start),  # Ensure offset is at least 1
                end_offset=max(1, end),
                page_number=page_num,
                chunk_index=len(chunks),
                metadata=metadata
            )
            chunks.append(chunk)

            start = end - overlap

        return chunks
