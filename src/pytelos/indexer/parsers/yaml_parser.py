"""YAML parser implementation using PyYAML.

Hidden design decisions:
- Using PyYAML library for YAML parsing
- Document-level vs key-level chunking strategy
- Text extraction and formatting
- Metadata extraction from YAML structure
"""

from pathlib import Path
from typing import Any

import yaml

from .base import DocumentChunk, DocumentParser


class YAMLParser(DocumentParser):
    """Parser for YAML files.

    Extracts text from YAML configuration files and chunks them
    by preserving document structure.
    """

    def __init__(self):
        """Initialize the YAML parser."""
        self._supported_extensions = {".yaml", ".yml"}

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the file.

        Args:
            file_path: Path to the file

        Returns:
            True if file has .yaml or .yml extension
        """
        return file_path.suffix.lower() in self._supported_extensions

    def get_file_type(self) -> str:
        """Get the file type this parser handles.

        Returns:
            'yaml'
        """
        return "yaml"

    def get_supported_extensions(self) -> set[str]:
        """Get file extensions supported by this parser.

        Returns:
            Set containing '.yaml' and '.yml'
        """
        return self._supported_extensions

    def extract_metadata(self, file_path: Path) -> dict[str, Any]:
        """Extract YAML metadata.

        Args:
            file_path: Path to the YAML file

        Returns:
            Dictionary with YAML metadata
        """
        try:
            with open(file_path, encoding='utf-8') as f:
                # Try to load first document
                data = yaml.safe_load(f)

            kind = None
            name = None
            namespace = None

            # Extract Kubernetes resource info if present
            if isinstance(data, dict):
                kind = data.get("kind", "")
                if "metadata" in data:
                    metadata = data["metadata"]
                    name = metadata.get("name", "")
                    namespace = metadata.get("namespace", "")

            return {
                "title": file_path.stem,
                "kind": kind,
                "name": name,
                "namespace": namespace,
                "file_type": "yaml"
            }
        except Exception as e:
            return {
                "title": file_path.stem,
                "error": str(e),
                "file_type": "yaml"
            }

    def parse_file(
        self,
        file_path: Path,
        chunk_size: int = 1000,
        preserve_structure: bool = True,
        **options: Any
    ) -> list[DocumentChunk]:
        """Parse YAML into chunks.

        Args:
            file_path: Path to the YAML file
            chunk_size: Target chunk size in characters
            preserve_structure: If True, chunk by documents; otherwise by size
            **options: Additional options

        Returns:
            List of document chunks

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not valid YAML
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.can_parse(file_path):
            raise ValueError(f"Not a YAML file: {file_path}")

        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Cannot read YAML file: {e}") from e

        metadata = self.extract_metadata(file_path)
        chunks = []

        if preserve_structure:
            # Chunk by YAML documents (separated by ---)
            chunks = self._chunk_by_documents(file_path, content, metadata)
        else:
            # Chunk by character count
            chunks = self._chunk_by_size(file_path, content, metadata, chunk_size)

        return chunks

    def _chunk_by_documents(
        self,
        file_path: Path,
        content: str,
        metadata: dict[str, Any]
    ) -> list[DocumentChunk]:
        """Create one chunk per YAML document.

        Args:
            file_path: Path to the YAML file
            content: File content
            metadata: Document metadata

        Returns:
            List of document-based chunks
        """
        chunks = []

        # Split by document separator
        documents = content.split("\n---\n")

        for doc_num, doc_text in enumerate(documents, start=1):
            if not doc_text.strip():
                continue

            # Try to parse document to get kind/name
            try:
                doc_data = yaml.safe_load(doc_text)
                doc_kind = doc_data.get("kind", "") if isinstance(doc_data, dict) else ""
                doc_name = ""
                if isinstance(doc_data, dict) and "metadata" in doc_data:
                    doc_name = doc_data["metadata"].get("name", "")

                doc_metadata = metadata.copy()
                doc_metadata["document_number"] = doc_num
                doc_metadata["kind"] = doc_kind
                doc_metadata["name"] = doc_name
            except Exception:
                doc_metadata = metadata.copy()
                doc_metadata["document_number"] = doc_num

            chunk = DocumentChunk(
                content=doc_text.strip(),
                file_path=str(file_path),
                start_offset=doc_num,
                end_offset=doc_num,
                page_number=doc_num,
                chunk_index=len(chunks),
                metadata=doc_metadata
            )
            chunks.append(chunk)

        return chunks

    def _chunk_by_size(
        self,
        file_path: Path,
        content: str,
        metadata: dict[str, Any],
        chunk_size: int
    ) -> list[DocumentChunk]:
        """Chunk by character count with overlap.

        Args:
            file_path: Path to the YAML file
            content: File content
            metadata: Document metadata
            chunk_size: Target chunk size

        Returns:
            List of size-based chunks
        """
        overlap = int(chunk_size * 0.1)
        chunks = []
        start = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end]

            chunk = DocumentChunk(
                content=chunk_text,
                file_path=str(file_path),
                start_offset=max(1, start),
                end_offset=max(1, end),
                page_number=None,
                chunk_index=len(chunks),
                metadata=metadata
            )
            chunks.append(chunk)

            start = end - overlap

        return chunks
