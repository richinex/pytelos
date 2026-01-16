"""Terraform HCL parser implementation using python-hcl2.

Hidden design decisions:
- Using python-hcl2 library for HCL2 parsing
- Resource-level vs file-level chunking strategy
- Text extraction and formatting
- Metadata extraction from HCL structure
"""

from pathlib import Path
from typing import Any

import hcl2

from .base import DocumentChunk, DocumentParser


class TerraformParser(DocumentParser):
    """Parser for Terraform HCL files.

    Extracts text from Terraform configuration files and chunks them
    by preserving resource structure.
    """

    def __init__(self):
        """Initialize the Terraform parser."""
        self._supported_extensions = {".tf", ".hcl"}

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the file.

        Args:
            file_path: Path to the file

        Returns:
            True if file has .tf or .hcl extension
        """
        return file_path.suffix.lower() in self._supported_extensions

    def get_file_type(self) -> str:
        """Get the file type this parser handles.

        Returns:
            'terraform'
        """
        return "terraform"

    def get_supported_extensions(self) -> set[str]:
        """Get file extensions supported by this parser.

        Returns:
            Set containing '.tf' and '.hcl'
        """
        return self._supported_extensions

    def extract_metadata(self, file_path: Path) -> dict[str, Any]:
        """Extract Terraform metadata.

        Args:
            file_path: Path to the Terraform file

        Returns:
            Dictionary with Terraform metadata
        """
        try:
            with open(file_path, encoding='utf-8') as f:
                data = hcl2.load(f)

            resource_count = len(data.get("resource", []))
            module_count = len(data.get("module", []))
            data_source_count = len(data.get("data", []))
            variable_count = len(data.get("variable", []))
            output_count = len(data.get("output", []))

            return {
                "title": file_path.stem,
                "resource_count": resource_count,
                "module_count": module_count,
                "data_source_count": data_source_count,
                "variable_count": variable_count,
                "output_count": output_count,
                "file_type": "terraform"
            }
        except Exception as e:
            return {
                "title": file_path.stem,
                "error": str(e),
                "file_type": "terraform"
            }

    def parse_file(
        self,
        file_path: Path,
        chunk_size: int = 1000,
        preserve_structure: bool = True,
        **options: Any
    ) -> list[DocumentChunk]:
        """Parse Terraform into chunks.

        Args:
            file_path: Path to the Terraform file
            chunk_size: Target chunk size in characters
            preserve_structure: If True, chunk by resources; otherwise by size
            **options: Additional options

        Returns:
            List of document chunks

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not valid HCL
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.can_parse(file_path):
            raise ValueError(f"Not a Terraform file: {file_path}")

        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Cannot read Terraform file: {e}") from e

        metadata = self.extract_metadata(file_path)
        chunks = []

        if preserve_structure:
            # Chunk by resources/blocks
            chunks = self._chunk_by_blocks(file_path, content, metadata)
        else:
            # Chunk by character count
            chunks = self._chunk_by_size(file_path, content, metadata, chunk_size)

        return chunks

    def _chunk_by_blocks(
        self,
        file_path: Path,
        content: str,
        metadata: dict[str, Any]
    ) -> list[DocumentChunk]:
        """Create chunks by Terraform blocks (resource, module, etc.).

        Args:
            file_path: Path to the Terraform file
            content: File content
            metadata: Document metadata

        Returns:
            List of block-based chunks
        """
        chunks = []

        # Simple approach: split by top-level blocks
        # More sophisticated parsing would use the HCL2 AST
        lines = content.split("\n")
        current_block = []
        block_start_line = 0
        in_block = False
        brace_count = 0

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Check if starting a new block
            if not in_block and any(stripped.startswith(keyword) for keyword in
                                   ["resource", "module", "data", "variable", "output", "locals", "terraform", "provider"]):
                in_block = True
                block_start_line = line_num
                current_block = [line]
                brace_count = line.count("{") - line.count("}")
            elif in_block:
                current_block.append(line)
                brace_count += line.count("{") - line.count("}")

                # Block complete when braces balanced
                if brace_count <= 0:
                    block_text = "\n".join(current_block)

                    # Extract block type and name
                    first_line = current_block[0].strip()
                    parts = first_line.split('"')
                    block_type = first_line.split()[0]
                    block_name = parts[1] if len(parts) > 1 else ""

                    block_metadata = metadata.copy()
                    block_metadata["block_type"] = block_type
                    block_metadata["block_name"] = block_name

                    chunk = DocumentChunk(
                        content=block_text,
                        file_path=str(file_path),
                        start_offset=block_start_line,
                        end_offset=line_num,
                        page_number=None,
                        chunk_index=len(chunks),
                        metadata=block_metadata
                    )
                    chunks.append(chunk)

                    in_block = False
                    current_block = []
            elif stripped and not stripped.startswith("#"):
                # Standalone line (comment, empty, etc.)
                pass

        # Handle any remaining block
        if current_block:
            block_text = "\n".join(current_block)
            chunk = DocumentChunk(
                content=block_text,
                file_path=str(file_path),
                start_offset=block_start_line,
                end_offset=len(lines),
                page_number=None,
                chunk_index=len(chunks),
                metadata=metadata
            )
            chunks.append(chunk)

        # If no blocks found, return whole file as one chunk
        if not chunks:
            chunks = [
                DocumentChunk(
                    content=content,
                    file_path=str(file_path),
                    start_offset=1,
                    end_offset=len(lines),
                    page_number=None,
                    chunk_index=0,
                    metadata=metadata
                )
            ]

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
            file_path: Path to the Terraform file
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
