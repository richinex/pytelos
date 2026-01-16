"""Markdown parser implementation using markdown-it-py.

Hidden design decisions:
- Using markdown-it-py for CommonMark-compliant parsing
- python-frontmatter for YAML extraction
- Token-based section extraction
- Heading-based vs size-based chunking strategy
"""

from pathlib import Path
from typing import Any

import frontmatter
from markdown_it import MarkdownIt

from .base import DocumentChunk, DocumentParser


class MarkdownParser(DocumentParser):
    """Parser for Markdown files.

    Chunks markdown files either by headings (preserving document structure)
    or by character count with overlap.
    """

    def __init__(self):
        """Initialize the Markdown parser."""
        self._supported_extensions = {".md", ".markdown", ".mdown", ".mkd"}
        self._md = MarkdownIt()

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the file.

        Args:
            file_path: Path to the file

        Returns:
            True if file has markdown extension
        """
        return file_path.suffix.lower() in self._supported_extensions

    def get_file_type(self) -> str:
        """Get the file type this parser handles.

        Returns:
            'markdown'
        """
        return "markdown"

    def get_supported_extensions(self) -> set[str]:
        """Get file extensions supported by this parser.

        Returns:
            Set of markdown extensions
        """
        return self._supported_extensions

    def extract_metadata(self, file_path: Path) -> dict[str, Any]:
        """Extract markdown metadata using python-frontmatter.

        Parses YAML frontmatter if present.

        Args:
            file_path: Path to the markdown file

        Returns:
            Dictionary with metadata
        """
        try:
            with open(file_path, encoding='utf-8') as f:
                post = frontmatter.load(f)

            # Start with frontmatter data
            metadata = dict(post.metadata) if post.metadata else {}

            # Add default fields
            metadata["file_type"] = "markdown"
            if "title" not in metadata:
                metadata["title"] = file_path.stem

            return metadata
        except Exception as e:
            return {
                "title": file_path.stem,
                "error": str(e),
                "file_type": "markdown"
            }

    def parse_file(
        self,
        file_path: Path,
        chunk_size: int = 1000,
        preserve_structure: bool = True,
        **options: Any
    ) -> list[DocumentChunk]:
        """Parse markdown into chunks.

        Args:
            file_path: Path to the markdown file
            chunk_size: Target chunk size in characters
            preserve_structure: If True, chunk by headings; otherwise by size
            **options: Additional options

        Returns:
            List of document chunks

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid markdown file
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not self.can_parse(file_path):
            raise ValueError(f"Not a markdown file: {file_path}")

        try:
            with open(file_path, encoding='utf-8') as f:
                post = frontmatter.load(f)
            # Get content without frontmatter
            content = post.content
        except Exception as e:
            raise ValueError(f"Failed to read markdown file: {e}") from e

        metadata = self.extract_metadata(file_path)
        chunks = []

        if preserve_structure:
            # Chunk by headings using markdown-it-py
            chunks = self._chunk_by_headings(file_path, content, metadata)
        else:
            # Chunk by size
            chunks = self._chunk_by_size(file_path, content, metadata, chunk_size)

        return chunks

    def _chunk_by_headings(
        self,
        file_path: Path,
        content: str,
        metadata: dict[str, Any]
    ) -> list[DocumentChunk]:
        """Chunk by markdown headings using markdown-it-py tokens.

        Args:
            file_path: Path to the file
            content: File content (without frontmatter)
            metadata: Document metadata

        Returns:
            List of heading-based chunks
        """
        sections = self._extract_sections_from_tokens(content)
        chunks = []

        for idx, section in enumerate(sections):
            # Skip empty sections
            if not section['content'].strip():
                continue

            chunk = DocumentChunk(
                content=section['content'],
                file_path=str(file_path),
                start_offset=section['start_offset'],
                end_offset=section['end_offset'],
                section_title=section['title'] or "Preamble",
                heading_level=section['level'] if section['level'] > 0 else None,
                chunk_index=idx,
                metadata=metadata
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
            file_path: Path to the file
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
                start_offset=start,
                end_offset=end,
                chunk_index=len(chunks),
                metadata=metadata
            )
            chunks.append(chunk)

            start = end - overlap

        return chunks

    def _extract_sections_from_tokens(self, content: str) -> list[dict[str, Any]]:
        """Extract sections using markdown-it-py token stream.

        Args:
            content: Markdown content

        Returns:
            List of section dictionaries with title, content, level, offsets
        """
        tokens = self._md.parse(content)
        sections = []
        current_section = {
            'title': '',
            'level': 0,
            'content_tokens': [],
            'start_offset': 0
        }

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == 'heading_open':
                # Save previous section
                if current_section['content_tokens'] or not sections:
                    section_text = self._tokens_to_text(current_section['content_tokens'], content)
                    sections.append({
                        'title': current_section['title'],
                        'level': current_section['level'],
                        'content': section_text,
                        'start_offset': current_section['start_offset'],
                        'end_offset': current_section['start_offset'] + len(section_text)
                    })

                # Start new section
                level = int(token.tag[1])  # h1 -> 1, h2 -> 2, etc.
                # Get heading text from next inline token
                heading_text = ''
                if i + 1 < len(tokens) and tokens[i + 1].type == 'inline':
                    heading_text = tokens[i + 1].content

                current_section = {
                    'title': heading_text,
                    'level': level,
                    'content_tokens': [token, tokens[i + 1] if i + 1 < len(tokens) else None, tokens[i + 2] if i + 2 < len(tokens) else None],
                    'start_offset': sum(len(s['content']) for s in sections)
                }
                i += 3  # Skip heading_open, inline, heading_close
                continue

            current_section['content_tokens'].append(token)
            i += 1

        # Add final section
        if current_section['content_tokens']:
            section_text = self._tokens_to_text(current_section['content_tokens'], content)
            sections.append({
                'title': current_section['title'],
                'level': current_section['level'],
                'content': section_text,
                'start_offset': current_section['start_offset'],
                'end_offset': current_section['start_offset'] + len(section_text)
            })

        return sections

    def _tokens_to_text(self, tokens: list, original_content: str) -> str:
        """Convert tokens back to text.

        Args:
            tokens: List of markdown-it tokens
            original_content: Original markdown content

        Returns:
            Reconstructed text from tokens
        """
        if not tokens:
            return ""

        # Use token map to extract original text
        lines = original_content.split('\n')
        result_lines = []

        for token in tokens:
            if token and token.map:
                start_line, end_line = token.map
                result_lines.extend(lines[start_line:end_line])

        return '\n'.join(result_lines)
