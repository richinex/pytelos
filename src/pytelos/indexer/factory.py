"""Parser factory for auto-detection and creation of parsers.

This module hides the parser selection logic and provides a unified
interface for parsing any supported file type.

Hidden design decisions:
- Parser registration and lookup
- File type detection order
- Fallback strategies
"""

from pathlib import Path
from typing import Any

from .base import Chunk, FileParser
from .parsers import MarkdownParser, PDFParser, PythonParser, TerraformParser, YAMLParser


class ParserFactory:
    """Factory for creating appropriate parsers based on file type.

    This class hides the decision of which parser to use for a given file,
    allowing easy extension with new parsers without affecting client code.
    """

    def __init__(self):
        """Initialize the parser factory."""
        self._parsers: list[FileParser] = []
        self._register_default_parsers()

    def _register_default_parsers(self) -> None:
        """Register all built-in parsers."""
        # Code parsers
        self.register_parser(PythonParser())

        # Document parsers
        self.register_parser(PDFParser())
        self.register_parser(MarkdownParser())

        # Infrastructure-as-Code parsers
        self.register_parser(YAMLParser())
        self.register_parser(TerraformParser())

    def register_parser(self, parser: FileParser) -> None:
        """Register a parser.

        Args:
            parser: Parser instance to register
        """
        self._parsers.append(parser)

    def get_parser(self, file_path: Path) -> FileParser | None:
        """Get appropriate parser for the file.

        Args:
            file_path: Path to the file

        Returns:
            Parser instance if found, None otherwise
        """
        for parser in self._parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    def can_parse(self, file_path: Path) -> bool:
        """Check if any registered parser can handle the file.

        Args:
            file_path: Path to the file

        Returns:
            True if a parser is available
        """
        return self.get_parser(file_path) is not None

    def parse_file(
        self,
        file_path: Path,
        **options: Any
    ) -> list[Chunk]:
        """Parse file using appropriate parser.

        Args:
            file_path: Path to the file
            **options: Parser-specific options

        Returns:
            List of chunks

        Raises:
            ValueError: If no parser available for the file type
            FileNotFoundError: If file doesn't exist
        """
        parser = self.get_parser(file_path)
        if parser is None:
            raise ValueError(
                f"No parser available for {file_path.suffix} files. "
                f"Supported extensions: {self.get_supported_extensions()}"
            )

        return parser.parse_file(file_path, **options)

    def get_supported_extensions(self) -> set[str]:
        """Get all supported file extensions.

        Returns:
            Set of all supported extensions across all parsers
        """
        extensions = set()
        for parser in self._parsers:
            extensions.update(parser.get_supported_extensions())
        return extensions

    def get_supported_file_types(self) -> list[str]:
        """Get all supported file types.

        Returns:
            List of file type identifiers (e.g., ['python', 'pdf', 'markdown'])
        """
        return [parser.get_file_type() for parser in self._parsers]


def create_parser_factory() -> ParserFactory:
    """Create a parser factory with all default parsers registered.

    Returns:
        Configured parser factory
    """
    return ParserFactory()
