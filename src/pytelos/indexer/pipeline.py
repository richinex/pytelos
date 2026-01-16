import time
from pathlib import Path
from typing import Any

from uuid_extensions import uuid7

from ..embedding import EmbeddingProvider
from ..storage import StorageBackend
from ..storage.models import CodeChunk as StorageCodeChunk
from .base import Chunk, CodeParser
from .factory import ParserFactory
from .models import ChunkingStrategy, IndexingResult, ParsedChunk
from .parsers.base import DocumentChunk


class IndexingPipeline:
    """High-level indexing pipeline.

    Orchestrates file parsing, embedding generation, and storage.
    Now supports multiple file types via ParserFactory.

    Hidden design decisions:
    - Coordination between parser, embedder, and storage
    - Error handling strategy
    - Progress tracking mechanism
    - Batch size for embedding and storage operations
    - Conversion from different chunk types to storage format
    """

    def __init__(
        self,
        parser_factory: ParserFactory,
        embedder: EmbeddingProvider,
        storage: StorageBackend,
        batch_size: int = 10
    ):
        """Initialize the indexing pipeline.

        Args:
            parser_factory: Parser factory for multi-file-type support
            embedder: Embedding provider instance
            storage: Storage backend instance
            batch_size: Number of chunks to process in a batch
        """
        self._parser_factory = parser_factory
        self._embedder = embedder
        self._storage = storage
        self._batch_size = batch_size

    async def index_file(
        self,
        file_path: Path,
        strategy: ChunkingStrategy = ChunkingStrategy.BY_FUNCTION,
        chunk_size: int = 1000,
        overlap: int = 200,
        preserve_structure: bool = True,
        **options: Any
    ) -> int:
        """Index a single file.

        Args:
            file_path: Path to file to index
            strategy: Chunking strategy (for code files)
            chunk_size: Target chunk size in characters
            overlap: Overlap size (for BY_LINES strategy or size-based chunking)
            preserve_structure: For documents, preserve pages/sections
            **options: Additional parser-specific options

        Returns:
            Number of chunks created and stored

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
        """
        # Use factory to auto-detect file type
        parser = self._parser_factory.get_parser(file_path)
        if parser is None:
            raise ValueError(
                f"No parser available for {file_path.suffix} files. "
                f"Supported: {self._parser_factory.get_supported_extensions()}"
            )

        # Different options for code vs documents
        if isinstance(parser, CodeParser):
            parsed_chunks = parser.parse_file(
                file_path,
                chunk_size=chunk_size,
                strategy=strategy,
                overlap=overlap,
                **options
            )
        else:
            # Document parser
            parsed_chunks = parser.parse_file(
                file_path,
                chunk_size=chunk_size,
                preserve_structure=preserve_structure,
                **options
            )

        # Process in batches
        total_stored = 0
        for i in range(0, len(parsed_chunks), self._batch_size):
            batch = parsed_chunks[i:i + self._batch_size]
            stored = await self._process_batch(batch)
            total_stored += stored

        return total_stored

    async def index_directory(
        self,
        directory: Path,
        recursive: bool = True,
        file_pattern: str = "**/*.py" if True else "*.py",
        strategy: ChunkingStrategy = ChunkingStrategy.BY_FUNCTION,
        max_chunk_size: int = 1000,
        overlap: int = 200
    ) -> IndexingResult:
        """Index all files in a directory.

        Args:
            directory: Path to directory
            recursive: Whether to recursively index subdirectories
            file_pattern: Glob pattern for files to index
            strategy: Chunking strategy
            max_chunk_size: Maximum chunk size (for BY_LINES strategy)
            overlap: Overlap size (for BY_LINES strategy)

        Returns:
            IndexingResult with statistics

        Raises:
            NotADirectoryError: If path is not a directory
        """
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        start_time = time.time()
        total_files = 0
        total_chunks = 0
        failed_files = []

        # Find all matching files
        if recursive:
            files = list(directory.glob(file_pattern))
        else:
            # Non-recursive: only direct children
            pattern = file_pattern.replace("**/", "")
            files = list(directory.glob(pattern))

        for file_path in files:
            if not file_path.is_file():
                continue

            try:
                chunks = await self.index_file(
                    file_path,
                    strategy=strategy,
                    max_chunk_size=max_chunk_size,
                    overlap=overlap
                )
                total_files += 1
                total_chunks += chunks
            except Exception as e:
                failed_files.append(f"{file_path}: {str(e)}")

        processing_time = time.time() - start_time

        return IndexingResult(
            total_files=total_files,
            total_chunks=total_chunks,
            failed_files=failed_files,
            processing_time_seconds=processing_time
        )

    async def _process_batch(self, batch: list[Chunk]) -> int:
        """Process a batch of parsed chunks.

        Handles different chunk types (ParsedChunk for code, DocumentChunk for docs).

        Args:
            batch: List of parsed chunks (any Chunk subtype)

        Returns:
            Number of chunks successfully stored
        """
        # Extract text for embedding
        texts = [chunk.content for chunk in batch]

        # Generate embeddings
        embeddings = await self._embedder.embed_batch(texts)

        # Store chunks with embeddings
        stored = 0
        for chunk, embedding in zip(batch, embeddings, strict=False):
            try:
                # Convert to storage format
                storage_chunk = self._convert_to_storage_chunk(chunk)
                chunk_id = await self._storage.store_chunk(storage_chunk, embedding)
                chunk.chunk_id = chunk_id
                stored += 1
            except Exception:
                # Log error but continue with other chunks
                # In a production system, you'd want proper logging here
                continue

        return stored

    def _convert_to_storage_chunk(self, chunk: Chunk) -> StorageCodeChunk:
        """Convert any chunk type to storage format.

        Args:
            chunk: Any Chunk subtype (ParsedChunk, DocumentChunk, etc.)

        Returns:
            StorageCodeChunk ready for database storage
        """
        # Generate UUIDv7 if not already set
        chunk_id = chunk.chunk_id or str(uuid7())

        # Handle different chunk types
        if isinstance(chunk, ParsedChunk):
            # Code chunk - has metadata object
            return StorageCodeChunk(
                id=chunk_id,
                file_path=chunk.metadata.file_path,
                chunk_text=chunk.content,
                start_line=chunk.metadata.start_line,
                end_line=chunk.metadata.end_line,
                language=chunk.metadata.language,
                metadata={
                    "chunk_type": "code",
                    "function_name": chunk.metadata.function_name,
                    "class_name": chunk.metadata.class_name,
                    "docstring": chunk.metadata.docstring,
                    "imports": chunk.metadata.imports,
                    "complexity": chunk.metadata.complexity,
                }
            )
        elif isinstance(chunk, DocumentChunk):
            # Document chunk - has different structure
            return StorageCodeChunk(
                id=chunk_id,
                file_path=chunk.file_path,
                chunk_text=chunk.content,
                # Map character offsets to pseudo line numbers
                start_line=chunk.start_offset,
                end_line=chunk.end_offset,
                language=chunk.metadata.get("file_type", "document"),
                metadata={
                    "chunk_type": "document",
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "heading_level": chunk.heading_level,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata  # Include document metadata
                }
            )
        else:
            # Generic Chunk - minimal conversion
            return StorageCodeChunk(
                id=chunk_id,
                file_path=chunk.file_path,
                chunk_text=chunk.content,
                start_line=chunk.start_offset,
                end_line=chunk.end_offset,
                language=chunk.chunk_type,
                metadata={
                    "chunk_type": chunk.chunk_type,
                    **chunk.metadata
                }
            )
