"""Indexing workflow for orchestrating the indexing process."""
import time
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..embedding import EmbeddingProvider
from ..indexer import ChunkingStrategy, IndexingPipeline, create_parser_factory
from ..storage import StorageBackend


class IndexingWorkflow:
    """Workflow for indexing a codebase.

    This module hides the orchestration logic for the indexing process.

    Hidden design decisions:
    - Progress tracking and reporting
    - Error handling and recovery
    - Resource management
    - Batching strategy
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedder: EmbeddingProvider,
        console: Console | None = None
    ):
        """Initialize the indexing workflow.

        Args:
            storage: Storage backend
            embedder: Embedding provider
            console: Optional Rich console for output
        """
        self._storage = storage
        self._embedder = embedder
        self._console = console or Console()

        # Create pipeline with parser factory (supports all file types)
        parser_factory = create_parser_factory()
        self._pipeline = IndexingPipeline(
            parser_factory=parser_factory,
            embedder=embedder,
            storage=storage,
            batch_size=10
        )

    async def index_directory(
        self,
        directory: Path,
        strategy: ChunkingStrategy = ChunkingStrategy.BY_FUNCTION,
        file_pattern: str = "**/*",
        preserve_structure: bool = True,
        on_progress: Callable[[int, int], None] | None = None
    ) -> dict:
        """Index an entire directory.

        Args:
            directory: Directory to index
            strategy: Chunking strategy (for code files)
            file_pattern: Glob pattern for files
            preserve_structure: For documents, preserve pages/sections
            on_progress: Optional progress callback (current, total)

        Returns:
            Dictionary with indexing statistics

        Raises:
            NotADirectoryError: If directory doesn't exist
        """
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        # Find all files
        files = list(directory.glob(file_pattern))
        total_files = len(files)

        if total_files == 0:
            self._console.print(f"[yellow]No files found matching {file_pattern}[/yellow]")
            return {
                "total_files": 0,
                "total_chunks": 0,
                "failed_files": [],
                "processing_time_seconds": 0
            }

        self._console.print(f"[bold]Indexing {total_files} files from {directory}[/bold]")

        total_chunks = 0
        failed_files = []
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self._console
        ) as progress:
            task = progress.add_task(
                "Processing files...",
                total=total_files
            )

            for i, file_path in enumerate(files, 1):
                try:
                    chunks = await self._pipeline.index_file(
                        file_path,
                        strategy=strategy,
                        preserve_structure=preserve_structure
                    )

                    total_chunks += chunks
                    progress.update(
                        task,
                        advance=1,
                        description=f"Processed {i}/{total_files} files ({total_chunks} chunks)"
                    )

                    if on_progress:
                        on_progress(i, total_files)

                except Exception as e:
                    failed_files.append(f"{file_path}: {str(e)}")
                    progress.update(task, advance=1)

        processing_time = time.time() - start_time

        # Print summary
        self._console.print("\n[bold green]Indexing Complete![/bold green]")
        self._console.print(f"  Total files processed: {total_files - len(failed_files)}")
        self._console.print(f"  Total chunks created: {total_chunks}")
        self._console.print(f"  Processing time: {processing_time:.2f}s")

        if failed_files:
            self._console.print(f"  [yellow]Failed files: {len(failed_files)}[/yellow]")
            for failed in failed_files[:5]:  # Show first 5
                self._console.print(f"    - {failed}")

        return {
            "total_files": total_files - len(failed_files),
            "total_chunks": total_chunks,
            "failed_files": failed_files,
            "processing_time_seconds": processing_time
        }

    async def reindex_file(
        self,
        file_path: Path,
        strategy: ChunkingStrategy = ChunkingStrategy.BY_FUNCTION
    ) -> int:
        """Reindex a single file.

        This will delete existing chunks for the file and create new ones.

        Args:
            file_path: Path to file to reindex
            strategy: Chunking strategy

        Returns:
            Number of chunks created

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self._console.print(f"[bold]Reindexing {file_path}[/bold]")

        # Delete existing chunks for this file
        deleted = await self._storage.delete_by_file(str(file_path))
        if deleted > 0:
            self._console.print(f"[dim]Deleted {deleted} existing chunks[/dim]")

        # Index the file
        chunks = await self._pipeline.index_file(file_path, strategy=strategy)

        self._console.print(f"[green]Created {chunks} chunks[/green]")

        return chunks

    async def get_indexing_stats(self) -> dict:
        """Get statistics about the indexed codebase.

        Returns:
            Dictionary with statistics
        """
        stats = await self._storage.get_stats()

        return {
            "total_chunks": stats.total_chunks,
            "total_files": stats.total_files
        }
