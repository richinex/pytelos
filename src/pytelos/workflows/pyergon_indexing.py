"""Distributed indexing workflow using Pyergon orchestration.

This module provides a Pyergon-powered workflow for distributed file indexing.
Compared to the sequential IndexingWorkflow, this provides:

- Parallel processing across multiple workers
- Automatic retry on transient failures (e.g., API rate limits)
- Durable state (resume after crashes)
- Better resource utilization

Hidden design decisions:
- Worker coordination strategy
- Task distribution algorithm
- Retry policies for each step
- Progress tracking mechanism
"""

import asyncio
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pyergon import Scheduler, Worker, flow, flow_type, step
from pyergon.core import RetryPolicy, TaskStatus
from pyergon.storage.sqlite import SqliteExecutionLog
from rich.console import Console
from rich.live import Live
from rich.table import Table
from uuid_extensions import uuid7

from ..embedding import EmbeddingProvider, create_embedding_provider
from ..indexer import ChunkingStrategy, create_parser_factory
from ..storage import CodeChunk, StorageBackend, create_storage_backend


class EmbeddingError(Exception):
    """Base class for embedding errors."""

    def is_retryable(self) -> bool:
        """Override in subclasses to control retry behavior."""
        return False


class RateLimitError(EmbeddingError):
    """Rate limit exceeded (retryable)."""

    def __init__(self, message: str):
        super().__init__(f"Rate limit exceeded: {message}")

    def is_retryable(self) -> bool:
        return True


class NetworkError(EmbeddingError):
    """Network or connection error (retryable)."""

    def __init__(self, message: str):
        super().__init__(f"Network error: {message}")

    def is_retryable(self) -> bool:
        return True


class InvalidInputError(EmbeddingError):
    """Invalid input data (non-retryable)."""

    def __init__(self, message: str, file_path: str = None):
        msg = f"Invalid input: {message}"
        if file_path:
            msg += f" (file: {file_path})"
        super().__init__(msg)
        self.file_path = file_path

    def is_retryable(self) -> bool:
        return False


@dataclass
@flow_type(invokable=str)
class FileIndexingFlow:
    """Durable flow for indexing a single file.

    This flow represents the complete indexing pipeline for one file:
    1. Parse file into chunks
    2. Generate embeddings for chunks
    3. Store chunks with embeddings in database

    Each step can be retried independently on failure.
    """
    file_path: str
    strategy: str  # ChunkingStrategy enum value
    preserve_structure: bool

    # Provider configurations (injected by scheduler)
    storage_config: dict[str, Any]
    embedder_config: dict[str, Any]

    @step
    async def parse_file(self) -> list[dict]:
        """Parse file into chunks.

        Returns:
            List of parsed chunks as dictionaries (for serialization)
        """
        parser_factory = create_parser_factory()
        file_path = Path(self.file_path)

        # Get appropriate parser for file type
        parser = parser_factory.get_parser(file_path)
        if not parser:
            raise ValueError(f"No parser available for {file_path}")

        # Parse with specified strategy
        strategy = ChunkingStrategy(self.strategy)
        chunks = parser.parse_file(
            file_path,
            strategy=strategy,
            preserve_structure=self.preserve_structure
        )

        # Convert to serializable format
        return [
            {
                "content": chunk.content,
                "metadata": chunk.metadata.model_dump(),
                "chunk_id": chunk.chunk_id
            }
            for chunk in chunks
        ]

    @step(retry_policy=RetryPolicy.STANDARD)
    async def embed_chunks(self, chunk_dicts: list[dict]) -> list[list[float]]:
        """Generate embeddings for chunks.

        Uses RetryPolicy.STANDARD to handle transient API failures
        (rate limits, network issues, etc.)

        Args:
            chunk_dicts: List of chunk dictionaries from parse_file

        Returns:
            List of embedding vectors as nested lists (for serialization)

        Raises:
            RateLimitError: When API rate limit is exceeded (retryable)
            NetworkError: On network/connection issues (retryable)
            InvalidInputError: On invalid input data (non-retryable)
        """
        # Recreate embedder from config
        embedder = create_embedding_provider(
            self.embedder_config["provider"],
            **self.embedder_config.get("kwargs", {})
        )

        try:
            # Extract text content
            texts = [chunk["content"] for chunk in chunk_dicts]

            # Filter out empty texts to avoid API errors
            if not texts or all(not text.strip() for text in texts):
                raise InvalidInputError(
                    "All chunks are empty or whitespace-only",
                    file_path=self.file_path
                )

            # Generate embeddings in batch
            try:
                embeddings = await embedder.embed_batch(texts)
            except Exception as e:
                error_msg = str(e).lower()

                # Classify error based on message
                if "rate" in error_msg and "limit" in error_msg or "429" in str(e):
                    raise RateLimitError(str(e))
                elif "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
                    raise NetworkError(str(e))
                elif "400" in str(e) or "invalid" in error_msg:
                    raise InvalidInputError(str(e), file_path=self.file_path)
                else:
                    raise

            # Convert numpy arrays to lists for serialization
            return [embedding.tolist() for embedding in embeddings]

        finally:
            await embedder.close()

    @step
    async def store_chunks(
        self,
        chunk_dicts: list[dict],
        embeddings: list[list[float]]
    ) -> int:
        """Store chunks with embeddings in database.

        Args:
            chunk_dicts: Parsed chunks
            embeddings: Corresponding embeddings

        Returns:
            Number of chunks stored
        """
        # Recreate storage from config
        storage = create_storage_backend(
            self.storage_config["backend"],
            **self.storage_config.get("kwargs", {})
        )

        try:
            await storage.connect()

            # Convert back to CodeChunk objects
            chunks = []
            for chunk_dict in chunk_dicts:
                metadata = chunk_dict["metadata"]
                # Generate UUIDv7 if not already set (same as sequential workflow)
                chunk_id = chunk_dict.get("chunk_id") or str(uuid7())

                chunks.append(CodeChunk(
                    id=chunk_id,
                    file_path=metadata["file_path"],
                    chunk_text=chunk_dict["content"],
                    start_line=metadata["start_line"],
                    end_line=metadata["end_line"],
                    language=metadata["language"],
                    metadata=metadata
                ))

            # Convert embeddings back to numpy arrays
            np_embeddings = [np.array(emb, dtype=np.float32) for emb in embeddings]

            # Store in batch
            await storage.store_chunks_batch(chunks, np_embeddings)

            return len(chunks)

        finally:
            await storage.disconnect()

    @flow(retry_policy=RetryPolicy.STANDARD)
    async def process(self) -> dict:
        """Main flow: parse -> embed -> store.

        Returns:
            Statistics about the indexing operation
        """
        # Parse
        chunk_dicts = await self.parse_file()

        # Embed
        embeddings = await self.embed_chunks(chunk_dicts)

        # Store
        chunk_count = await self.store_chunks(chunk_dicts, embeddings)

        return {
            "file_path": self.file_path,
            "chunks": chunk_count,
            "status": "success"
        }


class PyergonIndexingWorkflow:
    """Distributed indexing workflow using Pyergon.

    This workflow orchestrates parallel indexing across multiple workers,
    with automatic retry and durable state management.
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedder: EmbeddingProvider,
        console: Console | None = None,
        db_path: str = "data/pyergon_indexing.db"
    ):
        """Initialize the Pyergon workflow.

        Args:
            storage: Storage backend for indexed chunks
            embedder: Embedding provider
            console: Rich console for output
            db_path: Path to Pyergon execution log database
        """
        self._storage = storage
        self._embedder = embedder
        self._console = console or Console()
        self._db_path = db_path

        # Store configurations for injection into flows
        self._storage_config = {
            "backend": "postgres",  # TODO: Make dynamic
            "kwargs": {
                "host": storage._pool._connect_kwargs.get("host"),
                "port": storage._pool._connect_kwargs.get("port"),
                "database": storage._pool._connect_kwargs.get("database"),
                "user": storage._pool._connect_kwargs.get("user"),
                "password": storage._pool._connect_kwargs.get("password"),
            }
        }

        self._embedder_config = {
            "provider": "openai",  # TODO: Make dynamic
            "kwargs": {
                "api_key": embedder._client.api_key,
                "model": embedder._model
            }
        }

    async def index_directory(
        self,
        directory: Path,
        strategy: ChunkingStrategy = ChunkingStrategy.BY_FUNCTION,
        file_pattern: str = "**/*",
        preserve_structure: bool = True,
        num_workers: int = 4
    ) -> dict:
        """Index directory using distributed workers.

        Args:
            directory: Directory to index
            strategy: Chunking strategy
            file_pattern: Glob pattern for files
            preserve_structure: Preserve document structure
            num_workers: Number of parallel workers

        Returns:
            Indexing statistics
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

        self._console.print("[bold]Starting distributed indexing[/bold]")
        self._console.print(f"Files: {total_files}, Workers: {num_workers}\n")

        start_time = time.time()

        # Setup Pyergon
        execution_log = SqliteExecutionLog(self._db_path)
        await execution_log.connect()

        # Start workers
        workers = []
        worker_handles = []

        for i in range(num_workers):
            worker = Worker(execution_log, f"indexer-worker-{i}")
            await worker.register(FileIndexingFlow)
            handle = await worker.start()
            workers.append(worker)
            worker_handles.append(handle)

        self._console.print(f"[green]Started {num_workers} workers[/green]")

        try:
            # Schedule all files
            scheduler = Scheduler(execution_log).with_version("v1.0")
            task_ids = []

            for file_path in files:
                task_id = await scheduler.schedule(
                    FileIndexingFlow(
                        file_path=str(file_path),
                        strategy=strategy.value,
                        preserve_structure=preserve_structure,
                        storage_config=self._storage_config,
                        embedder_config=self._embedder_config
                    )
                )
                task_ids.append(task_id)

            self._console.print(f"[green]Scheduled {len(task_ids)} tasks[/green]\n")

            # Monitor progress
            stats = await self._monitor_progress(
                execution_log,
                task_ids,
                total_files
            )

            processing_time = time.time() - start_time

            # Print summary
            self._console.print("\n[bold green]Distributed Indexing Complete![/bold green]")
            self._console.print(f"  Total files processed: {stats['completed']}")
            self._console.print(f"  Total chunks created: {stats['total_chunks']}")
            self._console.print(f"  Failed files: {stats['failed']}")
            self._console.print(f"  Processing time: {processing_time:.2f}s")
            self._console.print(f"  Throughput: {stats['completed'] / processing_time:.2f} files/sec")

            return {
                "total_files": stats["completed"],
                "total_chunks": stats["total_chunks"],
                "failed_files": stats["failed_tasks"],
                "processing_time_seconds": processing_time
            }

        finally:
            # Cleanup
            for handle in worker_handles:
                await handle.shutdown()
            await execution_log.close()

    async def _monitor_progress(
        self,
        execution_log: SqliteExecutionLog,
        task_ids: list[str],
        total: int
    ) -> dict:
        """Monitor task progress and display live updates.

        Args:
            execution_log: Pyergon execution log
            task_ids: List of scheduled task IDs
            total: Total number of tasks

        Returns:
            Statistics dictionary
        """
        completed_tasks = set()
        failed_tasks = []
        total_chunks = 0

        status_notify = execution_log.status_notify()

        # Create initial progress table
        def create_progress_table(completed, failed, chunks):
            table = Table(title="Indexing Progress")
            table.add_column("Status", style="cyan")
            table.add_column("Count", style="yellow")
            table.add_column("Percentage", style="green")

            table.add_row(
                "Completed",
                str(completed),
                f"{(completed / total * 100):.1f}%"
            )
            table.add_row(
                "Failed",
                str(failed),
                f"{(failed / total * 100):.1f}%"
            )
            table.add_row(
                "Pending",
                str(total - completed - failed),
                f"{((total - completed - failed) / total * 100):.1f}%"
            )
            table.add_row(
                "Total Chunks",
                str(chunks),
                "-"
            )
            return table

        table = create_progress_table(0, 0, 0)

        with Live(table, console=self._console, refresh_per_second=4) as live:
            while len(completed_tasks) + len(failed_tasks) < total:
                # Check all tasks
                for task_id in task_ids:
                    task = await execution_log.get_scheduled_flow(task_id)

                    if task and task.status == TaskStatus.COMPLETE:
                        if task_id not in completed_tasks:
                            completed_tasks.add(task_id)
                            # Extract chunk count from invocation result
                            try:
                                inv = await execution_log.get_invocation(task.flow_id, 0)
                                if inv and inv.return_value:
                                    result = pickle.loads(inv.return_value)
                                    if isinstance(result, dict):
                                        total_chunks += result.get("chunks", 0)
                            except Exception:
                                pass  # Continue even if we can't get the result

                    elif task and task.status == TaskStatus.FAILED and task_id not in failed_tasks:
                        failed_tasks.append(task_id)

                # Update table
                completed = len(completed_tasks)
                failed = len(failed_tasks)
                table = create_progress_table(completed, failed, total_chunks)
                live.update(table)

                # Check if we're done before waiting
                if completed + failed >= total:
                    break

                # Wait for next status change with timeout (poll every 2 seconds)
                try:
                    await asyncio.wait_for(status_notify.wait(), timeout=2.0)
                    status_notify.clear()
                except TimeoutError:
                    # Timeout - will loop and check status again
                    pass

        return {
            "completed": len(completed_tasks),
            "failed": len(failed_tasks),
            "total_chunks": total_chunks,
            "failed_tasks": failed_tasks
        }
