"""Distributed reasoning agent using Pyergon orchestration.

This module provides a Pyergon-powered reasoning agent for distributed multi-step execution.
Compared to the sequential ReasoningAgent, this provides:

- Durable execution (resume after crashes)
- Automatic retry on transient failures (e.g., LLM rate limits)
- Persistent execution_trace (execution trace saved to SQLite)
- Better reliability for long-running tasks

Hidden design decisions:
- Worker coordination strategy
- Task distribution algorithm (currently single-task)
- Retry policies for each step
- Progress tracking mechanism
"""

import asyncio
import pickle
import time
from typing import Any

from pyergon import Scheduler, Worker
from pyergon.core import TaskStatus
from pyergon.storage.sqlite import SqliteExecutionLog
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ..llm import LLMProvider
from ..storage import StorageBackend
from .connection_pool import ConnectionPool
from .data_structures import Task, TaskResult
from .flows import ReasoningTaskFlow
from .tools import BaseTool


class PyergonReasoningAgent:
    """Distributed reasoning agent using Pyergon.

    This agent orchestrates durable multi-step reasoning across workers,
    with automatic retry and persistent execution state.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: list[BaseTool] | None = None,
        console: Console | None = None,
        db_path: str = "data/pyergon_reasoning.db",
        system_prompt: str | None = None
    ):
        """Initialize the Pyergon reasoning agent.

        Args:
            llm: LLM provider for generation
            tools: List of available tools
            console: Rich console for output
            db_path: Path to Pyergon execution log database
            system_prompt: Optional custom system prompt (or loaded from prompts/reasoning_agent.txt)
        """
        self._llm = llm
        self._tools = tools or []
        self._console = console or Console()
        self._db_path = db_path

        # Build tools description
        tools_desc = "\n".join([
            f"- {t.name}: {t.description}"
            for t in self._tools
        ])

        # Load prompt from file if not provided
        if system_prompt is None:
            from ..prompts import get_system_prompt
            system_prompt = get_system_prompt()

        self._system_prompt = system_prompt.format(tools_description=tools_desc or "None")

        # Store configurations for injection into flows
        self._llm_config = self._extract_llm_config()
        self._tools_config = self._extract_tools_config()

    def _extract_llm_config(self) -> dict[str, Any]:
        """Extract LLM configuration for serialization."""
        # Auto-detect provider from class name
        provider_class = self._llm.__class__.__name__
        if "DeepSeek" in provider_class:
            provider = "deepseek"
        elif "OpenAI" in provider_class:
            provider = "openai"
        else:
            # Fallback: extract from module path
            module_path = self._llm.__class__.__module__
            if "deepseek" in module_path:
                provider = "deepseek"
            elif "openai" in module_path:
                provider = "openai"
            else:
                provider = "unknown"

        # Extract client credentials
        client = getattr(self._llm, "_client", None)
        if client:
            api_key = client.api_key
            base_url = str(client.base_url) if client.base_url else None
        else:
            api_key = None
            base_url = None

        return {
            "provider": provider,
            "kwargs": {
                "api_key": api_key,
                "model": getattr(self._llm, "_model", "deepseek-chat"),
                "base_url": base_url
            }
        }

    def _extract_tools_config(self) -> dict[str, Any]:
        """Extract tools configuration for serialization."""
        config = {}
        for tool in self._tools:
            tool_config = {}

            # Extract tool-specific config
            if hasattr(tool, "_storage") and hasattr(tool, "_embedder"):
                # SearchCodebaseTool
                tool_config["search"] = {
                    "storage": self._extract_embedding_storage_config(tool._storage),
                    "embedder": self._extract_embedder_config(tool._embedder)
                }
            elif hasattr(tool, "_parser_factory"):
                # AnalyzeCodeTool
                tool_config["parser_factory"] = True

            config[tool.name] = tool_config

        return config

    def _extract_embedding_storage_config(self, storage: StorageBackend) -> dict[str, Any]:
        """Extract embedding storage configuration (PostgreSQL for code chunks/vectors)."""
        # Auto-detect backend from class name
        backend_class = storage.__class__.__name__
        if "Postgres" in backend_class:
            backend = "postgres"
        elif "SQLite" in backend_class:
            backend = "sqlite"
        else:
            backend = "unknown"

        return {
            "backend": backend,
            "kwargs": {
                "host": getattr(storage, "_host", "localhost"),
                "port": getattr(storage, "_port", 5433),
                "database": getattr(storage, "_database", "pytelos"),
                "user": getattr(storage, "_user", "pytelos"),
                "password": getattr(storage, "_password", ""),
            }
        }

    def _extract_embedder_config(self, embedder) -> dict[str, Any]:
        """Extract embedder configuration."""
        # Auto-detect provider from class name
        embedder_class = embedder.__class__.__name__
        if "OpenAI" in embedder_class:
            provider = "openai"
        elif "HuggingFace" in embedder_class:
            provider = "huggingface"
        else:
            # Fallback: extract from module path
            module_path = embedder.__class__.__module__
            provider = "openai" if "openai" in module_path else "unknown"

        return {
            "provider": provider,
            "kwargs": {
                "api_key": embedder._client.api_key,
                "model": embedder._model
            }
        }

    async def solve(
        self,
        task: Task,
        max_steps: int = 10,
        show_progress: bool = True,
        num_workers: int = 1  # Usually 1 for reasoning (sequential steps)
    ) -> TaskResult:
        """Solve task using durable reasoning flow.

        Args:
            task: The task to solve
            max_steps: Maximum number of reasoning steps
            show_progress: Show live progress updates
            num_workers: Number of parallel workers (default 1 for sequential reasoning)

        Returns:
            TaskResult with the solution
        """
        self._console.print("[bold]Starting durable reasoning[/bold]")
        self._console.print(f"Task: {task.instruction}")
        self._console.print(f"Max steps: {max_steps}, Workers: {num_workers}\n")

        start_time = time.time()

        # Setup Pyergon
        execution_log = SqliteExecutionLog(self._db_path)
        await execution_log.connect()

        # Warmup connection pool in parallel with worker startup
        self._console.print("[dim]Warming up connections...[/dim]")

        # Extract configs for warmup
        storage_config = None
        embedder_config = None
        for tool_name, tool_config in self._tools_config.items():
            if tool_name == "search_codebase" and "search" in tool_config:
                search_config = tool_config["search"]
                storage_config = search_config.get("storage")
                embedder_config = search_config.get("embedder")
                break

        # Start workers and warmup connections in parallel
        workers = []
        worker_handles = []

        async def start_worker(worker_id: int):
            worker = Worker(execution_log, f"reasoning-worker-{worker_id}")
            # Register flow types
            await worker.register(ReasoningTaskFlow)
            from .flows import StepFlow
            await worker.register(StepFlow)
            handle = await worker.start()
            return worker, handle

        # Parallel initialization
        warmup_task = ConnectionPool.warmup(
            llm_config=self._llm_config,
            storage_config=storage_config,
            embedder_config=embedder_config
        )

        worker_tasks = [start_worker(i) for i in range(num_workers)]

        # Wait for both warmup and workers
        results = await asyncio.gather(warmup_task, *worker_tasks)

        # Extract worker results (skip first result which is warmup)
        for worker, handle in results[1:]:
            workers.append(worker)
            worker_handles.append(handle)

        self._console.print(f"[green]Started {num_workers} worker(s) with warmed connections[/green]\n")

        try:
            # Schedule reasoning task
            scheduler = Scheduler(execution_log).with_version("v1.0")

            task_id = await scheduler.schedule(
                ReasoningTaskFlow(
                    task_instruction=task.instruction,
                    max_steps=max_steps,
                    llm_config=self._llm_config,
                    tools_config=self._tools_config,
                    system_prompt=self._system_prompt
                )
            )

            self._console.print("[green]Scheduled reasoning task[/green]\n")

            # Monitor progress
            if show_progress:
                result = await self._monitor_progress(execution_log, task_id, max_steps)
            else:
                # Just wait for completion
                result = await self._wait_for_completion(execution_log, task_id)

            processing_time = time.time() - start_time

            # Return result (progress bar already shows completion status)
            return TaskResult(
                task_id=task.id_,
                content=result.get("result", "No result available"),
                execution_trace=result.get("execution_trace", ""),
                metadata={
                    "steps": result.get("steps", 0),
                    "processing_time_seconds": processing_time,
                    "status": result.get("status", "unknown"),
                    "total_input_tokens": result.get("total_input_tokens", 0),
                    "total_output_tokens": result.get("total_output_tokens", 0)
                }
            )

        finally:
            # Cleanup
            for handle in worker_handles:
                await handle.shutdown()
            await execution_log.close()

            # Close connection pool
            await ConnectionPool.close_all()

    async def _get_step_count(
        self,
        execution_log: SqliteExecutionLog,
        flow_id: str
    ) -> int:
        """Get current step count by counting completed StepFlow invocations.

        Args:
            execution_log: Pyergon execution log
            flow_id: Flow ID of the parent ReasoningTaskFlow

        Returns:
            Number of completed reasoning steps
        """
        try:
            invocations = await execution_log.get_invocations_for_flow(flow_id)
            # Count completed child flow invocations where method_name = "invoke(StepFlow)"
            # (one per reasoning step)
            from pyergon.core import InvocationStatus
            completed_steps = sum(
                1 for inv in invocations
                if inv.method_name == "invoke(StepFlow)" and inv.status == InvocationStatus.COMPLETE
            )
            return completed_steps
        except Exception:
            return 0

    async def _monitor_progress(
        self,
        execution_log: SqliteExecutionLog,
        task_id: str,
        max_steps: int
    ) -> dict:
        """Monitor task progress with live updates using event-driven notifications.

        Args:
            execution_log: Pyergon execution log
            task_id: Scheduled task ID
            max_steps: Maximum number of steps for progress bar

        Returns:
            Result dictionary
        """
        status_notify = execution_log.status_notify()

        # Create progress bar with Rich (showing step count instead of percentage)
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            TextColumn("[progress.percentage]{task.completed}/{task.total} steps"),
            TimeElapsedColumn(),
            console=self._console
        )

        with progress:
            # Create progress task
            progress_task = progress.add_task(
                "[cyan]Reasoning...",
                total=max_steps
            )

            while True:
                # Check task status FIRST (after notification)
                task = await execution_log.get_scheduled_flow(task_id)

                # Get current step count from completed ExecutionFlow invocations
                step_count = 0
                if task:
                    step_count = await self._get_step_count(execution_log, task.flow_id)

                # Update progress bar with current status
                status_str = task.status.value if task else "Unknown"

                # Map status to color and description
                status_colors = {
                    "PENDING": "yellow",
                    "RUNNING": "cyan",
                    "SUSPENDED": "magenta",
                    "COMPLETE": "green",
                    "FAILED": "red"
                }
                color = status_colors.get(status_str, "white")

                progress.update(
                    progress_task,
                    completed=step_count,
                    description=f"[{color}]{status_str}[/{color}]"
                )

                # Check for terminal states AFTER updating display
                if task and task.status == TaskStatus.COMPLETE:
                    # Final update to show completion
                    progress.update(
                        progress_task,
                        completed=step_count,
                        description="[bold green]COMPLETE[/bold green]"
                    )

                    # Get result
                    try:
                        inv = await execution_log.get_invocation(task.flow_id, 0)
                        if inv and inv.return_value:
                            result = pickle.loads(inv.return_value)
                            return result
                    except Exception as e:
                        self._console.print(f"[red]Error retrieving result: {e}[/red]")
                        return {
                            "result": "Error retrieving result",
                            "steps": 0,
                            "execution_trace": "",
                            "status": "error"
                        }

                elif task and task.status == TaskStatus.FAILED:
                    # Final update to show failure
                    progress.update(
                        progress_task,
                        completed=step_count,
                        description="[bold red]FAILED[/bold red]"
                    )
                    return {
                        "result": "Task failed",
                        "steps": 0,
                        "execution_trace": "",
                        "status": "failed"
                    }

                # Wait for next status change notification (event-driven pattern)
                # This blocks until a status change occurs in the execution log
                await status_notify.wait()
                status_notify.clear()  # Reset for next notification

    async def _wait_for_completion(
        self,
        execution_log: SqliteExecutionLog,
        task_id: str
    ) -> dict:
        """Wait for task completion without live updates.

        Args:
            execution_log: Pyergon execution log
            task_id: Scheduled task ID

        Returns:
            Result dictionary
        """
        status_notify = execution_log.status_notify()

        while True:
            task = await execution_log.get_scheduled_flow(task_id)

            if task and task.status == TaskStatus.COMPLETE:
                inv = await execution_log.get_invocation(task.flow_id, 0)
                if inv and inv.return_value:
                    result = pickle.loads(inv.return_value)
                    return result

            elif task and task.status == TaskStatus.FAILED:
                return {
                    "result": "Task failed",
                    "steps": 0,
                    "execution_trace": "",
                    "status": "failed"
                }

            try:
                await asyncio.wait_for(status_notify.wait(), timeout=1.0)
                status_notify.clear()
            except TimeoutError:
                pass

    async def close(self) -> None:
        """Close resources."""
        await self._llm.close()
