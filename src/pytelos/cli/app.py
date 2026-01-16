"""Main CLI application using Typer."""
import asyncio
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..indexer import ChunkingStrategy
from ..search import SearchMode, SearchQuery, create_search_engine
from ..storage.models import IndexConfig
from .providers import get_embedder, get_llm, get_storage

# Load environment variables
load_dotenv()

# Create Typer app
app = typer.Typer(
    name="pytelos",
    help="Modular codebase indexer with semantic search and RAG capabilities",
    no_args_is_help=True,
    add_completion=True,
)

# Console for rich output
console = Console()


@app.command()
def init(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-initialization (WARNING: destroys existing data)"
    )
):
    """Initialize the database schema with pgvector and BM25 indexes."""
    async def _init():
        storage = get_storage()
        try:
            await storage.connect()

            if force:
                console.print("[yellow]WARNING: Force re-initialization will destroy existing data![/yellow]")
                confirm = typer.confirm("Are you sure you want to continue?")
                if not confirm:
                    console.print("[dim]Aborted.[/dim]")
                    return

            console.print("[dim]Initializing database...[/dim]")

            config = IndexConfig(
                chunk_size=500,
                overlap=50,
                embedding_model="text-embedding-3-small",
                embedding_dimensions=1536
            )

            if force:
                try:
                    await storage.drop_schema()
                except Exception:
                    pass  # Schema may not exist yet
            await storage.initialize_schema(config)

            console.print("[green]Database initialized successfully![/green]")
            console.print("[dim]You can now index code with: pytelos index <directory>[/dim]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)
        finally:
            await storage.disconnect()

    asyncio.run(_init())


@app.command()
def index(
    directory: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory to index"
    ),
    strategy: ChunkingStrategy = typer.Option(
        ChunkingStrategy.BY_FUNCTION,
        "--strategy",
        "-s",
        help="Chunking strategy to use"
    ),
    pattern: str = typer.Option(
        "**/*.py",
        "--pattern",
        "-p",
        help="File pattern to match"
    ),
    batch_size: int = typer.Option(
        10,
        "--batch-size",
        "-b",
        help="Number of chunks to process in a batch"
    ),
    distributed: bool = typer.Option(
        False,
        "--distributed",
        "-d",
        help="Use distributed Pyergon workflow for indexing"
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Number of parallel workers (only with --distributed)"
    )
):
    """Index a directory of code files."""
    async def _index():
        storage = get_storage()
        embedder = get_embedder(console)

        try:
            await storage.connect()

            console.print(f"[dim]Indexing {directory} with strategy: {strategy.value}[/dim]")
            console.print(f"[dim]Pattern: {pattern}[/dim]")

            if distributed:
                from ..workflows import PyergonIndexingWorkflow

                workflow = PyergonIndexingWorkflow(
                    storage=storage,
                    embedder=embedder,
                    console=console
                )

                result = await workflow.index_directory(
                    directory=Path(directory),
                    strategy=strategy,
                    file_pattern=pattern,
                    num_workers=workers
                )
            else:
                from ..workflows import IndexingWorkflow

                workflow = IndexingWorkflow(
                    storage=storage,
                    embedder=embedder,
                    console=console
                )

                result = await workflow.index_directory(
                    directory=Path(directory),
                    strategy=strategy,
                    file_pattern=pattern
                )

            # Handle both dict and object results
            if isinstance(result, dict):
                chunks = result.get("total_chunks", 0)
                files = result.get("total_files", 0)
                processing_time = result.get("processing_time", 0)
            else:
                chunks = result.chunks_indexed
                files = result.files_processed
                processing_time = result.total_time

            console.print(f"[green]Indexed {chunks} chunks from {files} files[/green]")
            console.print(f"[dim]Processing time: {processing_time:.2f}s[/dim]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise typer.Exit(code=1)
        finally:
            await storage.disconnect()
            await embedder.close()

    asyncio.run(_index())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    mode: SearchMode = typer.Option(
        SearchMode.HYBRID,
        "--mode",
        "-m",
        help="Search mode: vector, keyword, or hybrid"
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Maximum number of results"
    ),
    content: bool = typer.Option(
        False,
        "--content",
        "-c",
        help="Show full chunk content"
    )
):
    """Search indexed code using vector, keyword, or hybrid search."""
    async def _search():
        storage = get_storage()
        embedder = get_embedder(console)
        llm = get_llm(console)

        try:
            await storage.connect()

            console.print(f"[dim]Searching with mode: {mode.value}[/dim]\n")

            search_engine = create_search_engine(
                storage=storage,
                embedder=embedder,
                llm=llm
            )

            search_query = SearchQuery(
                query=query,
                mode=mode,
                limit=limit
            )

            response = await search_engine.search(search_query)

            if not response.results:
                console.print("[yellow]No results found[/yellow]")
                return

            console.print(f"[green]Found {response.total_results} results[/green]\n")

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Rank", style="dim", width=4)
            table.add_column("File", style="cyan")
            table.add_column("Type", style="yellow", width=15)
            table.add_column("Score", style="green", width=8)

            for i, result in enumerate(response.results, 1):
                file_path = Path(result.chunk.file_path).name
                chunk_type = result.chunk.metadata.get("chunk_type")
                if not chunk_type:
                    language = result.chunk.metadata.get("language", "unknown")
                    chunk_type = language if language else "unknown"
                score = f"{result.score:.4f}" if result.score else "N/A"

                table.add_row(str(i), file_path, chunk_type, score)

                if content:
                    console.print(Panel(
                        result.chunk.chunk_text[:500] + "..." if len(result.chunk.chunk_text) > 500 else result.chunk.chunk_text,
                        title=f"Content (Chunk {i})",
                        border_style="dim"
                    ))

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise typer.Exit(code=1)
        finally:
            await storage.disconnect()
            await embedder.close()
            if llm:
                await llm.close()

    asyncio.run(_search())


@app.command()
def stats():
    """Show indexing statistics from the database."""
    async def _stats():
        storage = get_storage()

        try:
            await storage.connect()

            stats = await storage.get_statistics()

            table = Table(show_header=False, box=None)
            table.add_column("Metric", style="bold cyan", width=15)
            table.add_column("Value")

            table.add_row("Total Chunks", str(stats.get("total_chunks", 0)))
            table.add_row("Total Files", str(stats.get("total_files", 0)))
            table.add_row("Total Size", f"{stats.get('total_size_mb', 0):.2f} MB")

            languages = stats.get("languages", {})
            lang_str = ", ".join([f"{k}: {v}" for k, v in languages.items()])
            table.add_row("Languages", lang_str or "None")

            last_indexed = stats.get("last_indexed", "Never")
            table.add_row("Last Indexed", str(last_indexed))

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)
        finally:
            await storage.disconnect()

    asyncio.run(_stats())


@app.command()
def health():
    """Check database connection and service health."""
    async def _health():
        all_healthy = True

        storage = get_storage()
        try:
            await storage.connect()
            console.print("[green]+[/green] Database connection: OK")
            await storage.disconnect()
        except Exception as e:
            console.print(f"[red]x[/red] Database connection: FAILED ({e})")
            all_healthy = False

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            console.print("[green]+[/green] OpenAI API key: SET")
        else:
            console.print("[yellow]![/yellow] OpenAI API key: NOT SET")

        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            console.print("[green]+[/green] DeepSeek API key: SET")
        else:
            console.print("[yellow]![/yellow] DeepSeek API key: NOT SET")

        if not all_healthy:
            raise typer.Exit(code=1)

    asyncio.run(_health())


@app.command()
def clear(
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt"
    )
):
    """Clear all indexed data from the database."""
    async def _clear():
        if not yes:
            console.print("[yellow]WARNING: This will delete all indexed data![/yellow]")
            confirm = typer.confirm("Are you sure you want to continue?")
            if not confirm:
                console.print("[dim]Aborted.[/dim]")
                return

        storage = get_storage()

        try:
            await storage.connect()

            console.print("[dim]Clearing all indexed data...[/dim]")
            deleted_count = await storage.clear_all()

            console.print(f"[green]Success! Deleted {deleted_count} chunks.[/green]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)
        finally:
            await storage.disconnect()

    asyncio.run(_clear())


@app.command()
def chat(
    max_steps: int = typer.Option(
        10,
        "--max-steps",
        "-s",
        help="Maximum reasoning steps"
    )
):
    """Interactive chat mode with the reasoning agent."""
    async def _chat():
        llm = get_llm(console)

        if not llm:
            console.print("[red]Error: LLM provider not configured[/red]")
            raise typer.Exit(code=1)

        storage = get_storage()
        embedder = get_embedder(console)

        try:
            await storage.connect()

            from ..reasoning_agent import (
                AnalyzeCodeTool,
                ReadFileTool,
                ReasoningAgent,
                SearchCodebaseTool,
            )

            search_tool = SearchCodebaseTool(storage, embedder)
            read_tool = ReadFileTool()
            analyze_tool = AnalyzeCodeTool()

            agent = ReasoningAgent(
                llm=llm,
                tools=[search_tool, read_tool, analyze_tool]
            )

            console.print("[bold cyan]Pytelos Interactive Chat[/bold cyan]")
            console.print("[dim]Type 'exit', 'quit', or 'q' to leave[/dim]\n")

            while True:
                try:
                    user_input = console.input("[bold yellow]You:[/bold yellow] ")

                    if not user_input.strip():
                        continue

                    if user_input.strip().lower() in ('exit', 'quit', 'q'):
                        console.print("[dim]Goodbye![/dim]")
                        break

                    from ..agent import Task

                    task = Task(instruction=user_input)
                    handler = agent.run(task, max_steps=max_steps)

                    result = await handler

                    console.print(f"[bold green]Agent:[/bold green] {result.content}\n")

                except KeyboardInterrupt:
                    console.print("\n[dim]Goodbye![/dim]")
                    break
                except EOFError:
                    console.print("\n[dim]Goodbye![/dim]")
                    break

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise typer.Exit(code=1)
        finally:
            await storage.disconnect()
            await embedder.close()
            await llm.close()

    asyncio.run(_chat())


@app.command(name="tui")
def tui_command(
    max_steps: int = typer.Option(
        10,
        "--max-steps",
        "-s",
        help="Maximum reasoning steps"
    ),
    durable: bool = typer.Option(
        False,
        "--durable",
        "-d",
        help="Use PyergonReasoningAgent for durable execution"
    ),
    memory_backend: str = typer.Option(
        "memory",
        "--memory",
        "-m",
        help="Conversation memory: 'memory' (session-only) or 'sqlite' (persistent)"
    ),
    memory_path: str | None = typer.Option(
        None,
        "--memory-path",
        help="Path for SQLite memory database (only with --memory sqlite)"
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Show log panel with level: debug (all), info, warning, or error"
    ),
):
    """Launch interactive TUI chat interface."""
    async def _tui():
        from ..ui import run_textual_tui

        llm = get_llm(console)

        if not llm:
            console.print("[red]Error: LLM provider not configured[/red]")
            raise typer.Exit(code=1)

        storage = get_storage()
        embedder = get_embedder(console)

        try:
            await storage.connect()

            await run_textual_tui(
                llm=llm,
                storage=storage,
                embedder=embedder,
                max_steps=max_steps,
                log_level=log_level,
                memory_backend=memory_backend,
                memory_path=memory_path,
                durable=durable,
            )
        finally:
            try:
                await storage.disconnect()
                await embedder.close()
                await llm.close()
            except BaseException:
                pass
            console.print("\n[dim]Goodbye![/dim]")

    try:
        asyncio.run(_tui())
    except KeyboardInterrupt:
        pass


@app.command()
def solve(
    task: str = typer.Argument(
        None,
        help="Task for the agent to solve (omit for interactive mode)"
    ),
    max_steps: int = typer.Option(
        10,
        "--max-steps",
        "-s",
        help="Maximum number of reasoning steps"
    ),
    distributed: bool = typer.Option(
        False,
        "--distributed",
        "-d",
        help="Use distributed Pyergon workflow for durable execution"
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help="Number of parallel workers (only with --distributed)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed execution steps"
    ),
):
    """Solve complex tasks using multi-step reasoning and tools.

    Run without arguments for interactive mode, or provide a task for single-shot mode.

    The reasoning agent can:
    - Search the codebase for relevant information
    - Read and analyze files
    - Break down complex tasks into steps
    - Use tools dynamically based on the task
    """

    async def _process_task(agent, task_text, agent_type):
        """Process a single task."""
        from ..agent import Task

        console.print(f"[dim]Starting reasoning with max {max_steps} steps...[/dim]\n")

        task_obj = Task(instruction=task_text)

        if agent_type == "distributed":
            result = await agent.solve(
                task_obj,
                max_steps=max_steps,
                show_progress=True,
                num_workers=workers
            )
        else:
            handler = agent.run(task_obj, max_steps=max_steps)

            last_step_count = 0
            while not handler.done():
                await asyncio.sleep(0.5)
                current_step = handler.step_counter
                if current_step > last_step_count:
                    console.print(f"[dim]Step {current_step} completed...[/dim]")
                    last_step_count = current_step

            result = await handler

        # Show detailed execution if verbose
        if verbose and hasattr(result, 'execution_trace') and result.execution_trace:
            console.print("\n[bold cyan]Execution Steps:[/bold cyan]")
            console.print("[dim]" + "=" * 80 + "[/dim]")
            console.print(result.execution_trace)
            console.print("[dim]" + "=" * 80 + "[/dim]")

        # Show result
        console.print("\n[bold green]Result:[/bold green]")
        console.print(result.content)

        # Display statistics
        if hasattr(result, 'metadata') and result.metadata:
            console.print()
            steps = result.metadata.get('steps', 0)
            processing_time = result.metadata.get('processing_time_seconds', 0)
            input_tokens = result.metadata.get('total_input_tokens', 0)
            output_tokens = result.metadata.get('total_output_tokens', 0)

            console.print(f"[dim]Steps: {steps}[/dim]")
            console.print(f"[dim]Processing time: {processing_time:.2f}s[/dim]")
            if input_tokens or output_tokens:
                console.print(f"[dim]Input tokens: {input_tokens:,}[/dim]")
                console.print(f"[dim]Output tokens: {output_tokens:,}[/dim]")

    async def _solve():
        llm = get_llm(console)

        if not llm:
            console.print("[red]Error: DEEPSEEK_API_KEY or OPENAI_API_KEY required[/red]")
            raise typer.Exit(code=1)

        storage = get_storage()
        embedder = get_embedder(console)

        try:
            await storage.connect()

            from ..reasoning_agent import (
                AnalyzeCodeTool,
                ReadFileTool,
                SearchCodebaseTool,
            )

            search_tool = SearchCodebaseTool(storage, embedder)
            read_tool = ReadFileTool()
            analyze_tool = AnalyzeCodeTool()

            if distributed:
                from ..reasoning_agent import PyergonReasoningAgent

                agent = PyergonReasoningAgent(
                    llm=llm,
                    tools=[search_tool, read_tool, analyze_tool],
                    console=console
                )
                agent_type = "distributed"
            else:
                from ..reasoning_agent import ReasoningAgent

                agent = ReasoningAgent(
                    llm=llm,
                    tools=[search_tool, read_tool, analyze_tool]
                )
                agent_type = "sequential"

            if task:
                console.print(f"[bold cyan]Task:[/bold cyan] {task}\n")
                await _process_task(agent, task, agent_type)
            else:
                # Interactive mode
                console.print("[bold cyan]Pytelos Solver[/bold cyan]")
                console.print("[dim]Type 'exit', 'quit', or 'q' to leave\n[/dim]")

                while True:
                    try:
                        user_input = console.input("[bold yellow]Task:[/bold yellow] ")

                        if not user_input.strip():
                            continue

                        if user_input.strip().lower() in ('exit', 'quit', 'q'):
                            console.print("[dim]Goodbye![/dim]")
                            break

                        console.print()
                        await _process_task(agent, user_input, agent_type)

                    except KeyboardInterrupt:
                        console.print("\n[dim]Goodbye![/dim]")
                        break
                    except EOFError:
                        console.print("\n[dim]Goodbye![/dim]")
                        break

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise typer.Exit(code=1)
        finally:
            if storage:
                await storage.disconnect()
            if embedder:
                await embedder.close()
            await agent.close()

    asyncio.run(_solve())


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
