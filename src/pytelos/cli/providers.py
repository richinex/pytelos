"""Provider factory functions for CLI.

Centralizes creation of storage, embedder, and LLM instances from environment variables.
Hides configuration details from command implementations.
"""

import os
from typing import Any

from rich.console import Console

from ..embedding import create_embedding_provider
from ..llm import create_llm_provider
from ..storage import create_storage_backend

# Default console for output
_console = Console()


def get_storage() -> Any:
    """Create storage backend from environment variables.

    Returns:
        PostgreSQL storage backend instance

    Environment variables:
        POSTGRES_HOST: Database host (default: localhost)
        POSTGRES_PORT: Database port (default: 5433)
        POSTGRES_DB: Database name (default: pytelos)
        POSTGRES_USER: Database user (default: pytelos)
        POSTGRES_PASSWORD: Database password (default: pytelos_dev)
    """
    return create_storage_backend(
        "postgres",
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5433")),
        database=os.getenv("POSTGRES_DB", "pytelos"),
        user=os.getenv("POSTGRES_USER", "pytelos"),
        password=os.getenv("POSTGRES_PASSWORD", "pytelos_dev")
    )


def get_embedder(console: Console | None = None) -> Any:
    """Create embedding provider from environment variables.

    Args:
        console: Optional Rich console for output

    Returns:
        OpenAI embedding provider instance

    Raises:
        SystemExit: If OPENAI_API_KEY is not set

    Environment variables:
        OPENAI_API_KEY: OpenAI API key (required)
    """
    import typer

    con = console or _console
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        con.print("[red]Error: OPENAI_API_KEY not set in environment[/red]")
        raise typer.Exit(code=1)

    return create_embedding_provider("openai", api_key=api_key)


def get_llm(console: Console | None = None) -> Any | None:
    """Create LLM provider from environment variables.

    Args:
        console: Optional Rich console for output

    Returns:
        LLM provider instance, or None if not configured

    Environment variables:
        LLM_PROVIDER: Provider type (openai, deepseek, anthropic, gemini; default: deepseek)
        OPENAI_API_KEY: OpenAI API key (for openai provider)
        OPENAI_CHAT_MODEL: OpenAI model (default: gpt-4o-mini)
        DEEPSEEK_API_KEY: DeepSeek API key (for deepseek provider)
        ANTHROPIC_API_KEY: Anthropic API key (for anthropic provider)
        ANTHROPIC_MODEL: Anthropic model (default: claude-sonnet-4-20250514)
        GEMINI_API_KEY: Gemini API key (for gemini provider)
        GEMINI_MODEL: Gemini model (default: gemini-2.5-flash)
    """
    con = console or _console
    llm_provider = os.getenv("LLM_PROVIDER", "deepseek").lower()

    if llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            con.print("[yellow]Warning: OPENAI_API_KEY not set, LLM features disabled[/yellow]")
            return None
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        return create_llm_provider("openai", api_key=api_key, model=model)

    elif llm_provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            con.print("[yellow]Warning: DEEPSEEK_API_KEY not set, LLM features disabled[/yellow]")
            return None
        return create_llm_provider("deepseek", api_key=api_key)

    elif llm_provider in ("anthropic", "claude"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            con.print("[yellow]Warning: ANTHROPIC_API_KEY not set, LLM features disabled[/yellow]")
            return None
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        return create_llm_provider("anthropic", api_key=api_key, model=model)

    elif llm_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            con.print("[yellow]Warning: GEMINI_API_KEY not set, LLM features disabled[/yellow]")
            return None
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return create_llm_provider("gemini", api_key=api_key, model=model)

    else:
        con.print(f"[red]Error: Unknown LLM provider: {llm_provider}[/red]")
        return None


def require_llm(console: Console | None = None) -> Any:
    """Get LLM provider, raising error if not configured.

    Args:
        console: Optional Rich console for output

    Returns:
        LLM provider instance

    Raises:
        SystemExit: If LLM provider is not configured
    """
    import typer

    con = console or _console
    llm = get_llm(con)
    if not llm:
        con.print("[red]Error: LLM provider not configured[/red]")
        raise typer.Exit(code=1)
    return llm
