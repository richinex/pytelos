"""Prompt management module.

Externalizes prompts to text files for easy customization.
Prompts can be overridden by placing files in the working directory.
"""

from functools import lru_cache
from pathlib import Path

# Default prompts directory (package location)
_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    """Load a prompt from file.

    Search order:
    1. Current working directory: ./prompts/{name}.txt
    2. Package prompts directory: pytelos/prompts/{name}.txt

    Args:
        name: Prompt name (without .txt extension)

    Returns:
        Prompt text content

    Raises:
        FileNotFoundError: If prompt file not found in any location
    """
    filename = f"{name}.txt"

    # Check working directory first (allows user overrides)
    local_path = Path.cwd() / "prompts" / filename
    if local_path.exists():
        return local_path.read_text(encoding="utf-8")

    # Fall back to package prompts
    package_path = _PROMPTS_DIR / filename
    if package_path.exists():
        return package_path.read_text(encoding="utf-8")

    raise FileNotFoundError(
        f"Prompt '{name}' not found. Searched:\n"
        f"  - {local_path}\n"
        f"  - {package_path}"
    )


def get_system_prompt() -> str:
    """Get the system prompt used by all agents."""
    return load_prompt("system")


def clear_cache() -> None:
    """Clear the prompt cache (useful after modifying prompt files)."""
    load_prompt.cache_clear()


__all__ = [
    "load_prompt",
    "get_system_prompt",
    "clear_cache",
]
