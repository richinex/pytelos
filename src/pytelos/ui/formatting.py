"""Text formatting utilities for the TUI.

Hides the details of markdown rendering and text cleanup.
"""

import re

from rich.markdown import Markdown
from rich.text import Text


def clean_latex(text: str) -> str:
    """Convert LaTeX notation to plain text equivalents.

    Handles common LaTeX patterns that Rich cannot render:
    - \\( ... \\) inline math -> just the content
    - \\[ ... \\] display math -> just the content
    - $...$ inline math -> just the content
    - $$...$$ display math -> just the content
    """
    # Remove \( ... \) inline math delimiters
    text = re.sub(r'\\\(\s*', '', text)
    text = re.sub(r'\s*\\\)', '', text)

    # Remove \[ ... \] display math delimiters
    text = re.sub(r'\\\[\s*', '', text)
    text = re.sub(r'\s*\\\]', '', text)

    # Remove $$ ... $$ display math delimiters (do this before single $)
    text = re.sub(r'\$\$\s*', '', text)

    # Remove $ ... $ inline math delimiters (but not escaped \$)
    text = re.sub(r'(?<!\\)\$([^$]+)(?<!\\)\$', r'\1', text)

    # Clean up common LaTeX commands
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', text)
    text = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', text)
    text = re.sub(r'\\sum', 'sum', text)
    text = re.sub(r'\\prod', 'product', text)
    text = re.sub(r'\\int', 'integral', text)
    text = re.sub(r'\\infty', 'infinity', text)
    text = re.sub(r'\\pi', 'pi', text)
    text = re.sub(r'\\times', 'x', text)
    text = re.sub(r'\\cdot', '*', text)
    text = re.sub(r'\\pm', '+/-', text)
    text = re.sub(r'\\leq', '<=', text)
    text = re.sub(r'\\geq', '>=', text)
    text = re.sub(r'\\neq', '!=', text)
    text = re.sub(r'\\approx', '~=', text)
    text = re.sub(r'\\ldots', '...', text)
    text = re.sub(r'\\cdots', '...', text)
    text = re.sub(r'\\quad', ' ', text)
    text = re.sub(r'\\qquad', '  ', text)
    text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textit\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\mathbf\{([^}]*)\}', r'\1', text)

    # Remove remaining backslash commands but keep the argument
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)

    # Clean up superscripts and subscripts
    text = re.sub(r'\^{([^}]*)}', r'^(\1)', text)
    text = re.sub(r'_{([^}]*)}', r'_(\1)', text)
    text = re.sub(r'\^(\w)', r'^\1', text)
    text = re.sub(r'_(\w)', r'_\1', text)

    return text


def render_markdown(text: str) -> Markdown:
    """Render text as markdown with LaTeX cleaned up."""
    cleaned = clean_latex(text)
    return Markdown(cleaned)


def render_text_styled(text: str, style: str = "") -> Text:
    """Render text with optional style and text wrapping.

    Handles Rich markup like [bold green]...[/] in the text.
    Falls back to plain text if markup parsing fails.
    """
    cleaned = clean_latex(text)
    try:
        # Use from_markup to handle Rich markup tags like [bold green]...[/]
        result = Text.from_markup(cleaned, overflow="fold")
    except Exception:
        # Fallback to plain text if markup parsing fails
        # (e.g., text contains invalid style names from tracebacks)
        result = Text(cleaned, overflow="fold")
    if style:
        result.stylize(style)
    return result


def is_tabular_data(data: object) -> bool:
    """Check if data is tabular (can be rendered as a table).

    Supports:
    - Dict with list values: {'col1': [1,2,3], 'col2': ['a','b','c']}
    - List of dicts: [{'col1': 1, 'col2': 'a'}, {'col1': 2, 'col2': 'b'}]
    """
    if isinstance(data, dict):
        # Dict with list values (columnar format)
        if data and all(isinstance(v, (list, tuple)) for v in data.values()):
            lengths = [len(v) for v in data.values()]
            return len(lengths) > 0 and len(set(lengths)) == 1
    elif isinstance(data, (list, tuple)) and data and all(isinstance(item, dict) for item in data):
        # List of dicts (row format)
        return True
    return False


def format_as_markdown_table(data: object) -> str:
    """Convert tabular data to markdown table format.

    Args:
        data: Dict with list values or list of dicts

    Returns:
        Markdown table string
    """
    if isinstance(data, dict) and all(isinstance(v, (list, tuple)) for v in data.values()):
        # Dict with list values -> columnar format
        headers = list(data.keys())
        if not headers:
            return str(data)

        rows = list(zip(*data.values(), strict=False))

        # Build markdown table
        lines = []
        # Header row
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        # Separator
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        # Data rows
        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(lines)

    elif isinstance(data, (list, tuple)) and data and all(isinstance(item, dict) for item in data):
        # List of dicts -> row format
        # Get all unique keys as headers
        headers = []
        for item in data:
            for key in item:
                if key not in headers:
                    headers.append(key)

        if not headers:
            return str(data)

        # Build markdown table
        lines = []
        # Header row
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        # Separator
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        # Data rows
        for item in data:
            row_values = [str(item.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(row_values) + " |")

        return "\n".join(lines)

    # Fallback to string representation
    return str(data)


def format_final_answer(data: object) -> str:
    """Format the final answer for display.

    Detects tabular data and formats as markdown table,
    otherwise returns string representation.

    Args:
        data: The final answer from agent execution

    Returns:
        Formatted string (may include markdown table)
    """
    # If already tabular data, format directly
    if is_tabular_data(data):
        return format_as_markdown_table(data)

    # If string, try to parse as Python literal (dict/list)
    if isinstance(data, str):
        import ast
        try:
            parsed = ast.literal_eval(data)
            if is_tabular_data(parsed):
                return format_as_markdown_table(parsed)
        except (ValueError, SyntaxError):
            pass  # Not a valid Python literal, use as-is

    return str(data)
