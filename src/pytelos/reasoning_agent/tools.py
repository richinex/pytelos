"""Tool infrastructure for reasoning agent."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..embedding import EmbeddingProvider
from ..search import SearchMode, SearchQuery
from ..storage import StorageBackend
from .data_structures import ToolCall, ToolCallResult


class BaseTool(ABC):
    """Abstract base class for tools.

    Following agent.txt tool patterns.
    """

    def __init__(self) -> None:
        self._debug_callback: Any | None = None

    def set_debug_callback(self, callback: Any) -> None:
        """Set debug callback for logging.

        Args:
            callback: Callable(level: str, component: str, message: str)
        """
        self._debug_callback = callback

    def _debug(self, level: str, component: str, message: str) -> None:
        """Send debug message if callback is set."""
        if self._debug_callback:
            self._debug_callback(level, component, message)

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the tool description for the LLM."""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """Get the JSON schema for tool parameters."""
        pass

    @abstractmethod
    async def execute(self, tool_call: ToolCall) -> ToolCallResult:
        """Execute the tool call.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolCallResult with the execution result
        """
        pass

    def to_llm_spec(self) -> dict[str, Any]:
        """Convert tool to LLM-friendly specification.

        Returns:
            Dictionary describing the tool for the LLM
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema
        }


class SearchCodebaseTool(BaseTool):
    """Tool for searching the indexed codebase.

    Allows the agent to search for relevant code/documents.
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedder: EmbeddingProvider
    ):
        """Initialize the search tool.

        Args:
            storage: Storage backend
            embedder: Embedding provider
        """
        super().__init__()
        self._storage = storage
        self._embedder = embedder

    @property
    def name(self) -> str:
        return "search_codebase"

    @property
    def description(self) -> str:
        return (
            "Search ALL indexed content including code files, PDFs, and documents. "
            "This tool performs semantic and keyword search across the entire indexed database. "
            "Use this to find information about functions, classes, concepts, documentation, "
            "or any topics from indexed PDFs and code files."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def execute(self, tool_call: ToolCall) -> ToolCallResult:
        """Execute the search."""
        try:
            from ..search import create_search_engine

            query = tool_call.arguments.get("query", "")
            limit = tool_call.arguments.get("limit", 5)

            if not query:
                return ToolCallResult(
                    tool_call_id=tool_call.id_,
                    content="Error: query parameter is required",
                    error=True
                )

            self._debug("info", "Search", f"Query: '{query[:50]}...' (limit={limit})")

            # Create search engine
            engine = create_search_engine(self._storage, self._embedder)

            # Execute search asynchronously (embedding happens inside engine)
            self._debug("info", "Search", "Executing hybrid search (embedding + vector + BM25)...")
            result = await engine.search(SearchQuery(
                query=query,
                mode=SearchMode.HYBRID,
                limit=limit
            ))
            self._debug("info", "Search", f"Search complete: {result.total_results} results found")

            # Format results
            if result.total_results == 0:
                content = f"No results found for query: {query}"
            else:
                results_list = []
                for i, res in enumerate(result.results, 1):
                    results_list.append({
                        "rank": i,
                        "file": res.chunk.file_path,
                        "lines": f"{res.chunk.start_line}-{res.chunk.end_line}",
                        "score": round(res.score, 4),
                        "content": res.chunk.chunk_text
                    })

                content = f"Found {result.total_results} results:\n"
                content += json.dumps(results_list, indent=2)

            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content=content,
                error=False
            )

        except Exception as e:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content=f"Error executing search: {str(e)}",
                error=True
            )


class ReadFileTool(BaseTool):
    """Tool for reading file contents.

    Allows the agent to read specific files from the codebase.
    """

    def __init__(self, base_path: Path | None = None):
        """Initialize the read file tool.

        Args:
            base_path: Base directory for file operations
        """
        super().__init__()
        self._base_path = base_path or Path.cwd()

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a file. "
            "Use this when you need to examine the full contents of a specific file."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional: Starting line number (1-indexed)",
                    "default": 1
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional: Ending line number",
                    "default": -1
                }
            },
            "required": ["file_path"]
        }

    async def execute(self, tool_call: ToolCall) -> ToolCallResult:
        """Execute the file read."""
        try:
            file_path = tool_call.arguments.get("file_path", "")
            start_line = tool_call.arguments.get("start_line", 1)
            end_line = tool_call.arguments.get("end_line", -1)

            if not file_path:
                return ToolCallResult(
                    tool_call_id=tool_call.id_,
                    content="Error: file_path parameter is required",
                    error=True
                )

            # Resolve path
            path = Path(file_path)
            if not path.is_absolute():
                path = self._base_path / path

            if not path.exists():
                return ToolCallResult(
                    tool_call_id=tool_call.id_,
                    content=f"Error: File not found: {file_path}",
                    error=True
                )

            # Read file
            with open(path, encoding='utf-8') as f:
                lines = f.readlines()

            # Apply line range
            if end_line == -1:
                end_line = len(lines)

            selected_lines = lines[start_line - 1:end_line]
            content = ''.join(selected_lines)

            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content=f"File: {file_path}\nLines {start_line}-{end_line}:\n\n{content}",
                error=False
            )

        except Exception as e:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content=f"Error reading file: {str(e)}",
                error=True
            )


class AnalyzeCodeTool(BaseTool):
    """Tool for analyzing code structure.

    Helps the agent understand code organization.
    """

    def __init__(self) -> None:
        """Initialize the analyze code tool."""
        super().__init__()

    @property
    def name(self) -> str:
        return "analyze_code"

    @property
    def description(self) -> str:
        return (
            "Analyze code to extract structure information like "
            "functions, classes, imports, and dependencies. "
            "Useful for understanding code organization."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to analyze"
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (default: python)",
                    "default": "python"
                }
            },
            "required": ["code"]
        }

    async def execute(self, tool_call: ToolCall) -> ToolCallResult:
        """Execute code analysis."""
        try:
            code = tool_call.arguments.get("code", "")
            language = tool_call.arguments.get("language", "python")

            if not code:
                return ToolCallResult(
                    tool_call_id=tool_call.id_,
                    content="Error: code parameter is required",
                    error=True
                )

            if language == "python":
                # Simple Python analysis using AST
                import ast

                try:
                    tree = ast.parse(code)
                except SyntaxError as e:
                    return ToolCallResult(
                        tool_call_id=tool_call.id_,
                        content=f"Syntax error in code: {str(e)}",
                        error=True
                    )

                analysis = {
                    "functions": [],
                    "classes": [],
                    "imports": []
                }

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        analysis["functions"].append({
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "line": node.lineno
                        })
                    elif isinstance(node, ast.ClassDef):
                        methods = [
                            n.name for n in node.body
                            if isinstance(n, ast.FunctionDef)
                        ]
                        analysis["classes"].append({
                            "name": node.name,
                            "methods": methods,
                            "line": node.lineno
                        })
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis["imports"].append(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        for alias in node.names:
                            analysis["imports"].append(f"{node.module}.{alias.name}")

                content = "Code Analysis:\n" + json.dumps(analysis, indent=2)

                return ToolCallResult(
                    tool_call_id=tool_call.id_,
                    content=content,
                    error=False
                )
            else:
                return ToolCallResult(
                    tool_call_id=tool_call.id_,
                    content=f"Analysis for {language} not yet implemented",
                    error=True
                )

        except Exception as e:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content=f"Error analyzing code: {str(e)}",
                error=True
            )
