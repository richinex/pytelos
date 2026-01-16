from pathlib import Path
from typing import Any

from tree_sitter import Language, Parser
from tree_sitter_python import language

from ..base import CodeParser
from ..models import ChunkingStrategy, CodeChunkMetadata, ParsedChunk


class PythonParser(CodeParser):
    """Python code parser using tree-sitter.

    Hidden design decisions:
    - Using tree-sitter for AST parsing
    - Specific node types to extract (functions, classes, methods)
    - Metadata extraction rules
    - Chunking algorithm for each strategy
    """

    def __init__(self):
        """Initialize the Python parser."""
        self._parser = Parser(Language(language()))
        self._supported_extensions = {".py", ".pyi"}

    def supports_language(self, language: str) -> bool:
        """Check if this parser supports Python.

        Args:
            language: Language identifier

        Returns:
            True if language is 'python'
        """
        return language.lower() == "python"

    def detect_language(self, file_path: Path) -> str | None:
        """Detect if file is Python based on extension.

        Args:
            file_path: Path to the file

        Returns:
            'python' if file has .py or .pyi extension, None otherwise
        """
        if file_path.suffix in self._supported_extensions:
            return "python"
        return None

    def get_file_type(self) -> str:
        """Get the file type this parser handles.

        Returns:
            'python'
        """
        return "python"

    def get_supported_extensions(self) -> set[str]:
        """Get file extensions supported by this parser.

        Returns:
            Set of Python file extensions
        """
        return self._supported_extensions

    def parse_file(
        self,
        file_path: Path,
        chunk_size: int = 1000,
        strategy: ChunkingStrategy = ChunkingStrategy.BY_FUNCTION,
        overlap: int = 200,
        **options: Any
    ) -> list[ParsedChunk]:
        """Parse a Python file into chunks.

        Args:
            file_path: Path to the Python file
            chunk_size: Maximum chunk size in characters (for BY_LINES)
            strategy: Chunking strategy to use
            overlap: Number of overlapping characters (for BY_LINES)
            **options: Additional parser options (currently unused)

        Returns:
            List of parsed chunks

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a Python file
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if self.detect_language(file_path) != "python":
            raise ValueError(f"Not a Python file: {file_path}")

        # Read file content
        content = file_path.read_text(encoding="utf-8")
        content_bytes = content.encode("utf-8")

        # Parse with tree-sitter
        tree = self._parser.parse(content_bytes)

        # Choose chunking strategy
        if strategy == ChunkingStrategy.BY_FUNCTION:
            return self._chunk_by_function(file_path, content, tree.root_node)
        elif strategy == ChunkingStrategy.BY_LINES:
            return self._chunk_by_lines(file_path, content, chunk_size, overlap)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(file_path, content, tree.root_node)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

    def _chunk_by_function(
        self,
        file_path: Path,
        content: str,
        root_node: any
    ) -> list[ParsedChunk]:
        """Chunk code by extracting functions and classes.

        Args:
            file_path: Path to the file
            content: File content
            root_node: Tree-sitter root node

        Returns:
            List of chunks, one per function/class
        """
        chunks = []
        lines = content.split("\n")

        def extract_node_content(node: any) -> str:
            """Extract text content from a node."""
            start_byte = node.start_byte
            end_byte = node.end_byte
            return content.encode("utf-8")[start_byte:end_byte].decode("utf-8")

        def strip_docstring_quotes(doc: str) -> str:
            """Strip docstring quote markers from a string."""
            # Check for triple quotes first (most common for docstrings)
            for quote in ('"""', "'''"):
                if doc.startswith(quote) and doc.endswith(quote):
                    return doc[3:-3]
            # Check for single quotes
            for quote in ('"', "'"):
                if doc.startswith(quote) and doc.endswith(quote):
                    return doc[1:-1]
            return doc

        def extract_docstring(node: any) -> str | None:
            """Extract docstring from a function/class node."""
            if node.child_count == 0:
                return None

            # Look for first expression_statement with a string
            for child in node.children:
                if child.type == "block":
                    for stmt in child.children:
                        if stmt.type == "expression_statement":
                            for expr_child in stmt.children:
                                if expr_child.type == "string":
                                    doc = extract_node_content(expr_child)
                                    return strip_docstring_quotes(doc)
            return None

        def visit_node(node: any, class_name: str | None = None):
            """Visit tree nodes and extract functions/classes."""
            if node.type == "function_definition":
                # Extract function
                func_name_node = node.child_by_field_name("name")
                func_name = extract_node_content(func_name_node) if func_name_node else "unknown"

                start_line = node.start_point[0] + 1  # 1-indexed
                end_line = node.end_point[0] + 1

                chunk_content = extract_node_content(node)
                docstring = extract_docstring(node)

                metadata = CodeChunkMetadata(
                    language="python",
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    function_name=func_name,
                    class_name=class_name,
                    docstring=docstring,
                    imports=[]  # Could extract from module level
                )

                chunks.append(ParsedChunk(content=chunk_content, metadata=metadata))

            elif node.type == "class_definition":
                # Extract class name
                class_name_node = node.child_by_field_name("name")
                current_class = extract_node_content(class_name_node) if class_name_node else "unknown"

                # Extract the class itself as a chunk
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                chunk_content = extract_node_content(node)
                docstring = extract_docstring(node)

                metadata = CodeChunkMetadata(
                    language="python",
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    class_name=current_class,
                    docstring=docstring,
                    imports=[]
                )

                chunks.append(ParsedChunk(content=chunk_content, metadata=metadata))

                # Visit children to get methods
                for child in node.children:
                    if child.type == "block":
                        for stmt in child.children:
                            visit_node(stmt, class_name=current_class)

            # Recurse for other nodes
            elif node.type != "function_definition" and class_name is None:
                for child in node.children:
                    visit_node(child, class_name)

        # Visit all top-level nodes
        visit_node(root_node)

        # If no chunks found (e.g., script with no functions), create one chunk for the whole file
        if not chunks:
            metadata = CodeChunkMetadata(
                language="python",
                file_path=str(file_path),
                start_line=1,
                end_line=len(lines),
                imports=[]
            )
            chunks.append(ParsedChunk(content=content, metadata=metadata))

        return chunks

    def _chunk_by_lines(
        self,
        file_path: Path,
        content: str,
        max_chunk_size: int,
        overlap: int
    ) -> list[ParsedChunk]:
        """Chunk code by lines with overlap.

        Args:
            file_path: Path to the file
            content: File content
            max_chunk_size: Maximum characters per chunk
            overlap: Overlapping characters between chunks

        Returns:
            List of line-based chunks
        """
        chunks = []
        lines = content.split("\n")

        current_chunk_lines = []
        current_size = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > max_chunk_size and current_chunk_lines:
                # Save current chunk
                chunk_content = "\n".join(current_chunk_lines)
                metadata = CodeChunkMetadata(
                    language="python",
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=i - 1,
                    imports=[]
                )
                chunks.append(ParsedChunk(content=chunk_content, metadata=metadata))

                # Calculate overlap
                overlap_chars = 0
                overlap_lines = []
                for prev_line in reversed(current_chunk_lines):
                    if overlap_chars + len(prev_line) + 1 <= overlap:
                        overlap_lines.insert(0, prev_line)
                        overlap_chars += len(prev_line) + 1
                    else:
                        break

                # Start new chunk with overlap
                current_chunk_lines = overlap_lines
                current_size = overlap_chars
                start_line = i - len(overlap_lines)

            current_chunk_lines.append(line)
            current_size += line_size

        # Add final chunk
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            metadata = CodeChunkMetadata(
                language="python",
                file_path=str(file_path),
                start_line=start_line,
                end_line=len(lines),
                imports=[]
            )
            chunks.append(ParsedChunk(content=chunk_content, metadata=metadata))

        return chunks

    def _chunk_semantic(
        self,
        file_path: Path,
        content: str,
        root_node: any
    ) -> list[ParsedChunk]:
        """Chunk code semantically based on cAST algorithm.

        Implements hierarchical semantic chunking:
        - Level 1: Module-level (imports, constants, type definitions)
        - Level 2: Class-level chunks
        - Level 3: Function-level chunks
        - Groups related constructs together

        Based on research from cAST paper (EMNLP 2025) and GitHub Copilot.

        Args:
            file_path: Path to the file
            content: File content
            root_node: Tree-sitter root node

        Returns:
            List of semantically grouped chunks
        """
        chunks = []
        lines = content.split("\n")

        def extract_node_content(node: any) -> str:
            """Extract text content from a node."""
            start_byte = node.start_byte
            end_byte = node.end_byte
            return content.encode("utf-8")[start_byte:end_byte].decode("utf-8")

        def strip_docstring_quotes(doc: str) -> str:
            """Strip docstring quote markers from a string."""
            for quote in ('"""', "'''"):
                if doc.startswith(quote) and doc.endswith(quote):
                    return doc[3:-3]
            for quote in ('"', "'"):
                if doc.startswith(quote) and doc.endswith(quote):
                    return doc[1:-1]
            return doc

        def extract_docstring(node: any) -> str | None:
            """Extract docstring from a function/class/module node."""
            if node.child_count == 0:
                return None

            for child in node.children:
                if child.type == "block":
                    for stmt in child.children:
                        if stmt.type == "expression_statement":
                            for expr_child in stmt.children:
                                if expr_child.type == "string":
                                    doc = extract_node_content(expr_child)
                                    return strip_docstring_quotes(doc)
                elif child.type == "expression_statement":
                    for expr_child in child.children:
                        if expr_child.type == "string":
                            doc = extract_node_content(expr_child)
                            return strip_docstring_quotes(doc)
            return None

        # Level 1: Extract module-level constructs
        module_level_chunks = self._extract_module_level(
            file_path, root_node, extract_node_content
        )
        chunks.extend(module_level_chunks)

        # Level 2: Extract classes
        class_chunks = self._extract_classes(
            file_path, content, root_node, extract_node_content, extract_docstring
        )
        chunks.extend(class_chunks)

        # Level 3: Extract standalone functions (not in classes)
        function_chunks = self._extract_standalone_functions(
            file_path, root_node, extract_node_content, extract_docstring
        )
        chunks.extend(function_chunks)

        # If no chunks found, create one for the whole file
        if not chunks:
            metadata = CodeChunkMetadata(
                language="python",
                file_path=str(file_path),
                start_line=1,
                end_line=len(lines),
                imports=[]
            )
            chunks.append(ParsedChunk(content=content, metadata=metadata))

        return chunks

    def _extract_module_level(
        self,
        file_path: Path,
        root_node: any,
        extract_node_content: callable
    ) -> list[ParsedChunk]:
        """Extract module-level constructs (imports, constants, type defs).

        Groups:
        1. Module docstring + imports (stdlib, third-party, local)
        2. Constants and configuration
        3. Type definitions (Enums, TypedDicts, Protocols)

        Returns:
            List of module-level chunks
        """
        chunks = []

        # Collect imports, constants, and type definitions
        imports = []
        constants = []
        type_definitions = []
        module_docstring = None

        for child in root_node.children:
            # Module docstring (first string in file)
            if child.type == "expression_statement" and module_docstring is None:
                for expr_child in child.children:
                    if expr_child.type == "string":
                        module_docstring = child

            # Imports
            elif child.type in ("import_statement", "import_from_statement", "future_import_statement"):
                imports.append(child)

            # Constants (module-level assignments with UPPERCASE names)
            elif child.type == "expression_statement":
                for expr_child in child.children:
                    if expr_child.type == "assignment":
                        left = expr_child.child_by_field_name("left")
                        if left and left.type == "identifier":
                            var_name = extract_node_content(left)
                            if var_name.isupper():
                                constants.append(child)
                                break

            # Type definitions (Enum, TypedDict, Protocol classes)
            elif child.type == "class_definition":
                superclasses = child.child_by_field_name("superclasses")
                if superclasses:
                    superclass_text = extract_node_content(superclasses)
                    if any(t in superclass_text for t in ["Enum", "TypedDict", "Protocol", "NamedTuple"]):
                        type_definitions.append(child)

        # Create chunk for imports (if any)
        if imports or module_docstring:
            import_nodes = []
            if module_docstring:
                import_nodes.append(module_docstring)
            import_nodes.extend(imports)

            if import_nodes:
                first_node = import_nodes[0]
                last_node = import_nodes[-1]

                chunk_content = extract_node_content(first_node)
                for node in import_nodes[1:]:
                    chunk_content += "\n" + extract_node_content(node)

                metadata = CodeChunkMetadata(
                    language="python",
                    file_path=str(file_path),
                    start_line=first_node.start_point[0] + 1,
                    end_line=last_node.end_point[0] + 1,
                    imports=[extract_node_content(imp) for imp in imports],
                    docstring=extract_node_content(module_docstring) if module_docstring else None
                )
                chunks.append(ParsedChunk(content=chunk_content, metadata=metadata))

        # Create chunk for constants (if any)
        if constants:
            first_node = constants[0]
            last_node = constants[-1]

            chunk_content = extract_node_content(first_node)
            for node in constants[1:]:
                chunk_content += "\n" + extract_node_content(node)

            metadata = CodeChunkMetadata(
                language="python",
                file_path=str(file_path),
                start_line=first_node.start_point[0] + 1,
                end_line=last_node.end_point[0] + 1,
                imports=[]
            )
            chunks.append(ParsedChunk(content=chunk_content, metadata=metadata))

        # Create chunks for type definitions (each separate)
        for type_def in type_definitions:
            class_name_node = type_def.child_by_field_name("name")
            class_name = extract_node_content(class_name_node) if class_name_node else "unknown"

            metadata = CodeChunkMetadata(
                language="python",
                file_path=str(file_path),
                start_line=type_def.start_point[0] + 1,
                end_line=type_def.end_point[0] + 1,
                class_name=class_name,
                imports=[]
            )
            chunks.append(ParsedChunk(
                content=extract_node_content(type_def),
                metadata=metadata
            ))

        return chunks

    def _extract_classes(
        self,
        file_path: Path,
        content: str,
        root_node: any,
        extract_node_content: callable,
        extract_docstring: callable
    ) -> list[ParsedChunk]:
        """Extract class definitions (excluding type definitions).

        Strategy:
        - Small classes (<1500 non-whitespace chars): Keep whole
        - Large classes: Split into class header + individual methods

        Returns:
            List of class chunks
        """
        chunks = []

        def count_non_whitespace(text: str) -> int:
            return len(''.join(text.split()))

        def is_type_definition(node: any) -> bool:
            """Check if class is a type definition (Enum, TypedDict, etc)."""
            superclasses = node.child_by_field_name("superclasses")
            if superclasses:
                superclass_text = extract_node_content(superclasses)
                return any(t in superclass_text for t in ["Enum", "TypedDict", "Protocol", "NamedTuple"])
            return False

        for child in root_node.children:
            if child.type == "class_definition" and not is_type_definition(child):
                class_content = extract_node_content(child)
                class_size = count_non_whitespace(class_content)

                class_name_node = child.child_by_field_name("name")
                class_name = extract_node_content(class_name_node) if class_name_node else "unknown"

                # Small class: keep whole
                if class_size < 1500:
                    metadata = CodeChunkMetadata(
                        language="python",
                        file_path=str(file_path),
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        class_name=class_name,
                        docstring=extract_docstring(child),
                        imports=[]
                    )
                    chunks.append(ParsedChunk(content=class_content, metadata=metadata))
                else:
                    # Large class: extract methods separately
                    # (Methods will be extracted in function extraction phase)
                    # Just create a chunk for class definition + __init__
                    class_body = child.child_by_field_name("body")
                    if class_body:
                        init_method = None
                        for stmt in class_body.children:
                            if stmt.type == "function_definition":
                                func_name_node = stmt.child_by_field_name("name")
                                if func_name_node:
                                    func_name = extract_node_content(func_name_node)
                                    if func_name == "__init__":
                                        init_method = stmt
                                        break

                        if init_method:
                            # Create chunk with class header + __init__
                            start_byte = child.start_byte
                            end_byte = init_method.end_byte
                            chunk_content = content.encode("utf-8")[start_byte:end_byte].decode("utf-8")

                            metadata = CodeChunkMetadata(
                                language="python",
                                file_path=str(file_path),
                                start_line=child.start_point[0] + 1,
                                end_line=init_method.end_point[0] + 1,
                                class_name=class_name,
                                function_name="__init__",
                                docstring=extract_docstring(child),
                                imports=[]
                            )
                            chunks.append(ParsedChunk(content=chunk_content, metadata=metadata))

        return chunks

    def _extract_standalone_functions(
        self,
        file_path: Path,
        root_node: any,
        extract_node_content: callable,
        extract_docstring: callable
    ) -> list[ParsedChunk]:
        """Extract standalone functions (not class methods).

        Returns:
            List of function chunks
        """
        chunks = []

        for child in root_node.children:
            # Check for decorated functions
            if child.type == "decorated_definition":
                definition = None
                for dec_child in child.children:
                    if dec_child.type == "function_definition":
                        definition = dec_child
                        break

                if definition:
                    func_name_node = definition.child_by_field_name("name")
                    func_name = extract_node_content(func_name_node) if func_name_node else "unknown"

                    # Use the entire decorated_definition node (includes decorators)
                    metadata = CodeChunkMetadata(
                        language="python",
                        file_path=str(file_path),
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        function_name=func_name,
                        docstring=extract_docstring(definition),
                        imports=[]
                    )
                    chunks.append(ParsedChunk(
                        content=extract_node_content(child),
                        metadata=metadata
                    ))

            # Standalone functions (no decorators)
            elif child.type == "function_definition":
                func_name_node = child.child_by_field_name("name")
                func_name = extract_node_content(func_name_node) if func_name_node else "unknown"

                metadata = CodeChunkMetadata(
                    language="python",
                    file_path=str(file_path),
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    function_name=func_name,
                    docstring=extract_docstring(child),
                    imports=[]
                )
                chunks.append(ParsedChunk(
                    content=extract_node_content(child),
                    metadata=metadata
                ))

        return chunks
