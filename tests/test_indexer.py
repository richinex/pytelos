"""Unit and property-based tests for the indexer module."""
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pytelos.indexer import (
    ChunkingStrategy,
    CodeChunkMetadata,
    ParsedChunk,
    PythonParser,
)


class TestPythonParser:
    """Tests for PythonParser."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return PythonParser()

    def test_supports_python_language(self, parser):
        """Test that parser recognizes Python language."""
        assert parser.supports_language("python")
        assert parser.supports_language("Python")
        assert parser.supports_language("PYTHON")
        assert not parser.supports_language("javascript")
        assert not parser.supports_language("rust")

    @given(st.text(min_size=1))
    def test_supports_language_with_random_input(self, language: str):
        """Property test: supports_language should only return True for python."""
        parser = PythonParser()
        result = parser.supports_language(language)
        assert isinstance(result, bool)
        if result:
            assert language.lower() == "python"

    def test_detect_language_python_files(self, parser, tmp_path):
        """Test language detection for Python files."""
        py_file = tmp_path / "test.py"
        py_file.write_text("print('hello')")

        pyi_file = tmp_path / "stub.pyi"
        pyi_file.write_text("def func() -> None: ...")

        assert parser.detect_language(py_file) == "python"
        assert parser.detect_language(pyi_file) == "python"

    def test_detect_language_non_python_files(self, parser, tmp_path):
        """Test language detection for non-Python files."""
        js_file = tmp_path / "test.js"
        js_file.write_text("console.log('hello');")

        assert parser.detect_language(js_file) is None

    def test_parse_simple_function(self, parser, tmp_path):
        """Test parsing a file with a simple function."""
        test_file = tmp_path / "simple.py"
        test_file.write_text("""
def hello_world():
    '''Say hello.'''
    print('Hello, World!')
""")

        chunks = parser.parse_file(test_file, strategy=ChunkingStrategy.BY_FUNCTION)

        assert len(chunks) == 1
        assert chunks[0].metadata.function_name == "hello_world"
        assert chunks[0].metadata.docstring == "Say hello."
        assert "Hello, World!" in chunks[0].content

    def test_parse_class_with_methods(self, parser, tmp_path):
        """Test parsing a file with a class and methods."""
        test_file = tmp_path / "class_test.py"
        test_file.write_text("""
class Calculator:
    '''A simple calculator.'''

    def add(self, a, b):
        '''Add two numbers.'''
        return a + b

    def subtract(self, a, b):
        return a - b
""")

        chunks = parser.parse_file(test_file, strategy=ChunkingStrategy.BY_FUNCTION)

        # Should get: class + method (class includes all methods)
        assert len(chunks) >= 1

        # Check class chunk
        class_chunk = chunks[0]
        assert class_chunk.metadata.class_name == "Calculator"
        assert class_chunk.metadata.docstring == "A simple calculator."

    def test_parse_file_not_found(self, parser):
        """Test that parsing non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("/nonexistent/file.py"))

    def test_parse_non_python_file(self, parser, tmp_path):
        """Test that parsing non-Python file raises error."""
        js_file = tmp_path / "test.js"
        js_file.write_text("console.log('test');")

        with pytest.raises(ValueError, match="Not a Python file"):
            parser.parse_file(js_file)

    def test_chunking_by_lines(self, parser, tmp_path):
        """Test chunking by lines strategy."""
        test_file = tmp_path / "lines_test.py"
        # Create content with known size
        content = "\n".join([f"# Line {i}" for i in range(50)])
        test_file.write_text(content)

        chunks = parser.parse_file(
            test_file,
            strategy=ChunkingStrategy.BY_LINES,
            chunk_size=200,
            overlap=50
        )

        # Should create multiple chunks
        assert len(chunks) > 1

        # Verify line ranges
        for chunk in chunks:
            assert chunk.metadata.start_line <= chunk.metadata.end_line

    def test_chunking_by_lines_various_sizes(self, parser, tmp_path):
        """Test chunking with various chunk sizes.

        This replaces a property test to avoid fixture incompatibility.
        """
        test_cases = [
            (100, 20),
            (500, 100),
            (1000, 200),
            (2000, 50),
        ]

        for chunk_size, overlap in test_cases:
            test_file = tmp_path / f"prop_test_{chunk_size}.py"
            content = "\n".join([f"x = {i}" for i in range(100)])
            test_file.write_text(content)

            chunks = parser.parse_file(
                test_file,
                strategy=ChunkingStrategy.BY_LINES,
                chunk_size=chunk_size,
                overlap=overlap
            )

            # Properties that should always hold:
            # 1. Should produce at least one chunk
            assert len(chunks) >= 1

            # 2. Chunks should be ordered by line number
            for i in range(len(chunks) - 1):
                assert chunks[i].metadata.start_line <= chunks[i + 1].metadata.start_line


class TestCodeChunkMetadata:
    """Tests for CodeChunkMetadata model."""

    def test_create_valid_metadata(self):
        """Test creating valid metadata."""
        metadata = CodeChunkMetadata(
            language="python",
            file_path="test.py",
            start_line=1,
            end_line=10,
            function_name="test_func",
            class_name=None,
            docstring="Test function",
            imports=[],
            complexity=None
        )

        assert metadata.language == "python"
        assert metadata.function_name == "test_func"
        assert metadata.start_line == 1
        assert metadata.end_line == 10

    @given(
        st.text(min_size=1),
        st.text(min_size=1),
        st.integers(min_value=1, max_value=10000),
        st.integers(min_value=1, max_value=10000)
    )
    def test_metadata_properties(
        self,
        language: str,
        file_path: str,
        line1: int,
        line2: int
    ):
        """Property test: metadata should accept valid inputs."""
        start = min(line1, line2)
        end = max(line1, line2)

        metadata = CodeChunkMetadata(
            language=language,
            file_path=file_path,
            start_line=start,
            end_line=end
        )

        assert metadata.start_line <= metadata.end_line
        assert metadata.language == language
        assert metadata.file_path == file_path


class TestParsedChunk:
    """Tests for ParsedChunk model."""

    def test_display_name_with_function(self):
        """Test display name for function chunk."""
        chunk = ParsedChunk(
            content="def test(): pass",
            metadata=CodeChunkMetadata(
                language="python",
                file_path="test.py",
                start_line=1,
                end_line=1,
                function_name="test"
            )
        )

        assert chunk.display_name == "test"

    def test_display_name_with_class_method(self):
        """Test display name for class method chunk."""
        chunk = ParsedChunk(
            content="def method(self): pass",
            metadata=CodeChunkMetadata(
                language="python",
                file_path="test.py",
                start_line=1,
                end_line=1,
                function_name="method",
                class_name="TestClass"
            )
        )

        assert chunk.display_name == "TestClass.method"

    def test_display_name_without_function(self):
        """Test display name for non-function chunk."""
        chunk = ParsedChunk(
            content="x = 1",
            metadata=CodeChunkMetadata(
                language="python",
                file_path="/path/to/test.py",
                start_line=5,
                end_line=5
            )
        )

        assert chunk.display_name == "test.py:5"
