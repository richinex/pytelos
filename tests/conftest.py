"""Pytest configuration and shared fixtures."""
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def api_keys():
    """Return API keys from environment."""
    return {
        "openai": os.getenv("OPENAI_API_KEY"),
        "deepseek": os.getenv("DEEPSEEK_API_KEY")
    }


@pytest.fixture(scope="session")
def postgres_config():
    """Return PostgreSQL configuration."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5433")),
        "database": os.getenv("POSTGRES_DB", "pytelos"),
        "user": os.getenv("POSTGRES_USER", "pytelos"),
        "password": os.getenv("POSTGRES_PASSWORD", "pytelos_dev")
    }


@pytest.fixture
def sample_python_code():
    """Return sample Python code for testing."""
    return '''
def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract two numbers."""
    return a - b

class Calculator:
    """A simple calculator."""

    def multiply(self, a, b):
        """Multiply two numbers."""
        return a * b

    def divide(self, a, b):
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
'''


@pytest.fixture
def sample_python_file(tmp_path, sample_python_code):
    """Create a temporary Python file with sample code."""
    test_file = tmp_path / "test_code.py"
    test_file.write_text(sample_python_code)
    return test_file
