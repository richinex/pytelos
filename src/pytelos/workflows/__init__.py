from .indexing import IndexingWorkflow

__all__ = ["IndexingWorkflow", "PyergonIndexingWorkflow"]


def __getattr__(name):
    """Lazy import for optional dependencies."""
    if name == "PyergonIndexingWorkflow":
        try:
            from .pyergon_indexing import PyergonIndexingWorkflow
            return PyergonIndexingWorkflow
        except ImportError as e:
            raise ImportError(
                "PyergonIndexingWorkflow requires 'pyergon' package. "
                "Install it from pyergon directory: cd pyergon && uv pip install -e ."
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
