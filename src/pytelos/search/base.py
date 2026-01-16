from abc import ABC, abstractmethod

from .models import SearchQuery, SearchResponse


class SearchEngine(ABC):
    """Abstract base class for search engines.

    This module hides the design decision of how to orchestrate search operations.

    Hidden design decisions:
    - Query preprocessing and expansion logic
    - Result ranking and re-ranking algorithms
    - Caching strategies
    - Performance optimization techniques
    """

    @abstractmethod
    async def search(self, query: SearchQuery) -> SearchResponse:
        """Execute a search query.

        Args:
            query: The search query with parameters

        Returns:
            SearchResponse with results and metadata

        Raises:
            ValueError: If query is invalid
        """
        pass

    @abstractmethod
    async def explain_query(self, query_text: str) -> str:
        """Explain how a query will be processed.

        Uses LLM to provide insights about the query.

        Args:
            query_text: The query text to explain

        Returns:
            Explanation of query processing

        Raises:
            Exception: If explanation fails
        """
        pass
