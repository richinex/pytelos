import time
from typing import Any

from ..embedding import EmbeddingProvider
from ..llm import ChatMessage, LLMProvider
from ..storage import StorageBackend
from ..storage.models import SearchResult as StorageSearchResult
from .base import SearchEngine
from .models import RerankStrategy, SearchMode, SearchQuery, SearchResponse


class DefaultSearchEngine(SearchEngine):
    """Default search engine implementation.

    Hidden design decisions:
    - Query expansion using LLM
    - Hybrid search with configurable weights
    - Reciprocal Rank Fusion for combining results
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedder: EmbeddingProvider,
        llm: LLMProvider | None = None,
        alpha: float = 0.5
    ):
        """Initialize the search engine.

        Args:
            storage: Storage backend for retrieving results
            embedder: Embedding provider for query vectorization
            llm: Optional LLM provider for query expansion and re-ranking
            alpha: Weight for hybrid search (0=keyword only, 1=vector only, 0.5=balanced)
        """
        self._storage = storage
        self._embedder = embedder
        self._llm = llm
        self._alpha = alpha

    async def search(self, query: SearchQuery) -> SearchResponse:
        """Execute a search query.

        Args:
            query: The search query with parameters

        Returns:
            SearchResponse with results and metadata
        """
        start_time = time.time()

        # Expand query if LLM is available
        expanded_query = None
        search_text = query.query
        if self._llm and query.mode in (SearchMode.VECTOR, SearchMode.HYBRID):
            expanded_query = await self._expand_query(query.query)
            search_text = expanded_query

        # Execute search based on mode
        if query.mode == SearchMode.VECTOR:
            results = await self._vector_search(search_text, query.limit)
        elif query.mode == SearchMode.KEYWORD:
            results = await self._keyword_search(search_text, query.limit)
        else:  # HYBRID
            results = await self._hybrid_search(search_text, query.limit)

        # Apply filters if provided
        if query.filters:
            results = self._apply_filters(results, query.filters)

        # Re-rank if requested
        if query.rerank != RerankStrategy.NONE:
            results = await self._rerank_results(results, query)

        # Ensure we don't exceed limit
        results = results[:query.limit]

        processing_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            processing_time_ms=processing_time_ms,
            expanded_query=expanded_query
        )

    async def explain_query(self, query_text: str) -> str:
        """Explain how a query will be processed.

        Args:
            query_text: The query text to explain

        Returns:
            Explanation of query processing
        """
        if not self._llm:
            return (
                f"Query: '{query_text}'\n"
                f"This query will be processed as-is (no LLM available for expansion).\n"
                f"Vector search will use embeddings, keyword search will use BM25."
            )

        messages = [
            ChatMessage(
                role="system",
                content="You are a code search assistant. Explain how to search for code based on the user's query."
            ),
            ChatMessage(
                role="user",
                content=f"Explain what kind of code this query is looking for: '{query_text}'"
            )
        ]

        response = await self._llm.chat_completion(messages, temperature=0.3)
        return response.content

    async def _expand_query(self, query: str) -> str:
        """Expand query using LLM.

        Args:
            query: Original query text

        Returns:
            Expanded query text
        """
        if not self._llm:
            return query

        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a code search assistant. Expand the user's query "
                    "to include relevant technical terms and synonyms. "
                    "Keep it concise and focused on code semantics."
                )
            ),
            ChatMessage(
                role="user",
                content=f"Expand this code search query: '{query}'"
            )
        ]

        response = await self._llm.chat_completion(messages, temperature=0.5, max_tokens=100)
        return response.content.strip()

    async def _vector_search(
        self,
        query: str,
        limit: int
    ) -> list[StorageSearchResult]:
        """Perform vector search.

        Args:
            query: Query text
            limit: Maximum results

        Returns:
            Search results
        """
        # Generate query embedding
        query_embedding = await self._embedder.embed_text(query)

        # Search storage
        results = await self._storage.vector_search(
            query_embedding=query_embedding,
            limit=limit
        )

        return results

    async def _keyword_search(
        self,
        query: str,
        limit: int
    ) -> list[StorageSearchResult]:
        """Perform keyword search using BM25.

        Args:
            query: Query text
            limit: Maximum results

        Returns:
            Search results
        """
        results = await self._storage.bm25_search(
            query=query,
            limit=limit
        )

        return results

    async def _hybrid_search(
        self,
        query: str,
        limit: int
    ) -> list[StorageSearchResult]:
        """Perform hybrid search combining vector and keyword.

        Uses the storage backend's hybrid search with Reciprocal Rank Fusion.

        Args:
            query: Query text
            limit: Maximum results

        Returns:
            Combined search results
        """
        # Generate query embedding
        query_embedding = await self._embedder.embed_text(query)

        # Use storage backend's hybrid search with RRF
        results = await self._storage.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            limit=limit,
            alpha=self._alpha
        )

        return results

    def _apply_filters(
        self,
        results: list[StorageSearchResult],
        filters: dict[str, Any]
    ) -> list[StorageSearchResult]:
        """Apply filters to search results.

        Args:
            results: Search results
            filters: Filters to apply

        Returns:
            Filtered results
        """
        filtered = []

        for result in results:
            matches = True

            # Check each filter
            if "language" in filters and result.chunk.language != filters["language"]:
                matches = False

            if "file_path" in filters and filters["file_path"] not in result.chunk.file_path:
                matches = False

            # Add more filter types as needed

            if matches:
                filtered.append(result)

        return filtered

    async def _rerank_results(
        self,
        results: list[StorageSearchResult],
        query: SearchQuery
    ) -> list[StorageSearchResult]:
        """Re-rank search results.

        Args:
            results: Initial search results
            query: Original query

        Returns:
            Re-ranked results
        """
        if query.rerank == RerankStrategy.RECIPROCAL_RANK_FUSION:
            # RRF is already applied in hybrid search at the database level
            return results

        return results
