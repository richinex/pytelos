"""Tools for the agent to interact with the document index."""


from ..embedding import EmbeddingProvider
from ..search import SearchEngine, SearchMode, SearchQuery
from ..storage import StorageBackend
from .data_structures import RetrievedChunk


class DocumentRetrievalTool:
    """Tool for retrieving relevant document chunks.

    Hidden design decisions:
    - Search engine configuration
    - Result formatting and filtering
    - Metadata extraction
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedder: EmbeddingProvider,
        search_engine: SearchEngine | None = None
    ):
        """Initialize the document retrieval tool.

        Args:
            storage: Storage backend instance
            embedder: Embedding provider instance
            search_engine: Optional pre-configured search engine
        """
        from ..search import create_search_engine

        self._storage = storage
        self._embedder = embedder
        self._search_engine = search_engine or create_search_engine(
            storage, embedder
        )

    async def retrieve(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        limit: int = 5,
        filters: dict[str, str] | None = None
    ) -> list[RetrievedChunk]:
        """Retrieve relevant document chunks.

        Args:
            query: Search query
            mode: Search mode (vector, keyword, hybrid)
            limit: Maximum number of results
            filters: Optional filters (e.g., language, file_path pattern)

        Returns:
            List of retrieved chunks with full content
        """
        search_query = SearchQuery(
            query=query,
            mode=mode,
            limit=limit,
            filters=filters
        )

        result = await self._search_engine.search(search_query)

        return [
            RetrievedChunk(
                content=res.chunk.chunk_text,
                source=res.chunk.file_path,
                lines=f"{res.chunk.start_line}-{res.chunk.end_line}",
                score=res.score,
                metadata=res.chunk.metadata
            )
            for res in result.results
        ]

    async def retrieve_with_context(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        limit: int = 5,
        filters: dict[str, str] | None = None
    ) -> tuple[list[RetrievedChunk], str]:
        """Retrieve chunks and format them as context for LLM.

        Args:
            query: Search query
            mode: Search mode
            limit: Maximum number of results
            filters: Optional filters

        Returns:
            Tuple of (retrieved chunks, formatted context string)
        """
        chunks = await self.retrieve(query, mode, limit, filters)

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Document {i}]\n"
                f"Source: {chunk.source} (lines {chunk.lines})\n"
                f"Content:\n{chunk.content}\n"
            )

        context = "\n---\n\n".join(context_parts)

        return chunks, context
