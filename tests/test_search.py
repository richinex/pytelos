"""Unit tests for the search module."""
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pytelos.search import (
    RerankStrategy,
    SearchEngine,
    SearchMode,
    SearchQuery,
    create_search_engine,
)


class TestSearchEngine:
    """Tests for SearchEngine interface."""

    def test_search_engine_is_abstract(self):
        """Test that SearchEngine cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SearchEngine()  # type: ignore


class TestSearchQuery:
    """Tests for SearchQuery model."""

    def test_create_basic_query(self):
        """Test creating a basic search query."""
        query = SearchQuery(query="find authentication code")

        assert query.query == "find authentication code"
        assert query.mode == SearchMode.HYBRID
        assert query.limit == 10
        assert query.filters is None
        assert query.rerank == RerankStrategy.NONE

    def test_create_query_with_options(self):
        """Test creating a query with all options."""
        query = SearchQuery(
            query="database connection",
            mode=SearchMode.VECTOR,
            limit=5,
            filters={"language": "python"},
            rerank=RerankStrategy.RECIPROCAL_RANK_FUSION
        )

        assert query.query == "database connection"
        assert query.mode == SearchMode.VECTOR
        assert query.limit == 5
        assert query.filters == {"language": "python"}
        assert query.rerank == RerankStrategy.RECIPROCAL_RANK_FUSION

    @given(st.integers(min_value=1, max_value=100))
    def test_query_limit_within_bounds(self, limit: int):
        """Property test: Query limit should accept 1-100."""
        query = SearchQuery(query="test", limit=limit)
        assert 1 <= query.limit <= 100

    def test_query_limit_below_minimum_fails(self):
        """Test that limit below 1 fails validation."""
        with pytest.raises(ValueError):
            SearchQuery(query="test", limit=0)

    def test_query_limit_above_maximum_fails(self):
        """Test that limit above 100 fails validation."""
        with pytest.raises(ValueError):
            SearchQuery(query="test", limit=101)


class TestSearchMode:
    """Tests for SearchMode enum."""

    def test_search_modes_exist(self):
        """Test that all search modes are defined."""
        assert SearchMode.VECTOR == "vector"
        assert SearchMode.KEYWORD == "keyword"
        assert SearchMode.HYBRID == "hybrid"

    @given(st.text())
    def test_search_mode_validation(self, mode_str: str):
        """Property test: Only valid modes should be accepted."""
        if mode_str in ("vector", "keyword", "hybrid"):
            mode = SearchMode(mode_str)
            assert mode in SearchMode
        else:
            with pytest.raises(ValueError):
                SearchMode(mode_str)


class TestRerankStrategy:
    """Tests for RerankStrategy enum."""

    def test_rerank_strategies_exist(self):
        """Test that all rerank strategies are defined."""
        assert RerankStrategy.NONE == "none"
        assert RerankStrategy.RECIPROCAL_RANK_FUSION == "rrf"


class TestSearchFactory:
    """Tests for search engine factory."""

    @pytest.mark.asyncio
    async def test_create_search_engine(self):
        """Test creating a search engine via factory."""
        from unittest.mock import Mock

        # Create mocks
        storage_mock = Mock()
        embedder_mock = Mock()

        # Create search engine
        engine = create_search_engine(
            storage=storage_mock,
            embedder=embedder_mock
        )

        # Verify it's the right type
        from pytelos.search.engine import DefaultSearchEngine
        assert isinstance(engine, DefaultSearchEngine)

    @pytest.mark.asyncio
    async def test_create_search_engine_with_llm(self):
        """Test creating a search engine with LLM."""
        from unittest.mock import Mock

        storage_mock = Mock()
        embedder_mock = Mock()
        llm_mock = Mock()

        engine = create_search_engine(
            storage=storage_mock,
            embedder=embedder_mock,
            llm=llm_mock
        )

        from pytelos.search.engine import DefaultSearchEngine
        assert isinstance(engine, DefaultSearchEngine)

    @pytest.mark.asyncio
    async def test_create_search_engine_with_custom_alpha(self):
        """Test creating a search engine with custom alpha."""
        from unittest.mock import Mock

        storage_mock = Mock()
        embedder_mock = Mock()

        engine = create_search_engine(
            storage=storage_mock,
            embedder=embedder_mock,
            alpha=0.7
        )

        assert engine._alpha == 0.7
