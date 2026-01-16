"""Unit tests for the embedding module."""
import pytest
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from pytelos.embedding import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider,
)


@pytest.fixture
async def mock_openai_provider():
    """Create a mock OpenAI provider for testing.

    This is an async fixture that doesn't actually call the API.
    """
    # For real tests, you'd use a mock here
    # For now, we'll skip tests that need real API keys
    pytest.skip("Requires OpenAI API key")


class TestEmbeddingProviderInterface:
    """Tests for the abstract EmbeddingProvider interface."""

    def test_provider_is_abstract(self):
        """Test that EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore


class TestOpenAIEmbeddingProvider:
    """Tests for OpenAIEmbeddingProvider."""

    def test_dimension_property(self):
        """Test that dimension property returns correct values."""
        provider_small = OpenAIEmbeddingProvider(
            api_key="fake-key",
            model="text-embedding-3-small"
        )
        assert provider_small.dimension == 1536

        provider_large = OpenAIEmbeddingProvider(
            api_key="fake-key",
            model="text-embedding-3-large"
        )
        assert provider_large.dimension == 3072

    def test_unknown_model_raises_error(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            OpenAIEmbeddingProvider(
                api_key="fake-key",
                model="unknown-model"
            )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embed_text_real_api(self, api_keys):
        """Integration test: Embed text with real API."""
        if not api_keys["openai"]:
            pytest.skip("OPENAI_API_KEY not set")

        provider = OpenAIEmbeddingProvider(
            api_key=api_keys["openai"],
            model="text-embedding-3-small"
        )

        try:
            text = "Hello, world!"
            embedding = await provider.embed_text(text)

            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype == np.float32
            assert embedding.shape[0] == 1536
            assert not np.all(embedding == 0)  # Should not be all zeros
        finally:
            await provider.close()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_embed_batch_real_api(self, api_keys):
        """Integration test: Embed batch with real API."""
        if not api_keys["openai"]:
            pytest.skip("OPENAI_API_KEY not set")

        provider = OpenAIEmbeddingProvider(
            api_key=api_keys["openai"],
            model="text-embedding-3-small"
        )

        try:
            texts = ["First text", "Second text", "Third text"]
            embeddings = await provider.embed_batch(texts)

            assert len(embeddings) == 3
            for embedding in embeddings:
                assert isinstance(embedding, np.ndarray)
                assert embedding.dtype == np.float32
                assert embedding.shape[0] == 1536
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self):
        """Test that embedding empty list returns empty list."""
        provider = OpenAIEmbeddingProvider(
            api_key="fake-key",
            model="text-embedding-3-small"
        )

        # This won't actually call the API
        result = await provider.embed_batch([])
        assert result == []


class TestEmbeddingFactory:
    """Tests for embedding factory function."""

    def test_create_openai_provider(self):
        """Test creating OpenAI provider via factory."""
        provider = create_embedding_provider(
            "openai",
            api_key="test-key",
            model="text-embedding-3-small"
        )

        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.dimension == 1536

    def test_create_provider_unknown_type(self):
        """Test that unknown provider type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_embedding_provider("unknown", api_key="test-key")

    def test_create_provider_missing_api_key(self):
        """Test that missing API key raises TypeError."""
        with pytest.raises(TypeError, match="requires 'api_key'"):
            create_embedding_provider("openai")

    @given(st.text(min_size=1))
    def test_factory_with_random_provider_names(self, provider_name: str):
        """Property test: Factory should only accept known providers."""
        if provider_name.lower() == "openai":
            # This should work (with fake key)
            provider = create_embedding_provider(provider_name, api_key="fake")
            assert isinstance(provider, OpenAIEmbeddingProvider)
        else:
            # All other names should fail
            with pytest.raises(ValueError):
                create_embedding_provider(provider_name, api_key="fake")
