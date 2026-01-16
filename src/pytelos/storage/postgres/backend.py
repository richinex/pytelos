"""PostgreSQL storage backend implementation."""

import json
from typing import Any
from uuid import UUID

import asyncpg
import numpy as np
from numpy.typing import NDArray
from pgvector.asyncpg import register_vector

from ..base import StorageBackend
from ..models import CodeChunk, IndexConfig, SearchResult, StorageStats
from . import schema


class PostgresBackend(StorageBackend):
    """
    PostgreSQL storage backend with pgvector and pg_textsearch.

    Hides all Postgres-specific details:
    - Connection pooling
    - SQL query construction
    - Vector and BM25 index management
    - Transaction handling
    """

    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
        embedding_dimension: int = 1536
    ):
        """
        Initialize Postgres backend.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_pool_size: Minimum connection pool size
            max_pool_size: Maximum connection pool size
            embedding_dimension: Expected embedding vector dimension
        """
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._embedding_dimension = embedding_dimension
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Establish database connection pool."""
        if self._pool is not None:
            return

        async def init(conn):
            """Initialize connection with pgvector support."""
            await register_vector(conn)

        try:
            self._pool = await asyncpg.create_pool(
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                password=self._password,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
                command_timeout=60.0,
                init=init
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}") from e

    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def initialize_schema(self, config: IndexConfig) -> None:
        """
        Initialize database schema with extensions and indexes.

        Args:
            config: Index configuration
        """
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        async with self._pool.acquire() as conn:
            # Enable extensions
            await conn.execute(schema.ENABLE_VECTOR_EXTENSION)
            await conn.execute(schema.ENABLE_PG_TEXTSEARCH_EXTENSION)

            # Create main table
            await conn.execute(
                schema.CREATE_CHUNKS_TABLE.format(
                    dimension=self._embedding_dimension
                )
            )

            # Create standard indexes
            await conn.execute(schema.CREATE_FILE_PATH_INDEX)
            await conn.execute(schema.CREATE_LANGUAGE_INDEX)
            await conn.execute(schema.CREATE_CREATED_AT_INDEX)
            await conn.execute(schema.CREATE_METADATA_GIN_INDEX)

            # Create vector index based on config
            if config.vector_index_type == "hnsw":
                await conn.execute(
                    schema.CREATE_HNSW_INDEX.format(
                        m=config.hnsw_m,
                        ef_construction=config.hnsw_ef_construction
                    )
                )
            else:  # ivfflat
                await conn.execute(
                    schema.CREATE_IVFFLAT_INDEX.format(
                        lists=config.ivfflat_lists
                    )
                )

            # Create BM25 index
            await conn.execute(
                schema.CREATE_BM25_INDEX.format(
                    text_config=config.text_config,
                    k1=config.bm25_k1,
                    b=config.bm25_b
                )
            )

            # Create triggers and views
            await conn.execute(schema.CREATE_UPDATED_AT_TRIGGER)
            await conn.execute(schema.CREATE_STATS_VIEW)

    async def store_chunk(
        self,
        chunk: CodeChunk,
        embedding: NDArray[np.float32]
    ) -> str:
        """Store single code chunk with embedding."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        if embedding.shape[0] != self._embedding_dimension:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} doesn't match "
                f"expected {self._embedding_dimension}"
            )

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO code_chunks (
                    id, file_path, chunk_text, start_line, end_line,
                    language, metadata, embedding
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (id) DO UPDATE SET
                    file_path = EXCLUDED.file_path,
                    chunk_text = EXCLUDED.chunk_text,
                    start_line = EXCLUDED.start_line,
                    end_line = EXCLUDED.end_line,
                    language = EXCLUDED.language,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding
                """,
                UUID(chunk.id),
                chunk.file_path,
                chunk.chunk_text,
                chunk.start_line,
                chunk.end_line,
                chunk.language,
                json.dumps(chunk.metadata),
                embedding.tolist()
            )

        return chunk.id

    async def store_chunks_batch(
        self,
        chunks: list[CodeChunk],
        embeddings: list[NDArray[np.float32]]
    ) -> list[str]:
        """Store multiple chunks efficiently in a batch."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        if self._pool is None:
            raise RuntimeError("Not connected to database")

        # Validate all embeddings
        for i, emb in enumerate(embeddings):
            if emb.shape[0] != self._embedding_dimension:
                raise ValueError(
                    f"Embedding {i} dimension {emb.shape[0]} doesn't match "
                    f"expected {self._embedding_dimension}"
                )

        async with self._pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                """
                    INSERT INTO code_chunks (
                        id, file_path, chunk_text, start_line, end_line,
                        language, metadata, embedding
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (id) DO UPDATE SET
                        file_path = EXCLUDED.file_path,
                        chunk_text = EXCLUDED.chunk_text,
                        start_line = EXCLUDED.start_line,
                        end_line = EXCLUDED.end_line,
                        language = EXCLUDED.language,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding
                    """,
                [
                    (
                        UUID(chunk.id),
                        chunk.file_path,
                        chunk.chunk_text,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.language,
                        json.dumps(chunk.metadata),
                        emb.tolist()
                    )
                    for chunk, emb in zip(chunks, embeddings, strict=True)
                ]
            )

        return [chunk.id for chunk in chunks]

    async def get_chunk(self, chunk_id: str) -> CodeChunk | None:
        """Retrieve chunk by ID."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, file_path, chunk_text, start_line, end_line,
                       language, metadata, created_at
                FROM code_chunks
                WHERE id = $1
                """,
                UUID(chunk_id)
            )

        if row is None:
            return None

        return CodeChunk(
            id=str(row["id"]),
            file_path=row["file_path"],
            chunk_text=row["chunk_text"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            language=row["language"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"]
        )

    async def delete_chunk(self, chunk_id: str) -> bool:
        """Delete chunk by ID."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM code_chunks WHERE id = $1",
                UUID(chunk_id)
            )

        return result != "DELETE 0"

    async def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks from a specific file."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM code_chunks WHERE file_path = $1",
                file_path
            )

        # Extract count from result string like "DELETE 5"
        return int(result.split()[-1]) if result else 0

    async def vector_search(
        self,
        query_embedding: NDArray[np.float32],
        limit: int = 10,
        file_filter: str | None = None,
        language_filter: str | None = None
    ) -> list[SearchResult]:
        """Perform vector similarity search using cosine distance."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        if query_embedding.shape[0] != self._embedding_dimension:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[0]} "
                f"doesn't match expected {self._embedding_dimension}"
            )

        # Build WHERE clause dynamically
        where_conditions = []
        params: list[Any] = [query_embedding.tolist(), limit]
        param_idx = 3

        if file_filter:
            where_conditions.append(f"file_path = ${param_idx}")
            params.append(file_filter)
            param_idx += 1

        if language_filter:
            where_conditions.append(f"language = ${param_idx}")
            params.append(language_filter)
            param_idx += 1

        where_clause = (
            "WHERE " + " AND ".join(where_conditions)
            if where_conditions else ""
        )

        query = f"""
            SELECT
                id, file_path, chunk_text, start_line, end_line,
                language, metadata, created_at,
                embedding <=> $1 as distance
            FROM code_chunks
            {where_clause}
            ORDER BY embedding <=> $1
            LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results = []
        for rank, row in enumerate(rows, 1):
            chunk = CodeChunk(
                id=str(row["id"]),
                file_path=row["file_path"],
                chunk_text=row["chunk_text"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                language=row["language"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"]
            )
            # Convert distance to similarity score (1 - cosine distance)
            score = 1.0 - float(row["distance"])
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    search_type="vector"
                )
            )

        return results

    async def bm25_search(
        self,
        query: str,
        limit: int = 10,
        file_filter: str | None = None,
        language_filter: str | None = None
    ) -> list[SearchResult]:
        """Perform BM25 keyword search."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        # Build WHERE clause dynamically
        where_conditions = []
        params: list[Any] = [query, limit]
        param_idx = 3

        if file_filter:
            where_conditions.append(f"file_path = ${param_idx}")
            params.append(file_filter)
            param_idx += 1

        if language_filter:
            where_conditions.append(f"language = ${param_idx}")
            params.append(language_filter)
            param_idx += 1

        where_clause = (
            "AND " + " AND ".join(where_conditions)
            if where_conditions else ""
        )

        # Note: <@> returns negative BM25 scores, lower is better
        query_sql = f"""
            SELECT
                id, file_path, chunk_text, start_line, end_line,
                language, metadata, created_at,
                -(chunk_text <@> to_bm25query($1, 'idx_chunks_bm25')) as bm25_score
            FROM code_chunks
            WHERE chunk_text <@> to_bm25query($1, 'idx_chunks_bm25') < 0  -- Only return matches
            {where_clause}
            ORDER BY chunk_text <@> to_bm25query($1, 'idx_chunks_bm25')
            LIMIT $2
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query_sql, *params)

        results = []
        for rank, row in enumerate(rows, 1):
            chunk = CodeChunk(
                id=str(row["id"]),
                file_path=row["file_path"],
                chunk_text=row["chunk_text"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                language=row["language"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"]
            )
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(row["bm25_score"]),
                    rank=rank,
                    search_type="bm25"
                )
            )

        return results

    async def hybrid_search(
        self,
        query: str,
        query_embedding: NDArray[np.float32],
        limit: int = 10,
        alpha: float = 0.5,
        file_filter: str | None = None,
        language_filter: str | None = None
    ) -> list[SearchResult]:
        """
        Perform hybrid search using Reciprocal Rank Fusion (RRF).

        Implementation follows the approach from pg_search.txt documentation.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        if self._pool is None:
            raise RuntimeError("Not connected to database")

        if query_embedding.shape[0] != self._embedding_dimension:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[0]} "
                f"doesn't match expected {self._embedding_dimension}"
            )

        # Build WHERE clause
        where_conditions = []
        params: list[Any] = [query_embedding.tolist(), query, limit]
        param_idx = 4

        if file_filter:
            where_conditions.append(f"file_path = ${param_idx}")
            params.append(file_filter)
            param_idx += 1

        if language_filter:
            where_conditions.append(f"language = ${param_idx}")
            params.append(language_filter)
            param_idx += 1

        where_clause = (
            "WHERE " + " AND ".join(where_conditions)
            if where_conditions else ""
        )

        # Reciprocal Rank Fusion query
        # k=60 is the standard RRF constant
        query_sql = f"""
            WITH vector_results AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY embedding <=> $1) as rank
                FROM code_chunks
                {where_clause}
                LIMIT 20
            ),
            bm25_results AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (ORDER BY chunk_text <@> to_bm25query($2, 'idx_chunks_bm25')) as rank
                FROM code_chunks
                {where_clause}
                LIMIT 20
            ),
            fused AS (
                SELECT
                    COALESCE(v.id, b.id) as id,
                    ({alpha} * 1.0 / (60 + COALESCE(v.rank, 999)) +
                     {1.0 - alpha} * 1.0 / (60 + COALESCE(b.rank, 999))) as score
                FROM vector_results v
                FULL OUTER JOIN bm25_results b USING (id)
            )
            SELECT
                c.id, c.file_path, c.chunk_text, c.start_line, c.end_line,
                c.language, c.metadata, c.created_at,
                f.score
            FROM fused f
            JOIN code_chunks c ON f.id = c.id
            ORDER BY f.score DESC
            LIMIT $3
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query_sql, *params)

        results = []
        for rank, row in enumerate(rows, 1):
            chunk = CodeChunk(
                id=str(row["id"]),
                file_path=row["file_path"],
                chunk_text=row["chunk_text"],
                start_line=row["start_line"],
                end_line=row["end_line"],
                language=row["language"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"]
            )
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(row["score"]),
                    rank=rank,
                    search_type="hybrid"
                )
            )

        return results

    async def get_stats(self) -> StorageStats:
        """Get storage statistics from the stats view."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM code_stats")

        if row is None:
            return StorageStats(
                total_chunks=0,
                total_files=0,
                languages={},
                total_size_bytes=0,
                last_indexed=None
            )

        return StorageStats(
            total_chunks=row["total_chunks"],
            total_files=row["total_files"],
            languages=json.loads(row["languages"]) if row["languages"] else {},
            total_size_bytes=row["total_size_bytes"] or 0,
            last_indexed=row["last_indexed"]
        )

    async def health_check(self) -> bool:
        """Check if database connection is healthy."""
        if self._pool is None:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def drop_schema(self) -> None:
        """Drop all database schema (tables, indexes, views)."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        try:
            async with self._pool.acquire() as conn:
                # Drop views first
                await conn.execute(schema.DROP_STATS_VIEW)

                # Drop main table (CASCADE will drop indexes)
                await conn.execute(schema.DROP_CHUNKS_TABLE)

        except Exception as e:
            raise RuntimeError(f"Failed to drop schema: {e}") from e

    async def clear_all(self) -> int:
        """Delete all chunks from the database."""
        if self._pool is None:
            raise RuntimeError("Not connected to database")

        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute("DELETE FROM code_chunks")

            # Extract count from result string like "DELETE 129"
            return int(result.split()[-1]) if result else 0

        except Exception as e:
            raise RuntimeError(f"Failed to clear chunks: {e}") from e
