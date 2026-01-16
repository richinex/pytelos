"""PostgreSQL schema definitions."""

from typing import Final

# Table creation SQL
CREATE_CHUNKS_TABLE: Final[str] = """
CREATE TABLE IF NOT EXISTS code_chunks (
    id UUID PRIMARY KEY,
    file_path TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    start_line INTEGER NOT NULL CHECK (start_line > 0),
    end_line INTEGER NOT NULL CHECK (end_line >= start_line),
    language TEXT NOT NULL,
    metadata JSONB DEFAULT '{{}}'::jsonb,
    embedding vector({dimension}),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

# Indexes
CREATE_FILE_PATH_INDEX: Final[str] = """
CREATE INDEX IF NOT EXISTS idx_chunks_file_path
ON code_chunks (file_path);
"""

CREATE_LANGUAGE_INDEX: Final[str] = """
CREATE INDEX IF NOT EXISTS idx_chunks_language
ON code_chunks (language);
"""

CREATE_CREATED_AT_INDEX: Final[str] = """
CREATE INDEX IF NOT EXISTS idx_chunks_created_at
ON code_chunks (created_at DESC);
"""

CREATE_METADATA_GIN_INDEX: Final[str] = """
CREATE INDEX IF NOT EXISTS idx_chunks_metadata
ON code_chunks USING GIN (metadata);
"""

# Vector index (HNSW)
CREATE_HNSW_INDEX: Final[str] = """
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON code_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = {m}, ef_construction = {ef_construction});
"""

# Vector index (IVFFlat)
CREATE_IVFFLAT_INDEX: Final[str] = """
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat
ON code_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = {lists});
"""

# BM25 index (pg_textsearch)
CREATE_BM25_INDEX: Final[str] = """
CREATE INDEX IF NOT EXISTS idx_chunks_bm25
ON code_chunks
USING bm25 (chunk_text)
WITH (text_config = '{text_config}', k1 = {k1}, b = {b});
"""

# Update trigger for updated_at
CREATE_UPDATED_AT_TRIGGER: Final[str] = """
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_chunks_updated_at ON code_chunks;

CREATE TRIGGER trigger_update_chunks_updated_at
BEFORE UPDATE ON code_chunks
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();
"""

# Statistics view
CREATE_STATS_VIEW: Final[str] = """
CREATE OR REPLACE VIEW code_stats AS
SELECT
    COUNT(*) as total_chunks,
    COUNT(DISTINCT file_path) as total_files,
    jsonb_object_agg(language, count) as languages,
    SUM(pg_column_size(chunk_text)) as total_size_bytes,
    MAX(created_at) as last_indexed
FROM (
    SELECT
        language,
        COUNT(*) as count,
        created_at,
        chunk_text,
        file_path
    FROM code_chunks
    GROUP BY language, created_at, chunk_text, file_path
) subquery;
"""

# Enable extensions
ENABLE_VECTOR_EXTENSION: Final[str] = """
CREATE EXTENSION IF NOT EXISTS vector;
"""

ENABLE_PG_TEXTSEARCH_EXTENSION: Final[str] = """
CREATE EXTENSION IF NOT EXISTS pg_textsearch;
"""

# Cleanup
DROP_CHUNKS_TABLE: Final[str] = """
DROP TABLE IF EXISTS code_chunks CASCADE;
"""

DROP_STATS_VIEW: Final[str] = """
DROP VIEW IF EXISTS code_stats;
"""
