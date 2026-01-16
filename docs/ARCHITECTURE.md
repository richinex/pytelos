# Pytelos Architecture Document

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Module Structure](#3-module-structure)
4. [Core Data Flows](#4-core-data-flows)
5. [Storage Layer](#5-storage-layer)
6. [Embedding Layer](#6-embedding-layer)
7. [Search Engine](#7-search-engine)
8. [Indexing Pipeline](#8-indexing-pipeline)
9. [LLM Integration](#9-llm-integration)
10. [Reasoning Agent](#10-reasoning-agent)
11. [Pyergon Durable Execution](#11-pyergon-durable-execution)
12. [Memory & Conversation Persistence](#12-memory--conversation-persistence)
13. [User Interfaces](#13-user-interfaces)
14. [Configuration](#14-configuration)
15. [Error Handling](#15-error-handling)
16. [Extensibility](#16-extensibility)

---

## 1. Executive Summary

Pytelos is a **modular codebase indexer with semantic search and RAG capabilities**. It follows **Parnas's information hiding principles** where each module encapsulates a specific design decision.

### Core Capabilities

```
+------------------+     +------------------+     +------------------+
|    INDEXING      |     |     SEARCH       |     |   REASONING      |
+------------------+     +------------------+     +------------------+
| - Python (AST)   |     | - Vector Search  |     | - Multi-step     |
| - PDF Documents  |     | - BM25 Keyword   |     | - Tool Calling   |
| - Markdown       |     | - Hybrid Fusion  |     | - Streaming      |
| - YAML           |     | - Query Expand   |     | - Memory         |
| - Terraform      |     | - Re-ranking     |     | - Durable (opt)  |
+------------------+     +------------------+     +------------------+
         |                       |                       |
         v                       v                       v
+--------------------------------------------------------------+
|                    STORAGE (PostgreSQL + pgvector)            |
|  - Code chunks with embeddings (1536-dim vectors)             |
|  - HNSW/IVFFlat vector indexes                                |
|  - pg_textsearch for BM25                                     |
+--------------------------------------------------------------+
```

### Technology Stack

| Layer | Technology |
|-------|------------|
| Storage | PostgreSQL + pgvector + pg_textsearch |
| Embeddings | OpenAI (text-embedding-3-small/large) |
| LLM | OpenAI, DeepSeek, Anthropic, Gemini |
| Parsing | tree-sitter (Python), pypdf, markdown-it-py |
| Orchestration | Pyergon (optional, for durable execution) |
| CLI | Typer |
| TUI | Textual |

---

## 2. System Overview

### High-Level Architecture

```
                              +------------------+
                              |      USER        |
                              +--------+---------+
                                       |
              +------------------------+------------------------+
              |                        |                        |
              v                        v                        v
    +------------------+     +------------------+     +------------------+
    |       CLI        |     |       TUI        |     |  Programmatic    |
    |  (cli/app.py)    |     |  (ui/app.py)     |     |      API         |
    +--------+---------+     +--------+---------+     +--------+---------+
              |                        |                        |
              +------------------------+------------------------+
                                       |
                                       v
    +--------------------------------------------------------------+
    |                     REASONING AGENT                           |
    |  +------------------+  +------------------+  +---------------+ |
    |  | ReasoningAgent   |  | PyergonReasoning |  | DocumentAgent | |
    |  | (sequential)     |  | Agent (durable)  |  | (retrieval)   | |
    |  +--------+---------+  +--------+---------+  +-------+-------+ |
    |           |                     |                    |         |
    |           +---------------------+--------------------+         |
    +------------------------------|----------------------------------+
                                   |
                    +--------------+--------------+
                    |              |              |
                    v              v              v
          +-----------+    +-----------+    +-----------+
          |   TOOLS   |    |    LLM    |    |  MEMORY   |
          +-----------+    +-----------+    +-----------+
          | search_   |    | OpenAI    |    | In-Memory |
          | codebase  |    | DeepSeek  |    | SQLite    |
          | read_file |    | Anthropic |    +-----------+
          | analyze_  |    | Gemini    |
          | code      |    +-----------+
          +-----------+
                |
                v
    +--------------------------------------------------------------+
    |                      SEARCH ENGINE                            |
    |  +------------------+  +------------------+  +---------------+ |
    |  | Vector Search    |  | Keyword Search   |  | Hybrid + RRF  | |
    |  | (embeddings)     |  | (BM25)           |  | (fusion)      | |
    |  +--------+---------+  +--------+---------+  +-------+-------+ |
    +------------------------------|----------------------------------+
                                   |
    +--------------------------------------------------------------+
    |                      STORAGE BACKEND                          |
    |  +------------------+  +------------------+  +---------------+ |
    |  | PostgreSQL       |  | pgvector         |  | pg_textsearch | |
    |  | (chunks, meta)   |  | (embeddings)     |  | (BM25 index)  | |
    |  +------------------+  +------------------+  +---------------+ |
    +--------------------------------------------------------------+
```

---

## 3. Module Structure

### Directory Layout

```
pytelos/
|
+-- storage/                    # Storage abstraction layer
|   +-- base.py                 # StorageBackend ABC
|   +-- models.py               # CodeChunk, SearchResult, StorageStats, IndexConfig
|   +-- factory.py              # create_storage_backend()
|   +-- postgres/
|       +-- backend.py          # PostgresBackend implementation
|       +-- schema.py           # SQL schema definitions
|
+-- embedding/                  # Embedding provider abstraction
|   +-- base.py                 # EmbeddingProvider ABC
|   +-- models.py               # EmbeddingResponse
|   +-- factory.py              # create_embedding_provider()
|   +-- providers/
|       +-- openai.py           # OpenAIEmbeddingProvider
|
+-- search/                     # Search engine abstraction
|   +-- base.py                 # SearchEngine ABC
|   +-- models.py               # SearchQuery, SearchResponse, SearchMode
|   +-- engine.py               # DefaultSearchEngine (hybrid search)
|   +-- factory.py              # create_search_engine()
|
+-- indexer/                    # Multi-format indexing pipeline
|   +-- base.py                 # FileParser, CodeParser ABCs
|   +-- models.py               # ChunkingStrategy, ParsedChunk, CodeChunkMetadata
|   +-- pipeline.py             # IndexingPipeline
|   +-- document_parser.py      # DocumentParser ABC, DocumentChunk
|   +-- factory.py              # ParserFactory
|   +-- parsers/
|       +-- python.py           # PythonParser (tree-sitter)
|       +-- pdf.py              # PDFParser
|       +-- markdown_parser.py  # MarkdownParser
|       +-- yaml_parser.py      # YAMLParser
|       +-- terraform_parser.py # TerraformParser
|
+-- llm/                        # LLM provider abstraction
|   +-- base.py                 # LLMProvider ABC
|   +-- models.py               # ChatMessage, LLMResponse
|   +-- factory.py              # create_llm_provider()
|   +-- providers/
|       +-- openai.py           # OpenAIProvider
|       +-- deepseek.py         # DeepSeekProvider
|       +-- anthropic.py        # AnthropicProvider
|       +-- gemini.py           # GeminiProvider
|
+-- reasoning_agent/            # Multi-step reasoning with tools
|   +-- reasoning_agent.py      # ReasoningAgent (main implementation)
|   +-- pyergon_reasoning_agent.py  # PyergonReasoningAgent (durable)
|   +-- data_structures.py      # Task, TaskResult, ToolCall, NextStepDecision
|   +-- tools.py                # BaseTool, SearchCodebaseTool, ReadFileTool
|   +-- flows.py                # Pyergon flow definitions
|   +-- connection_pool.py      # Singleton connection pooling
|
+-- agent/                      # Document-based Q&A agent
|   +-- document_agent.py       # DocumentAgent
|   +-- data_structures.py      # Task, TaskResult, RetrievedChunk
|   +-- tools.py                # DocumentRetrievalTool
|
+-- memory/                     # Conversation memory abstraction
|   +-- base.py                 # ConversationMemory ABC
|   +-- models.py               # ConversationState, TaskRecord
|   +-- factory.py              # create_conversation_memory()
|   +-- in_memory.py            # InMemoryConversationMemory
|   +-- sqlite.py               # SQLiteConversationMemory
|
+-- workflows/                  # Orchestration workflows
|   +-- indexing.py             # IndexingWorkflow
|   +-- pyergon_indexing.py     # PyergonIndexingWorkflow (distributed)
|
+-- ui/                         # Terminal User Interface (Textual)
|   +-- app.py                  # ReasoningTextualApp
|   +-- widgets.py              # ChatHistoryWidget, ExecutionLog, MetricsPanel
|   +-- callbacks.py            # TUICallback
|   +-- styles.py               # CSS styling
|   +-- themes.py               # Color themes
|   +-- screens.py              # Modal dialogs
|   +-- formatting.py           # Output formatting
|   +-- config.py               # UI configuration
|
+-- cli/                        # Command-line interface (Typer)
|   +-- app.py                  # CLI commands
|   +-- providers.py            # Environment configuration
|
+-- prompts/                    # Externalized prompts
    +-- *.txt                   # Prompt templates
```

### Module Dependencies

```
                    +-------------+
                    |   prompts   |
                    +------+------+
                           |
    +----------------------+----------------------+
    |                      |                      |
    v                      v                      v
+-------+            +---------+            +----------+
|  llm  |            | storage |            | embedding|
+---+---+            +----+----+            +-----+----+
    |                     |                       |
    |    +----------------+----------------+      |
    |    |                                 |      |
    v    v                                 v      v
+----------+                          +-----------+
|  search  |<-------------------------+  indexer  |
+----+-----+                          +-----+-----+
     |                                      |
     +------------------+-------------------+
                        |
                        v
              +------------------+
              | reasoning_agent  |
              +--------+---------+
                       |
         +-------------+-------------+
         |             |             |
         v             v             v
    +--------+    +--------+    +--------+
    | memory |    |  agent |    |workflows|
    +--------+    +--------+    +--------+
         |             |             |
         +-------------+-------------+
                       |
         +-------------+-------------+
         |                           |
         v                           v
      +-----+                     +-----+
      | cli |                     | ui  |
      +-----+                     +-----+
```

---

## 4. Core Data Flows

### 4.1 Indexing Flow

```
+--------+     +-------------+     +-------------+     +-----------+
|  User  | --> | CLI/Index   | --> | ParserFact  | --> | Parser    |
| (dir)  |     | Command     |     | .get_parser |     | .parse    |
+--------+     +------+------+     +------+------+     +-----+-----+
                      |                   |                  |
                      v                   v                  v
               +------+------+     +------+------+     +-----+-----+
               | Indexing    |     | Auto-detect |     | Chunks    |
               | Workflow    |     | file type   |     | with meta |
               +------+------+     +-------------+     +-----+-----+
                      |                                      |
                      +--------------------------------------+
                      |
                      v
               +------+------+     +-------------+     +-----------+
               | Indexing    | --> | Embedding   | --> | OpenAI    |
               | Pipeline    |     | Provider    |     | API       |
               +------+------+     +------+------+     +-----+-----+
                      |                   |                  |
                      v                   v                  v
               +------+------+     +------+------+     +-----+-----+
               | Batch       |     | embed_batch |     | 1536-dim  |
               | (size=10)   |     | (texts)     |     | vectors   |
               +------+------+     +------+------+     +-----+-----+
                      |                   |                  |
                      +-------------------+------------------+
                      |
                      v
               +------+------+     +-------------+
               | Storage     | --> | PostgreSQL  |
               | Backend     |     | + pgvector  |
               +-------------+     +-------------+
```

**Sequence Diagram: Indexing a Python File**

```
User          CLI           ParserFactory    PythonParser    Embedder      Storage
  |            |                 |                |              |            |
  |--index---->|                 |                |              |            |
  |            |--get_parser()-->|                |              |            |
  |            |                 |--detect .py-->|              |            |
  |            |<--PythonParser--|                |              |            |
  |            |                 |                |              |            |
  |            |--parse_file()------------------>|              |            |
  |            |                 |                |--tree-sitter |            |
  |            |                 |                |   parse AST  |            |
  |            |<--[ParsedChunk]-----------------|              |            |
  |            |                 |                |              |            |
  |            |--embed_batch([texts])------------------------>|            |
  |            |                 |                |              |--OpenAI-->|
  |            |<--[embeddings]--------------------------------|            |
  |            |                 |                |              |            |
  |            |--store_chunks_batch([chunks], [embeddings])-------------->|
  |            |                 |                |              |            |
  |<--done-----|                 |                |              |            |
```

### 4.2 Search Flow

```
+--------+     +-------------+     +-------------+
|  User  | --> | SearchQuery | --> | Search      |
| (query)|     | (mode,limit)|     | Engine      |
+--------+     +------+------+     +------+------+
                                          |
                     +--------------------+--------------------+
                     |                    |                    |
                     v                    v                    v
              +------+------+      +------+------+      +------+------+
              | VECTOR mode |      |KEYWORD mode |      | HYBRID mode |
              +------+------+      +------+------+      +------+------+
                     |                    |                    |
                     v                    v                    v
              +------+------+      +------+------+      +------+------+
              | embed_text  |      | BM25 search |      | Both paths  |
              | (query)     |      | (tsquery)   |      | + RRF fusion|
              +------+------+      +------+------+      +------+------+
                     |                    |                    |
                     v                    v                    v
              +------+------+      +------+------+      +------+------+
              | pgvector    |      | pg_text     |      | Reciprocal  |
              | <-> cosine  |      | search      |      | Rank Fusion |
              +------+------+      +------+------+      +------+------+
                     |                    |                    |
                     +--------------------+--------------------+
                                          |
                                          v
                                   +------+------+
                                   | Apply       |
                                   | Filters     |
                                   +------+------+
                                          |
                                          v
                                   +------+------+
                                   | Re-rank     |
                                   | (optional)  |
                                   +------+------+
                                          |
                                          v
                                   +------+------+
                                   | Search      |
                                   | Response    |
                                   +-------------+
```

**Hybrid Search with Reciprocal Rank Fusion**

```python
# From search/engine.py - DefaultSearchEngine._hybrid_search()

def _reciprocal_rank_fusion(
    vector_results: list[SearchResult],
    keyword_results: list[SearchResult],
    k: int = 60
) -> list[SearchResult]:
    """
    RRF Formula: score(d) = sum( 1 / (k + rank(d)) )

    For each document d:
      - Get its rank in vector results (or infinity if not present)
      - Get its rank in keyword results (or infinity if not present)
      - Combined score = 1/(k+rank_vector) + 1/(k+rank_keyword)
    """
    scores = {}

    for rank, result in enumerate(vector_results, 1):
        chunk_id = result.chunk.id_
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    for rank, result in enumerate(keyword_results, 1):
        chunk_id = result.chunk.id_
        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

    # Sort by combined score descending
    return sorted(all_results, key=lambda r: scores[r.chunk.id_], reverse=True)
```

### 4.3 Reasoning Agent Flow

```
+--------+     +-------------+     +-------------+
|  User  | --> | Task        | --> | Reasoning   |
| (task) |     | (instruct)  |     | Agent       |
+--------+     +------+------+     +------+------+
                                          |
                                          v
                                   +------+------+
                                   | Processing  |
                                   | Loop        |
                                   +------+------+
                                          |
            +-----------------------------+-----------------------------+
            |                             |                             |
            v                             v                             v
     +------+------+               +------+------+               +------+------+
     | get_next_   |               | run_step    |               | final_result|
     | step (LLM)  |               | (LLM+tools) |               | (complete)  |
     +------+------+               +------+------+               +-------------+
            |                             |
            v                             v
     +------+------+               +------+------+
     | NextStep    |               | Execute     |
     | Decision    |               | Tools       |
     +------+------+               +------+------+
            |                             |
            |    +------------------------+
            |    |
            v    v
     +------+------+
     | Execution   |
     | Trace       |
     +------+------+
            |
            v
     +------+------+
     | TaskResult  |
     +-------------+
```

**Sequence Diagram: Multi-Step Reasoning**

```
User        Agent         LLM          SearchTool      ReadTool     Storage
  |           |            |               |              |            |
  |--task---->|            |               |              |            |
  |           |            |               |              |            |
  |           |--Step 1: get_next_step()-->|              |            |
  |           |            |<--"search for auth code"----|            |
  |           |            |               |              |            |
  |           |--search_codebase("auth")-->|              |            |
  |           |            |               |--query------>|            |
  |           |            |               |<--results----|            |
  |           |<--[chunks]-|---------------|              |            |
  |           |            |               |              |            |
  |           |--Step 2: run_step(results)|              |            |
  |           |            |<--"read auth.py"------------|            |
  |           |            |               |              |            |
  |           |--read_file("auth.py")----->|------------->|            |
  |           |            |               |              |--read---->|
  |           |<--[content]|---------------|--------------|            |
  |           |            |               |              |            |
  |           |--Step 3: get_next_step()-->|              |            |
  |           |            |<--"final_result: analysis"--|            |
  |           |            |               |              |            |
  |<--result--|            |               |              |            |
```

---

## 5. Storage Layer

### Class Hierarchy

```
StorageBackend (ABC)                    # storage/base.py
|
+-- PostgresBackend                     # storage/postgres/backend.py
    |
    +-- Uses: asyncpg (connection pool)
    +-- Uses: pgvector (vector storage)
    +-- Uses: pg_textsearch (BM25)
```

### StorageBackend Interface

```python
# storage/base.py

class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def initialize_schema(self, config: IndexConfig) -> None: ...

    @abstractmethod
    async def store_chunk(
        self, chunk: CodeChunk, embedding: ndarray
    ) -> str: ...  # Returns chunk_id

    @abstractmethod
    async def store_chunks_batch(
        self, chunks: list[CodeChunk], embeddings: list[ndarray]
    ) -> list[str]: ...

    @abstractmethod
    async def vector_search(
        self, query_embedding: ndarray, limit: int, filters: dict | None
    ) -> list[SearchResult]: ...

    @abstractmethod
    async def bm25_search(
        self, query_text: str, limit: int, filters: dict | None
    ) -> list[SearchResult]: ...

    @abstractmethod
    async def hybrid_search(
        self, query_text: str, query_embedding: ndarray,
        limit: int, alpha: float
    ) -> list[SearchResult]: ...
```

### Data Models

```python
# storage/models.py

@dataclass
class CodeChunk:
    id_: str                    # UUIDv7
    file_path: str              # Source file path
    chunk_text: str             # Content
    start_line: int             # Starting line number
    end_line: int               # Ending line number
    language: str               # Programming language
    metadata: dict              # Additional metadata (function_name, class_name, etc.)
    created_at: datetime        # Timestamp

@dataclass
class SearchResult:
    chunk: CodeChunk            # The matched chunk
    score: float                # Relevance score
    rank: int                   # Position in results
    search_type: str            # "vector", "keyword", or "hybrid"

@dataclass
class IndexConfig:
    vector_index_type: str      # "hnsw" or "ivfflat"
    hnsw_m: int                 # HNSW graph degree (default: 16)
    hnsw_ef_construction: int   # HNSW construction parameter (default: 64)
    ivfflat_lists: int          # IVFFlat cluster count
    bm25_k1: float              # BM25 term frequency saturation (default: 1.2)
    bm25_b: float               # BM25 length normalization (default: 0.75)
    text_config: str            # PostgreSQL text search config (default: "english")
```

### Database Schema

```sql
-- storage/postgres/schema.py

-- Main chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    file_path TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    language TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector embeddings table (pgvector)
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id UUID PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    embedding vector(1536) NOT NULL
);

-- HNSW index for fast vector search
CREATE INDEX IF NOT EXISTS chunk_embeddings_hnsw_idx
ON chunk_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- BM25 text search index
CREATE INDEX IF NOT EXISTS chunks_tsvector_idx
ON chunks
USING GIN (to_tsvector('english', chunk_text));

-- Composite indexes
CREATE INDEX IF NOT EXISTS chunks_language_idx ON chunks(language);
CREATE INDEX IF NOT EXISTS chunks_file_path_idx ON chunks(file_path);
```

### Vector Search Implementation

```python
# storage/postgres/backend.py

async def vector_search(
    self,
    query_embedding: ndarray,
    limit: int = 10,
    filters: dict | None = None
) -> list[SearchResult]:
    """
    Perform vector similarity search using pgvector.
    Uses cosine distance: embedding <=> query_embedding
    """
    query = """
        SELECT c.*, ce.embedding,
               1 - (ce.embedding <=> $1::vector) as score
        FROM chunks c
        JOIN chunk_embeddings ce ON c.id = ce.chunk_id
        WHERE 1=1
    """
    params = [query_embedding.tolist()]

    # Apply filters
    if filters:
        if "language" in filters:
            query += " AND c.language = $2"
            params.append(filters["language"])
        if "file_path" in filters:
            query += " AND c.file_path LIKE $3"
            params.append(f"%{filters['file_path']}%")

    query += " ORDER BY ce.embedding <=> $1::vector LIMIT $4"
    params.append(limit)

    rows = await self._pool.fetch(query, *params)
    return [self._row_to_search_result(row, "vector") for row in rows]
```

---

## 6. Embedding Layer

### Class Hierarchy

```
EmbeddingProvider (ABC)                 # embedding/base.py
|
+-- OpenAIEmbeddingProvider             # embedding/providers/openai.py
```

### EmbeddingProvider Interface

```python
# embedding/base.py

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @abstractmethod
    async def embed_text(self, text: str) -> ndarray:
        """Embed a single text string."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[ndarray]:
        """Embed multiple texts in batch for efficiency."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...
```

### OpenAI Implementation

```python
# embedding/providers/openai.py

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding-3 models."""

    _MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        if model not in self._MODEL_DIMENSIONS:
            raise ValueError(f"Unknown model: {model}")
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

    @property
    def dimension(self) -> int:
        return self._MODEL_DIMENSIONS[self._model]

    async def embed_batch(self, texts: list[str]) -> list[ndarray]:
        """Batch embedding for efficiency."""
        if not texts:
            return []

        response = await self._client.embeddings.create(
            model=self._model,
            input=texts
        )

        return [
            np.array(item.embedding, dtype=np.float32)
            for item in response.data
        ]
```

---

## 7. Search Engine

### Class Hierarchy

```
SearchEngine (ABC)                      # search/base.py
|
+-- DefaultSearchEngine                 # search/engine.py
```

### Search Modes

```python
# search/models.py

class SearchMode(Enum):
    VECTOR = "vector"       # Semantic similarity search
    KEYWORD = "keyword"     # BM25 keyword search
    HYBRID = "hybrid"       # Combined with RRF fusion

class RerankStrategy(Enum):
    NONE = "none"                           # No re-ranking
    RECIPROCAL_RANK_FUSION = "rrf"          # RRF (used in hybrid)
    LLM_BASED = "llm"                       # LLM scores results
```

### DefaultSearchEngine Implementation

```python
# search/engine.py

class DefaultSearchEngine(SearchEngine):
    """
    Hybrid search engine combining vector and keyword search.

    Features:
    - Vector search via embeddings
    - BM25 keyword search
    - Hybrid search with Reciprocal Rank Fusion
    - Optional LLM query expansion
    - Optional LLM-based re-ranking
    """

    def __init__(
        self,
        storage: StorageBackend,
        embedder: EmbeddingProvider,
        llm: LLMProvider | None = None,
        alpha: float = 0.5  # 0=keyword only, 1=vector only
    ):
        self._storage = storage
        self._embedder = embedder
        self._llm = llm
        self._alpha = alpha

    async def search(self, query: SearchQuery) -> SearchResponse:
        """Execute search based on query mode."""
        start_time = time.time()

        # Optional query expansion
        expanded_query = query.query
        if self._llm and query.expand_query:
            expanded_query = await self._expand_query(query.query)

        # Execute search based on mode
        if query.mode == SearchMode.VECTOR:
            results = await self._vector_search(expanded_query, query.limit)
        elif query.mode == SearchMode.KEYWORD:
            results = await self._keyword_search(expanded_query, query.limit)
        else:  # HYBRID
            results = await self._hybrid_search(expanded_query, query.limit)

        # Apply filters
        if query.filters:
            results = self._apply_filters(results, query.filters)

        # Optional re-ranking
        if query.rerank == RerankStrategy.LLM_BASED and self._llm:
            results = await self._rerank_with_llm(results, query.query)

        return SearchResponse(
            query=query.query,
            results=results[:query.limit],
            total_results=len(results),
            processing_time_ms=(time.time() - start_time) * 1000,
            expanded_query=expanded_query if expanded_query != query.query else None
        )
```

### Query Expansion

```python
# search/engine.py

async def _expand_query(self, query: str) -> str:
    """Use LLM to expand query with related terms."""
    prompt = f"""Expand this search query with related technical terms.

Query: {query}

Return ONLY the expanded query, no explanation.
Include: synonyms, related concepts, alternative phrasings.
Example: "auth" -> "authentication authorization login OAuth JWT token"
"""
    messages = [ChatMessage(role="user", content=prompt)]
    response = await self._llm.chat_completion(messages)
    return response.content.strip()
```

---

## 8. Indexing Pipeline

### Parser Hierarchy

```
FileParser (ABC)                        # indexer/base.py
|
+-- CodeParser (ABC)                    # For source code files
|   |
|   +-- PythonParser                    # tree-sitter based
|   +-- YAMLParser
|   +-- TerraformParser
|
+-- DocumentParser (ABC)                # For documents
    |
    +-- PDFParser
    +-- MarkdownParser
```

### Chunking Strategies

```python
# indexer/models.py

class ChunkingStrategy(Enum):
    BY_FUNCTION = "by_function"     # Chunk at function/class boundaries
    BY_LINES = "by_lines"           # Fixed-size chunks with overlap
    SEMANTIC = "semantic"           # Structure-aware chunking
```

### PythonParser Implementation

```python
# indexer/parsers/python.py

class PythonParser(CodeParser):
    """
    Python parser using tree-sitter for AST-based chunking.

    Extracts:
    - Function definitions with docstrings
    - Class definitions with methods
    - Import statements
    - Top-level code blocks
    """

    def __init__(self):
        import tree_sitter_python as tspython
        self._language = tspython.language()
        self._parser = Parser(self._language)

    def parse_file(
        self,
        file_path: Path,
        chunk_size: int = 1000,
        strategy: ChunkingStrategy = ChunkingStrategy.BY_FUNCTION,
        overlap: int = 200,
        **options
    ) -> list[ParsedChunk]:
        """Parse Python file into chunks."""
        content = file_path.read_text(encoding="utf-8")
        tree = self._parser.parse(content.encode("utf-8"))

        if strategy == ChunkingStrategy.BY_FUNCTION:
            return self._chunk_by_function(file_path, content, tree.root_node)
        elif strategy == ChunkingStrategy.BY_LINES:
            return self._chunk_by_lines(file_path, content, chunk_size, overlap)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(file_path, content, tree.root_node)

    def _chunk_by_function(
        self, file_path: Path, content: str, root_node
    ) -> list[ParsedChunk]:
        """Extract chunks at function/class boundaries."""
        chunks = []

        for node in self._walk_tree(root_node):
            if node.type in ("function_definition", "class_definition"):
                chunk_text = content[node.start_byte:node.end_byte]

                # Extract metadata
                name = self._get_name(node)
                docstring = self._get_docstring(node)

                metadata = CodeChunkMetadata(
                    language="python",
                    file_path=str(file_path),
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    function_name=name if node.type == "function_definition" else None,
                    class_name=name if node.type == "class_definition" else None,
                    docstring=docstring,
                    imports=self._get_imports(root_node, content),
                )

                chunks.append(ParsedChunk(
                    content=chunk_text,
                    metadata=metadata
                ))

        return chunks
```

### Indexing Pipeline

```python
# indexer/pipeline.py

class IndexingPipeline:
    """
    Orchestrates the indexing workflow:
    1. Parse files into chunks
    2. Generate embeddings in batches
    3. Store chunks with embeddings
    """

    def __init__(
        self,
        parser_factory: ParserFactory,
        embedder: EmbeddingProvider,
        storage: StorageBackend,
        batch_size: int = 10
    ):
        self._parser_factory = parser_factory
        self._embedder = embedder
        self._storage = storage
        self._batch_size = batch_size

    async def index_directory(
        self,
        directory: Path,
        recursive: bool = True,
        file_pattern: str = "**/*",
        strategy: ChunkingStrategy = ChunkingStrategy.BY_FUNCTION
    ) -> IndexingResult:
        """Index all supported files in a directory."""
        files = list(directory.glob(file_pattern if recursive else file_pattern.split("/")[-1]))

        total_chunks = 0
        failed_files = []

        for file_path in files:
            parser = self._parser_factory.get_parser(file_path)
            if parser is None:
                continue

            try:
                chunks = parser.parse_file(file_path, strategy=strategy)

                # Process in batches
                for i in range(0, len(chunks), self._batch_size):
                    batch = chunks[i:i + self._batch_size]
                    count = await self._process_batch(batch)
                    total_chunks += count

            except Exception as e:
                failed_files.append((file_path, str(e)))

        return IndexingResult(
            total_files=len(files),
            total_chunks=total_chunks,
            failed_files=failed_files
        )

    async def _process_batch(self, batch: list[ParsedChunk]) -> int:
        """Process a batch of chunks: embed and store."""
        texts = [chunk.content for chunk in batch]
        embeddings = await self._embedder.embed_batch(texts)

        storage_chunks = [
            self._convert_to_storage_chunk(chunk)
            for chunk in batch
        ]

        chunk_ids = await self._storage.store_chunks_batch(
            storage_chunks, embeddings
        )

        return len(chunk_ids)
```

---

## 9. LLM Integration

### Provider Hierarchy

```
LLMProvider (ABC)                       # llm/base.py
|
+-- OpenAIProvider                      # llm/providers/openai.py
+-- DeepSeekProvider                    # llm/providers/deepseek.py
+-- AnthropicProvider                   # llm/providers/anthropic.py
+-- GeminiProvider                      # llm/providers/gemini.py
```

### LLMProvider Interface

```python
# llm/base.py

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a chat completion."""
        ...

    @abstractmethod
    async def chat_completion_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming chat completion."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...
```

### Data Models

```python
# llm/models.py

@dataclass
class ChatMessage:
    role: str       # "user", "assistant", or "system"
    content: str    # Message content

@dataclass
class LLMResponse:
    content: str                    # Generated text
    model: str                      # Model used
    usage: dict[str, int] | None    # Token usage stats
```

### Factory Pattern

```python
# llm/factory.py

def create_llm_provider(provider: str, **config) -> LLMProvider:
    """
    Create an LLM provider instance.

    Args:
        provider: Provider name ("openai", "deepseek", "anthropic", "gemini")
        **config: Provider-specific configuration

    Returns:
        Configured LLMProvider instance
    """
    provider = provider.lower()

    if provider == "openai":
        return OpenAIProvider(
            api_key=config.get("api_key"),
            model=config.get("model", "gpt-4o-mini")
        )
    elif provider == "deepseek":
        return DeepSeekProvider(
            api_key=config.get("api_key"),
            model=config.get("model", "deepseek-chat")
        )
    elif provider == "anthropic":
        return AnthropicProvider(
            api_key=config.get("api_key"),
            model=config.get("model", "claude-3-5-sonnet-20241022")
        )
    elif provider == "gemini":
        return GeminiProvider(
            api_key=config.get("api_key"),
            model=config.get("model", "gemini-2.0-flash-exp")
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
```

---

## 10. Reasoning Agent

### Architecture

```
+------------------------------------------------------------------+
|                        ReasoningAgent                             |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+    +------------------+    +--------------+ |
|  |   LLMProvider    |    |   Tools Registry |    | System Prompt| |
|  +--------+---------+    +--------+---------+    +------+-------+ |
|           |                       |                     |         |
|           +---------------+-------+---------------------+         |
|                           |                                       |
|                           v                                       |
|                  +--------+--------+                              |
|                  | Processing Loop |                              |
|                  +--------+--------+                              |
|                           |                                       |
|           +---------------+---------------+                       |
|           |               |               |                       |
|           v               v               v                       |
|    +------+------+ +------+------+ +------+------+                |
|    | get_next_   | | run_step    | | execute_    |                |
|    | step()      | | ()          | | tools()     |                |
|    +-------------+ +-------------+ +-------------+                |
|                                                                   |
+------------------------------------------------------------------+
```

### Tool System

```python
# reasoning_agent/tools.py

class BaseTool(ABC):
    """Abstract base class for reasoning agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for LLM function calling."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM context."""
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict:
        """JSON Schema for tool parameters."""
        ...

    @abstractmethod
    async def execute(self, tool_call: ToolCall) -> ToolCallResult:
        """Execute the tool with given arguments."""
        ...

    def to_llm_spec(self) -> dict:
        """Convert to LLM function calling format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema
        }


class SearchCodebaseTool(BaseTool):
    """Search indexed codebase using semantic + keyword search."""

    @property
    def name(self) -> str:
        return "search_codebase"

    @property
    def description(self) -> str:
        return (
            "Search the indexed codebase for relevant code, documents, and files. "
            "Supports semantic search (meaning-based) and keyword search. "
            "Use this to find implementations, documentation, or any indexed content."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (natural language or keywords)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 5
                },
                "file_extension": {
                    "type": "string",
                    "description": "Filter by file extension (e.g., '.py', '.md')"
                }
            },
            "required": ["query"]
        }

    async def execute(self, tool_call: ToolCall) -> ToolCallResult:
        """Execute codebase search."""
        query = tool_call.arguments.get("query", "")
        limit = tool_call.arguments.get("limit", 5)

        search_query = SearchQuery(
            query=query,
            mode=SearchMode.HYBRID,
            limit=limit
        )

        response = await self._search_engine.search(search_query)

        # Format results as JSON
        results = [
            {
                "rank": i + 1,
                "file": r.chunk.file_path,
                "lines": f"{r.chunk.start_line}-{r.chunk.end_line}",
                "score": round(r.score, 3),
                "content": r.chunk.chunk_text[:500]
            }
            for i, r in enumerate(response.results)
        ]

        return ToolCallResult(
            tool_call_id=tool_call.id_,
            content=json.dumps(results, indent=2),
            error=False
        )
```

### Data Structures

```python
# reasoning_agent/data_structures.py

@dataclass
class Task:
    id_: str                # UUIDv7
    instruction: str        # User's task/question

@dataclass
class NextStepDecision:
    kind: str               # "next_step" or "final_result"
    content: str            # Next instruction or final answer

@dataclass
class ToolCall:
    id_: str                # Tool call ID
    tool_name: str          # Name of tool to execute
    arguments: dict         # Tool arguments

@dataclass
class ToolCallResult:
    tool_call_id: str       # Matching tool call ID
    content: str            # Tool output
    error: bool             # Whether execution failed

@dataclass
class TaskResult:
    task_id: str            # Original task ID
    content: str            # Final answer
    execution_trace: str    # Full step-by-step trace
    metadata: dict          # Stats (steps, tokens, time)
```

### ReasoningAgent Implementation

```python
# reasoning_agent/reasoning_agent.py

class ReasoningAgent:
    """
    Multi-step reasoning agent with tool calling.

    Processing loop:
    1. Get next step decision from LLM
    2. If next_step: extract tool calls, execute, run step
    3. If final_result: return answer
    4. Repeat until done or max_steps reached
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None
    ):
        self._llm = llm
        self._tools = tools or []
        self._tools_registry = {t.name: t for t in self._tools}

        # Load system prompt
        if system_prompt is None:
            from ..prompts import get_system_prompt
            system_prompt = get_system_prompt()

        # Format with tools description
        tools_desc = "\n".join([
            f"- {t.name}: {t.description}"
            for t in self._tools
        ])
        self._system_prompt = system_prompt.format(
            tools_description=tools_desc or "None"
        )

    def run(
        self,
        task: Task,
        max_steps: int = 10,
        conversation_context: str | None = None
    ) -> TaskHandler:
        """
        Run the reasoning agent on a task.

        Returns a TaskHandler (Future) that can be awaited.
        """
        handler = TaskHandler(
            agent=self,
            task=task,
            max_steps=max_steps,
            conversation_context=conversation_context
        )
        handler.start()
        return handler


class TaskHandler(asyncio.Future):
    """Async handler for reasoning task execution."""

    async def _execute_processing_loop(self) -> TaskResult:
        """Main processing loop."""
        execution_trace = ""
        step_results = []

        while self._step_counter < self._max_steps:
            # Get next step decision
            decision = await self._get_next_step(step_results)

            if decision.kind == "final_result":
                return TaskResult(
                    task_id=self._task.id_,
                    content=decision.content,
                    execution_trace=execution_trace,
                    metadata={
                        "steps": self._step_counter,
                        "total_input_tokens": self._usage.total_input_tokens,
                        "total_output_tokens": self._usage.total_output_tokens
                    }
                )

            # Execute step with tool calls
            step_result = await self._run_step(decision.content, step_results)
            step_results.append(step_result)

            # Update trace
            execution_trace += f"\n=== Step {self._step_counter + 1} ===\n"
            execution_trace += f"{step_result.content}\n"

            self._step_counter += 1

        # Max steps reached
        return TaskResult(
            task_id=self._task.id_,
            content="Maximum steps reached without final answer.",
            execution_trace=execution_trace,
            metadata={"steps": self._step_counter}
        )
```

---

## 11. Pyergon Durable Execution

### Architecture

```
+------------------------------------------------------------------+
|                   PyergonReasoningAgent                           |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+    +------------------+    +--------------+ |
|  |    Scheduler     |    |     Workers      |    | Execution    | |
|  |                  |    |   (1 or more)    |    | Log (SQLite) | |
|  +--------+---------+    +--------+---------+    +------+-------+ |
|           |                       |                     |         |
|           v                       v                     v         |
|  +--------+--------------------------------------------------+   |
|  |                    ReasoningTaskFlow                       |   |
|  |  +------------------+  +------------------+  +------------+|   |
|  |  | @flow solve()    |  | invoke(StepFlow) |  | Checkpoint ||   |
|  |  +--------+---------+  +--------+---------+  +------+-----+|   |
|  +-----------|----------------------|------------------|------+   |
|              |                      |                  |          |
|              +----------------------+------------------+          |
|                                     |                             |
+-------------------------------------|-----------------------------+
                                      |
                                      v
                          +-----------+-----------+
                          |       StepFlow        |
                          +-----------+-----------+
                          | - Decision (LLM)      |
                          | - Execution (LLM+Tool)|
                          | - RetryPolicy         |
                          +-----------------------+
```

### Flow Definitions

```python
# reasoning_agent/flows.py

@dataclass
@flow_type(invokable=str)
class ReasoningTaskFlow:
    """
    Durable flow for complete multi-step reasoning.

    Orchestrates:
    1. Execute step (decision + action)
    2. Update execution trace
    3. Check for completion
    4. Persist state after each step
    """
    task_instruction: str
    llm_config: dict[str, Any]
    tools_config: dict[str, Any]
    system_prompt: str
    max_steps: int = 10

    def __post_init__(self):
        self.execution_trace = ""
        self.step_count = 0

    @flow
    async def solve(self) -> dict:
        """Main reasoning flow."""
        total_input_tokens = 0
        total_output_tokens = 0

        while self.step_count < self.max_steps:
            # Invoke child flow (checkpointed)
            step_pending = self.invoke(StepFlow(
                task_instruction=self.task_instruction,
                execution_trace=self.execution_trace,
                step_count=self.step_count,
                max_steps=self.max_steps,
                llm_config=self.llm_config,
                tools_config=self.tools_config,
                system_prompt=self.system_prompt
            ))
            step_result = await step_pending.result()

            # Track usage
            if "usage" in step_result:
                total_input_tokens += step_result["usage"].get("input_tokens", 0)
                total_output_tokens += step_result["usage"].get("output_tokens", 0)

            # Check completion
            if step_result["kind"] == "final_result":
                return {
                    "result": step_result["content"],
                    "steps": self.step_count,
                    "execution_trace": self.execution_trace,
                    "status": "completed",
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens
                }

            # Update state
            self.execution_trace += f"\n=== Step {self.step_count + 1} ===\n"
            self.execution_trace += f"Response: {step_result['content']}\n"
            self.step_count += 1

        return {
            "result": "Maximum steps reached.",
            "steps": self.step_count,
            "execution_trace": self.execution_trace,
            "status": "max_steps_reached"
        }


@dataclass
@flow_type(invokable=str)
class StepFlow:
    """
    Combined decision and execution flow.

    Single LLM call that:
    1. Decides next action
    2. Executes tool if needed
    3. Returns result
    """
    task_instruction: str
    execution_trace: str
    step_count: int
    max_steps: int
    llm_config: dict[str, Any]
    tools_config: dict[str, Any]
    system_prompt: str

    @flow
    async def execute(self) -> dict:
        """Execute a single reasoning step."""
        # Get LLM from connection pool
        llm = await ConnectionPool.get_llm(self.llm_config)

        # Build prompt with context
        prompt = self._build_prompt()

        # Get LLM response
        messages = [ChatMessage(role="user", content=prompt)]
        response = await llm.chat_completion(messages)

        # Parse decision
        decision = self._parse_decision(response.content)

        if decision["kind"] == "final_result":
            return decision

        # Extract and execute tool call if present
        tool_call = self._extract_tool_call(response.content)
        if tool_call:
            tool_result = await self._execute_tool(tool_call)
            decision["used_tool"] = tool_call["tool_name"]
            decision["tool_result"] = tool_result

        return decision
```

### Retry Policies

```python
# reasoning_agent/flows.py

class LLMRateLimitError(ReasoningError):
    """LLM API rate limit exceeded (retryable)."""
    def is_retryable(self) -> bool:
        return True

class LLMNetworkError(ReasoningError):
    """Network or connection error (retryable)."""
    def is_retryable(self) -> bool:
        return True

class InvalidPromptError(ReasoningError):
    """Invalid prompt (non-retryable)."""
    def is_retryable(self) -> bool:
        return False

# Usage with Pyergon RetryPolicy
@dataclass
@flow_type(invokable=str)
class ReasoningStepFlow:

    @step(retry_policy=RetryPolicy.STANDARD)  # 3 retries with backoff
    async def get_llm_response(self) -> dict:
        """Get LLM response with automatic retry on transient failures."""
        try:
            response = await self._llm.chat_completion(messages)
            return {"content": response.content}
        except RateLimitError as e:
            raise LLMRateLimitError(str(e))  # Will be retried
        except ConnectionError as e:
            raise LLMNetworkError(str(e))    # Will be retried
```

### Connection Pool

```python
# reasoning_agent/connection_pool.py

class ConnectionPool:
    """
    Singleton connection pool for worker processes.

    Pools:
    - LLM clients (by provider:model)
    - Storage backends (by backend:host:port:db)
    - Embedding providers (by provider:model)
    """

    _llm_pool: dict[str, LLMProvider] = {}
    _storage_pool: dict[str, StorageBackend] = {}
    _embedder_pool: dict[str, EmbeddingProvider] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def get_llm(cls, config: dict) -> LLMProvider:
        """Get or create LLM provider."""
        key = f"{config['provider']}:{config.get('kwargs', {}).get('model', 'default')}"

        async with cls._lock:
            if key not in cls._llm_pool:
                cls._llm_pool[key] = create_llm_provider(
                    config["provider"],
                    **config.get("kwargs", {})
                )
            return cls._llm_pool[key]

    @classmethod
    async def warmup(
        cls,
        llm_config: dict | None = None,
        storage_config: dict | None = None,
        embedder_config: dict | None = None
    ) -> None:
        """Pre-warm connections in parallel."""
        tasks = []
        if llm_config:
            tasks.append(cls.get_llm(llm_config))
        if storage_config:
            tasks.append(cls.get_storage(storage_config))
        if embedder_config:
            tasks.append(cls.get_embedder(embedder_config))

        await asyncio.gather(*tasks)

    @classmethod
    async def close_all(cls) -> None:
        """Close all pooled connections."""
        for llm in cls._llm_pool.values():
            await llm.close()
        for storage in cls._storage_pool.values():
            await storage.disconnect()

        cls._llm_pool.clear()
        cls._storage_pool.clear()
        cls._embedder_pool.clear()
```

---

## 12. Memory & Conversation Persistence

### Class Hierarchy

```
ConversationMemory (ABC)                # memory/base.py
|
+-- InMemoryConversationMemory          # memory/in_memory.py
|   (session-only, lost on exit)
|
+-- SQLiteConversationMemory            # memory/sqlite.py
    (persistent across sessions)
```

### Data Models

```python
# memory/models.py

@dataclass
class TaskRecord:
    task_id: str            # UUIDv7
    task: str               # User's question/task
    answer: str             # Agent's response
    files_read: list[str]   # Files accessed during task
    timestamp: datetime     # When task was completed

@dataclass
class ConversationState:
    session_id: str
    history: list[TaskRecord]
    files_read: set[str]
    last_task: str | None
    last_answer: str | None
    created_at: datetime
    updated_at: datetime

    def add_task(self, task: str, answer: str, files: list[str]) -> None:
        """Add a completed task to history."""
        self.history.append(TaskRecord(
            task_id=str(uuid7()),
            task=task,
            answer=answer,
            files_read=files,
            timestamp=datetime.now()
        ))
        self.files_read.update(files)
        self.last_task = task
        self.last_answer = answer
        self.updated_at = datetime.now()

    def to_context_string(self, limit: int = 5) -> str:
        """Format recent history for LLM context injection."""
        recent = self.history[-limit:]

        context_parts = ["Previous conversation:"]
        for record in recent:
            context_parts.append(f"Q: {record.task}")
            context_parts.append(f"A: {record.answer[:500]}...")

        return "\n".join(context_parts)
```

### SQLite Implementation

```python
# memory/sqlite.py

class SQLiteConversationMemory(ConversationMemory):
    """
    Persistent conversation memory using SQLite.

    Schema:
    - sessions: session metadata
    - task_records: individual task/answer pairs
    """

    def __init__(
        self,
        db_path: str = "./conversation_memory.db",
        default_session_id: str | None = None
    ):
        self._db_path = Path(db_path)
        self._default_session_id = default_session_id or str(uuid7())
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Connect to SQLite database."""
        self._connection = await aiosqlite.connect(self._db_path)
        await self._initialize_schema()

    async def _initialize_schema(self) -> None:
        """Create tables if they don't exist."""
        await self._connection.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS task_records (
                task_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                task TEXT NOT NULL,
                answer TEXT NOT NULL,
                files_read TEXT,  -- JSON array
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_task_records_session
            ON task_records(session_id);
        """)
        await self._connection.commit()

    async def add_task_record(
        self,
        task: str,
        answer: str,
        files_read: list[str] | None = None,
        session_id: str | None = None
    ) -> TaskRecord:
        """Add a task record to the database."""
        session_id = session_id or self._default_session_id
        task_id = str(uuid7())

        await self._connection.execute(
            """
            INSERT INTO task_records (task_id, session_id, task, answer, files_read)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task_id, session_id, task, answer, json.dumps(files_read or []))
        )
        await self._connection.commit()

        return TaskRecord(
            task_id=task_id,
            task=task,
            answer=answer,
            files_read=files_read or [],
            timestamp=datetime.now()
        )
```

### Memory Integration Flow

```
User: "Question 1"
    |
    v
ReasoningAgent.run(task1)
    |
    v
TaskResult(answer1)
    |
    v
Memory.add_task_record(task1, answer1)
    |
    +---> TaskRecord stored
    |
    v
User: "Follow-up question"
    |
    v
context = Memory.get_state().to_context_string()
    |
    v
ReasoningAgent.run(task2, conversation_context=context)
    |
    +---> LLM receives:
    |     - System prompt
    |     - Previous Q&A history
    |     - Current task
    |
    v
TaskResult(answer2, references answer1)
```

---

## 13. User Interfaces

### CLI (Typer)

```python
# cli/app.py

app = typer.Typer(name="pytelos")

@app.command()
def init(force: bool = False):
    """Initialize database schema."""
    storage = get_storage()
    asyncio.run(storage.connect())
    asyncio.run(storage.initialize_schema(IndexConfig()))

@app.command()
def index(
    directory: Path,
    pattern: str = "**/*",
    strategy: str = "by_function"
):
    """Index a directory."""
    # Implementation...

@app.command()
def search(
    query: str,
    mode: str = "hybrid",
    limit: int = 10
):
    """Search the index."""
    # Implementation...

@app.command()
def chat(question: str):
    """Interactive Q&A with reasoning agent."""
    # Implementation...
```

### TUI (Textual)

```
+------------------------------------------------------------------+
|  Pytelos - model | environment | max 20 steps | sqlite           |
+------------------------------------------------------------------+
|                              |                                    |
|  +------------------------+  |  +------------------------------+  |
|  |      Chat Panel        |  |  |       Execution Panel        |  |
|  +------------------------+  |  +------------------------------+  |
|  |                        |  |  |                              |  |
|  | < Assistant [12:34:56] |  |  | LLM Response (streaming):    |  |
|  | Welcome to Pytelos!    |  |  | ============ Step 1/20 ===== |  |
|  |                        |  |  | {                            |  |
|  | > You [12:35:01]       |  |  |   "tool_name": "search...",  |  |
|  | How does auth work?    |  |  |   "arguments": {...}         |  |
|  |                        |  |  | }                            |  |
|  | < Assistant [12:35:10] |  |  |                              |  |
|  | The authentication...  |  |  | ============ Step 2/20 ===== |  |
|  |                        |  |  | ...                          |  |
|  +------------------------+  |  +------------------------------+  |
|                              |                                    |
|                              |  +------------------------------+  |
|                              |  |         Log Panel            |  |
|                              |  +------------------------------+  |
|                              |  | 12:35:02 INFO [LLM] Calling..|  |
|                              |  | 12:35:05 DEBUG [Step] Tool...|  |
|                              |  +------------------------------+  |
|                              |                                    |
+------------------------------------------------------------------+
|  Iter: 2  Code: 0  Time: 8.5s  Main: 2,500 (2,100/400)           |
+------------------------------------------------------------------+
|  [                    Chat input...                    ] [ Send ] |
+------------------------------------------------------------------+
| ^L Clear Log | ^K Clear Chat | ^B Max Chat | ^E Max Log | ESC    |
+------------------------------------------------------------------+
```

### Widget Structure

```python
# ui/widgets.py

class ChatHistoryWidget(Static):
    """Displays chat messages with user/assistant styling."""

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat history."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if role == "user":
            prefix = f"> You [{timestamp}]"
            css_class = "user-message"
        else:
            prefix = f"< Assistant [{timestamp}]"
            css_class = "assistant-message"

        message = ChatMessageWidget(
            header=prefix,
            content=content,
            classes=css_class
        )
        self.mount(message)
        self.scroll_end()


class ExecutionLog(RichLog):
    """Real-time execution log with streaming support."""

    def start_execution(self) -> None:
        """Mark start of execution."""
        self.border_subtitle = "Running..."
        self.add_class("executing")

    def end_execution(self, success: bool = True) -> None:
        """Mark end of execution."""
        self.border_subtitle = "Complete" if success else "Failed"
        self.remove_class("executing")


class MetricsPanel(Static):
    """Displays execution metrics."""

    def update_metrics(
        self,
        iterations: int = 0,
        code_blocks: int = 0,
        time: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0
    ) -> None:
        """Update displayed metrics."""
        self.update(
            f"[cyan]Iter:[/] {iterations}  "
            f"[green]Code:[/] {code_blocks}  "
            f"[yellow]Time:[/] {time:.2f}s  "
            f"[magenta]Main:[/] {input_tokens + output_tokens} "
            f"({input_tokens}/{output_tokens})"
        )
```

### TUICallback Integration

```python
# ui/callbacks.py

class TUICallback:
    """
    Callback handler for ReasoningAgent execution updates.

    Thread-safe updates from worker threads to UI.
    """

    def __init__(
        self,
        log: ExecutionLog,
        plot_panel: PlotPanel | None = None,
        metrics_panel: MetricsPanel | None = None,
        app: App | None = None
    ):
        self.log = log
        self.plot_panel = plot_panel
        self.metrics_panel = metrics_panel
        self.app = app
        self._step_count = 0
        self._start_time: float | None = None

    def _call_thread_safe(self, func, *args, **kwargs) -> None:
        """Call function from worker thread safely."""
        if self.app and self.app._thread_id != threading.get_ident():
            self.app.call_from_thread(func, *args, **kwargs)
        else:
            func(*args, **kwargs)

    def _update_metrics(self) -> None:
        """Update metrics panel with current progress."""
        if self.metrics_panel is None:
            return

        elapsed = time.time() - self._start_time if self._start_time else 0

        self._call_thread_safe(
            self.metrics_panel.update_metrics,
            iterations=self._step_count,
            time=elapsed
        )

    def print_step_progress(self, step: int, max_steps: int) -> None:
        """Print step progress and update metrics."""
        self._step_count = step
        self._update_metrics()
        self._call_thread_safe(
            self.log.write,
            f"[cyan]{'='*15}[/] [yellow]Step {step}/{max_steps}[/] [cyan]{'='*15}[/]",
            markup=True
        )

    def handle_stream_chunk(self, chunk: str) -> None:
        """Handle streaming LLM tokens."""
        if chunk == "__START__":
            self._call_thread_safe(
                self.log.write,
                "[cyan]LLM Response (streaming):[/]",
                markup=True
            )
        elif chunk == "__END__":
            self._call_thread_safe(self.log.write, "\n")
        else:
            self._call_thread_safe(self.log.write, chunk, markup=False)
```

---

## 14. Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | localhost | PostgreSQL host |
| `POSTGRES_PORT` | 5433 | PostgreSQL port |
| `POSTGRES_DB` | pytelos | Database name |
| `POSTGRES_USER` | pytelos | Database user |
| `POSTGRES_PASSWORD` | pytelos_dev | Database password |
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_CHAT_MODEL` | gpt-4o-mini | OpenAI chat model |
| `DEEPSEEK_API_KEY` | - | DeepSeek API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `LLM_PROVIDER` | deepseek | Default LLM provider |
| `EMBEDDING_MODEL` | text-embedding-3-small | Embedding model |

### Prompt Customization

```
prompts/
|
+-- system.txt              # Main system prompt (with {tools_description})
+-- reasoning_agent.txt     # Reasoning agent specific
+-- document_agent.txt      # Document QA specific
```

**Loading Priority**:
1. `./prompts/{name}.txt` (local override)
2. `pytelos/prompts/{name}.txt` (package default)

```python
# prompts/__init__.py

def load_prompt(name: str) -> str:
    """Load prompt from file."""
    # Check local first
    local_path = Path(f"./prompts/{name}.txt")
    if local_path.exists():
        return local_path.read_text()

    # Fall back to package
    package_path = Path(__file__).parent / f"{name}.txt"
    if package_path.exists():
        return package_path.read_text()

    raise FileNotFoundError(f"Prompt not found: {name}")

def get_system_prompt() -> str:
    """Get the main system prompt."""
    return load_prompt("system")
```

### Index Configuration

```python
# storage/models.py

@dataclass
class IndexConfig:
    # Vector index settings
    vector_index_type: str = "hnsw"     # "hnsw" or "ivfflat"
    hnsw_m: int = 16                    # HNSW graph connections
    hnsw_ef_construction: int = 64      # HNSW construction parameter
    ivfflat_lists: int = 100            # IVFFlat cluster count

    # BM25 settings
    bm25_k1: float = 1.2                # Term frequency saturation
    bm25_b: float = 0.75                # Length normalization
    text_config: str = "english"        # PostgreSQL text search config
```

---

## 15. Error Handling

### Error Categories

```
+------------------+     +------------------+     +------------------+
|   Retryable      |     |  Non-Retryable   |     |   User Errors    |
+------------------+     +------------------+     +------------------+
| - Rate limits    |     | - Invalid prompt |     | - File not found |
| - Network errors |     | - Auth failure   |     | - Invalid query  |
| - Timeouts       |     | - Schema errors  |     | - Bad config     |
+------------------+     +------------------+     +------------------+
         |                        |                       |
         v                        v                       v
   Auto-retry with         Fail immediately        Return error
   exponential backoff     with error message      to user
```

### Pyergon Error Handling

```python
# reasoning_agent/flows.py

class ReasoningError(Exception):
    """Base class for reasoning errors."""

    def is_retryable(self) -> bool:
        return False

class LLMRateLimitError(ReasoningError):
    """Retryable: API rate limit."""
    def is_retryable(self) -> bool:
        return True

class LLMNetworkError(ReasoningError):
    """Retryable: Network failure."""
    def is_retryable(self) -> bool:
        return True

class InvalidPromptError(ReasoningError):
    """Non-retryable: Bad prompt."""
    def is_retryable(self) -> bool:
        return False

class ToolExecutionError(ReasoningError):
    """Non-retryable: Tool failed."""
    def is_retryable(self) -> bool:
        return False
```

### Graceful Degradation

```python
# reasoning_agent/reasoning_agent.py

async def _run_step(self, instruction: str, results: list) -> TaskStepResult:
    """Execute a step with graceful error handling."""
    try:
        # Execute tools
        for tool_call in tool_calls:
            try:
                result = await self._execute_tool(tool_call)
            except ToolExecutionError as e:
                # Log error but continue
                result = ToolCallResult(
                    tool_call_id=tool_call.id_,
                    content=f"Error: {e}",
                    error=True
                )
            tool_results.append(result)

        # Continue with LLM even if some tools failed
        return await self._llm_completion(instruction, tool_results)

    except Exception as e:
        # Return partial result on failure
        return TaskStepResult(
            task_step_id=str(uuid7()),
            content=f"Step failed: {e}"
        )
```

---

## 16. Extensibility

### Adding a New Parser

```python
# indexer/parsers/custom.py

from pytelos.indexer.base import CodeParser
from pytelos.indexer.models import ParsedChunk, CodeChunkMetadata, ChunkingStrategy

class CustomParser(CodeParser):
    """Parser for custom file format."""

    def supports_language(self, language: str) -> bool:
        return language.lower() == "custom"

    def detect_language(self, file_path: Path) -> str | None:
        if file_path.suffix == ".custom":
            return "custom"
        return None

    def get_file_type(self) -> str:
        return "custom"

    def get_supported_extensions(self) -> set[str]:
        return {".custom"}

    def parse_file(
        self,
        file_path: Path,
        chunk_size: int = 1000,
        strategy: ChunkingStrategy = ChunkingStrategy.BY_LINES,
        overlap: int = 200,
        **options
    ) -> list[ParsedChunk]:
        content = file_path.read_text()
        # Custom parsing logic...
        return chunks

# Register with factory
from pytelos.indexer import create_parser_factory

factory = create_parser_factory()
factory.register_parser(CustomParser())
```

### Adding a New Tool

```python
# reasoning_agent/tools.py (or custom module)

from pytelos.reasoning_agent.tools import BaseTool
from pytelos.reasoning_agent.data_structures import ToolCall, ToolCallResult

class CustomTool(BaseTool):
    """Custom tool for specific functionality."""

    @property
    def name(self) -> str:
        return "custom_tool"

    @property
    def description(self) -> str:
        return "Description for LLM to understand when to use this tool."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "integer", "description": "Second parameter"}
            },
            "required": ["param1"]
        }

    async def execute(self, tool_call: ToolCall) -> ToolCallResult:
        param1 = tool_call.arguments.get("param1")
        param2 = tool_call.arguments.get("param2", 0)

        # Custom logic...
        result = f"Processed {param1} with {param2}"

        return ToolCallResult(
            tool_call_id=tool_call.id_,
            content=result,
            error=False
        )

# Add to agent
agent = ReasoningAgent(llm=llm)
agent.add_tool(CustomTool())
```

### Adding a New LLM Provider

```python
# llm/providers/custom.py

from pytelos.llm.base import LLMProvider
from pytelos.llm.models import ChatMessage, LLMResponse

class CustomLLMProvider(LLMProvider):
    """Custom LLM provider."""

    def __init__(self, api_key: str, model: str = "custom-model"):
        self._api_key = api_key
        self._model = model
        # Initialize client...

    async def chat_completion(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> LLMResponse:
        # Custom API call...
        return LLMResponse(
            content=response_text,
            model=model or self._model,
            usage={"prompt_tokens": ..., "completion_tokens": ...}
        )

    async def chat_completion_stream(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs
    ) -> AsyncIterator[str]:
        # Streaming implementation...
        async for chunk in stream:
            yield chunk

    async def close(self) -> None:
        # Cleanup...
        pass

# Register in factory (llm/factory.py)
def create_llm_provider(provider: str, **config) -> LLMProvider:
    if provider.lower() == "custom":
        return CustomLLMProvider(**config)
    # ... existing providers
```

### Adding a New Memory Backend

```python
# memory/custom.py

from pytelos.memory.base import ConversationMemory
from pytelos.memory.models import ConversationState, TaskRecord

class CustomMemoryBackend(ConversationMemory):
    """Custom memory backend (e.g., Redis, MongoDB)."""

    async def connect(self) -> None:
        # Connect to backend...
        pass

    async def disconnect(self) -> None:
        # Disconnect...
        pass

    async def get_state(self, session_id: str | None = None) -> ConversationState:
        # Retrieve state...
        pass

    async def save_state(self, state: ConversationState) -> None:
        # Persist state...
        pass

    async def add_task_record(
        self,
        task: str,
        answer: str,
        files_read: list[str] | None = None,
        session_id: str | None = None
    ) -> TaskRecord:
        # Add record...
        pass

    @property
    def backend_type(self) -> str:
        return "custom"

# Register in factory (memory/factory.py)
def create_conversation_memory(backend: str, **config) -> ConversationMemory:
    if backend.lower() == "custom":
        return CustomMemoryBackend(**config)
    # ... existing backends
```

---

## Summary

Pytelos is a production-grade codebase indexer following information hiding principles:

| Module | Responsibility | Key Classes |
|--------|---------------|-------------|
| `storage` | Data persistence | `StorageBackend`, `PostgresBackend` |
| `embedding` | Vector generation | `EmbeddingProvider`, `OpenAIEmbeddingProvider` |
| `search` | Retrieval | `SearchEngine`, `DefaultSearchEngine` |
| `indexer` | File parsing | `ParserFactory`, `PythonParser`, `IndexingPipeline` |
| `llm` | LLM integration | `LLMProvider`, `OpenAI/DeepSeek/Anthropic/GeminiProvider` |
| `reasoning_agent` | Multi-step reasoning | `ReasoningAgent`, `BaseTool`, `PyergonReasoningAgent` |
| `memory` | Conversation persistence | `ConversationMemory`, `SQLiteConversationMemory` |
| `ui` | Terminal interface | `ReasoningTextualApp`, `TUICallback` |
| `cli` | Command line | Typer commands |

Each module exposes abstract interfaces and factory functions, allowing easy extension and customization while hiding implementation details.
