# Unified RAG Runtime Capabilities

Use the capabilities endpoint to discover optional RAG features, defaults, limits, and deployment-specific support at runtime.

```bash
curl -s http://127.0.0.1:8000/api/v1/rag/capabilities \
  -H "X-API-KEY: your-api-key" | jq
```

Related discovery routes:

- `GET /api/v1/rag/features`
- `GET /api/v1/rag/vlm/backends`
- `GET /api/v1/rag/health/simple`
- `GET /api/v1/rag/health`

## Search Modes

Request `search_mode` accepts:

- `fts` - SQLite FTS/BM25-style full-text retrieval
- `vector` - vector similarity retrieval
- `hybrid` - combined full-text and vector retrieval

Capabilities responses should use these exact request values when describing active search modes. Clients should send only `fts`, `vector`, or `hybrid` in requests. `search_mode` is accepted globally, but vector behavior depends on the selected sources and configured vector adapters; sources without vector indexing use their source-specific retrieval path.

## Public Sources

Public request `sources` are:

- `media_db`
- `notes`
- `chats`
- `characters`
- `kanban`
- `prompts`
- `world_books`
- `dictionaries`
- `sql`

`characters` and `chats` are separate public source values. `characters` maps to character-card data, while `chats` maps to chat history. Retrieval availability remains source-specific; clients should use `metadata.source_status` from search responses to decide whether a requested source was searched, empty, or unavailable in the current deployment.

Generated, test, and workspace-scoped artifacts are excluded from normal search results unless the request includes an explicit `workspace_id` scope.

## Vector Store Support

- ChromaDB is the default vector store adapter.
- PGVector is conditional and available only when the optional import succeeds and configuration selects it.
- Other declared vector-store enum values are not active adapters unless the runtime capabilities response says they are available.

## What To Discover At Runtime

Clients should use `/capabilities` or `/features` for:

- enabled reranking strategies
- VLM/table-processing backends
- streaming support and event types
- batch limits
- cache support
- generation and citation features
- deployment limits such as `top_k`, timeouts, and token budgets

Do not treat this page as a static replacement for runtime discovery. It describes how to interpret the active discovery endpoints.
