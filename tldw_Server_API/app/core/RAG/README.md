# RAG Module

This package contains the backend implementation for unified Retrieval-Augmented Generation in `tldw_server`.

For canonical contributor guidance, see `Docs/Code_Documentation/RAG-Developer-Guide.md`.

## Active Entrypoints

Programmatic entrypoints:

- `rag_service/unified_pipeline.py` - `unified_rag_pipeline(...)`
- `rag_service/unified_pipeline.py` - `unified_batch_pipeline(...)`
- `rag_service/unified_pipeline.py` - `simple_search(...)`
- `rag_service/unified_pipeline.py` - `advanced_search(...)`

HTTP entrypoints:

- `app/api/v1/endpoints/rag_unified.py` - active `/api/v1/rag` unified routes
- `app/api/v1/endpoints/rag_health.py` - active `/api/v1/rag` health and operations routes
- `app/api/v1/schemas/rag_schemas_unified.py` - request and response schemas

## Module Map

- `rag_service/unified_pipeline.py` - unified request execution, batch execution, convenience wrappers
- `rag_service/database_retrievers.py` - datastore retrieval adapters and source fan-out
- `rag_service/advanced_reranking.py` - reranking strategies
- `rag_service/query_expansion.py` - query expansion strategies
- `rag_service/vector_stores/` - vector store factory and adapters
- `rag_service/citations.py` - citation helpers
- `rag_service/generation.py` - answer generation helpers
- `rag_service/health_check.py` - health checks used by operations routes
- `exceptions.py` - RAG-specific exceptions

## Public Request Surface

Public `search_mode` values:

- `fts`
- `vector`
- `hybrid`

Public request `sources`:

- `media_db`
- `notes`
- `characters`
- `chats`
- `kanban`
- `sql`

ChromaDB is the default vector store adapter. PGVector is available only when its optional import succeeds and the deployment is configured for it.

## Related Documentation

- `Docs/Code_Documentation/RAG-Developer-Guide.md` - canonical developer and contributor guide
- `Docs/API-related/RAG_API_Documentation.md` - concise endpoint reference
- `Docs/API-related/RAG-API-Guide.md` - consumer examples
- `tldw_Server_API/app/core/RAG/CAPABILITIES.md` - runtime capability discovery notes
