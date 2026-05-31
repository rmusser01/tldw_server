# RAG Service Package

`rag_service/` contains internal modules that support `unified_pipeline.py`. Prefer the public API routes for application clients. Backend code that must call RAG directly should use the active entrypoints in `unified_pipeline.py`.

## Internal Flow

`HTTP request -> request_bundle.build_request_bundle -> unified_rag_pipeline (retrieval + generation) -> post_retrieval_coordinator.coordinate_standard_result_evidence -> response_mapping.rag_result_to_response`

## Active Entrypoints

- `unified_pipeline.py` - `unified_rag_pipeline(...)`
- `unified_pipeline.py` - `unified_batch_pipeline(...)`
- `unified_pipeline.py` - `simple_search(...)`
- `unified_pipeline.py` - `advanced_search(...)`

## Service Module Map

- `database_retrievers.py` - retrieval across media, notes, characters, chats, kanban, and SQL-backed sources
- `query_expansion.py` - acronym, synonym, domain, and entity expansion helpers
- `semantic_cache.py` and `advanced_cache.py` - cache support
- `advanced_reranking.py` - reranking implementations
- `vector_stores/` - ChromaDB default adapter and conditional PGVector adapter
- `citations.py` - citation formatting and source mapping
- `generation.py` - answer generation support
- `feedback_system.py` - feedback collection support
- `batch_processing.py` - batch execution support
- `health_check.py` - service health checks
- `resilience.py` - retry and circuit-breaker support
- `types.py` - shared RAG data types
- `utils.py` - small helper functions

## Documentation

- `Docs/Code_Documentation/RAG-Developer-Guide.md` - canonical developer guide
- `Docs/API-related/RAG_API_Documentation.md` - endpoint reference
- `tldw_Server_API/app/core/RAG/README.md` - module orientation
