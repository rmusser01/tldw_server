# RAG Module

This README is a contributor orientation for the backend RAG module. It is not the full API reference or a complete implementation guide; use it to find the active code paths and then follow the linked docs or source files for detail.

## Start Here

Active entrypoints:

- API routers: `rag_unified.py`, `rag_health.py`
- Public schemas: `UnifiedRAGRequest`, `UnifiedRAGResponse`, `UnifiedBatchRequest`, `UnifiedBatchResponse`
- Core pipeline: `unified_rag_pipeline()`, `unified_batch_pipeline()`
- Convenience functions: `simple_search()`, `advanced_search()`

Primary source paths:

- Routers: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` and `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
- Schemas: `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- Pipeline: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`

## Request Flow

Primary search flow:

```text
POST /api/v1/rag/search
  -> UnifiedRAGRequest validation
  -> auth/rate-limit/permission dependencies
  -> profile/default resolution
  -> per-user DB adapter/path injection
  -> unified_rag_pipeline(...)
  -> UnifiedRAGResponse mapping
```

The endpoint layer handles FastAPI concerns: auth dependencies, request validation, source normalization, profile/default application, user-scoped database adapters, and response conversion. The pipeline layer receives explicit inputs and returns the search result carrier used to build `UnifiedRAGResponse`.

Adjacent paths share the same module boundary. `POST /api/v1/rag/batch` uses `UnifiedBatchRequest` and `unified_batch_pipeline()`, `POST /api/v1/rag/search/stream` streams generated answer events, `GET /api/v1/rag/capabilities` and `GET /api/v1/rag/features` expose discovery data, and health/operations routes live in `rag_health.py`.

## Module Map

- `API_DOCUMENTATION.md`: in-module pointer to the maintained API reference in `Docs/API-related/RAG_API_Documentation.md`.
- `CAPABILITIES.md`: feature discovery and capability summary.
- `UNIFIED_PIPELINE_EXAMPLES.md`: request and pipeline examples to verify against current schema.
- `exceptions.py`: RAG-specific exception types.
- `rag_custom_metrics.py`: RAG metrics helpers.
- `rag_service/`: implementation package for retrieval, generation, guardrails, citations, profiles, vector stores, metrics, and utilities.

Inside `rag_service/`, start with `unified_pipeline.py` for orchestration, `database_retrievers.py` for source retrieval, `advanced_reranking.py` for reranking, `query_expansion.py` for expansion, `profiles.py` for presets, and `vector_stores/` for vector adapter contracts and implementations.

## Common Contributor Tasks

- Add retrieval behavior: `rag_service/database_retrievers.py` and the source registry, then update discovery output and tests.
- Add vector-store support: `rag_service/vector_stores/` by implementing the shared adapter contract and registering the adapter in the factory.
- Adjust profiles/defaults: `rag_service/profiles.py` plus endpoint payload handling in `rag_unified.py`.
- Adjust request/response contract: `rag_schemas_unified.py` and endpoint response mapping.
- Adjust reranking: `rag_service/advanced_reranking.py`.
- Adjust generation/streaming: `rag_service/generation.py`, `rag_service/response_writer.py`, and `POST /api/v1/rag/search/stream`.
- Adjust guardrails/citations/claims: `guardrails.py`, `citations.py`, `claims.py`, `faithfulness.py`, and related tests.
- Adjust route behavior in `rag_unified.py` or `rag_health.py`; keep transport, dependency, and response conversion at the endpoint boundary.

## Current Endpoints

Search and generation routes:

- `POST /api/v1/rag/search`: primary unified RAG search.
- `POST /api/v1/rag/search/stream`: NDJSON streaming search for generated answers.
- `POST /api/v1/rag/batch`: concurrent multi-query RAG search.
- `POST /api/v1/rag/batch/resume/{checkpoint_id}`: resume a checkpointed batch run.
- `GET /api/v1/rag/simple`: lightweight convenience search.
- `GET /api/v1/rag/advanced`: convenience search with common advanced options enabled.

Discovery routes:

- `GET /api/v1/rag/capabilities`: runtime capability, default, and limit discovery.
- `GET /api/v1/rag/features`: feature groups and related request parameter names.
- `GET /api/v1/rag/vlm/backends`: VLM/table-processing backend availability.

Feedback and experimentation routes:

- `POST /api/v1/rag/feedback/implicit`: implicit interaction feedback capture.
- `POST /api/v1/rag/ablate`: compare retrieval/generation variants for one query.

Health routes:

- `GET /api/v1/rag/health`, `GET /api/v1/rag/health/live`, `GET /api/v1/rag/health/ready`
- `GET /api/v1/rag/health/simple`: lightweight unified-pipeline health check.

Operations routes:

- `GET /api/v1/rag/cache/stats`, `POST /api/v1/rag/cache/clear`, `GET /api/v1/rag/cache/warm`
- `GET /api/v1/rag/metrics/summary`, `GET /api/v1/rag/costs/summary`, `GET /api/v1/rag/batch/jobs`
- `POST /api/v1/rag/quality-gate`, `POST /api/v1/rag/baseline/save`, `GET /api/v1/rag/regression/check`, `POST /api/v1/rag/regression/check`

## Configuration And Profiles

Configuration enters through request fields, profile defaults, and application settings. Endpoint resolution follows this precedence: explicit request value, profile default, search/default helper value, then schema default.

Public `search_mode` values are `fts`, `vector`, and `hybrid`. Public request `sources` are `media_db`, `notes`, `chats`, `characters`, `kanban`, `prompts`, `world_books`, `dictionaries`, and `sql`.

Public request `rag_profile` values are `fast`, `balanced`, and `accuracy`. `rag_service/profiles.py` also contains internal helper profiles such as `production`, `research`, and `cheap`; do not assume they are accepted directly by `UnifiedRAGRequest`.

Explicit request fields override profile defaults. `index_namespace` can be set directly or via `corpus`; endpoint payload handling maps the alias before calling the pipeline. Production deployments should prefer injected DB adapters and resolved per-user paths over raw fallback assumptions.

ChromaDB is the default vector store adapter. PGVector is conditional on optional import and configuration. Treat other declared vector-store types as unavailable until an adapter is implemented, registered, and tested.

## Testing

For this README, verify the heading structure and confirm no retired RAG guides, old router names, or superseded pipeline names are referenced before committing. The heading check should show only the orientation sections in this file.

For RAG code changes, activate the project virtual environment first:

```bash
source .venv/bin/activate
```

Focused RAG test entry points:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_rag_request_schema_profiles.py -v
python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_rag_profiles.py -v
python -m pytest tldw_Server_API/tests/RAG_NEW/integration/test_rag_health_endpoints.py -v
python -m pytest tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py -v
python -m pytest tldw_Server_API/tests/RAG/test_rag_sources_sql_validation.py -v
```

Broader RAG suites:

```bash
python -m pytest tldw_Server_API/tests/RAG_NEW/unit -v
python -m pytest tldw_Server_API/tests/RAG_NEW/integration -v
python -m pytest tldw_Server_API/tests/RAG -v
```

Use narrower tests while iterating: schema/profile tests for request changes, response-mapping tests for endpoint conversion, pipeline tests for orchestration changes, vector-store tests for adapter work, and AuthNZ permission tests when RAG access control changes.

## Advanced Notes

`strategy` currently selects between `standard` and `agentic` request handling. The agentic branch builds a query-time synthetic chunk path, while the default path builds explicit pipeline kwargs and calls `unified_rag_pipeline()`.

Streaming search requires `enable_generation=true`; the endpoint rejects streaming requests without generation enabled. Batch search and batch resume use the batch schema/checkpoint path rather than requiring clients to loop over single-search calls.

Vector search depends on embeddings and vector-store availability. Hybrid search can still use FTS when vector retrieval is unavailable or disabled.

Citation, hard-citation, claim, numeric fidelity, and post-verification features may abstain or ask for clarification when evidence is weak. Reranking options carry cost and latency tradeoffs; `two_tier` reranking is quality-oriented because it combines a fast pass with LLM scoring.

Multi-tenant or production usage should avoid assumptions about raw SQL fallbacks and should preserve user and tenant scoping through injected adapters, paths, and permissions.

The public source values are normalized before retrieval: `media_db`, `notes`, `chats`, `characters`, `kanban`, `prompts`, `world_books`, `dictionaries`, and `sql`. `characters` and `chats` are separate public source values; `characters` maps to character-card data, while `chats` maps to chat history. `metadata.source_status` reports each requested source as `searched`, `empty`, or `unavailable` with counts and clear reasons. Generated, test, and workspace-scoped artifacts are filtered from global retrieval unless `workspace_id` is provided.

## Related Documentation

- `Docs/Code_Documentation/RAG-Developer-Guide.md`: primary deeper contributor guide for architecture, extension points, profiles, and testing.
- `Docs/API-related/RAG_API_Documentation.md`: maintained endpoint reference.
- `Docs/API-related/RAG-API-Guide.md`: consumer examples and client-oriented usage notes.
- `tldw_Server_API/app/core/RAG/API_DOCUMENTATION.md`: in-module pointer to the maintained API reference.
- `tldw_Server_API/app/core/RAG/CAPABILITIES.md`: runtime capability interpretation notes.
- `tldw_Server_API/app/core/RAG/UNIFIED_PIPELINE_EXAMPLES.md`: request and programmatic examples to compare with current schemas.
