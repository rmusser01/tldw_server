# RAG Developer Guide

## Purpose And Ownership

This is the canonical developer and contributor guide for the current unified RAG module. It maps the active FastAPI routes, schema boundaries, pipeline orchestration, retriever setup, vector store adapters, profiles, health endpoints, and test locations that contributors need when changing RAG behavior.

Use this guide for architecture and extension decisions. Do not treat it as the schema reference. For exhaustive parameter constraints, read UnifiedRAGRequest directly or use the generated OpenAPI schema; this guide intentionally documents stable groups and extension-relevant fields rather than copying the full schema.

Primary ownership areas:

- API surface: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` and `tldw_Server_API/app/api/v1/endpoints/rag_health.py`.
- Public schemas: `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`.
- Core orchestration: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`.
- Retrieval and indexing: `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py` and `tldw_Server_API/app/core/RAG/rag_service/vector_stores/`.
- Profiles: `tldw_Server_API/app/core/RAG/rag_service/profiles.py`.
- Tests: `tldw_Server_API/tests/RAG/`, `tldw_Server_API/tests/RAG_NEW/`, RAG-related integration/e2e tests, and AuthNZ permission tests that exercise RAG access.

## Current Architecture

The active public RAG API is split between the unified query router and the health/observability router. Both use the `/api/v1/rag` prefix.

Endpoint modules:

- `rag_unified.py`: router prefix `/api/v1/rag`, tag `rag-unified`.
- `rag_health.py`: router prefix `/api/v1/rag`, tag `rag-health`.

Active unified query routes:

- `POST /ablate` -> `rag_ablate`
- `GET /capabilities` -> `get_capabilities`
- `GET /vlm/backends` -> `list_vlm_backends`
- `POST /search` -> `unified_search_endpoint`
- `POST /feedback/implicit` -> `rag_implicit_feedback`
- `POST /batch` -> `unified_batch_endpoint`
- `GET /simple` -> `simple_search_endpoint`
- `POST /batch/resume/{checkpoint_id}` -> `resume_batch_endpoint`
- `POST /search/stream` -> `unified_search_stream_endpoint`
- `GET /advanced` -> `advanced_search_endpoint`
- `GET /features` -> `list_features`
- `GET /health/simple` -> `unified_health_simple`

Active health and observability routes:

- Health: `GET /health`, `GET /health/live`, `GET /health/ready`
- Cache: `GET /cache/stats`, `POST /cache/clear`, `GET /cache/warm`
- Metrics/cost/batch visibility: `GET /metrics/summary`, `GET /costs/summary`, `GET /batch/jobs`
- Quality and regression: `POST /quality-gate`, `POST /baseline/save`, `GET /regression/check`, `POST /regression/check`

Source map:

- Schemas: `UnifiedRAGRequest`, `UnifiedRAGResponse`, `UnifiedBatchRequest`, and `UnifiedBatchResponse` live in `rag_schemas_unified.py`.
- Pipeline functions: `unified_rag_pipeline`, `unified_batch_pipeline`, `simple_search`, and `advanced_search` live under `rag_service/`.
- Result mapping: `UnifiedSearchResult` is the pipeline result carrier; `rag_unified.py::convert_result_to_response` maps it into `UnifiedRAGResponse`.
- Retrieval: `database_retrievers.py` owns `DataSource`, retriever classes, `RetrievalConfig`, and `MultiDatabaseRetriever`.
- Source normalization: `core/Text2SQL/source_registry.py` normalizes public aliases before schemas and endpoints hand sources to retrievers.
- Vector stores: `rag_service/vector_stores/base.py`, `chromadb_adapter.py`, `pgvector_adapter.py`, `factory.py`, and `__init__.py`.
- Profiles: `profiles.py` defines built-in profile defaults and merge helpers.
- Health/observability: `rag_health.py`, metrics/cost/cache helpers under `rag_service/`, and regression/quality gate tests.
- Tests: older `tests/RAG/` coverage remains relevant for source behavior and feedback; current unified API and pipeline coverage is concentrated in `tests/RAG_NEW/`.

Public request sources advertised by `UnifiedRAGRequest` are `media_db`, `notes`, `characters`, `chats`, `kanban`, and `sql`. The canonical source registry also supports internal normalized sources `prompts` and `claims`. `characters` and `chats` are routed to the character-card/ChaChaNotes-backed retriever path in the current inspected setup. `DataSource` includes `CHAT_HISTORY` and `WEB_CONTENT`, but those enum values are not active `MultiDatabaseRetriever` keys in the inspected retriever setup; do not document or expose them as fully wired public sources without adding retriever wiring and tests.

Vector store support is adapter-based. `VectorStoreType` declares `chromadb`, `pinecone`, `weaviate`, `qdrant`, `milvus`, `faiss`, and `pgvector`, but only `ChromaDBAdapter` is registered by default. `PGVectorAdapter` is conditionally registered if its import succeeds. Treat the other enum values as declared/future types until a concrete adapter is implemented, registered, and tested.

## Request Flow

The standard search request path is:

`POST /api/v1/rag/search -> dependencies -> UnifiedRAGRequest validation/profile/default resolution -> per-user DB adapter injection -> unified_rag_pipeline(...) -> convert_result_to_response(...) -> UnifiedRAGResponse`

The endpoint layer is responsible for request validation, authentication dependencies, user-scoped database adapter selection, source normalization, and profile/default application. The pipeline layer should receive an explicit payload and dependencies, then return `UnifiedSearchResult` without knowing about FastAPI response models.

Important route variants:

- `/search/stream` requires `enable_generation=true`; the endpoint rejects streaming search when generation is disabled.
- `/batch` and `/batch/resume/{checkpoint_id}` use `UnifiedBatchRequest` and the batch pipeline/checkpoint path, not repeated ad hoc calls from client code.
- `/simple` is a GET wrapper around `simple_search` for lightweight retrieval.
- `/advanced` is a GET wrapper around `advanced_search` for richer retrieval options.
- `/capabilities` and `/features` are discovery endpoints. Keep them in sync when adding externally visible features.
- Health, cache, metrics, cost, quality gate, baseline, and regression routes live in `rag_health.py`, not in the unified search router.

## Core Components

`rag_unified.py` is the FastAPI boundary. It owns route declarations, request dependency handling, profile/default resolution, batch/resume request handling, streaming checks, and response conversion. Keep endpoint logic focused on transport concerns and compatibility translation.

`rag_schemas_unified.py` is the public schema boundary. `UnifiedRAGRequest` groups query text, sources, strategy/profile selection, retrieval controls, cache controls, reranking controls, generation controls, advanced retrieval options, observability/debug flags, and evaluation metrics inputs. `UnifiedRAGResponse` groups retrieved documents, query metadata, timings, citations, feedback identifiers, answer/generation output, cache state, and retrieval metrics. Batch schemas mirror the same knobs at batch granularity.

`unified_pipeline.py` is the orchestration boundary. It coordinates retrieval, optional query expansion, optional reranking, optional generation, metadata/timing collection, and batch execution. Pipeline additions should surface through explicit request fields, profile defaults, or internal kwargs instead of implicit global behavior.

`database_retrievers.py` is the retrieval source boundary. `MultiDatabaseRetriever` configures available retrievers from database paths and injected adapters. Current active keys include media, notes, character cards for `characters`/`chats`, kanban, SQL when an SQL retriever is provided, and internal prompts/claims when their DB paths are present.

`rag_service/vector_stores/` is the vector abstraction boundary. `base.py` defines the adapter contract and shared config/result types. `factory.py` selects and registers concrete adapters. Adapter modules should translate the common interface to backend-specific collection, upsert, delete, and search operations.

`advanced_reranking.py` owns reranking implementations and strategy selection. The request schema exposes reranking enablement, strategy, model override, rerank count, and relevance thresholds; the pipeline decides when to apply them.

`query_expansion.py` owns query expansion behavior. Keep expansion strategies isolated there and wire new controls through the schema/profile/pipeline path.

`profiles.py` owns named presets. Built-in profiles are `production`, `research`, `cheap`, `fast`, `balanced`, and `accuracy`. Public request-time `rag_profile` currently accepts `fast`, `balanced`, and `accuracy`; lower-level profile helpers can build kwargs for all registered profiles.

## Extension Points

Retrieval source:

1. Add or extend the retriever implementation in `rag_service/database_retrievers.py`.
2. Add the source to `DataSource` only if a retriever will be wired in `MultiDatabaseRetriever`.
3. Add alias/public or internal normalization in `core/Text2SQL/source_registry.py`.
4. Update endpoint source normalization and any discovery payloads in `rag_unified.py`.
5. Add unit tests for the retriever and endpoint/integration tests for source selection and authorization.

Vector store:

1. Implement `VectorStoreAdapter` from `rag_service/vector_stores/base.py`.
2. Put backend-specific code in a dedicated adapter module.
3. Register the adapter in `rag_service/vector_stores/factory.py`.
4. Add factory tests, adapter contract tests, and any backend-specific integration tests behind appropriate fixtures.
5. Do not imply support for a declared `VectorStoreType` until the adapter is registered and covered.

Reranker:

1. Add strategy-specific implementation in `advanced_reranking.py`.
2. Wire strategy selection through the pipeline.
3. Expose only stable knobs in `UnifiedRAGRequest` and batch schema.
4. Add tests for ordering, truncation, score metadata, and failure fallback.

Query expansion:

1. Add expansion logic in `query_expansion.py`.
2. Keep strategy-specific dependencies optional and testable.
3. Wire the strategy through profile defaults or explicit request fields.
4. Verify expanded query metadata is preserved in `UnifiedSearchResult` and response mapping.

Profile:

1. Add or modify profile defaults in `profiles.py`.
2. Ensure endpoint profile application preserves explicit user-supplied fields over profile defaults.
3. Update `/capabilities` or `/features` only when the profile becomes a supported public contract.
4. Add focused tests for profile metadata and override behavior.

Response metadata:

1. Add pipeline-level metadata to `UnifiedSearchResult`.
2. Map the metadata in `convert_result_to_response`.
3. Add the response field to `UnifiedRAGResponse` only when clients need a stable typed contract; otherwise prefer namespaced `metadata`.
4. Test both direct pipeline output and API response mapping.

## Configuration And Profiles

Configuration enters RAG from three places: request fields, profile defaults, and application settings. Endpoint logic applies precedence in this order: explicit request value, profile default, search/default helper value, schema default.

Stable request field groups:

- Query and source selection: `query`, `sources`, `sql_target_id`, `corpus`, and `index_namespace`.
- Strategy and profiles: `strategy` and `rag_profile`.
- Retrieval knobs: `top_k`, score thresholds, hybrid/FTS/vector choices, filters, temporal/selection controls, and advanced retrieval feature toggles.
- Cache knobs: `enable_cache`, `cache_threshold`, `adaptive_cache`, semantic cache and agentic cache controls.
- Reranking knobs: `enable_reranking`, `reranking_strategy`, `rerank_top_k`, `reranking_model`, and relevance thresholds.
- Generation knobs: `enable_generation`, model/provider options, answer style, citation/grounding controls, pre-retrieval clarification, post-verification, numeric fidelity, and strict extractive behavior.
- Observability and evaluation knobs: debug flags, monitoring, tracing, retrieval metrics inputs, and quality gate inputs.

Built-in profiles:

- `production`: conservative defaults for hybrid retrieval, cache, reranking, and guardrails.
- `research`: broader experimental retrieval and verification features.
- `cheap`: lower cost and latency with fewer expensive extras.
- `fast`: latency-first public preset.
- `balanced`: public preset balancing latency and retrieval quality.
- `accuracy`: quality-first public preset with stronger retrieval/reranking.

When changing profile defaults, check both the direct helpers in `profiles.py` and endpoint application in `rag_unified.py`. The public schema currently constrains `rag_profile` to `fast`, `balanced`, and `accuracy`, so exposing `production`, `research`, or `cheap` through the request model is a separate API compatibility decision.

## Testing Guide

Use the project virtual environment before running Python tests:

```bash
source .venv/bin/activate
```

Focused RAG test entry points:

```bash
python -m pytest tldw_Server_API/tests/RAG_NEW/unit -v
python -m pytest tldw_Server_API/tests/RAG_NEW/integration -v
python -m pytest tldw_Server_API/tests/RAG -v
```

Run narrower tests while iterating:

- Schema/profile behavior: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_request_schema_profiles.py`, `test_rag_profiles.py`, and `test_unified_pipeline_profile_metadata.py`.
- Response mapping: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_response_mapping.py`.
- Pipeline behavior: `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline.py`, `test_unified_pipeline_focused.py`, and related feature-specific tests.
- Vector stores: `tldw_Server_API/tests/RAG_NEW/unit/test_vector_store_parity.py`, `test_vector_store_list_vectors.py`, and integration tests for PGVector when available.
- Endpoints: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_integration.py`, `test_rag_stream_parity.py`, `test_rag_batch_resume_api.py`, `test_rag_batch_checkpoint_api.py`, `test_rag_capabilities_styles.py`, `test_rag_unified_features_endpoint.py`, and `test_rag_health_endpoints.py`.
- Source and SQL hardening: `tldw_Server_API/tests/RAG/test_rag_sources_sql_validation.py`, `test_sql_retriever_hardening.py`, and SQL integration tests.
- Auth and permissions: `tldw_Server_API/tests/AuthNZ/integration/test_rag_media_permissions_claims.py`, `test_auth_principal_media_rag_invariants.py`, and SQLite equivalents.

For API changes, include at least one endpoint-level test and one lower-level test near the component you changed. For new retrieval sources or vector stores, test the contract in isolation and then test the endpoint path that normalizes and activates it.

Before considering RAG code changes complete, run Bandit on touched Python paths as required by the repository security policy. Documentation-only changes normally do not need Bandit unless they include executable examples that are copied into source files.

## Known Pitfalls

- Do not add public source names only to `DataSource`; they also need normalization, retriever registration, endpoint discovery updates, and tests.
- Do not present `CHAT_HISTORY` or `WEB_CONTENT` as public wired sources until `MultiDatabaseRetriever` has active retrievers for them.
- `characters` and `chats` are public source aliases, but the current retriever setup maps them through character-card/ChaChaNotes-backed retrieval rather than a distinct chat-history retriever key.
- Declared vector store enum values are not equivalent to implemented adapter support. ChromaDB is the default adapter; PGVector is optional; other declared values need adapter work.
- `/search/stream` is generation-only. Retrieval-only callers should use `/search`, `/simple`, or `/advanced`.
- Profile defaults must not override explicit user request fields.
- Avoid putting transport concerns inside `unified_rag_pipeline`; keep FastAPI dependencies and response model conversion in endpoint code.
- Keep response metadata names stable. Prefer namespaced metadata for experimental details until the contract is ready for typed schema fields.
- Health and observability endpoints are split into `rag_health.py`; adding health-like routes to `rag_unified.py` makes the API harder to discover.

## Related Documentation

- `tldw_Server_API/app/core/RAG/README.md`: deeper module notes and examples for the current RAG service.
- `tldw_Server_API/app/core/RAG/UNIFIED_PIPELINE_EXAMPLES.md`: request and pipeline examples.
- `Docs/Published/User_Guides/Server/RAG_Production_Configuration_Guide.md`: operator-facing configuration guidance.
- `Docs/RAG/`: research notes, benchmark manifests, and design background.
- `Docs/Code_Documentation/Embeddings-Developer-Guide.md`: embedding subsystem details used by vector retrieval.
- Generated OpenAPI schema: authoritative endpoint and field constraints for clients.
