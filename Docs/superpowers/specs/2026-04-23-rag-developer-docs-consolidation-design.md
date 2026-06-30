# RAG Developer Documentation Consolidation Design

Date: 2026-04-23
Topic: RAG developer documentation accuracy and consolidation
Status: Approved design

## Objective

Improve the RAG module documentation for developers and contributors by making one current, accurate contributor guide the source of truth and reducing conflicting or stale guidance across overlapping RAG docs.

The immediate problem is documentation drift: several RAG docs still describe the archived functional-pipeline shape (`functional_pipeline.py`, `rag_api.py`, `standard_pipeline`, `minimal_pipeline`, and related examples), while the active implementation is centered on `rag_unified.py`, `rag_health.py`, `UnifiedRAGRequest`, `UnifiedRAGResponse`, and `unified_pipeline.py`.

## Scope

Primary canonical document:

- `Docs/Code_Documentation/RAG-Developer-Guide.md`

Supporting docs to align or narrow:

- `Docs/API-related/RAG_API_Documentation.md`
- `Docs/API-related/RAG-API-Guide.md`
- `Docs/Code_Documentation/RAG-Functional-Pipeline-Guide.md`
- `tldw_Server_API/app/core/RAG/README.md`
- `tldw_Server_API/app/core/RAG/rag_service/README.md`
- `tldw_Server_API/app/core/RAG/API_DOCUMENTATION.md`
- `tldw_Server_API/app/core/RAG/CAPABILITIES.md`
- `tldw_Server_API/app/core/RAG/UNIFIED_PIPELINE_EXAMPLES.md`

Canonical code paths for verification:

- `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
- `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
- `tldw_Server_API/app/core/RAG/rag_service/vector_stores/`
- `tldw_Server_API/app/core/RAG/rag_service/profiles.py`

Out of scope:

- RAG code changes, unless a tiny typo blocks accurate documentation.
- Runtime behavior changes.
- Full documentation-site restructuring beyond RAG docs.
- Editing unrelated dirty worktree files.

## Approaches Considered

### Recommended: Canonical developer guide plus thin pointers

Rewrite `Docs/Code_Documentation/RAG-Developer-Guide.md` as the contributor-facing source of truth. Align adjacent docs so they have narrower responsibilities and point to the guide for implementation details.

Why this is preferred:

- gives contributors one reliable path through the module
- removes the worst stale functional-pipeline guidance
- limits churn compared with a full docs-site restructure
- reduces future drift by defining document ownership

### Alternative: Patch existing docs in place

Fix obvious inaccuracies in every file without changing ownership.

Trade-offs:

- fastest first edit
- preserves duplicated explanations
- leaves contributors with multiple sources that can drift again

### Alternative: Full RAG docs restructure

Create a new RAG docs index with separate architecture, API, extension, testing, deployment, and troubleshooting pages.

Trade-offs:

- cleanest long-term shape
- more disruptive than needed for this pass
- higher chance of scope creep

## Chosen Design

Use the recommended canonical guide plus thin pointers approach.

`Docs/Code_Documentation/RAG-Developer-Guide.md` will become the main contributor guide. It should describe the current unified architecture and avoid archived API examples except where explicitly called out as migration context.

Supporting docs should be reduced or corrected according to role:

- API docs: endpoint contracts and consumer examples.
- Core RAG README: internal module map and quick orientation.
- `rag_service/README.md`: short package-level implementation map.
- Functional Pipeline Guide: legacy/archived migration context, not an active contributor guide.
- Capabilities and examples docs: runtime discovery examples and request examples only, with terminology aligned to active schema values.

## Canonical Guide Structure

The rewritten developer guide should use this structure:

1. Current Architecture
   Active files, endpoint routers, schema models, and the unified pipeline path.
2. Request Flow
   `POST /api/v1/rag/search` through endpoint dependencies, profile/default resolution, DB adapter injection, `unified_rag_pipeline()`, response mapping, and optional batch/stream paths.
3. Core Components
   Retrieval, vector store adapters, query expansion, reranking, generation, guardrails, citations, feedback, observability, and agentic mode.
4. Extension Points
   How to add a retrieval source, vector adapter, reranker, query expansion strategy, profile, or response metadata field.
5. Configuration And Profiles
   Request defaults, `rag_profile`, env/config defaults, tenant/user DB handling, production-mode adapter expectations, and relevant safety switches.
6. Testing Guide
   Which test folders cover unit/integration/e2e behavior, plus targeted commands for schema, endpoint, retrieval, reranking, streaming, and health checks.
7. Known Pitfalls
   Stale source names, vector search needing embeddings, notes/characters sharing ChaCha DB, production raw-SQL fallback restrictions, streaming requiring `enable_generation=true`, and duplicated docs to avoid editing as source of truth.

## Validation Plan

This is a docs-only task. Validate by source inspection and consistency checks:

- confirm active endpoint list from `rag_unified.py` and `rag_health.py`
- confirm public request/response fields from `rag_schemas_unified.py`
- confirm active pipeline entrypoints from `unified_pipeline.py`
- confirm source and vector-store terminology from retriever and vector-store modules
- confirm profile names and precedence behavior from `profiles.py` and endpoint request handling
- use `rg` to catch stale active-guide references to archived names such as `functional_pipeline.py`, `rag_api.py`, `standard_pipeline`, `minimal_pipeline`, `quality_pipeline`, and `/agent`

No Bandit or RAG test-suite run is required unless implementation touches Python code.

## Success Criteria

- Developers can identify the active RAG request path from endpoint to pipeline without reading stale functional-pipeline examples.
- The canonical developer guide points to current files and current schema names.
- Adjacent docs no longer compete as full contributor guides.
- Legacy functional-pipeline material is clearly marked as legacy or replaced with current unified-pipeline guidance.
- Cross-links make document ownership obvious.
- No unrelated worktree changes are modified.
