# RAG Module README Orientation Design

Date: 2026-04-24
Topic: RAG module README accuracy and contributor orientation
Status: Approved design

## Objective

Update `tldw_Server_API/app/core/RAG/README.md` so it is accurate, current, and useful as a quick contributor orientation for the RAG module.

The README should not be the canonical deep implementation guide or the full API reference. Its job is to help a contributor landing in `app/core/RAG/` quickly understand what is active, where the main request path starts, which files matter, how the pieces fit together, and where to go next for deeper detail.

## Scope

Primary file to update:

- `tldw_Server_API/app/core/RAG/README.md`

Source files to verify against:

- `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
- `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- `tldw_Server_API/app/core/RAG/rag_service/profiles.py`
- `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
- `tldw_Server_API/app/core/RAG/rag_service/vector_stores/`
- Current RAG tests under `tldw_Server_API/tests/RAG_NEW`, `tldw_Server_API/tests/RAG`, and relevant e2e test files.

Supporting docs may be linked but should not be rewritten as part of this task unless a link target needs a tiny correction:

- `Docs/Code_Documentation/RAG-Developer-Guide.md`
- `Docs/API-related/RAG_API_Documentation.md`
- `Docs/API-related/RAG-API-Guide.md`
- `tldw_Server_API/app/core/RAG/API_DOCUMENTATION.md`
- `tldw_Server_API/app/core/RAG/CAPABILITIES.md`
- `tldw_Server_API/app/core/RAG/UNIFIED_PIPELINE_EXAMPLES.md`
- `tldw_Server_API/app/core/RAG/rag_service/README.md`

Out of scope:

- Python behavior changes.
- Endpoint/schema changes.
- Broad RAG documentation consolidation beyond this README.
- Updating unrelated dirty worktree files.
- Reworking archived functional-pipeline docs.

## Approaches Considered

### Recommended: Orientation plus curated appendix

Replace the current long README with a concise orientation page, then keep a short advanced appendix with only source-verified notes that are useful to contributors working inside `rag_service/`.

Why this is preferred:

- Preserves the README as a fast entrypoint.
- Avoids carrying forward stale examples and missing-doc links.
- Keeps advanced RAG concepts discoverable without making the README a full manual.
- Leaves canonical deep-dive and API details in dedicated docs.

### Alternative: Short orientation only

Replace the README with a very short module map and links.

Trade-offs:

- Lowest maintenance burden.
- Risks being too thin for contributors who need immediate context on streaming, guardrails, reranking, citations, and production gotchas.

### Alternative: Correct the current long README in place

Keep most sections and patch stale references.

Trade-offs:

- Least disruptive to existing document shape.
- Preserves duplication with API and developer docs.
- Higher risk of future drift because the README remains too broad.

## Chosen Design

Use the recommended orientation plus curated appendix approach.

The updated README should be concise at the top and explicit about its role. It should state that deeper implementation and API references live elsewhere, then focus on active code paths and contributor navigation.

## Proposed README Structure

1. `# RAG Module`
   - Short description of the active unified RAG module.
   - One sentence explaining that this README is a contributor orientation, not the complete API reference.

2. `## Start Here`
   - Active API router files.
   - Active schema file.
   - Active pipeline entrypoints.
   - Link to deeper developer/API docs.

3. `## Request Flow`
   - `POST /api/v1/rag/search` receives a `UnifiedRAGRequest`.
   - Endpoint dependencies apply auth/rate limits and resolve profile/default behavior.
   - User/tenant database adapters are passed into the pipeline where available.
   - `unified_rag_pipeline()` runs retrieval, optional expansion, reranking, guardrails, generation, citations, and metadata collection.
   - Endpoint maps pipeline output to `UnifiedRAGResponse`.
   - Batch, streaming, health, and capability endpoints are called out as adjacent flows.

4. `## Module Map`
   - `README.md`, `API_DOCUMENTATION.md`, `CAPABILITIES.md`, `UNIFIED_PIPELINE_EXAMPLES.md`.
   - `exceptions.py`, `rag_custom_metrics.py`.
   - `rag_service/` with a concise map of the most important implementation files.
   - `rag_service/vector_stores/` adapters.

5. `## Common Contributor Tasks`
   - Add or adjust retrieval source.
   - Add vector-store adapter.
   - Add reranking strategy.
   - Add or adjust profile/default behavior.
   - Adjust guardrails, citations, generation, claims, or streaming behavior.
   - Add tests and choose the right test location.

6. `## Current Endpoints`
   - Summarize active RAG endpoints from `rag_unified.py` and `rag_health.py`.
   - Avoid brittle source line numbers.
   - Point to API docs for full request and response examples.

7. `## Configuration And Profiles`
   - Explain request-level options, `rag_profile`, environment/config defaults, and production adapter expectations at a high level.
   - Link deeper config details instead of duplicating every knob.

8. `## Testing`
   - List targeted test folders and representative commands.
   - Keep commands limited to docs-relevant validation and focused RAG checks.

9. `## Advanced Notes`
   - Curated, source-verified notes only.
   - Include short notes on streaming, citations, claims/factuality, guardrails, reranking, vector search, and production gotchas.
   - Avoid long cURL examples and parameter catalogs that belong in API docs.

10. `## Related Documentation`
    - Link canonical deeper docs and examples.
    - Make ownership clear so future contributors know where to edit.

## Accuracy Rules

- Verify active endpoints from `rag_unified.py` and `rag_health.py`.
- Verify public request/response names from `rag_schemas_unified.py`.
- Verify pipeline entrypoints from `unified_pipeline.py`.
- Verify profile names and precedence behavior from `profiles.py` and endpoint handling.
- Verify retrieval and vector-store terminology from retriever and vector-store modules.
- Verify current test paths with `rg --files`.
- Do not preserve source line references in the README; they drift quickly.
- Do not link missing files such as `IMPLEMENTATION_STATUS.md` or `DEPRECATION_NOTICE.md` unless they exist when implementation happens.
- Mention archived functional pipelines only as legacy context, not as active contributor guidance.
- Omit, shorten, or label uncertain features instead of presenting them as guaranteed behavior.

## Validation Plan

This is a docs-only change.

Validation steps:

- Inspect active endpoint decorators in `rag_unified.py` and `rag_health.py`.
- Inspect schema, profile, pipeline, retriever, and vector-store files listed above.
- Use `rg` against the edited README for stale active-guide terms:
  - `functional_pipeline.py`
  - `rag_api.py`
  - `standard_pipeline`
  - `minimal_pipeline`
  - `quality_pipeline`
  - `IMPLEMENTATION_STATUS.md`
  - `DEPRECATION_NOTICE.md`
- Review Markdown headings and local links for readability.

No Bandit run is required because no Python code changes are planned.

## Success Criteria

- The README quickly orients contributors to the active unified RAG path.
- Stale functional-pipeline guidance is removed or clearly marked as legacy context.
- Missing-doc links and brittle line-number references are removed.
- The README points API users and deep implementation readers to the right dedicated docs.
- The advanced appendix contains only concise, source-verified notes.
- No unrelated dirty worktree files are modified.
