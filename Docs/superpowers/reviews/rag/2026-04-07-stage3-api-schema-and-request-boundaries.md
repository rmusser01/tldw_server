# Stage 3 API Schema and Request Boundaries

## Scope

Review the RAG endpoint, schema, profile-default, and request-mapping boundary files for ownership drift across defaults, payload construction, and response mapping.

## Code Paths Reviewed

- Reviewed: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Reviewed: `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
- Reviewed: `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- Reviewed: `tldw_Server_API/app/api/v1/schemas/rag_schemas_simple.py`
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/profiles.py`
- Reviewed: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- Reviewed: `tldw_Server_API/app/api/v1/utils/rag_cache.py`
- Reviewed: `tldw_Server_API/app/core/Embeddings/services/jobs_worker.py`

Boundary map:
- `UnifiedRAGRequest` is the main request contract and owns schema-time aliasing and validation for `rag_profile`, `corpus` -> `index_namespace`, `min_relevance_score` -> `min_score`, source normalization, and chunk overlap validation.
- `_apply_search_agent_defaults()`, `_apply_rag_profile_defaults()`, `_build_effective_request_payload()`, and `_build_unified_pipeline_kwargs()` in `rag_unified.py` are the effective default-resolution boundary for the standard `/search` path and the retrieval half of `/search/stream`.
- `profiles.py` owns the switchable profile presets; endpoint code decides when those presets are actually applied.
- `convert_result_to_response()` in `rag_unified.py` owns API response shaping from `UnifiedSearchResult` to `UnifiedRAGResponse`.
- `rag_health.py` is a separate reporting boundary that instantiates cache and batch-monitoring singletons directly in the API layer.
- `rag_cache.py` is nominally an API utility, but it reaches into semantic cache, agentic chunk vectors, vector-store factories, and collection naming.

## Tests Reviewed

- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_response_mapping.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_request_schema_profiles.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_profiles.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_batch_round2_flags.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_health_endpoints.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_unified_features_endpoint.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_capabilities_styles.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py`

## Validation Commands

- Mapping command:
  - `rg -n "@router\\.|def _|model_validator|field_validator|rag_profile|get_profile_kwargs|invalidate_rag_caches|delete_media_vectors|UnifiedRAGRequest|UnifiedRAGResponse|UnifiedBatchRequest|ImplicitFeedbackEvent" tldw_Server_API/app/api/v1/endpoints/rag_unified.py tldw_Server_API/app/api/v1/endpoints/rag_health.py tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py tldw_Server_API/app/api/v1/schemas/rag_schemas_simple.py tldw_Server_API/app/core/RAG/rag_service/profiles.py tldw_Server_API/app/api/v1/utils/rag_cache.py`
- Supporting evidence command for findings 3 and 4:
  - `rg -n "verification_report|result\\.metadata\\[\\\"verification_report\\\"\\]|from tldw_Server_API\\.app\\.api\\.v1\\.utils\\.rag_cache import invalidate_rag_caches|invalidate_rag_caches\\(" tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py tldw_Server_API/app/core/Embeddings/services/jobs_worker.py`
- Result:
  - `unified_pipeline.py` matched the `generate_verification_report` gate and `result.metadata["verification_report"]` write path; `jobs_worker.py` matched the API-layer import plus the three invalidation call sites cited in finding 4.
- Targeted tests:
  - `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_response_mapping.py tldw_Server_API/tests/RAG_NEW/unit/test_rag_request_schema_profiles.py tldw_Server_API/tests/RAG_NEW/unit/test_rag_profiles.py tldw_Server_API/tests/RAG_NEW/unit/test_batch_round2_flags.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_health_endpoints.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_unified_features_endpoint.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_capabilities_styles.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py -v`
- Result:
  - `37 passed, 612 warnings in 133.84s (0:02:13)`
- Docs-scope Bandit:
  - `source ../../.venv/bin/activate && python -m bandit -r Docs/superpowers/reviews/rag -f json -o /tmp/bandit_rag_stage3.json`
- Result:
  - `JSON output written to /tmp/bandit_rag_stage3.json; jq confirmed results=0 and errors=0`

## Findings

1. High: non-streaming agentic `/search` bypasses the central request-default boundary, so the advertised precedence only holds for the standard path and the streaming path, not for agentic search requests.
   - In the standard path, precedence is explicit request fields > profile defaults > Search-Agent defaults > schema defaults via `_build_effective_request_payload()` and `_build_unified_pipeline_kwargs()` in `tldw_Server_API/app/api/v1/endpoints/rag_unified.py:153-289`.
   - In the agentic branch of `/api/v1/rag/search`, the endpoint builds `AgenticConfig` and `agentic_rag_pipeline(...)` directly from raw `request` attributes instead of the resolved payload, in `tldw_Server_API/app/api/v1/endpoints/rag_unified.py:1204-1277`.
   - That means `rag_profile` defaults from `tldw_Server_API/app/core/RAG/rag_service/profiles.py:196-255` and Search-Agent defaults from `tldw_Server_API/app/api/v1/endpoints/rag_unified.py:153-209` do not shape the non-streaming agentic request, even though they do shape standard `/search` and streaming retrieval.
   - The targeted suite proves the precedence helper itself is correct (`tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py:266-312`) and, on the streaming agentic path, specifically proves that `rag_profile="fast"` propagates the profile-resolved `top_k=6` into the downstream `agentic_rag_pipeline` call (`tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py:472-538`). There is still no direct test of non-streaming agentic parity. This leaves two effective policies for the same public request schema.

2. Medium: the batch request schema is a second source of truth for the API contract and has already drifted from the single-request contract and capability surface.
   - `UnifiedBatchRequest` repeats a large subset of `UnifiedRAGRequest` instead of reusing a shared request model in `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py:1788-1935`.
   - The endpoint description says batch exposes "all parameters from the single search endpoint," but batch omits `rag_profile` entirely and therefore cannot participate in the same profile-default boundary as single search. The single-request schema exposes `rag_profile` at `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py:75-85`; the batch schema does not.
   - Batch also diverges on concrete literals and defaults. Example: single-request `citation_style` allows `ieee` at `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py:953`, capabilities advertises lowercase styles including `ieee` in `tldw_Server_API/app/api/v1/endpoints/rag_unified.py:844-847`, but batch only allows `apa|mla|chicago|harvard` at `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py:1905-1908`.
   - Source defaults also differ without an obvious ownership rule: single request defaults to `["media_db"]` at `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py:63-68`, while batch defaults to `["media_db", "notes", "characters"]` at `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py:1812-1817`.
   - Existing tests cover only the newer round-2 flags and their forwarding (`tldw_Server_API/tests/RAG_NEW/unit/test_batch_round2_flags.py:10-56`, `tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py:231-264`). They do not protect full single-vs-batch contract parity.

3. Medium: response mapping duplicates internal metadata knowledge in the endpoint and already misses at least one declared response field.
   - `UnifiedRAGResponse` declares first-class response fields such as `research_summary`, `suggestions`, `images`, `videos`, and `verification_report` in `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py:1686-1747`.
   - `convert_result_to_response()` re-derives those fields from `UnifiedSearchResult.metadata` keys such as `"research"`, `"suggestions"`, `"images"`, `"videos"`, `"academic_citations"`, `"chunk_citations"`, `"retrieval_metrics"`, and `"faithfulness"` in `tldw_Server_API/app/api/v1/endpoints/rag_unified.py:497-587`.
   - This duplicates internal pipeline metadata naming at the API boundary instead of consuming a stable result model. The duplication is partial: `verification_report` is declared in the response schema but never populated by `convert_result_to_response()`, even though the pipeline stores it in `result.metadata["verification_report"]` in `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py:5714-5726`.
   - The current tests only protect the round-2 response keys (`tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_response_mapping.py:11-76`). There is no targeted coverage for `verification_report`, `academic_citations`, `chunk_citations`, or other schema-level response fields that depend on endpoint-owned metadata mapping.

4. Medium: cache invalidation ownership is misplaced in an API utility and already leaks back into core jobs code.
   - `tldw_Server_API/app/api/v1/utils/rag_cache.py:35-116` owns vector-store deletion, semantic-cache clearing, agentic intra-doc vector invalidation, and the collection naming convention `user_{user_id}_media_embeddings`.
   - That helper is not just used by API endpoints. Core worker code imports the API utility directly in `tldw_Server_API/app/core/Embeddings/services/jobs_worker.py:42-50` and calls it during embeddings jobs at `tldw_Server_API/app/core/Embeddings/services/jobs_worker.py:713-715`, `802-804`, and `940-941`.
   - This is a boundary inversion: a module under `app/api/v1/utils` now owns RAG cache semantics for non-API callers, and it embeds core-RAG assumptions such as semantic-cache namespaces and agentic chunk invalidation (`tldw_Server_API/app/api/v1/utils/rag_cache.py:93-115`).
   - `rag_health.py` is related but lower risk: it instantiates cache and batch-monitoring singletons inside the endpoint module (`tldw_Server_API/app/api/v1/endpoints/rag_health.py:29-47`) and exposes shape-level operational data, but the reviewed tests only assert JSON shape and status classes, not ownership boundaries (`tldw_Server_API/tests/RAG_NEW/integration/test_rag_health_endpoints.py:44-98`).

## Suggested Refactor/Actions

- Move all request-default resolution behind one boundary helper that every `/search` mode uses, including non-streaming agentic and batch. The endpoint should consume a resolved request object or resolved kwargs map, not reconstruct agentic config from raw request attributes.
- Collapse single-request and batch-request shared fields into one reusable request-shaping layer. If batch intentionally differs, document that explicitly in schema descriptions and capability docs instead of claiming full parity.
- Replace endpoint-owned metadata unpacking with a typed internal response/result contract. At minimum, define one canonical mapping point for `UnifiedSearchResult` -> API response fields and add coverage for every first-class response field declared in `UnifiedRAGResponse`.
- Move `invalidate_rag_caches()` and `delete_media_vectors()` out of `app/api/v1/utils` into a core RAG invalidation service. API endpoints and jobs workers should both depend on that core service rather than on an API utility module.
- Keep `rag_health.py` focused on transport and authorization. Long-lived singleton construction and cache recommendation policy should be injected from core services if this surface keeps growing.

## Coverage Gaps

- `test_rag_unified_search_agent_defaults.py`
  - Protects explicit > profile > Search-Agent > schema precedence at the helper level, plus env-over-config precedence and user-id normalization.
  - Boundary depth: helper-level only; it does not exercise `/api/v1/rag/search` end-to-end.
  - Missing negative case: non-streaming agentic `/search` does not have a parity test proving it uses the same precedence rules.
- `test_rag_unified_response_mapping.py`
  - Protects mapping of round-2 metadata keys into first-class response fields.
  - Boundary depth: direct endpoint helper test.
  - Missing negative case: no test for `verification_report` mapping, or for mismatch between declared response fields and metadata-owned keys.
- `test_rag_request_schema_profiles.py`
  - Protects allowed `rag_profile` literals for `UnifiedRAGRequest` and the raised `max_generation_tokens` bound.
  - Boundary depth: schema-direct.
  - Missing negative case: no batch-schema parity test for `rag_profile` because batch does not expose it.
- `test_rag_profiles.py`
  - Protects profile registration, concrete design-target defaults, override merging, and one pipeline-facing retrieval mapping.
  - Boundary depth: profiles module and one mocked pipeline interaction.
  - Missing negative case: no test that profile defaults are applied consistently by every public endpoint path.
- `test_batch_round2_flags.py`
  - Protects acceptance and forwarding of round-2 flags through batch processing.
  - Boundary depth: schema-direct plus mocked pipeline forwarding.
  - Missing negative case: no contract-parity coverage for batch omissions such as `rag_profile`, `fts_level`, `sql_target_id`, or `citation_style="ieee"`.
- `test_rag_health_endpoints.py`
  - Protects health/readiness/cache-stats endpoint status and broad JSON shape.
  - Boundary depth: direct HTTP integration with auth override.
  - Missing negative case: no assertions about singleton ownership, cache recommendation policy, or degraded/unhealthy component composition beyond shape.
- `test_rag_unified_features_endpoint.py`
  - Protects that `list_features()` includes clarification and research-action-dedup parameters.
  - Boundary depth: direct function call, not HTTP.
  - Missing negative case: no parity check against the real request schemas, so stale feature lists can still pass if they only contain the targeted strings.
- `test_rag_capabilities_styles.py`
  - Protects lowercase citation styles in `/api/v1/rag/capabilities`.
  - Boundary depth: direct HTTP integration.
  - Missing negative case: no comparison against batch schema literals, which already diverge.
- `test_rag_stream_parity.py`
  - Protects streaming retrieval config forwarding, generation-provider/model resolution, research-progress event ordering, claim-overlay event preservation, and profile-default effects on streaming generation and streaming agentic top-k.
  - Boundary depth: HTTP integration with patched pipeline/generation components.
  - Missing negative case: no equivalent non-streaming `/search` agentic parity test, and no test that streaming/non-streaming responses expose the same declared first-class metadata fields.

## Exit Note

Stage 3 confirms the request-boundary hotspot is not the schema alone; it is the combination of `UnifiedRAGRequest`, endpoint-side default application, and endpoint-owned result mapping. Stage 4 should not re-audit this layer. It should instead settle retrieval-side questions that remain after boundary shaping is done: where retrieval policy actually becomes fixed inside `unified_pipeline.py`, whether retrieval inputs are normalized once or repeatedly, and which retrieval outputs are authoritative before the API response mapper touches them.
