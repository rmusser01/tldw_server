# Stage 6 Test Gaps and Synthesis

## Scope

Consolidate cross-stage findings, compare the reviewed architecture against the test surface, and identify the most important structural blind spots in the current tests.

## Code Paths Reviewed

- Cross-stage synthesis over Stage 1-5 review outputs, with the main architectural claims anchored to:
  - `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
  - `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
  - `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
  - `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py`
  - `tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py`
  - `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`
  - `tldw_Server_API/app/core/RAG/rag_service/response_writer.py`
  - `tldw_Server_API/app/core/RAG/rag_service/guardrails.py`
  - `tldw_Server_API/app/api/v1/utils/rag_cache.py`

## Tests Reviewed

- Canonical test evidence inherited from the targeted test slices recorded in the Stage 2-5 reports. Stage 6 confidence ratings are based on that broader reviewed test base plus the representative sanity-pack rerun below.
- `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_decomposition.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_retrieval.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_two_tier_reranker.py`

## Validation Commands

- `rg -n "^## Findings|^## Suggested Refactor/Actions|^## Coverage Gaps|^## Exit Note" Docs/superpowers/reviews/rag/2026-04-07-stage1-architecture-survey-and-inventory.md Docs/superpowers/reviews/rag/2026-04-07-stage2-unified-pipeline-orchestration.md Docs/superpowers/reviews/rag/2026-04-07-stage3-api-schema-and-request-boundaries.md Docs/superpowers/reviews/rag/2026-04-07-stage4-retrieval-boundaries-and-data-sources.md Docs/superpowers/reviews/rag/2026-04-07-stage5-reranking-and-post-retrieval-composition.md`
  - This command is an index into the canonical Stage 1-5 findings/actions/gaps/exit notes; the detailed evidence remains in those stage reports rather than in the heading scan itself.
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_decomposition.py tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py tldw_Server_API/tests/RAG_NEW/unit/test_retrieval.py tldw_Server_API/tests/RAG_NEW/unit/test_two_tier_reranker.py -v`
  - Result in this environment: `38 passed, 96 warnings in 1.60s`
- `source ../../.venv/bin/activate && python -m bandit -r Docs/superpowers/reviews/rag -f json -o /tmp/bandit_stage6_rag_synthesis.json`
  - Result in this environment: Bandit completed successfully and the JSON report contained `0` errors and `0` results.

## Findings

1. High severity, high confidence: the main architectural risk is still unstable ownership of request policy and evidence state across phases, not a lack of local feature tests.
   - Stage 2 and Stage 5 converge on the same problem from opposite sides: `unified_rag_pipeline()` is both coordinator and policy owner, and later post-retrieval flows can mutate, regenerate, or replace the working document set through numeric retry, adaptive verification, and recursive rerun behavior.
   - The requested sanity pack does not materially constrain that seam. `test_unified_pipeline_decomposition.py` only proves one mocked retrieval subfeature, and `test_two_tier_reranker.py` only proves one gated reranker path with fake rerankers. Neither test pins a one-way phase graph or a stable distinction between retrieved evidence and derived evidence.

2. High severity, high confidence: the public RAG contract still has multiple effective owners, so helper-level green tests overstate boundary confidence.
   - Stage 3 showed that request-default precedence, profile shaping, batch behavior, and response-field mapping are split across endpoint helpers, schemas, profiles, and direct agentic call construction.
   - `test_rag_unified_search_agent_defaults.py` is useful but narrow: it strongly protects helper-level precedence and profile-vs-default resolution, yet it does not prove non-streaming agentic `/search` parity, batch parity with single-search, or full response-field mapping from pipeline metadata to `UnifiedRAGResponse`.

3. High severity, high confidence: retrieval behavior is well tested where it is already concrete, but that strength exposes a deeper ownership problem rather than resolving it.
   - `test_retrieval.py` gives the strongest signal in the sanity pack. It meaningfully constrains Media DB fallback logic, chunk-level late chunking, scoped vector filters, bounded-term retry, allowed-media filtering, and some `MultiDatabaseRetriever` dispatch behavior.
   - That strength is local to the current concrete design. It does not prove that retrieval policy is resolved once upstream, that namespace/collection ownership is centralized, that `collection_prefix` matters, or that PGVector-backed behavior is equivalent in this environment.

4. Medium severity, high confidence: post-retrieval grounding remains fragmented across reranking, citations, response writing, and verification, with little structural protection in the representative slice.
   - Stage 5 showed three parallel citation systems and multiple response-writing/control-flow owners.
   - The requested tests barely touch this area beyond reranker gating metadata. There is no structural protection here for citation-system consistency, `verification_report` exposure, or the boundary between utility guardrails and orchestration-owned retry/abstention policy.

5. Medium severity, medium confidence: secondary entry points are still treated more like facades than explicit product surfaces, which keeps drift cheap.
   - Batch, non-streaming agentic search, streaming agentic search, and research-side flows share helpers and internals but do not obviously share one canonical request/result contract.
   - Earlier stages already showed the concrete drift points. Stage 6 concludes the current tests mostly validate behavior inside each path rather than enforcing shared ownership across them.

## Suggested Refactor/Actions

1. Define one canonical contract chain and move everything else behind it: resolved request -> retrieval plan -> retrieved evidence -> derived evidence -> API response.
   - This is the shortest path to collapsing the cross-stage duplication. It addresses Stage 2 orchestration leakage, Stage 3 endpoint/schema drift, Stage 4 namespace ambiguity, and Stage 5 evidence mutation with one ownership model.

2. Make every public `/search` mode consume the same resolved request object before any agentic, batch, or streaming branch-specific logic runs.
   - Non-streaming agentic parity and batch drift are the clearest contract regressions already identified. They should stop reconstructing policy from raw request fields.

3. Extract a post-retrieval coordinator that owns retry, abstention, repair, and evidence-set replacement decisions.
   - `guardrails.py`, `citations.py`, and verifier helpers can stay utility-oriented, but `unified_pipeline.py` and `agentic_chunker.py` should stop being hidden second orchestrators of retrieval and regeneration.

4. Centralize namespace, collection, and invalidation ownership in core RAG services.
   - Retrievers, vector adapters, and cache invalidation should all depend on one resolver/service instead of conventionally rebuilding `user_{user_id}_media_embeddings` and related aliases in multiple layers.

5. Add structural tests in blast-radius order rather than feature-file order.
   - First: one-way phase-ownership tests for recursive rerun, numeric retry, and derived-vs-retrieved document sets.
   - Second: endpoint parity tests for standard, non-streaming agentic, streaming agentic, and batch request resolution.
   - Third: namespace/invalidation ownership tests, including PGVector parity when the environment supports it.
   - Fourth: citation/response-mapping consistency tests, including `verification_report` and shared evidence provenance.

## Coverage Gaps

1. Highest blast radius: no representative test proves one authoritative cross-phase state model.
   - Missing protection includes adaptive recursive rerun, numeric-repair document replacement, and explicit surfacing of retrieved vs derived evidence sets after Stage 4 retrieval completes.
   - Current signal: weak. `test_unified_pipeline_decomposition.py` and `test_two_tier_reranker.py` are behavior checks around isolated subpaths, not ownership tests.

2. High blast radius: endpoint and schema parity still lack structural protection.
   - Missing protection includes non-streaming agentic parity with helper-resolved defaults, batch-vs-single contract parity, and response-field completeness for first-class schema fields.
   - Current signal: mixed. `test_rag_unified_search_agent_defaults.py` is strong for helper precedence, weak for full API-surface parity.

3. High-to-medium blast radius: retrieval source routing is better protected than retrieval ownership.
   - Missing protection includes centralized namespace/collection ownership, `collection_prefix` effectiveness, API/core invalidation parity, and PGVector-backed parity in this environment.
   - Current signal: strongest in the slice. `test_retrieval.py` protects many concrete Media DB behaviors, but mostly through mocked vector adapters or SQLite-backed fixtures.

4. Medium blast radius: post-retrieval grounding and response composition remain largely unconstrained by the representative pack.
   - Missing protection includes citation-system consistency, `verification_report` mapping, response-writer authority, and consistent provenance for repaired/agentic evidence.
   - Current signal: weak. The slice validates reranker gating but not downstream grounding ownership.

## Exit Note

Stage 6 does not overturn any earlier stage. It narrows them into one synthesis: the RAG module is best understood as a single large orchestration surface with several hidden secondary owners, and the current tests are strongest where policy has already hardened into concrete local code rather than where ownership is architecturally supposed to live.

Open questions that still block a stronger claim:
- whether the recursive/adaptive rerun branch is intended to preserve prior retrieved evidence as first-class state or to replace it;
- whether agentic search is supposed to be contract-compatible with standard `/search` or a separate product surface;
- whether namespace/collection naming is meant to be a public core contract or an implementation convention;
- whether the response schema is intended to mirror a typed internal result model or remain metadata-derived at the endpoint boundary.

The canonical evidence remains in the Stage 1-5 reports. This document is the ranked, deduplicated synthesis and test-gap summary for that review set.
