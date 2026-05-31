# Stage 2 Unified Pipeline Orchestration

## Scope

Review `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py` as the central orchestration layer, plus the closest contract and type modules that materially shape orchestration behavior.

## Code Paths Reviewed

- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
  - Public entry points: `unified_rag_pipeline()` (`1160+`), `unified_batch_pipeline()` (`6556+`), `simple_search()` (`6843+`), `advanced_search()` (`6889+`)
  - Runtime response carrier: `UnifiedSearchResult` (`1048-1068`)
  - Retrieval start and retrieval-owned fan-out: `2756-3278`
  - Reranking handoff and profile degradation: `4301-4586`
  - Generation and generation-side metadata shaping: `5006-5298`
  - Post-generation verification and recursive adaptive rerun: `5841-6109`
  - Final schema/dict response shaping: `6513-6545`
- `tldw_Server_API/app/core/RAG/rag_service/types.py`
  - Shared document/search contracts plus the lightweight `RAGPipelineContext` state carrier (`255-271`)
- Adjacent helpers were reviewed only through `unified_pipeline.py` call sites so Stage 2 stays on orchestration ownership. Stage 4/5 seams remain routed forward per Stage 1.

## Tests Reviewed

- `tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py`
  - Protects the public entry-point contract, common parameter acceptance, rerank-debug metadata hiding, citation shaping, and broad smoke/integration scenarios.
  - Mostly happy-path coverage; useful for downgrading user-facing regression risk, not for proving clean phase boundaries.
- `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline.py`
  - Protects classification skip-search bypass, research-loop retrieval bypass, external-prefetch bypass, security-filter fallbacks, citation/highlighting hooks, metadata passthrough, and streaming toggles.
  - Mixed happy-path plus a few real boundary checks; downgrades risk around retrieval-bypass semantics more than overall decomposition quality.
- `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_decomposition.py`
  - Protects the hidden decomposition sub-pipeline: secondary subquery retrieval must add documents and publish `metadata.decomposition`, even when base retrieval is empty.
  - Explicit decomposition-boundary coverage; materially downgrades risk of accidental removal of the subquery merge behavior.
- `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_focused.py`
  - Protects cache read/write compatibility, request-scoped `chacha_db` reuse, managed-media-db claims verification, reranking basics, and common user flows.
  - Contains several real ownership checks for cache and claims paths; still mostly outcome-driven rather than phase-order driven.
- `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_profile_metadata.py`
  - Protects profile-resolution metadata and the two-tier-to-hybrid degradation contract.
  - Explicit contract coverage; downgrades risk of silent profile fallback regressions.
- `tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_generation_controls.py`
  - Protects abstention-on-gated-generation, multi-turn synthesis metadata, and provider/model passthrough.
  - Boundary-focused on generation controls; downgrades risk of generation-policy regressions.
- `tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_post_verification_metadata.py`
  - Protects `post_verification` metadata attachment and repaired-answer adoption.
  - Explicit verification-boundary coverage; does not cover the recursive adaptive rerun branch.
- `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_structured_writer.py`
  - Protects structured-writer prompt selection and depth-policy metadata under different token budgets.
  - Boundary-focused for one generation sub-mode; lowers risk of prompt/policy metadata drift.

## Validation Commands

- Entry-point/flag map:
  - `rg -n "async def unified_rag_pipeline|async def unified_batch_pipeline|def simple_search|def advanced_search|enable_|search_mode|debug_mode|metadata|generated_answer|cache_hit" tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py tldw_Server_API/app/core/RAG/rag_service/types.py`
- Targeted test run:
  - `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline.py tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_decomposition.py tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_focused.py tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_profile_metadata.py tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_generation_controls.py tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_post_verification_metadata.py tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_structured_writer.py -v`
  - Result from rerun in this worktree: `71 passed, 508 warnings`.
  - The warning volume is non-trivial and the command output also included config/auth/test-environment warnings during startup and teardown; they did not fail the suite, but they are still part of the observed validation result.
- Docs-scope security check:
  - `source ../../.venv/bin/activate && python -m bandit -r Docs/superpowers/reviews/rag -f json -o /tmp/bandit_stage2_rag_orchestration.json`
  - Result from rerun in this worktree: JSON report written to `/tmp/bandit_stage2_rag_orchestration.json` with `0 errors` and `0 results`.

## Findings

- High severity, high confidence: `unified_rag_pipeline()` is not just a coordinator; it is the policy engine, phase controller, and recovery loop.
  - Entry normalization mutates request policy in place through presets and profile metadata (`1581-1684`), query-classification/research logic can bypass or replace retrieval while preserving later stages (`1890-2283`), and post-verification can recursively call `unified_rag_pipeline()` with a copied argument set (`5916-6109`).
  - This makes phase ownership non-local. Generation and verification are not leaf stages; they can become hidden orchestrators of a second pipeline pass.
- Medium severity, high confidence: retrieval is a stacked sub-pipeline, not a single handoff from “query prep” to “documents”.
  - Query preparation effectively ends when cache/retrieval gating is resolved and the retrieval block begins (`2531-2757`).
  - Inside retrieval, the function owns direct retrieval, Media DB fallback FTS, HyDE merge, expansion fan-out, PRF second pass, and concurrent query decomposition before publishing `result.documents` and retrieval metadata (`2756-3278`).
  - After that, more pre-rerank mutation still occurs inline through multi-vector passages, gap analysis, filtering, security policy, table/VLM processing, evidence accumulation, personalization, grading, and rewrite-loop logic (`3402-4300`), so the “retrieval” phase boundary is operationally fuzzy.
- Medium severity, high confidence: response shaping and metadata ownership live inside the orchestrator instead of behind a typed contract.
  - `UnifiedSearchResult` in `unified_pipeline.py` duplicates runtime response fields (`metadata`, `generated_answer`, `cache_hit`, `errors`) while `UnifiedPipelineResult = Any` explicitly accepts mixed shapes (`1048-1068`).
  - `types.py` still carries a lighter `RAGPipelineContext` (`255-271`), but that is not the runtime handoff contract used by the main orchestration path.
  - Final schema conversion happens only at the end after hundreds of metadata writes (`6513-6545`), and `unified_batch_pipeline()` manually clones selected fields back into new `UnifiedSearchResult` instances (`6793-6821`).
  - The result contract is therefore “whatever metadata the monolith happened to attach”, not a narrow phase-by-phase model.
- Low severity, medium confidence: the secondary entry points are facades, not isolation layers.
  - `simple_search()` and `advanced_search()` only preset arguments into `unified_rag_pipeline()` (`6843-6919`).
  - `unified_batch_pipeline()` adds dedupe/clustering and result reconstruction, but it still inherits the mixed result contract and manual field copying (`6556-6838`).
  - Stage 3 should treat these as thin veneers over the same orchestration surface, not as separate integration boundaries.

## Suggested Refactor/Actions

- Introduce an explicit request-normalization stage result that owns profile resolution, preset application, source normalization, and bypass decisions before retrieval starts. That would move the top-of-function flag mutation out of the main pipeline body without changing feature behavior.
- Split the runtime state into narrower phase outputs, at minimum: retrieval output, generation output, and verification output. Keep metadata appenders separate from the phase executors so response-shaping stops being incidental mutation on one shared `result`.
- Pull adaptive rerun into a dedicated recovery coordinator instead of recursively re-entering `unified_rag_pipeline()` with a copied argument list. The current branch is functionally powerful but architecturally the clearest ownership leak.
- Centralize final response serialization and batch-result cloning behind one adapter. That reduces drift between `UnifiedSearchResult`, `UnifiedRAGResponse`, wrapper helpers, and test assumptions.

## Coverage Gaps

- The reviewed tests strongly protect behavior and metadata keys, but they do not pin a clean one-way phase graph from query prep -> retrieval -> reranking -> generation -> verification. A refactor could preserve outputs while keeping the same hidden coupling.
- No reviewed test directly exercises the adaptive recursive rerun branch (`5916-6109`), which is the highest-leverage orchestration leak in this stage.
- Batch orchestration is only indirectly covered. There is no targeted test in the requested slice for query clustering/reuse semantics in `unified_batch_pipeline()`.
- Contract coverage is broad but still output-oriented. There is no single test asserting that the runtime state carrier and final API schema remain aligned as new metadata fields are added.

## Exit Note

Stage 2 confirms that `unified_pipeline.py` is the orchestration hotspot identified in Stage 1, and that several “features” inside it actually behave like hidden sub-pipelines. Stage 3 should verify the endpoint boundary in `rag_unified.py`: where API request shaping stops, which defaults and aliases are resolved before the pipeline is called, and whether the endpoint layer adds a second source of truth for response metadata or parameter policy.
