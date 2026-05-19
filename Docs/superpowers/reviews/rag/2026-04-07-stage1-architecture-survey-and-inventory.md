# Stage 1 Architecture Survey and Inventory

## Scope

Create the review scaffold, capture the scoped RAG source and test inventory, record size and churn baselines for the main hotspots, and route any secondary seams into later stages before deeper review begins.

The workspace safety check confirmed the isolated worktree path at `/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review`. Before the review scaffold was created, `git status --short` showed only the copied execution plan as an untracked file. The scaffold itself remained docs-only, and the workspace-safety artifact preserves the observed pre-scaffold state, the literal current `git status --short` clean observation at the latest Task 1 head, the fixed Task 1 commit trail through `9494d0fed`, and a path-limited `git log` reproduction rule anchored at `9494d0fed` for the later Task 1 stabilization tail across the Stage 1 evidence files.

## Code Paths Reviewed

- `tldw_Server_API/app/core/RAG/README.md`
- `tldw_Server_API/app/core/RAG/rag_service/README.md`
- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
- `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py`
- `tldw_Server_API/app/core/RAG/rag_service/generation.py`
- `tldw_Server_API/app/core/RAG/rag_service/guardrails.py`
- `tldw_Server_API/app/core/RAG/rag_service/citations.py`
- `tldw_Server_API/app/core/RAG/rag_service/profiles.py`
- `tldw_Server_API/app/core/RAG/rag_service/types.py`
- `tldw_Server_API/app/core/RAG/rag_service/semantic_cache.py`
- `tldw_Server_API/app/core/RAG/rag_service/vector_stores/factory.py`
- `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`
- `tldw_Server_API/app/core/RAG/rag_service/research_agent.py`
- `tldw_Server_API/app/core/RAG/rag_service/response_writer.py`
- `tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py`
- `tldw_Server_API/app/core/RAG/rag_service/query_expansion.py`
- `tldw_Server_API/app/core/RAG/rag_service/web_fallback.py`
- `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
- `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- `tldw_Server_API/app/api/v1/schemas/rag_schemas_simple.py`
- `tldw_Server_API/app/api/v1/utils/rag_cache.py`

## Tests Reviewed

- None executed in this stage.
- Scoped inventory captured across `tldw_Server_API/tests/RAG`, `tldw_Server_API/tests/RAG_NEW`, `tldw_Server_API/tests/e2e`, and `tldw_Server_API/tests/server_e2e_tests`.

## Validation Commands

All commands below were run from the worktree root:
`/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review`

```bash
git rev-parse --show-toplevel
git status --short
mkdir -p Docs/superpowers/reviews/rag
source ../../.venv/bin/activate && rg --files \
  tldw_Server_API/app/core/RAG \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/app/api/v1/endpoints/rag_health.py \
  tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py \
  tldw_Server_API/app/api/v1/schemas/rag_schemas_simple.py \
  tldw_Server_API/app/api/v1/utils/rag_cache.py | sort
source ../../.venv/bin/activate && rg --files \
  tldw_Server_API/tests/RAG \
  tldw_Server_API/tests/RAG_NEW \
  tldw_Server_API/tests/e2e \
  tldw_Server_API/tests/server_e2e_tests | rg 'rag|RAG|search' | sort
wc -l \
  tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py \
  tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py
git log --oneline -n 20 -- tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py
git log --oneline -n 20 -- tldw_Server_API/app/api/v1/endpoints/rag_unified.py
git log --oneline -n 20 -- tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py
git log --oneline -n 20 -- tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py
rg -n "async def unified_rag_pipeline|async def agentic_rag_pipeline|def _build_effective_request_payload|def get_profile_kwargs|class Document|class DataSource|def invalidate_rag_caches|create_from_settings_for_user" \
  tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py \
  tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py \
  tldw_Server_API/app/core/RAG/rag_service/profiles.py \
  tldw_Server_API/app/core/RAG/rag_service/types.py \
  tldw_Server_API/app/core/RAG/rag_service/vector_stores/factory.py \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/app/api/v1/utils/rag_cache.py | sort
```

## Workspace Safety Evidence

- Worktree root: `/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review`
- Preserved safety artifact: [`2026-04-07-stage1-workspace-safety.txt`](./2026-04-07-stage1-workspace-safety.txt)
- The pre-scaffold snapshot shows only the copied execution plan as untracked.
- The current worktree state after the Task 1 review-loop fix commits is preserved as a literal `git status --short` observation with no output.
- The docs-only footprint of Task 1 is established by the fixed Task 1 commit trail recorded in the safety artifact plus the anchored reproduction rule for the later Task 1 stabilization tail across the Stage 1 evidence files.

## Preserved Source Inventory

The verbatim output of `rg --files ... | sort` is retained in [`2026-04-07-stage1-source-inventory.txt`](./2026-04-07-stage1-source-inventory.txt).

## Preserved Test Inventory

The verbatim output of `rg --files ... | rg 'rag|RAG|search' | sort` is retained in [`2026-04-07-stage1-test-inventory.txt`](./2026-04-07-stage1-test-inventory.txt).
This is a raw grep capture, so it includes helper and support files such as `conftest.py`, `__init__.py`, and `TEST_STRATEGY.md` in addition to executable pytest modules.

## Preserved Baselines

- Hotspot size baseline: [`2026-04-07-stage1-hotspot-sizes.txt`](./2026-04-07-stage1-hotspot-sizes.txt)
- Churn baseline by hotspot: [`2026-04-07-stage1-churn-baseline.txt`](./2026-04-07-stage1-churn-baseline.txt)
- Seed-set ownership map: [`2026-04-07-stage1-seed-set-ownership-map.txt`](./2026-04-07-stage1-seed-set-ownership-map.txt)
- The seed-set ownership artifact is the sorted output of the documented `rg -n ... | sort` command so the preserved evidence stays stable across ripgrep file-order variation.

## Findings

- No confirmed defects at survey depth.
- Hotspot inventory by size: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py` is the dominant orchestrator at 6977 LOC; `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py` is the next largest core retrieval module at 3590 LOC; `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` is the main API boundary at 2445 LOC; and `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py` is the post-retrieval ranking hotspot at 1704 LOC.
- Hotspot inventory by churn: the preserved per-file recent-history baseline now attributes change activity to each named hotspot. `unified_pipeline.py` shows the densest recent orchestration churn, including rerank-debug gating, retrieval-tuning work, database-lifecycle refactors, and SQL-retriever integration; `rag_unified.py` shows a narrower API-boundary history centered on default-precedence and boundary cleanup; `database_retrievers.py` carries retrieval fallback and factory-routing changes; and `advanced_reranking.py` shows an older, more isolated reranking-focused change pattern. That is still lightweight churn evidence rather than a full ownership map, but it is now attributable to the exact hotspot files selected for deeper review.
- Hotspot inventory by centrality: `unified_pipeline.py` is the primary fan-in/fan-out orchestrator; `rag_unified.py` owns the request/response boundary; `profiles.py` and `types.py` define shared request and result contracts; `rag_cache.py` and `vector_stores/factory.py` mediate request-time adapter selection and invalidation; and `database_retrievers.py` sits on the retrieval composition seam. Those are the ownership points that later stages should treat as structurally significant.
- Probable risk: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py` and `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` are large enough to concentrate orchestration and boundary ownership, so later stages need explicit seam ownership to avoid drift.
- Probable risk: retrieval-adjacent support files such as `query_expansion.py`, `hyde.py`, `query_classifier.py`, `semantic_cache.py`, `response_writer.py`, `post_generation_verifier.py`, `agentic_chunker.py`, and `research_agent.py` are part of the active path and should not remain implicit spillover areas.

## Suggested Refactor/Actions

- Route the orchestration core to Stage 2.
- Route API, schema, profile, and request-mapping ownership to Stage 3.
- Route retrieval, cache, and vector-store seams to Stage 4.
- Route reranking, generation, response writing, verification, and agentic side paths to Stage 5.
- Keep Stage 6 focused on test gaps and synthesis rather than re-litigating stage ownership.

## Secondary Hotspot Routing

| File | Stage | Why it belongs there |
| --- | --- | --- |
| `tldw_Server_API/app/core/RAG/rag_service/profiles.py` | Stage 3 | request-profile defaults and payload shaping |
| `tldw_Server_API/app/core/RAG/rag_service/types.py` | Stage 3 | shared data contracts consumed by the API and pipeline |
| `tldw_Server_API/app/api/v1/utils/rag_cache.py` | Stage 3 | request-time cache invalidation and adapter selection |
| `tldw_Server_API/app/core/RAG/rag_service/query_expansion.py` | Stage 4 | retrieval-time query shaping and expansion |
| `tldw_Server_API/app/core/RAG/rag_service/hyde.py` | Stage 4 | HYDE retrieval expansion and merge behavior |
| `tldw_Server_API/app/core/RAG/rag_service/query_classifier.py` | Stage 4 | retrieval-time routing and classification policy |
| `tldw_Server_API/app/core/RAG/rag_service/semantic_cache.py` | Stage 4 | retrieval cache behavior and scoping |
| `tldw_Server_API/app/core/RAG/rag_service/vector_stores/factory.py` | Stage 4 | data-source adapter creation and selection |
| `tldw_Server_API/app/core/RAG/rag_service/web_fallback.py` | Stage 4 | fallback retrieval path and source routing |
| `tldw_Server_API/app/core/RAG/rag_service/response_writer.py` | Stage 5 | post-retrieval response composition |
| `tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py` | Stage 5 | generation verification and post-processing |
| `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py` | Stage 5 | agentic post-retrieval execution path |
| `tldw_Server_API/app/core/RAG/rag_service/research_agent.py` | Stage 5 | research-side orchestration after retrieval |

## Coverage Gaps

- No runtime checks or tests were run in this stage.
- The legacy `tldw_Server_API/tests/RAG` suite and the newer `tldw_Server_API/tests/RAG_NEW` suite should both remain visible in later stages so coverage gaps do not get masked by one green path.
- The inventory surfaced adjacent support seams beyond the initial seed set, especially `query_expansion.py`, `hyde.py`, `query_classifier.py`, `response_writer.py`, `post_generation_verifier.py`, `semantic_cache.py`, `agentic_chunker.py`, and `research_agent.py`.

## Exit Note

Stage 1 is complete. The review ledger exists, the scoped inventories are captured, and no important seam is left unowned: Stage 2 owns the pipeline core, Stage 3 owns API/schema/profile boundaries, Stage 4 owns retrieval and data-source seams, Stage 5 owns reranking and post-retrieval composition, and Stage 6 owns the final test-gap synthesis.
