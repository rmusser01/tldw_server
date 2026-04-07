# Stage 1 Architecture Survey and Inventory

## Scope

Create the review scaffold, capture the scoped RAG source and test inventory, record size and churn baselines for the main hotspots, and route any secondary seams into later stages before deeper review begins.

The workspace safety check confirmed the isolated worktree path at `/Users/appledev/Documents/GitHub/tldw_server/.worktrees/rag-module-review`. Before the review scaffold was created, `git status --short` showed only the copied execution plan as an untracked file, so this run remained docs-only and did not touch source code.

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

```bash
mkdir -p Docs/superpowers/reviews/rag
git rev-parse --show-toplevel
git status --short
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
git log --oneline -n 20 -- \
  tldw_Server_API/app/core/RAG \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py
rg -n "async def unified_rag_pipeline|async def agentic_rag_pipeline|def _build_effective_request_payload|def get_profile_kwargs|class Document|class DataSource|def invalidate_rag_caches|create_from_settings_for_user" \
  tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py \
  tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py \
  tldw_Server_API/app/core/RAG/rag_service/profiles.py \
  tldw_Server_API/app/core/RAG/rag_service/types.py \
  tldw_Server_API/app/core/RAG/rag_service/vector_stores/factory.py \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/app/api/v1/utils/rag_cache.py
```

## Findings

- No confirmed defects at survey depth.
- Hotspot baseline: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py` is the largest clear orchestrator at 6977 LOC, followed by `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py` at 3590 LOC, `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` at 2445 LOC, and `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py` at 1704 LOC.
- Churn baseline: the last 20 commits touching the scoped surface include rerank-debug snapshot gating, review-feedback cleanup, and RAG tuning/recipe work, which means the active surface is still moving and later stages should expect recent-commit context to matter.
- Probable risk: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py` and `tldw_Server_API/app/api/v1/endpoints/rag_unified.py` are large enough to concentrate orchestration and boundary ownership, so later stages need explicit seam ownership to avoid drift.
- Probable risk: retrieval-adjacent support files such as `query_expansion.py`, `semantic_cache.py`, `response_writer.py`, `post_generation_verifier.py`, `agentic_chunker.py`, and `research_agent.py` are part of the active path and should not remain implicit spillover areas.

## Suggested Refactor/Actions

- Route the orchestration core to Stage 2.
- Route API, schema, profile, and request-mapping ownership to Stage 3.
- Route retrieval, cache, and vector-store seams to Stage 4.
- Route reranking, generation, response writing, verification, and agentic side paths to Stage 5.
- Keep Stage 6 focused on test gaps and synthesis rather than re-litigating stage ownership.

## Coverage Gaps

- No runtime checks or tests were run in this stage.
- The legacy `tldw_Server_API/tests/RAG` suite and the newer `tldw_Server_API/tests/RAG_NEW` suite should both remain visible in later stages so coverage gaps do not get masked by one green path.
- The inventory surfaced adjacent support seams beyond the initial seed set, especially `query_expansion.py`, `response_writer.py`, `post_generation_verifier.py`, `semantic_cache.py`, `agentic_chunker.py`, and `research_agent.py`.

## Exit Note

Stage 1 is complete. The review ledger exists, the scoped inventories are captured, and no important seam is left unowned: Stage 2 owns the pipeline core, Stage 3 owns API/schema/profile boundaries, Stage 4 owns retrieval and data-source seams, Stage 5 owns reranking and post-retrieval composition, and Stage 6 owns the final test-gap synthesis.
