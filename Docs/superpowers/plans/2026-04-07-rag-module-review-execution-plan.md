# RAG Module Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved RAG architecture and maintainability review and produce one evidence-backed stage report per slice plus a final synthesis for the `tldw_Server_API/app/core/RAG` surface and its direct API boundaries.

**Architecture:** This is a read-first, staged audit plan. Execution starts with a broad architecture survey and fixed review scaffold, then moves through the unified pipeline, API/schema boundaries, retrieval seams, and reranking or post-retrieval seams in order. Each stage writes findings before suggested actions, uses only the smallest relevant test set to confirm or sharpen claims, and commits docs-only review artifacts that later remediation work can reference. Prefer executing the review in a dedicated git worktree; if that is not available, keep the run docs-only and never revert or disturb unrelated workspace changes. In this isolated worktree, activate the shared project virtualenv with `source ../../.venv/bin/activate` from the worktree root before running Python or pytest commands.

**Tech Stack:** Python 3, FastAPI, SQLite, ChromaDB/pgvector adapters, pytest, ripgrep, git, Markdown

---

## Review File Map

**Create during execution:**
- `Docs/superpowers/reviews/rag/README.md`
- `Docs/superpowers/reviews/rag/2026-04-07-stage1-architecture-survey-and-inventory.md`
- `Docs/superpowers/reviews/rag/2026-04-07-stage2-unified-pipeline-orchestration.md`
- `Docs/superpowers/reviews/rag/2026-04-07-stage3-api-schema-and-request-boundaries.md`
- `Docs/superpowers/reviews/rag/2026-04-07-stage4-retrieval-boundaries-and-data-sources.md`
- `Docs/superpowers/reviews/rag/2026-04-07-stage5-reranking-and-post-retrieval-composition.md`
- `Docs/superpowers/reviews/rag/2026-04-07-stage6-test-gaps-and-synthesis.md`

**Primary source files to inspect during the review:**
- `tldw_Server_API/app/core/RAG/README.md`
- `tldw_Server_API/app/core/RAG/rag_service/README.md`
- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
- `tldw_Server_API/app/core/RAG/rag_service/query_expansion.py`
- `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py`
- `tldw_Server_API/app/core/RAG/rag_service/generation.py`
- `tldw_Server_API/app/core/RAG/rag_service/guardrails.py`
- `tldw_Server_API/app/core/RAG/rag_service/citations.py`
- `tldw_Server_API/app/core/RAG/rag_service/profiles.py`
- `tldw_Server_API/app/core/RAG/rag_service/types.py`
- `tldw_Server_API/app/core/RAG/rag_service/semantic_cache.py`
- `tldw_Server_API/app/core/RAG/rag_service/response_writer.py`
- `tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py`
- `tldw_Server_API/app/core/RAG/rag_service/vector_stores/base.py`
- `tldw_Server_API/app/core/RAG/rag_service/vector_stores/factory.py`
- `tldw_Server_API/app/core/RAG/rag_service/vector_stores/chromadb_adapter.py`
- `tldw_Server_API/app/core/RAG/rag_service/vector_stores/pgvector_adapter.py`
- `tldw_Server_API/app/core/RAG/rag_service/media_search.py`
- `tldw_Server_API/app/core/RAG/rag_service/web_fallback.py`
- `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`
- `tldw_Server_API/app/core/RAG/rag_service/research_agent.py`
- `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
- `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- `tldw_Server_API/app/api/v1/schemas/rag_schemas_simple.py`
- `tldw_Server_API/app/api/v1/utils/rag_cache.py`

**High-value existing tests to reuse during the review:**
- `tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_decomposition.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_focused.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_profile_metadata.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_generation_controls.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_post_verification_metadata.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_structured_writer.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_response_mapping.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_request_schema_profiles.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_profiles.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_batch_round2_flags.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_health_endpoints.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_unified_features_endpoint.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_capabilities_styles.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_retrieval.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_vector_store_parity.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_vector_store_admin_guardrails.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_vector_retriever_hyde.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_hyde_retrieval_merge.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_tenant_scoping.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_persistence.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_path_sanitization.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_corpus_synonyms_expansion.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_query_classifier.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_media_search.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_retriever_pgvector_multi_search.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_adapter_guards.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_two_tier_reranker.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_two_tier_gate.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_reranker_metrics.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_response_writer.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_quotes_and_numeric.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_injection_numeric.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_hard_citations_golden.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_post_verifier.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_strict_extractive_and_citations.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_failures_and_fallbacks.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_cache_invalidation.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_golden_citations.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_strict_extractive_nli_api.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_research_agent_loop.py`
- `tldw_Server_API/tests/e2e/test_rag_generation_grounding_smoke.py`
- `tldw_Server_API/tests/e2e/test_rag_post_verification_smoke.py`

## Stage Overview

## Stage 1: Architecture Survey and Inventory
**Goal:** Create the review scaffold, capture the exact source and test surface, record hotspot size and churn, and assign any secondary hotspots to later stages so no important seam is left unowned.
**Success Criteria:** Review artifacts exist under `Docs/superpowers/reviews/rag/`, the seed-set inventory and git-history baseline are recorded, hotspot modules are named explicitly, and the Stage 1 exit note routes any newly discovered hotspot into a later stage.
**Tests:** None
**Status:** Complete

## Stage 2: Unified Pipeline Orchestration
**Goal:** Review `unified_pipeline.py` and its closest type contracts as the central orchestration layer.
**Success Criteria:** Phase ordering, ownership boundaries, parameter sprawl, and metadata or response-shaping leakage are documented with evidence and tied to concrete tests where available.
**Tests:** `test_unified_pipeline.py`, `unit/test_unified_pipeline.py`, `unit/test_unified_pipeline_decomposition.py`, `unit/test_unified_pipeline_focused.py`, `unit/test_unified_pipeline_profile_metadata.py`, `unit/test_pipeline_generation_controls.py`, `unit/test_pipeline_post_verification_metadata.py`, `unit/test_unified_pipeline_structured_writer.py`
**Status:** Complete

## Stage 3: API, Schema, and Request Boundaries
**Goal:** Review endpoint, schema, profile-default, and request-mapping ownership around the RAG API surfaces.
**Success Criteria:** Default precedence, response mapping, schema sprawl, and endpoint-to-pipeline coupling are recorded with evidence and validated against the targeted API/schema tests.
**Tests:** `unit/test_rag_unified_search_agent_defaults.py`, `unit/test_rag_unified_response_mapping.py`, `unit/test_rag_request_schema_profiles.py`, `unit/test_rag_profiles.py`, `unit/test_batch_round2_flags.py`, `integration/test_rag_health_endpoints.py`, `integration/test_rag_unified_features_endpoint.py`, `integration/test_rag_capabilities_styles.py`, `integration/test_rag_stream_parity.py`
**Status:** Not Started

## Stage 4: Retrieval Boundaries and Data Sources
**Goal:** Review retrieval composition across database retrievers, query expansion, caches, vector stores, and data-source adapters.
**Success Criteria:** Source ownership, extension seams, cache scoping, vector-store contracts, and retrieval-time fallback behavior are documented with evidence and validated against targeted retrieval tests.
**Tests:** `unit/test_retrieval.py`, `unit/test_vector_store_parity.py`, `unit/test_vector_store_admin_guardrails.py`, `unit/test_vector_retriever_hyde.py`, `unit/test_hyde_retrieval_merge.py`, `unit/test_semantic_cache_tenant_scoping.py`, `unit/test_semantic_cache_persistence.py`, `unit/test_semantic_cache_path_sanitization.py`, `unit/test_corpus_synonyms_expansion.py`, `unit/test_query_classifier.py`, `unit/test_media_search.py`, `integration/test_retriever_pgvector_multi_search.py`, `integration/test_adapter_guards.py`
**Status:** Not Started

## Stage 5: Reranking and Post-Retrieval Composition
**Goal:** Review reranking, generation, citations, guardrails, verification, response writing, and the agentic or research side paths that touch the active RAG request surface.
**Success Criteria:** Ownership boundaries after retrieval completes are documented with evidence, including where reranking, generation, verification, and agentic execution overlap or leak responsibilities.
**Tests:** `unit/test_two_tier_reranker.py`, `unit/test_pipeline_two_tier_gate.py`, `unit/test_reranker_metrics.py`, `unit/test_response_writer.py`, `unit/test_guardrails_quotes_and_numeric.py`, `unit/test_guardrails_injection_numeric.py`, `unit/test_guardrails_hard_citations_golden.py`, `unit/test_post_verifier.py`, `unit/test_strict_extractive_and_citations.py`, `unit/test_agentic_failures_and_fallbacks.py`, `unit/test_agentic_cache_invalidation.py`, `unit/test_agentic_golden_citations.py`, `integration/test_rag_agentic_api.py`, `integration/test_rag_strict_extractive_nli_api.py`, `integration/test_research_agent_loop.py`, `e2e/test_rag_generation_grounding_smoke.py`, `e2e/test_rag_post_verification_smoke.py`
**Status:** Not Started

## Stage 6: Test Gaps and Final Synthesis
**Goal:** Consolidate cross-stage findings into one ranked synthesis and identify the most important architectural blind spots in the current tests.
**Success Criteria:** Duplicate findings are removed, test gaps are prioritized by blast radius, disputed claims are either downgraded or rechecked with a minimal representative test pack, and the final synthesis points back to the canonical stage artifacts.
**Tests:** Re-run a representative cross-slice sanity pack before finalizing the synthesis.
**Status:** Not Started

### Task 1: Prepare Review Artifacts and Complete Stage 1 Survey

**Files:**
- Create: `Docs/superpowers/reviews/rag/README.md`
- Create: `Docs/superpowers/reviews/rag/2026-04-07-stage1-architecture-survey-and-inventory.md`
- Create: `Docs/superpowers/reviews/rag/2026-04-07-stage2-unified-pipeline-orchestration.md`
- Create: `Docs/superpowers/reviews/rag/2026-04-07-stage3-api-schema-and-request-boundaries.md`
- Create: `Docs/superpowers/reviews/rag/2026-04-07-stage4-retrieval-boundaries-and-data-sources.md`
- Create: `Docs/superpowers/reviews/rag/2026-04-07-stage5-reranking-and-post-retrieval-composition.md`
- Create: `Docs/superpowers/reviews/rag/2026-04-07-stage6-test-gaps-and-synthesis.md`
- Test: none

- [x] **Step 1: Create the review output directory**

Run:
```bash
mkdir -p Docs/superpowers/reviews/rag
```

Expected: the `Docs/superpowers/reviews/rag` directory exists and no source files change.

- [x] **Step 1.5: Verify the execution environment is safe for a docs-only staged review**

Run:
```bash
git rev-parse --show-toplevel
git status --short
```

Expected: ideally this runs inside a dedicated worktree; if it does not, record that the review must remain docs-only and that unrelated local changes must not be touched or reverted.

- [x] **Step 2: Create one markdown file per stage with a fixed review template**

Each stage file should contain:
```markdown
# Stage N Title

## Scope
## Code Paths Reviewed
## Tests Reviewed
## Validation Commands
## Findings
## Suggested Refactor/Actions
## Coverage Gaps
## Exit Note
```

- [x] **Step 3: Write `Docs/superpowers/reviews/rag/README.md`**

Document:
- the stage order `1 -> 2 -> 3 -> 4 -> 5 -> 6`
- the path to each stage report
- the rule that findings must be written before suggested actions
- the rule that uncertain items are labeled as probable risks or assumptions instead of confirmed defects
- the rule that later stage summaries point back to the stage files instead of replacing them

- [x] **Step 4: Capture the scoped source inventory**

Run:
```bash
source ../../.venv/bin/activate
rg --files \
  tldw_Server_API/app/core/RAG \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/app/api/v1/endpoints/rag_health.py \
  tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py \
  tldw_Server_API/app/api/v1/schemas/rag_schemas_simple.py \
  tldw_Server_API/app/api/v1/utils/rag_cache.py | sort
```

Expected: a stable source list that captures the core RAG tree plus the direct API boundary files.

- [x] **Step 5: Capture the scoped test inventory**

Run:
```bash
source ../../.venv/bin/activate
rg --files \
  tldw_Server_API/tests/RAG \
  tldw_Server_API/tests/RAG_NEW \
  tldw_Server_API/tests/e2e \
  tldw_Server_API/tests/server_e2e_tests | rg 'rag|RAG|search' | sort
```

Expected: a stable list of direct RAG and RAG-adjacent tests that later stages can cite.

- [x] **Step 6: Capture hotspot size and recent-history baselines**

Run:
```bash
wc -l \
  tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py \
  tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py

git log --oneline -n 20 -- \
  tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py
git log --oneline -n 20 -- \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py
git log --oneline -n 20 -- \
  tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py
git log --oneline -n 20 -- \
  tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py
```

Expected: a size map and churn baseline that justify which files deserve deeper review first.

- [x] **Step 7: Map the initial seed-set ownership surface**

Run:
```bash
rg -n "async def unified_rag_pipeline|async def agentic_rag_pipeline|def _build_effective_request_payload|def get_profile_kwargs|class Document|class DataSource|def invalidate_rag_caches|create_from_settings_for_user" \
  tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py \
  tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py \
  tldw_Server_API/app/core/RAG/rag_service/profiles.py \
  tldw_Server_API/app/core/RAG/rag_service/types.py \
  tldw_Server_API/app/core/RAG/rag_service/vector_stores/factory.py \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/app/api/v1/utils/rag_cache.py | sort
```

Expected: a compact map of the ownership seams that Stage 1 must route into later stages.

- [x] **Step 8: Write the Stage 1 report**

Record:
- the scoped source and test inventories
- the hotspot files by size, churn, or centrality
- any newly discovered secondary hotspots and the stage they belong to
- the Stage 1 exit note that assigns unowned seams before deeper review begins

- [x] **Step 9: Verify the workspace starts in a safe state**

Run:
```bash
git status --short
```

Expected: unrelated local changes may exist, but the RAG review setup itself only adds docs under `Docs/superpowers/reviews/rag/` and this plan file.

- [x] **Step 10: Commit the scaffold and Stage 1 survey**

Run:
```bash
git add \
  Docs/superpowers/reviews/rag \
  Docs/superpowers/plans/2026-04-07-rag-module-review-execution-plan.md
git commit -m "docs: scaffold rag review artifacts"
```

Expected: one docs-only commit captures the review workspace and Stage 1 inventory.

### Task 2: Execute Stage 2 Unified Pipeline Orchestration Review

**Files:**
- Modify: `Docs/superpowers/reviews/rag/2026-04-07-stage2-unified-pipeline-orchestration.md`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/types.py`
- Test: `tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_decomposition.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_focused.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_profile_metadata.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_generation_controls.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_post_verification_metadata.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_structured_writer.py`

- [x] **Step 1: Map the orchestration entry points and phase toggles**

Run:
```bash
rg -n "async def unified_rag_pipeline|async def unified_batch_pipeline|def simple_search|def advanced_search|enable_|search_mode|debug_mode|metadata|generated_answer|cache_hit" \
  tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py \
  tldw_Server_API/app/core/RAG/rag_service/types.py
```

Expected: a compact map of the public entry points, major flags, and response-shaping fields.

- [x] **Step 2: Trace phase ownership inside `unified_pipeline.py`**

Confirm:
- where query preparation ends and retrieval begins
- where retrieval hands off to reranking, generation, and verification
- where metadata and response-shaping logic are embedded instead of delegated
- whether internal helper sections behave like hidden sub-pipelines

- [x] **Step 3: Review the focused pipeline tests and extract the protected invariants**

For each listed test file, record:
- the main orchestration invariant it protects
- whether it checks decomposition boundaries or only happy-path behavior
- which probable risks can be upgraded or downgraded because of it

- [x] **Step 4: Run the targeted unified-pipeline tests**

Run:
```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/test_unified_pipeline.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_decomposition.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_focused.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_profile_metadata.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_generation_controls.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_post_verification_metadata.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_structured_writer.py -v
```

Expected: tests collect and mostly pass; any failure either sharpens a pipeline finding or must be explained as environment-specific noise.

- [x] **Step 5: Write the Stage 2 report**

Record:
- ranked findings with severity and confidence
- the specific points where orchestration ownership leaks across phases
- suggested refactor directions that reduce coupling without turning into a rewrite plan
- the exit note for what Stage 3 must verify at the endpoint boundary

- [x] **Step 6: Commit the Stage 2 report**

Run:
```bash
git add Docs/superpowers/reviews/rag/2026-04-07-stage2-unified-pipeline-orchestration.md
git commit -m "docs: record rag pipeline architecture findings"
```

Expected: one docs-only commit contains the Stage 2 report.

### Task 3: Execute Stage 3 API, Schema, and Request Boundary Review

**Files:**
- Modify: `Docs/superpowers/reviews/rag/2026-04-07-stage3-api-schema-and-request-boundaries.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/rag_schemas_simple.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/profiles.py`
- Inspect: `tldw_Server_API/app/api/v1/utils/rag_cache.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_response_mapping.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_request_schema_profiles.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_profiles.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_batch_round2_flags.py`
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_health_endpoints.py`
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_unified_features_endpoint.py`
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_capabilities_styles.py`
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py`

- [ ] **Step 1: Map endpoint, schema, profile, and cache boundary entry points**

Run:
```bash
rg -n "@router\\.|def _|model_validator|field_validator|rag_profile|get_profile_kwargs|invalidate_rag_caches|delete_media_vectors|UnifiedRAGRequest|UnifiedRAGResponse|UnifiedBatchRequest|ImplicitFeedbackEvent" \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/app/api/v1/endpoints/rag_health.py \
  tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py \
  tldw_Server_API/app/api/v1/schemas/rag_schemas_simple.py \
  tldw_Server_API/app/core/RAG/rag_service/profiles.py \
  tldw_Server_API/app/api/v1/utils/rag_cache.py
```

Expected: a compact map of the public request and response boundary plus the default-resolution helpers.

- [ ] **Step 2: Trace request-default precedence and response ownership**

Confirm:
- how explicit request fields, profile defaults, search-agent defaults, and schema defaults combine
- where the endpoint owns pipeline knowledge that should arguably live elsewhere
- where response mapping duplicates internal data-model knowledge
- whether cache invalidation helpers are clearly owned by the API boundary or leak core-RAG assumptions

- [ ] **Step 3: Review the targeted tests and extract the protected invariants**

For each listed test file, record:
- the exact precedence, mapping, or contract invariant it checks
- whether it validates the boundary directly or only via mocks
- any major negative case that remains untested

- [ ] **Step 4: Run the targeted API and schema tests**

Run:
```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_response_mapping.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_request_schema_profiles.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_profiles.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_batch_round2_flags.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_health_endpoints.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_unified_features_endpoint.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_capabilities_styles.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py -v
```

Expected: tests collect and mostly pass; any failure either confirms a boundary mismatch or must be explained in the report.

- [ ] **Step 5: Write the Stage 3 report**

Record:
- ranked findings with file references
- where endpoint, schema, and profile ownership are too tightly coupled
- suggested refactor directions that reduce drift or duplicate defaults
- the exit note for retrieval-specific questions Stage 4 must settle

- [ ] **Step 6: Commit the Stage 3 report**

Run:
```bash
git add Docs/superpowers/reviews/rag/2026-04-07-stage3-api-schema-and-request-boundaries.md
git commit -m "docs: record rag api boundary findings"
```

Expected: one docs-only commit contains the Stage 3 report.

### Task 4: Execute Stage 4 Retrieval Boundary and Data Source Review

**Files:**
- Modify: `Docs/superpowers/reviews/rag/2026-04-07-stage4-retrieval-boundaries-and-data-sources.md`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/query_expansion.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/hyde.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/query_classifier.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/semantic_cache.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/media_search.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/web_fallback.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/vector_stores/base.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/vector_stores/factory.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/vector_stores/chromadb_adapter.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/vector_stores/pgvector_adapter.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_retrieval.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_vector_store_parity.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_vector_store_admin_guardrails.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_vector_retriever_hyde.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_hyde_retrieval_merge.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_tenant_scoping.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_persistence.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_path_sanitization.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_corpus_synonyms_expansion.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_query_classifier.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_media_search.py`
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_retriever_pgvector_multi_search.py`
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_adapter_guards.py`

- [ ] **Step 1: Map the retrieval, expansion, cache, and vector-store seams**

Run:
```bash
rg -n "class (.*Retriever|.*Adapter)|def (retrieve|search|expand|initialize|create_|delete_by_filter|get_shared_cache|lookup|store|merge)|DataSource|index_namespace|vector_store_type" \
  tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py \
  tldw_Server_API/app/core/RAG/rag_service/query_expansion.py \
  tldw_Server_API/app/core/RAG/rag_service/hyde.py \
  tldw_Server_API/app/core/RAG/rag_service/query_classifier.py \
  tldw_Server_API/app/core/RAG/rag_service/semantic_cache.py \
  tldw_Server_API/app/core/RAG/rag_service/media_search.py \
  tldw_Server_API/app/core/RAG/rag_service/web_fallback.py \
  tldw_Server_API/app/core/RAG/rag_service/vector_stores/base.py \
  tldw_Server_API/app/core/RAG/rag_service/vector_stores/factory.py \
  tldw_Server_API/app/core/RAG/rag_service/vector_stores/chromadb_adapter.py \
  tldw_Server_API/app/core/RAG/rag_service/vector_stores/pgvector_adapter.py
```

Expected: a concise map of the retrieval seams, factory entry points, and cache or adapter ownership.

- [ ] **Step 2: Trace source ownership and extension boundaries**

Confirm:
- how database retrievers expose source-specific behavior
- where query expansion, HYDE, query classification, or fallback logic leaks retrieval-policy knowledge
- how vector-store factories and adapters couple settings, user namespaces, and collection names
- whether semantic caching is owned cleanly or bleeds assumptions across the API and core layers

- [ ] **Step 3: Review the targeted retrieval tests and extract the protected invariants**

For each listed test file, record:
- the main invariant it protects
- which retrieval seam it actually constrains
- whether the test covers only the happy path or also the failure or multi-tenant path

- [ ] **Step 4: Run the targeted retrieval tests**

Run:
```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_retrieval.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_vector_store_parity.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_vector_store_admin_guardrails.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_vector_retriever_hyde.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_hyde_retrieval_merge.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_tenant_scoping.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_persistence.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_path_sanitization.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_corpus_synonyms_expansion.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_query_classifier.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_media_search.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_retriever_pgvector_multi_search.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_adapter_guards.py -v
```

Expected: tests collect and mostly pass; failures either validate retrieval-boundary concerns or must be explained explicitly.

- [ ] **Step 5: Write the Stage 4 report**

Record:
- ranked findings with severity and confidence
- the exact seams where retrieval policy, cache policy, or adapter ownership blur together
- suggested refactor directions that reduce coupling at the retriever and adapter boundary
- the exit note for which post-retrieval seams Stage 5 must verify

- [ ] **Step 6: Commit the Stage 4 report**

Run:
```bash
git add Docs/superpowers/reviews/rag/2026-04-07-stage4-retrieval-boundaries-and-data-sources.md
git commit -m "docs: record rag retrieval boundary findings"
```

Expected: one docs-only commit contains the Stage 4 report.

### Task 5: Execute Stage 5 Reranking and Post-Retrieval Composition Review

**Files:**
- Modify: `Docs/superpowers/reviews/rag/2026-04-07-stage5-reranking-and-post-retrieval-composition.md`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/generation.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/guardrails.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/citations.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/response_writer.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`
- Inspect: `tldw_Server_API/app/core/RAG/rag_service/research_agent.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_two_tier_reranker.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_two_tier_gate.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_reranker_metrics.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_response_writer.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_quotes_and_numeric.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_injection_numeric.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_hard_citations_golden.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_post_verifier.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_strict_extractive_and_citations.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_failures_and_fallbacks.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_cache_invalidation.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_golden_citations.py`
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py`
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_strict_extractive_nli_api.py`
- Test: `tldw_Server_API/tests/RAG_NEW/integration/test_research_agent_loop.py`
- Test: `tldw_Server_API/tests/e2e/test_rag_generation_grounding_smoke.py`
- Test: `tldw_Server_API/tests/e2e/test_rag_post_verification_smoke.py`

- [ ] **Step 1: Map the post-retrieval control points**

Run:
```bash
rg -n "class |def (rerank|generate|stream|verify|gate|cite|write|agentic_|research_|invalidate_|quote_|check_)" \
  tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py \
  tldw_Server_API/app/core/RAG/rag_service/generation.py \
  tldw_Server_API/app/core/RAG/rag_service/guardrails.py \
  tldw_Server_API/app/core/RAG/rag_service/citations.py \
  tldw_Server_API/app/core/RAG/rag_service/response_writer.py \
  tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py \
  tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py \
  tldw_Server_API/app/core/RAG/rag_service/research_agent.py
```

Expected: a compact map of the reranking, generation, verification, response-writing, and agentic side-path entry points.

- [ ] **Step 2: Trace ownership after retrieval completes**

Confirm:
- where reranking ends and generation or verification begins
- whether citations and guardrails are leaf utilities or hidden orchestrators
- how response writing duplicates or centralizes formatting logic
- whether the agentic or research path shares clean contracts with the main pipeline or forks incompatible behavior

- [ ] **Step 3: Review the targeted post-retrieval tests and extract the protected invariants**

For each listed test file, record:
- the main post-retrieval invariant it protects
- which seam it actually constrains
- whether it checks structure, behavior, or only metadata shape

- [ ] **Step 4: Run the targeted post-retrieval tests**

Run:
```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_two_tier_reranker.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_pipeline_two_tier_gate.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_reranker_metrics.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_response_writer.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_quotes_and_numeric.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_injection_numeric.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_guardrails_hard_citations_golden.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_post_verifier.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_strict_extractive_and_citations.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_failures_and_fallbacks.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_cache_invalidation.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_golden_citations.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_agentic_api.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_strict_extractive_nli_api.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_research_agent_loop.py \
  tldw_Server_API/tests/e2e/test_rag_generation_grounding_smoke.py \
  tldw_Server_API/tests/e2e/test_rag_post_verification_smoke.py -v
```

Expected: tests collect and mostly pass; failures either validate a structural concern or must be explained as environment-specific.

- [ ] **Step 5: Write the Stage 5 report**

Record:
- ranked findings with file references
- where reranking, generation, verification, and agentic logic overlap or conflict
- suggested refactor directions that reduce post-retrieval feature tangling
- the exit note for what the final synthesis should emphasize

- [ ] **Step 6: Commit the Stage 5 report**

Run:
```bash
git add Docs/superpowers/reviews/rag/2026-04-07-stage5-reranking-and-post-retrieval-composition.md
git commit -m "docs: record rag post-retrieval findings"
```

Expected: one docs-only commit contains the Stage 5 report.

### Task 6: Execute Stage 6 Test-Gap Pass and Final Synthesis

**Files:**
- Modify: `Docs/superpowers/reviews/rag/2026-04-07-stage6-test-gaps-and-synthesis.md`
- Inspect: `Docs/superpowers/reviews/rag/2026-04-07-stage1-architecture-survey-and-inventory.md`
- Inspect: `Docs/superpowers/reviews/rag/2026-04-07-stage2-unified-pipeline-orchestration.md`
- Inspect: `Docs/superpowers/reviews/rag/2026-04-07-stage3-api-schema-and-request-boundaries.md`
- Inspect: `Docs/superpowers/reviews/rag/2026-04-07-stage4-retrieval-boundaries-and-data-sources.md`
- Inspect: `Docs/superpowers/reviews/rag/2026-04-07-stage5-reranking-and-post-retrieval-composition.md`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_decomposition.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_retrieval.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_two_tier_reranker.py`

- [ ] **Step 1: Collate findings, coverage gaps, and exit notes from every stage**

Run:
```bash
rg -n "^## Findings|^## Suggested Refactor/Actions|^## Coverage Gaps|^## Exit Note" \
  Docs/superpowers/reviews/rag/2026-04-07-stage1-architecture-survey-and-inventory.md \
  Docs/superpowers/reviews/rag/2026-04-07-stage2-unified-pipeline-orchestration.md \
  Docs/superpowers/reviews/rag/2026-04-07-stage3-api-schema-and-request-boundaries.md \
  Docs/superpowers/reviews/rag/2026-04-07-stage4-retrieval-boundaries-and-data-sources.md \
  Docs/superpowers/reviews/rag/2026-04-07-stage5-reranking-and-post-retrieval-composition.md
```

Expected: a compact view of the material that must be deduplicated and ranked in the final synthesis.

- [ ] **Step 2: Map the highest-risk findings to the tests that actually constrain them**

Record:
- which findings are well-covered by current tests
- which findings are only weakly covered by mocked or narrow tests
- which high-blast-radius seams still have little or no structural protection

- [ ] **Step 3: Run a representative cross-slice sanity pack**

Run:
```bash
source ../../.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_decomposition.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_search_agent_defaults.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_retrieval.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_two_tier_reranker.py -v
```

Expected: the representative tests pass or fail in ways that sharpen the final confidence ratings.

- [ ] **Step 4: Write the Stage 6 synthesis**

Record:
- the final ranked findings list ordered by severity
- grouped test gaps by blast radius
- the most useful suggested refactor or action sequence
- any open questions that block a stronger architectural claim

- [ ] **Step 5: Commit the Stage 6 synthesis**

Run:
```bash
git add Docs/superpowers/reviews/rag/2026-04-07-stage6-test-gaps-and-synthesis.md
git commit -m "docs: finalize rag review synthesis"
```

Expected: one docs-only commit contains the final synthesis.
