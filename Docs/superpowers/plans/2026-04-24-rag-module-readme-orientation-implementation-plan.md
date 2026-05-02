# RAG Module README Orientation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `tldw_Server_API/app/core/RAG/README.md` into a concise, accurate contributor orientation with a curated advanced appendix.

**Architecture:** This is a docs-only replacement of an overgrown module README. The implementation gathers current facts from active RAG endpoint/schema/pipeline files, rewrites the README around active unified RAG paths, and validates that stale functional-pipeline guidance is no longer presented as active contributor guidance.

**Tech Stack:** Markdown, FastAPI route/source inspection, Pydantic schema inspection, `rg`, project virtualenv for optional Bandit validation.

---

## File Structure

- Modify: `tldw_Server_API/app/core/RAG/README.md`
  - Responsibility: quick contributor orientation for the active RAG module.
  - Final shape: start-here section, request flow, module map, common contributor tasks, endpoint summary, configuration/profile notes, testing commands, curated advanced notes, and accurately labeled related docs.

- Reference only: `Docs/superpowers/specs/2026-04-24-rag-module-readme-orientation-design.md`
  - Responsibility: approved design and accuracy rules.

- Reference only: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
  - Responsibility: active unified RAG API router.

- Reference only: `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
  - Responsibility: active RAG operational health/cache/metrics router.

- Reference only: `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
  - Responsibility: public unified RAG request/response schemas.

- Reference only: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
  - Responsibility: active RAG pipeline entrypoints.

- Reference only: `tldw_Server_API/app/core/RAG/rag_service/profiles.py`
  - Responsibility: internal profile helper definitions and profile merge semantics.

- Reference only: `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
  - Responsibility: retrieval source names and retriever classes.

- Reference only: `tldw_Server_API/app/core/RAG/rag_service/vector_stores/`
  - Responsibility: vector-store adapter names and extension point.

No Python source files should be modified.

---

### Task 1: Gather Current RAG Facts

**Files:**
- Read: `Docs/superpowers/specs/2026-04-24-rag-module-readme-orientation-design.md`
- Read: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Read: `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
- Read: `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- Read: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- Read: `tldw_Server_API/app/core/RAG/rag_service/profiles.py`
- Read: `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
- Read: `tldw_Server_API/app/core/RAG/rag_service/vector_stores/`
- Test: source-inspection commands only

- [ ] **Step 1: Confirm the approved scope**

Run:

```bash
sed -n '1,240p' Docs/superpowers/specs/2026-04-24-rag-module-readme-orientation-design.md
```

Expected: spec describes a README-only orientation update, no Python behavior changes, and the link caveat for stale deep docs.

- [ ] **Step 2: Capture active endpoint inventory**

Run:

```bash
rg -n '@router\.(get|post)\(' -A 8 tldw_Server_API/app/api/v1/endpoints/rag_unified.py tldw_Server_API/app/api/v1/endpoints/rag_health.py
```

Expected: output includes `/search`, `/search/stream`, `/batch`, `/batch/resume/{checkpoint_id}`, `/feedback/implicit`, `/capabilities`, `/features`, `/simple`, `/advanced`, `/vlm/backends`, `/ablate`, and health/cache/metrics endpoints.

- [ ] **Step 3: Confirm router prefixes**

Run:

```bash
rg -n 'router = APIRouter|include_router\(.*rag|rag_unified|rag_health' tldw_Server_API/app/api/v1/endpoints/rag_unified.py tldw_Server_API/app/api/v1/endpoints/rag_health.py tldw_Server_API/app/main.py
```

Expected: both routers use `/api/v1/rag` and are included from `main.py`.

- [ ] **Step 4: Confirm public schema and profile values**

Run:

```bash
rg -n 'class UnifiedRAGRequest|class UnifiedRAGResponse|class UnifiedBatchRequest|class UnifiedBatchResponse|rag_profile' tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py
```

Expected: public schema classes exist and `rag_profile` accepts `fast`, `balanced`, and `accuracy`.

- [ ] **Step 5: Confirm internal profile helper names**

Run:

```bash
sed -n '1,340p' tldw_Server_API/app/core/RAG/rag_service/profiles.py
```

Expected: internal helpers include additional profile names such as `production`, `research`, and `cheap`; README must distinguish these from public request `rag_profile` values if mentioned.

- [ ] **Step 6: Confirm active pipeline entrypoints**

Run:

```bash
rg -n 'async def unified_rag_pipeline|async def unified_batch_pipeline|async def simple_search|async def advanced_search' tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py
```

Expected: active entrypoints are `unified_rag_pipeline`, `unified_batch_pipeline`, `simple_search`, and `advanced_search`.

- [ ] **Step 7: Confirm retrieval and vector-store terminology**

Run:

```bash
rg -n 'class .*Retriever|DataSource|RetrievalConfig|media_db|notes|characters|chats|kanban|sql' tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py
find tldw_Server_API/app/core/RAG/rag_service/vector_stores -maxdepth 1 -type f -name '*.py' -print | sort
```

Expected: current retrieval sources and vector-store adapters are visible; README should use these names rather than older aliases.

- [ ] **Step 8: Confirm current test locations**

Run:

```bash
rg --files tldw_Server_API/tests/RAG_NEW tldw_Server_API/tests/RAG tldw_Server_API/tests/e2e | rg 'rag|RAG'
```

Expected: current unit/integration/e2e RAG tests are found; README should list representative folders and targeted commands without inventing missing files.

---

### Task 2: Replace README With Orientation Structure

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/README.md`
- Test: stale-term check should fail before the rewrite and pass after the rewrite

- [ ] **Step 1: Run stale active-guide check before editing**

Run:

```bash
rg -n 'functional_pipeline\.py|rag_api\.py|standard_pipeline|minimal_pipeline|quality_pipeline|IMPLEMENTATION_STATUS\.md|DEPRECATION_NOTICE\.md' tldw_Server_API/app/core/RAG/README.md
```

Expected: current README reports stale or missing-doc references. This confirms the rewrite is addressing known drift.

- [ ] **Step 2: Replace the top-level README with the approved structure**

Edit `tldw_Server_API/app/core/RAG/README.md` with these sections in this order:

```markdown
# RAG Module

## Start Here

## Request Flow

## Module Map

## Common Contributor Tasks

## Current Endpoints

## Configuration And Profiles

## Testing

## Advanced Notes

## Related Documentation
```

Expected: the old long README content is replaced rather than patched in place, with only concise source-verified advanced notes carried forward.

- [ ] **Step 3: Write the opening role statement**

Include text equivalent to:

```markdown
This README is a contributor orientation for the backend RAG module. It is not the full API reference or a complete implementation guide; use it to find the active code paths and then follow the linked docs or source files for detail.
```

Expected: readers know the README's scope immediately.

- [ ] **Step 4: Add `Start Here`**

Document active entrypoints:

```markdown
- API routers: `rag_unified.py`, `rag_health.py`
- Public schemas: `UnifiedRAGRequest`, `UnifiedRAGResponse`, `UnifiedBatchRequest`, `UnifiedBatchResponse`
- Core pipeline: `unified_rag_pipeline()`, `unified_batch_pipeline()`
- Convenience functions: `simple_search()`, `advanced_search()`
```

Expected: no source line numbers and no archived functional-pipeline names.

- [ ] **Step 5: Add `Request Flow`**

Summarize the active flow:

```markdown
POST /api/v1/rag/search
  -> UnifiedRAGRequest validation
  -> auth/rate-limit/permission dependencies
  -> profile/default resolution
  -> per-user DB adapter/path injection
  -> unified_rag_pipeline(...)
  -> UnifiedRAGResponse mapping
```

Expected: the flow mentions adjacent batch, streaming, capability, and health paths but does not over-explain every internal branch.

- [ ] **Step 6: Add `Module Map`**

Group files by responsibility:

```markdown
- `API_DOCUMENTATION.md`: local endpoint/parameter reference.
- `CAPABILITIES.md`: feature discovery and capability summary.
- `UNIFIED_PIPELINE_EXAMPLES.md`: request and pipeline examples to verify against current schema.
- `exceptions.py`: RAG-specific exception types.
- `rag_custom_metrics.py`: RAG metrics helpers.
- `rag_service/`: implementation package for retrieval, generation, guardrails, citations, profiles, vector stores, metrics, and utilities.
```

Expected: the map is concise and avoids claiming stale docs are authoritative.

---

### Task 3: Fill Contributor Tasks, Endpoints, Profiles, Tests, And Advanced Notes

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/README.md`
- Test: source-inspection and stale-term commands

- [ ] **Step 1: Add `Common Contributor Tasks`**

Map common tasks to files:

```markdown
- Add retrieval behavior: `rag_service/database_retrievers.py`
- Add vector-store support: `rag_service/vector_stores/`
- Adjust profiles/defaults: `rag_service/profiles.py` plus endpoint payload handling in `rag_unified.py`
- Adjust request/response contract: `rag_schemas_unified.py` and endpoint response mapping
- Adjust reranking: `rag_service/advanced_reranking.py`
- Adjust generation/streaming: `rag_service/generation.py`, `rag_service/response_writer.py`, and `/search/stream`
- Adjust guardrails/citations/claims: `guardrails.py`, `citations.py`, `claims.py`, `faithfulness.py`, and related tests
```

Expected: contributor task mappings are practical and source-verified.

- [ ] **Step 2: Add grouped `Current Endpoints`**

Group endpoints by purpose rather than by source line:

```markdown
- Primary search: `POST /api/v1/rag/search`
- Streaming search: `POST /api/v1/rag/search/stream`
- Convenience search: `GET /api/v1/rag/simple`, `GET /api/v1/rag/advanced`
- Batch: `POST /api/v1/rag/batch`, `POST /api/v1/rag/batch/resume/{checkpoint_id}`
- Feedback and experiments: `POST /api/v1/rag/feedback/implicit`, `POST /api/v1/rag/ablate`
- Discovery: `GET /api/v1/rag/capabilities`, `GET /api/v1/rag/features`, `GET /api/v1/rag/vlm/backends`
- Operational: `GET /api/v1/rag/health`, `GET /api/v1/rag/health/live`, `GET /api/v1/rag/health/ready`, cache/metrics/cost/batch-job/regression endpoints from `rag_health.py`
```

Expected: endpoint names match the router decorators inspected in Task 1.

- [ ] **Step 3: Add `Configuration And Profiles`**

Document profile behavior carefully:

```markdown
- Public request `rag_profile` values are `fast`, `balanced`, and `accuracy`.
- `profiles.py` also contains internal helper profiles such as `production`, `research`, and `cheap`; do not assume they are accepted directly by `UnifiedRAGRequest`.
- Explicit request fields override profile defaults.
- `index_namespace` can be set directly or via `corpus`.
- Production deployments should prefer injected DB adapters/paths over raw fallback assumptions.
```

Expected: README avoids confusing public request schema with internal helpers.

- [ ] **Step 4: Add `Testing`**

Include representative focused commands:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_rag_request_schema_profiles.py -v
python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_rag_profiles.py -v
python -m pytest tldw_Server_API/tests/RAG_NEW/integration/test_rag_health_endpoints.py -v
python -m pytest tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py -v
python -m pytest tldw_Server_API/tests/RAG/test_rag_sources_sql_validation.py -v
```

Expected: commands reference existing files only and are clearly examples, not mandatory for a docs-only README edit.

- [ ] **Step 5: Add `Advanced Notes`**

Keep short notes only on:

```markdown
- Streaming requires `enable_generation=true`.
- Vector search depends on embeddings/vector-store availability; hybrid search can still use FTS.
- Citations, hard-citation gating, claims, numeric fidelity, and post-verification can abstain or ask for clarification when evidence is weak.
- Reranking options can carry cost/latency tradeoffs; two-tier reranking is quality-oriented.
- Multi-tenant or production usage should avoid assumptions about raw SQL fallbacks and should preserve user/tenant scoping.
```

Expected: appendix is concise and source-verified; long cURL examples and parameter catalogs stay out of the README.

- [ ] **Step 6: Add `Related Documentation` with accurate role labels**

Use labels like:

```markdown
- `API_DOCUMENTATION.md`: local endpoint/parameter reference.
- `CAPABILITIES.md`: capability summary.
- `UNIFIED_PIPELINE_EXAMPLES.md`: examples to verify against the current schema.
- `rag_service/README.md`: package-level implementation map.
- `Docs/API-related/RAG_API_Documentation.md`: broader API doc; verify against current schema before treating examples as definitive.
- `Docs/Code_Documentation/RAG-Developer-Guide.md`: broader guide that may contain legacy context; prefer current source files for active behavior.
```

Expected: no stale broader guide is called canonical, authoritative, or source of truth.

---

### Task 4: Validate And Commit README Rewrite

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/README.md`
- Test: Markdown/source-inspection commands, optional Bandit record

- [ ] **Step 1: Run stale active-guide check after editing**

Run:

```bash
rg -n 'functional_pipeline\.py|rag_api\.py|standard_pipeline|minimal_pipeline|quality_pipeline|IMPLEMENTATION_STATUS\.md|DEPRECATION_NOTICE\.md' tldw_Server_API/app/core/RAG/README.md
```

Expected: no output. If output exists, it must be clearly marked legacy context; prefer removing it entirely from this README.

- [ ] **Step 2: Check that stale authority language is absent**

Run:

```bash
rg -n 'canonical|authoritative|source of truth' tldw_Server_API/app/core/RAG/README.md
```

Expected: no output, unless the term appears only in a warning not to treat stale docs that way. Prefer no output.

- [ ] **Step 3: Check endpoint and profile terms**

Run:

```bash
rg -n '/api/v1/rag/(search|search/stream|batch|batch/resume|simple|advanced|capabilities|features|vlm/backends|health)' tldw_Server_API/app/core/RAG/README.md
rg -n 'fast|balanced|accuracy|production|research|cheap' tldw_Server_API/app/core/RAG/README.md
```

Expected: active endpoint references are present; public profiles are distinguished from internal helper profiles if both are mentioned.

- [ ] **Step 4: Review final README headings**

Run:

```bash
rg -n '^#{1,3} ' tldw_Server_API/app/core/RAG/README.md
```

Expected: headings match the planned structure and the document is much shorter than the previous long README.

- [ ] **Step 5: Optional project security checklist record**

Run only if needed for the final project checklist:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/RAG -f json -o /tmp/bandit_rag_readme_orientation.json
```

Expected: any Bandit findings are baseline findings in untouched Python files. A README-only change should not introduce Python findings.

- [ ] **Step 6: Review diff**

Run:

```bash
git diff -- tldw_Server_API/app/core/RAG/README.md
```

Expected: diff replaces stale long-form content with a concise orientation and does not touch unrelated files.

- [ ] **Step 7: Commit README update**

Run:

```bash
git add tldw_Server_API/app/core/RAG/README.md
git commit -m "docs: refresh rag module readme orientation"
```

Expected: commit contains only `tldw_Server_API/app/core/RAG/README.md`.
