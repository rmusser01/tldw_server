# RAG Developer Docs Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate and correct the RAG developer documentation so contributors have one current guide for the active unified RAG architecture.

**Architecture:** `Docs/Code_Documentation/RAG-Developer-Guide.md` becomes the canonical contributor guide. Adjacent RAG docs are narrowed by role instead of duplicating internals, and published mirrors under `Docs/Published` are refreshed from source docs with the existing refresh script rather than edited directly.

**Tech Stack:** Markdown, FastAPI source references, Pydantic schema references, shell/rg checks, MkDocs source publishing workflow

---

## Review Feedback Incorporated

This plan explicitly addresses the pre-planning review findings:

- Published docs: source docs live under `Docs/`, but `Docs/mkdocs.yml` builds from `Docs/Published`; run `Helper_Scripts/refresh_docs_published.sh` after source-doc edits and review generated mirror changes.
- Per-file disposition: each target doc has a concrete action below so the implementation does not turn into broad churn.
- Schema duplication: the developer guide documents stable field groups, defaults, and extension-relevant knobs only; exhaustive request/response parameter listings stay in API/OpenAPI-style docs.

## File Disposition Map

**Canonical rewrite**
- `Docs/Code_Documentation/RAG-Developer-Guide.md`: Rewrite as the contributor-facing source of truth for the current unified RAG implementation.

**Legacy marker**
- `Docs/Code_Documentation/RAG-Functional-Pipeline-Guide.md`: Replace or heavily shorten into a legacy note that points contributors to the canonical developer guide. Keep only migration context if useful.

**Consumer/API docs**
- `Docs/API-related/RAG_API_Documentation.md`: Keep as concise endpoint/API reference. Align endpoint list and add a clear pointer to the developer guide for implementation internals.
- `Docs/API-related/RAG-API-Guide.md`: Keep as consumer guide. Remove or clearly demote deprecated `/agent` examples in favor of `/search`, `/search/stream`, `/batch`, and `strategy="agentic"` where applicable.

**Internal package docs**
- `tldw_Server_API/app/core/RAG/README.md`: Narrow to module orientation, active entrypoints, quick code pointers, and links. Remove stale line-number claims and references to nonexistent or obsolete status/deprecation docs unless files exist.
- `tldw_Server_API/app/core/RAG/rag_service/README.md`: Keep as package-level map for service internals. Remove active-use examples for archived preset pipelines.
- `tldw_Server_API/app/core/RAG/API_DOCUMENTATION.md`: Keep only if useful as internal detailed API notes; otherwise replace with a short pointer to `Docs/API-related/RAG_API_Documentation.md` and `Docs/Code_Documentation/RAG-Developer-Guide.md`.
- `tldw_Server_API/app/core/RAG/CAPABILITIES.md`: Keep as runtime capabilities endpoint guide. Align terminology with active schema values (`fts`, `vector`, `hybrid`; public sources from schema normalization).
- `tldw_Server_API/app/core/RAG/UNIFIED_PIPELINE_EXAMPLES.md`: Keep as examples cookbook. Ensure examples use active schema fields and avoid obsolete functional-pipeline imports.

**Generated mirrors**
- `Docs/Published/API-related/RAG_API_Documentation.md`
- `Docs/Published/API-related/RAG-API-Guide.md`
- `Docs/Published/Code_Documentation/RAG-Developer-Guide.md`
- `Docs/Published/Code_Documentation/RAG-Functional-Pipeline-Guide.md`

Do not edit these by hand. Refresh them from source with `Helper_Scripts/refresh_docs_published.sh` after source docs are updated.

## Task 1: Reconfirm Current RAG Source Surface

**Files:**
- Read: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Read: `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
- Read: `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- Read: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- Read: `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
- Read: `tldw_Server_API/app/core/RAG/rag_service/profiles.py`
- Read: `tldw_Server_API/app/core/RAG/rag_service/vector_stores/`
- Test: none

- [ ] **Step 1: Capture active endpoint decorators**

Run:
```bash
rg -n 'router = APIRouter|@router\.(get|post|put|delete|patch)' \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  tldw_Server_API/app/api/v1/endpoints/rag_health.py
```

Expected: confirms `/api/v1/rag` routers and active endpoint paths including `/search`, `/search/stream`, `/batch`, `/batch/resume/{checkpoint_id}`, `/simple`, `/advanced`, `/capabilities`, `/features`, `/health/simple`, `/health`, `/health/live`, `/health/ready`, cache/metrics/cost/batch/quality/regression health endpoints, `/feedback/implicit`, `/ablate`, and `/vlm/backends`.

- [ ] **Step 2: Capture active schema and pipeline entrypoints**

Run:
```bash
rg -n 'class UnifiedRAGRequest|class UnifiedRAGResponse|class UnifiedBatchRequest|class UnifiedBatchResponse|async def unified_rag_pipeline|async def unified_batch_pipeline|async def simple_search|async def advanced_search' \
  tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py \
  tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py
```

Expected: confirms current public schemas and internal entrypoints without relying on archived functional pipelines.

- [ ] **Step 3: Capture current profiles and data-source terminology**

Run:
```bash
rg -n 'ProfileName|_PROFILES|class DataSource' \
  tldw_Server_API/app/core/RAG/rag_service/profiles.py \
  tldw_Server_API/app/core/RAG/rag_service/types.py
```

Expected: confirms built-in profile names and internal `DataSource` enum values before writing docs.

- [ ] **Step 4: Record key facts in scratch notes**

Write a temporary local checklist in your own working notes, not a repo file. Include endpoint names, schema class names, profile names, public source names, and active internal extension modules.

Expected: later docs edits use source-backed facts.

## Task 2: Rewrite Canonical Developer Guide

**Files:**
- Modify: `Docs/Code_Documentation/RAG-Developer-Guide.md`
- Test: `rg` stale-term checks against this file

- [ ] **Step 1: Replace old functional-pipeline structure with current architecture outline**

Edit `Docs/Code_Documentation/RAG-Developer-Guide.md` to use these sections:

```markdown
# RAG Module Developer Guide

## Purpose And Ownership
## Current Architecture
## Request Flow
## Core Components
## Extension Points
## Configuration And Profiles
## Testing Guide
## Known Pitfalls
## Related Documentation
```

Expected: the guide no longer presents `functional_pipeline.py`, `rag_api.py`, `minimal_pipeline`, `standard_pipeline`, `quality_pipeline`, or `enhanced_pipeline` as active contributor APIs.

- [ ] **Step 2: Document current architecture without duplicating full schema**

In `Current Architecture`, include a concise source map:

```markdown
- API router: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Health/ops router: `tldw_Server_API/app/api/v1/endpoints/rag_health.py`
- API schemas: `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- Pipeline orchestration: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- Retrieval composition: `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
- Vector stores: `tldw_Server_API/app/core/RAG/rag_service/vector_stores/`
- Profiles: `tldw_Server_API/app/core/RAG/rag_service/profiles.py`
```

Expected: contributors can find active files quickly.

- [ ] **Step 3: Document request flow**

Describe the active flow in prose:

```text
POST /api/v1/rag/search
  -> FastAPI dependencies for auth, permissions, rate limits, and quota
  -> UnifiedRAGRequest validation and profile/default resolution
  -> per-user MediaDatabase and CharactersRAGDB injection
  -> unified_rag_pipeline(...)
  -> convert_result_to_response(...)
  -> UnifiedRAGResponse
```

Also document:
- `/search/stream` requires `enable_generation=true`
- `/batch` and `/batch/resume/{checkpoint_id}` use `UnifiedBatchRequest`
- `/simple` and `/advanced` are convenience wrappers, not separate architectures
- `/capabilities` and `/features` are discovery surfaces
- `/health` routes live in `rag_health.py`

Expected: request flow is accurate enough for contributors changing endpoint or pipeline code.

- [ ] **Step 4: Document field groups, not every field**

Create a section that groups schema knobs by concern:

```markdown
- Sources and strategy: `sources`, `sql_target_id`, `strategy`, `rag_profile`, `corpus`, `index_namespace`
- Retrieval: `search_mode`, `fts_level`, `hybrid_alpha`, `top_k`, `min_score`
- Query enhancement: `expand_query`, `expansion_strategies`, `spell_check`, PRF/HyDE/decomposition flags
- Context and chunking: late chunking, parent/sibling expansion, multi-vector spans
- Reranking: `enable_reranking`, `reranking_strategy`, model and threshold overrides
- Generation and guardrails: generation provider/model, hard citations, numeric fidelity, injection/content filters
- Feedback and observability: feedback, monitoring, trace, cost, debug flags
```

Add a sentence:

```markdown
For exhaustive parameter constraints, read `UnifiedRAGRequest` directly or use the generated OpenAPI schema; this guide intentionally documents stable groups and extension-relevant fields rather than copying the full schema.
```

Expected: avoids creating another stale full schema reference.

- [ ] **Step 5: Document extension points**

Include concrete locations:
- retrieval source: `database_retrievers.py`, `DataSource`, endpoint source normalization
- vector store: `rag_service/vector_stores/base.py`, `factory.py`, adapter module
- reranker: `advanced_reranking.py` and pipeline wiring
- query expansion: `query_expansion.py`
- profile: `profiles.py` and endpoint profile application
- response metadata: `UnifiedSearchResult`, `convert_result_to_response`, `UnifiedRAGResponse`

Expected: contributors know where to make common RAG changes.

- [ ] **Step 6: Document targeted tests**

Include commands:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_rag_request_schema_profiles.py -v
python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_rag_profiles.py -v
python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_rag_unified_response_mapping.py -v
python -m pytest tldw_Server_API/tests/RAG_NEW/integration/test_rag_health_endpoints.py -v
python -m pytest tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py -v
```

Expected: test guidance points to current tests, not removed `test_functional_pipeline.py`.

- [ ] **Step 7: Check canonical guide for stale active API references**

Run:
```bash
rg -n 'functional_pipeline|rag_api\.py|standard_pipeline|minimal_pipeline|quality_pipeline|enhanced_pipeline|test_functional_pipeline|/agent' Docs/Code_Documentation/RAG-Developer-Guide.md
```

Expected: no output, unless a clearly labeled legacy note is intentionally kept.

## Task 3: Narrow Legacy And Supporting Source Docs

**Files:**
- Modify: `Docs/Code_Documentation/RAG-Functional-Pipeline-Guide.md`
- Modify: `Docs/API-related/RAG_API_Documentation.md`
- Modify: `Docs/API-related/RAG-API-Guide.md`
- Modify: `tldw_Server_API/app/core/RAG/README.md`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/README.md`
- Modify: `tldw_Server_API/app/core/RAG/API_DOCUMENTATION.md`
- Modify: `tldw_Server_API/app/core/RAG/CAPABILITIES.md`
- Modify: `tldw_Server_API/app/core/RAG/UNIFIED_PIPELINE_EXAMPLES.md`
- Test: `rg` stale-term and broken-reference checks

- [ ] **Step 1: Convert functional pipeline guide to legacy context**

Replace `Docs/Code_Documentation/RAG-Functional-Pipeline-Guide.md` with a short legacy page:

```markdown
# RAG Functional Pipeline Guide (Legacy)

The functional preset pipelines are legacy documentation. New contributor work should use the unified RAG pipeline documented in `RAG-Developer-Guide.md`.

Current active entrypoints:
- `unified_rag_pipeline(...)`
- `unified_batch_pipeline(...)`
- `simple_search(...)`
- `advanced_search(...)`

Use this page only as migration context when reviewing older code or historical plans.
```

Expected: no active examples import `minimal_pipeline`, `standard_pipeline`, `quality_pipeline`, or `enhanced_pipeline`.

- [ ] **Step 2: Align API reference docs by role**

For `Docs/API-related/RAG_API_Documentation.md`:
- keep endpoint contracts
- ensure endpoint list matches active router names
- add a pointer to `../Code_Documentation/RAG-Developer-Guide.md`
- avoid internal implementation walkthroughs

For `Docs/API-related/RAG-API-Guide.md`:
- keep consumer examples
- remove or clearly label `/agent` and `/agent/advanced` as obsolete/deprecated if those endpoints are not active
- prefer `/api/v1/rag/search` with `strategy: "agentic"` for agentic-mode examples

Expected: API docs remain useful to consumers but do not compete with the developer guide.

- [ ] **Step 3: Narrow internal package README files**

For `tldw_Server_API/app/core/RAG/README.md`:
- replace stale line-number endpoint references with path-only references
- remove links to `IMPLEMENTATION_STATUS.md` or `DEPRECATION_NOTICE.md` if those files do not exist
- keep module map, active entrypoints, and contributor pointers
- link to `Docs/Code_Documentation/RAG-Developer-Guide.md`

For `tldw_Server_API/app/core/RAG/rag_service/README.md`:
- keep short service-module map
- remove active-use functional-pipeline examples
- document that package internals support `unified_pipeline.py`

Expected: internal README files orient code readers without becoming duplicate full guides.

- [ ] **Step 4: Decide internal API documentation disposition**

Inspect `tldw_Server_API/app/core/RAG/API_DOCUMENTATION.md`.

If it mostly duplicates `Docs/API-related/RAG_API_Documentation.md`, replace it with a short pointer:

```markdown
# RAG API Documentation

The maintained API reference lives at `Docs/API-related/RAG_API_Documentation.md`.
The contributor implementation guide lives at `Docs/Code_Documentation/RAG-Developer-Guide.md`.
```

If it contains unique internal notes that are still accurate, keep those notes but remove stale schema copies and link to the maintained docs.

Expected: one API contract source remains clearly preferred.

- [ ] **Step 5: Align capabilities and examples docs**

For `tldw_Server_API/app/core/RAG/CAPABILITIES.md`:
- use request `search_mode` values `fts`, `vector`, and `hybrid`
- clarify any user-facing aliases only if active normalization supports them
- keep the page about runtime discovery, not internals

For `tldw_Server_API/app/core/RAG/UNIFIED_PIPELINE_EXAMPLES.md`:
- keep examples that call `/api/v1/rag/search` or `unified_rag_pipeline(...)`
- remove stale imports or unsupported fields discovered during Task 1
- include a link to the canonical developer guide for extension work

Expected: examples remain current and role-specific.

- [ ] **Step 6: Check supporting docs for stale active references**

Run:
```bash
rg -n 'functional_pipeline|rag_api\.py|standard_pipeline|minimal_pipeline|quality_pipeline|enhanced_pipeline|test_functional_pipeline|/agent' \
  Docs/Code_Documentation/RAG-Functional-Pipeline-Guide.md \
  Docs/API-related/RAG_API_Documentation.md \
  Docs/API-related/RAG-API-Guide.md \
  tldw_Server_API/app/core/RAG/README.md \
  tldw_Server_API/app/core/RAG/rag_service/README.md \
  tldw_Server_API/app/core/RAG/API_DOCUMENTATION.md \
  tldw_Server_API/app/core/RAG/CAPABILITIES.md \
  tldw_Server_API/app/core/RAG/UNIFIED_PIPELINE_EXAMPLES.md
```

Expected: no output except intentionally labeled legacy/deprecated mentions.

## Task 4: Refresh Published Documentation Mirrors

**Files:**
- Generated/refresh: `Docs/Published/API-related/RAG_API_Documentation.md`
- Generated/refresh: `Docs/Published/API-related/RAG-API-Guide.md`
- Generated/refresh: `Docs/Published/Code_Documentation/RAG-Developer-Guide.md`
- Generated/refresh: `Docs/Published/Code_Documentation/RAG-Functional-Pipeline-Guide.md`
- Test: source/published diff checks

- [ ] **Step 1: Confirm `Docs/Published` is generated from source**

Run:
```bash
sed -n '1,50p' Docs/Code_Documentation/Docs_Site_Guide.md
sed -n '1,80p' Helper_Scripts/refresh_docs_published.sh
```

Expected: confirms source of truth is `Docs/`, and `Docs/Published` is refreshed by script.

- [ ] **Step 2: Refresh published docs**

Run:
```bash
bash Helper_Scripts/refresh_docs_published.sh
```

Expected: `Docs/Published/API-related` and `Docs/Published/Code_Documentation` mirror the updated source docs. The script may also refresh other curated docs if source and published mirrors were already out of sync; review `git diff --name-status Docs/Published` before staging.

- [ ] **Step 3: Confirm updated RAG source docs match published mirrors**

Run:
```bash
diff -q Docs/Code_Documentation/RAG-Developer-Guide.md Docs/Published/Code_Documentation/RAG-Developer-Guide.md
diff -q Docs/Code_Documentation/RAG-Functional-Pipeline-Guide.md Docs/Published/Code_Documentation/RAG-Functional-Pipeline-Guide.md
diff -q Docs/API-related/RAG_API_Documentation.md Docs/Published/API-related/RAG_API_Documentation.md
diff -q Docs/API-related/RAG-API-Guide.md Docs/Published/API-related/RAG-API-Guide.md
```

Expected: no output.

- [ ] **Step 4: Avoid staging unrelated published drift**

Run:
```bash
git diff --name-status Docs/Published
```

Expected: review all refreshed files. If the refresh touches many unrelated files, stage only RAG-related published mirrors and document the excluded generated drift in the final response.

## Task 5: Validate Documentation Consistency

**Files:**
- Read/validate: all modified source and published RAG docs
- Test: shell consistency checks

- [ ] **Step 1: Run source stale-reference check**

Run:
```bash
rg -n 'functional_pipeline|rag_api\.py|standard_pipeline|minimal_pipeline|quality_pipeline|enhanced_pipeline|test_functional_pipeline|/agent' \
  Docs/Code_Documentation/RAG-Developer-Guide.md \
  Docs/Code_Documentation/RAG-Functional-Pipeline-Guide.md \
  Docs/API-related/RAG_API_Documentation.md \
  Docs/API-related/RAG-API-Guide.md \
  tldw_Server_API/app/core/RAG/README.md \
  tldw_Server_API/app/core/RAG/rag_service/README.md \
  tldw_Server_API/app/core/RAG/API_DOCUMENTATION.md \
  tldw_Server_API/app/core/RAG/CAPABILITIES.md \
  tldw_Server_API/app/core/RAG/UNIFIED_PIPELINE_EXAMPLES.md
```

Expected: no output except intentionally labeled legacy/deprecated context.

- [ ] **Step 2: Run published stale-reference check**

Run:
```bash
rg -n 'functional_pipeline|rag_api\.py|standard_pipeline|minimal_pipeline|quality_pipeline|enhanced_pipeline|test_functional_pipeline|/agent' \
  Docs/Published/Code_Documentation/RAG-Developer-Guide.md \
  Docs/Published/Code_Documentation/RAG-Functional-Pipeline-Guide.md \
  Docs/Published/API-related/RAG_API_Documentation.md \
  Docs/Published/API-related/RAG-API-Guide.md
```

Expected: no output except intentionally labeled legacy/deprecated context.

- [ ] **Step 3: Confirm canonical guide links to source-backed active files**

Run:
```bash
rg -n 'rag_unified\.py|rag_health\.py|rag_schemas_unified\.py|unified_pipeline\.py|database_retrievers\.py|vector_stores|profiles\.py' Docs/Code_Documentation/RAG-Developer-Guide.md
```

Expected: all major active code paths are referenced.

- [ ] **Step 4: Confirm docs do not copy the full schema**

Run:
```bash
rg -n '"enable_|`enable_|UnifiedRAGRequest|For exhaustive parameter constraints' Docs/Code_Documentation/RAG-Developer-Guide.md
```

Expected: the guide includes grouped field discussion and a pointer to `UnifiedRAGRequest`/OpenAPI, not a long exhaustive JSON request body.

- [ ] **Step 5: Optional MkDocs build if dependencies are available**

Run:
```bash
cd Docs && python -m mkdocs build --strict
```

Expected: build succeeds. If `mkdocs` or plugins are not installed, record the missing dependency and rely on the source/published consistency checks.

## Task 6: Review, Stage, And Commit Docs Changes

**Files:**
- Stage: modified RAG source docs
- Stage: RAG-related published mirrors refreshed from source
- Do not stage: unrelated existing dirty files

- [ ] **Step 1: Inspect changed files**

Run:
```bash
git status --short
git diff --name-status
```

Expected: RAG docs and refreshed RAG published mirrors are visible; unrelated pre-existing dirty files remain untouched.

- [ ] **Step 2: Review the main canonical guide diff**

Run:
```bash
git diff -- Docs/Code_Documentation/RAG-Developer-Guide.md
```

Expected: the diff shows a current unified RAG contributor guide, not a small patch over stale functional-pipeline docs.

- [ ] **Step 3: Stage only intended files**

Run:
```bash
git add \
  Docs/Code_Documentation/RAG-Developer-Guide.md \
  Docs/Code_Documentation/RAG-Functional-Pipeline-Guide.md \
  Docs/API-related/RAG_API_Documentation.md \
  Docs/API-related/RAG-API-Guide.md \
  tldw_Server_API/app/core/RAG/README.md \
  tldw_Server_API/app/core/RAG/rag_service/README.md \
  tldw_Server_API/app/core/RAG/API_DOCUMENTATION.md \
  tldw_Server_API/app/core/RAG/CAPABILITIES.md \
  tldw_Server_API/app/core/RAG/UNIFIED_PIPELINE_EXAMPLES.md \
  Docs/Published/Code_Documentation/RAG-Developer-Guide.md \
  Docs/Published/Code_Documentation/RAG-Functional-Pipeline-Guide.md \
  Docs/Published/API-related/RAG_API_Documentation.md \
  Docs/Published/API-related/RAG-API-Guide.md
```

Expected: only intended RAG documentation files are staged.

- [ ] **Step 4: Confirm staged set**

Run:
```bash
git diff --cached --name-status
```

Expected: staged files match the intended RAG documentation set.

- [ ] **Step 5: Commit**

Run:
```bash
git commit -m "docs: consolidate RAG developer documentation"
```

Expected: commit succeeds with docs-only changes.
