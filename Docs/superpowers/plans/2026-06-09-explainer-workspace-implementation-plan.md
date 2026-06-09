# Explainer Workspace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a persisted `/explainer` WebUI workspace that can recursively explain either a learning goal or selected sources, with Jobs-backed generation and single-item Chatbook export.

**Architecture:** Explainer uses a first-class per-user `Explainer.db` with a focused repository/service layer, FastAPI endpoints under `/api/v1/explainer`, and Jobs workers for generation. The WebUI owns the interactive tree/detail experience, but all session state, source selections, citations, generation metadata, and export payloads are persisted server-side. Chatbooks gets a first-class `explainer_session` content type, while import also recognizes the compatibility fallback `generated_document` with `metadata.subtype = "explainer_session"`.

**Tech Stack:** FastAPI, Pydantic, SQLite, existing `DatabasePaths`, existing AuthNZ `get_request_user`, core `JobManager`/`WorkerSDK`, existing RAG/media search, Chatbook service, Next.js page shim, React, Ant Design, lucide-react, TanStack Query, Vitest, Playwright.

---

## File Structure

### Backend Persistence And API

- Create `tldw_Server_API/app/api/v1/schemas/explainer.py`
  - Pydantic request/response models for sessions, nodes, sources, citations, generation metadata, expansion jobs, and Chatbook export.
- Create `tldw_Server_API/app/api/v1/endpoints/explainer.py`
  - FastAPI router for `/api/v1/explainer`; performs auth, dependency wiring, HTTP error mapping, and response serialization.
- Create `tldw_Server_API/app/api/v1/API_Deps/Explainer_DB_Deps.py`
  - Cached per-user `ExplainerDatabase` dependency, similar to the existing Slides/ChaCha dependency style.
- Modify `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
  - Add `DatabasePaths.get_explainer_db_path(user_id)`.
- Create `tldw_Server_API/app/core/DB_Management/Explainer_DB.py`
  - SQLite schema creation/migration and low-level transaction helpers for sessions, nodes, sources, citations, and generation metadata.
- Create `tldw_Server_API/app/core/Explainer/models.py`
  - Dataclasses/enums for core Explainer entities and status values.
- Create `tldw_Server_API/app/core/Explainer/repository.py`
  - Ownership-aware CRUD, tree loading, soft archive/delete, and citation snapshot persistence.
- Create `tldw_Server_API/app/core/Explainer/service.py`
  - Session creation, settings updates, question answers, node creation/deletion, expansion enqueueing, retry setup, and export orchestration.
- Modify `tldw_Server_API/app/api/v1/router_groups/content.py`
  - Register the Explainer router as a content/workspace router with route key `explainer`.
- Modify `tldw_Server_API/app/api/v1/router_groups/minimal.py`
  - Add Explainer to optional minimal routers for endpoint integration tests without forcing heavy generation imports.

### Backend Generation Jobs

- Create `tldw_Server_API/app/core/Explainer/jobs.py`
  - Domain constants, enqueue helper, job payload validation, and job handler entry point.
- Create `tldw_Server_API/app/core/Explainer/jobs_worker.py`
  - `WorkerSDK` runner for local/ops execution.
- Create `tldw_Server_API/app/core/Explainer/prompting.py`
  - Versioned prompt builders for clarify, explain, plan, and both intents.
- Create `tldw_Server_API/app/core/Explainer/grounding.py`
  - Source-only/source-led/open validation and evidence-state resolution.
- Create `tldw_Server_API/app/core/Explainer/retrieval.py`
  - Selected-source ownership validation and source context retrieval through existing media/RAG APIs.

### Chatbook Integration

- Create `tldw_Server_API/app/core/Explainer/chatbook_adapter.py`
  - Serialize a complete Explainer session to structured JSON plus rendered reading form; restore a session on import.
- Modify `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`
  - Add `ContentType.EXPLAINER_SESSION`, `ChatbookContent.explainer_sessions`, manifest item support, and ID aggregation.
- Modify `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
  - Expose the new content type in request/response schemas.
- Modify `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
  - Add `_collect_explainer_sessions`, import support, generated-document fallback routing, and manifest statistics/metadata.
- Modify `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
  - Only if endpoint validation or preview selection explicitly enumerates content types.

### Frontend API And Route

- Create `apps/tldw-frontend/pages/explainer.tsx`
  - Next.js dynamic page shim.
- Create `apps/tldw-frontend/extension/routes/option-explainer.tsx`
  - WebUI route wrapper using `OptionLayout`, `RouteErrorBoundary`, and `PageShell`.
- Modify `apps/tldw-frontend/extension/routes/route-registry.tsx`
  - Add lazy import, nav entry in `workspace`, and a lucide icon such as `SplitSquareVertical` or `Workflow`.
- Modify `apps/packages/ui/src/routes/route-metadata.ts`
  - Add `/explainer` to `AUDITED_ROOT_ROUTE_PATHS` and `ROUTE_METADATA`.
- Modify `apps/packages/ui/src/services/tldw/openapi-guard.ts`
  - Add all `/api/v1/explainer` paths used by the client.
- Modify `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Add typed `Explainer*` interfaces and client methods.
- Create `apps/packages/ui/src/components/Option/Explainer/`
  - Workspace components, hooks, API adapters, tree utilities, and tests.
- Modify `apps/packages/ui/src/public/_locales/en/option.json`
  - Add navigation and Explainer UI strings.
- Modify `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
  - Include `/explainer` in smoke coverage if route metadata does not already cover it.
- Create `apps/tldw-frontend/e2e/utils/page-objects/ExplainerPage.ts`
  - Page object for mocked Explainer workflows.

### Tests

- Create `tldw_Server_API/tests/Explainer/test_explainer_repository.py`
- Create `tldw_Server_API/tests/Explainer/test_explainer_endpoints.py`
- Create `tldw_Server_API/tests/Explainer/test_explainer_jobs.py`
- Create `tldw_Server_API/tests/Explainer/test_explainer_chatbook_export.py`
- Modify or add Chatbook tests under `tldw_Server_API/tests/Chatbooks/` for `explainer_session` import/export.
- Create `apps/packages/ui/src/components/Option/Explainer/__tests__/ExplainerWorkspace.test.tsx`
- Create `apps/packages/ui/src/components/Option/Explainer/__tests__/explainer-tree.test.ts`
- Create `apps/packages/ui/src/services/__tests__/tldw-api-client.explainer.test.ts`
- Create `apps/tldw-frontend/e2e/explainer.spec.ts`

---

## Design Decisions Locked By This Plan

- Use a separate per-user `Explainer.db`, not the existing ChaChaNotes database. This limits schema coupling and keeps future import/restore work reversible.
- Add a first-class Chatbook content type `explainer_session` now. Keep generated-document subtype import compatibility for older or fallback exports.
- Add `GET /api/v1/explainer/jobs/{job_id}` even though the design spec only said to follow existing Jobs status. There is no single safe global Jobs status endpoint for this UI, so Explainer needs an ownership-checked status endpoint.
- Treat `Source-only` insufficient retrieval as a completed `insufficient` child node, not a failed job.
- Keep generated content and source excerpts out of logs. Tests should assert that error payloads do not include prompt text or citation excerpts.
- First release source picker supports ingested media/documents and notes. Web URLs are supported only after they are ingested into the library as media/web records.

---

### Task 1: Backend Persistence And CRUD API

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/Explainer_DB.py`
- Create: `tldw_Server_API/app/core/Explainer/models.py`
- Create: `tldw_Server_API/app/core/Explainer/repository.py`
- Create: `tldw_Server_API/app/core/Explainer/service.py`
- Create: `tldw_Server_API/app/api/v1/API_Deps/Explainer_DB_Deps.py`
- Create: `tldw_Server_API/app/api/v1/schemas/explainer.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/explainer.py`
- Modify: `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Test: `tldw_Server_API/tests/Explainer/test_explainer_repository.py`
- Test: `tldw_Server_API/tests/Explainer/test_explainer_endpoints.py`
- Test: `tldw_Server_API/tests/Services/test_router_groups_contract.py`

- [x] **Step 1: Create an implementation Backlog task**

Create a task before code edits, for example:

```text
Title: Implement Explainer backend persistence and API
References: TASK-546, TASK-547, Docs/superpowers/specs/2026-06-09-explainer-workspace-design.md
Modified files: all Task 1 files
```

- [x] **Step 2: Write failing repository tests**

Add tests that define the persistence contract before implementation:

```python
def test_create_goal_session_persists_root_node(tmp_path):
    db = ExplainerDatabase(tmp_path / "Explainer.db")
    repo = ExplainerRepository(db)

    session = repo.create_session(
        owner_user_id="7",
        title="Learn attention",
        mode="goal",
        output_intent="explain",
        grounding="open",
        depth_preset="standard",
        selected_sources=[],
        root_prompt="Explain transformer attention",
    )

    loaded = repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert loaded.root_node_ids
    assert loaded.nodes[loaded.root_node_ids[0]].title == "Explain transformer attention"
```

Also include:

```python
def test_repository_rejects_cross_user_session_access(tmp_path):
    db = ExplainerDatabase(tmp_path / "Explainer.db")
    repo = ExplainerRepository(db)
    session = repo.create_session(
        owner_user_id="7",
        title="Private",
        mode="goal",
        output_intent="plan",
        grounding="open",
        depth_preset="quick",
        selected_sources=[],
        root_prompt="Private topic",
    )

    assert repo.get_session(session.id, owner_user_id="8") is None
```

- [x] **Step 3: Run repository tests and verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer/test_explainer_repository.py -v
```

Expected: FAIL because `ExplainerDatabase` and `ExplainerRepository` do not exist.

- [x] **Step 4: Implement the database schema and repository**

Create tables with explicit ownership columns and soft delete fields:

```sql
CREATE TABLE IF NOT EXISTS explainer_sessions (
  id TEXT PRIMARY KEY,
  owner_user_id TEXT NOT NULL,
  title TEXT NOT NULL,
  mode TEXT NOT NULL CHECK (mode IN ('goal', 'sources')),
  status TEXT NOT NULL CHECK (status IN ('draft', 'active', 'archived', 'error')),
  output_intent TEXT NOT NULL CHECK (output_intent IN ('explain', 'plan', 'both')),
  grounding TEXT NOT NULL CHECK (grounding IN ('source_only', 'source_led', 'open')),
  depth_preset TEXT NOT NULL CHECK (depth_preset IN ('quick', 'standard', 'deep')),
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  archived_at TEXT
);

CREATE TABLE IF NOT EXISTS explainer_nodes (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES explainer_sessions(id) ON DELETE CASCADE,
  parent_id TEXT REFERENCES explainer_nodes(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  title TEXT NOT NULL,
  body TEXT,
  kind TEXT NOT NULL,
  intent TEXT NOT NULL,
  status TEXT NOT NULL,
  evidence_state TEXT NOT NULL,
  outside_knowledge_used INTEGER NOT NULL DEFAULT 0,
  question_options_json TEXT,
  selected_option_id TEXT,
  selected_custom_answer TEXT,
  generation_metadata_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  deleted_at TEXT
);
```

Add companion tables for selected sources and citations rather than storing them only as JSON, so ownership checks and export serialization stay reliable.

- [x] **Step 5: Make repository tests pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer/test_explainer_repository.py -v
```

Expected: PASS.

- [x] **Step 6: Write failing endpoint tests**

Use the existing FastAPI test-client patterns from nearby endpoint tests. Cover:

- `POST /api/v1/explainer/sessions` creates a persisted Goal session.
- `POST /api/v1/explainer/sessions` rejects `source_only` with no selected sources.
- `GET /api/v1/explainer/sessions` lists only the current user's sessions.
- `PATCH /api/v1/explainer/sessions/{session_id}` updates output intent and grounding.
- `DELETE /api/v1/explainer/sessions/{session_id}` archives rather than hard-deleting.

- [x] **Step 7: Run endpoint tests and verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer/test_explainer_endpoints.py -v
```

Expected: FAIL because the endpoint router is not implemented/registered.

- [x] **Step 8: Implement schemas, dependency, service, and router**

Endpoint surface:

```text
POST   /api/v1/explainer/sessions
GET    /api/v1/explainer/sessions
GET    /api/v1/explainer/sessions/{session_id}
PATCH  /api/v1/explainer/sessions/{session_id}
DELETE /api/v1/explainer/sessions/{session_id}
POST   /api/v1/explainer/sessions/{session_id}/nodes
PATCH  /api/v1/explainer/sessions/{session_id}/nodes/{node_id}
DELETE /api/v1/explainer/sessions/{session_id}/nodes/{node_id}
```

Endpoint dependencies should use:

```python
current_user: User = Depends(get_request_user)
db: ExplainerDatabase = Depends(get_explainer_db)
```

Do not put LLM or retrieval imports in `endpoints/explainer.py`; keep them in the later jobs layer so the router remains lightweight.

- [x] **Step 9: Register routers and run router contract tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -v
```

Expected: PASS and route key `explainer` is present where expected.

- [x] **Step 10: Run Task 1 tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer/test_explainer_repository.py tldw_Server_API/tests/Explainer/test_explainer_endpoints.py -v
```

Expected: PASS.

- [x] **Step 11: Commit Task 1**

```bash
git add tldw_Server_API/app/core/DB_Management/Explainer_DB.py \
  tldw_Server_API/app/core/DB_Management/db_path_utils.py \
  tldw_Server_API/app/core/Explainer \
  tldw_Server_API/app/api/v1/API_Deps/Explainer_DB_Deps.py \
  tldw_Server_API/app/api/v1/schemas/explainer.py \
  tldw_Server_API/app/api/v1/endpoints/explainer.py \
  tldw_Server_API/app/api/v1/router_groups/content.py \
  tldw_Server_API/app/api/v1/router_groups/minimal.py \
  tldw_Server_API/tests/Explainer \
  tldw_Server_API/tests/Services/test_router_groups_contract.py
git commit -m "feat: add explainer persistence api"
```

---

### Task 2: Jobs-Backed Expansion, Grounding, And Job Status

**Files:**
- Create: `tldw_Server_API/app/core/Explainer/jobs.py`
- Create: `tldw_Server_API/app/core/Explainer/jobs_worker.py`
- Create: `tldw_Server_API/app/core/Explainer/prompting.py`
- Create: `tldw_Server_API/app/core/Explainer/grounding.py`
- Create: `tldw_Server_API/app/core/Explainer/retrieval.py`
- Modify: `tldw_Server_API/app/core/Explainer/service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/explainer.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/explainer.py`
- Test: `tldw_Server_API/tests/Explainer/test_explainer_jobs.py`
- Test: `tldw_Server_API/tests/Explainer/test_explainer_endpoints.py`

- [x] **Step 1: Create or update Backlog task**

Either create a new implementation task for Jobs generation or update the Task 1 implementation task if the same PR slice is still small.

- [x] **Step 2: Write failing service/job tests**

Cover:

```python
def test_expand_marks_node_queued_and_creates_job(fake_job_manager, explainer_repo):
    service = ExplainerService(repo=explainer_repo, job_manager=fake_job_manager)
    accepted = service.enqueue_node_expansion(
        session_id="session-1",
        node_id="node-1",
        owner_user_id="7",
        intent="both",
    )

    assert accepted.status == "queued"
    assert accepted.job_id
    assert explainer_repo.get_node("session-1", "node-1", owner_user_id="7").status == "queued"
```

Also cover:

- The job handler writes child nodes and `generation_metadata`.
- Provider failure marks the target node `error`.
- `source_only` insufficient retrieval creates a child node with `evidence_state = "insufficient"` and `outside_knowledge_used = False`.
- Cross-user job status returns 404.

- [x] **Step 3: Run job tests and verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer/test_explainer_jobs.py -v
```

Expected: FAIL because Jobs integration does not exist.

- [x] **Step 4: Implement enqueue and status endpoints**

Add endpoints:

```text
POST /api/v1/explainer/sessions/{session_id}/nodes/{node_id}/expand
POST /api/v1/explainer/sessions/{session_id}/nodes/{node_id}/answer-question
GET  /api/v1/explainer/jobs/{job_id}
```

`expand` returns:

```json
{
  "jobId": "123",
  "sessionId": "session-1",
  "nodeId": "node-1",
  "status": "queued"
}
```

`GET /jobs/{job_id}` must load the Jobs row and verify `owner_user_id == current_user.id` and `domain == "explainer"` before returning status.

- [x] **Step 5: Implement Explainer Jobs constants and handler**

Use the existing Jobs domain pattern:

```python
EXPLAINER_DOMAIN = "explainer"
EXPLAINER_JOB_TYPE = "node_expansion"
EXPLAINER_QUEUE = "default"
```

Use an idempotency key with session, node, intent, and answer version:

```python
idempotency_key=f"explainer:{session_id}:{node_id}:{intent}:{answer_revision}"
```

- [x] **Step 6: Implement prompt builders with deterministic test seams**

`prompting.py` should expose pure functions:

```python
def build_node_expansion_prompt(*, session, node, source_context, intent, grounding) -> ExplainerPrompt:
    ...
```

Tests should inject a fake generator into the job handler rather than calling a real LLM.

- [x] **Step 7: Implement retrieval and grounding validation**

`retrieval.py` should validate selected source ownership before returning excerpts. `grounding.py` should convert handler output to one of:

```text
supported
partially_supported
uncited
insufficient
```

For `source_only`, reject or rewrite uncited generated claims into an `insufficient` node. Do not silently downgrade to outside knowledge.

- [x] **Step 8: Run Task 2 backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer/test_explainer_jobs.py tldw_Server_API/tests/Explainer/test_explainer_endpoints.py -v
```

Expected: PASS.

- [x] **Step 9: Commit Task 2**

```bash
git add tldw_Server_API/app/core/Explainer \
  tldw_Server_API/app/api/v1/endpoints/explainer.py \
  tldw_Server_API/app/api/v1/schemas/explainer.py \
  tldw_Server_API/tests/Explainer
git commit -m "feat: add explainer expansion jobs"
```

---

### Task 3: Chatbook Export And Import

**Files:**
- Create: `tldw_Server_API/app/core/Explainer/chatbook_adapter.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_models.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/explainer.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/explainer.py`
- Test: `tldw_Server_API/tests/Explainer/test_explainer_chatbook_export.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py`

- [x] **Step 1: Create or update Backlog task**

Use a dedicated task if this becomes a standalone PR slice:

```text
Title: Add Explainer Chatbook export and import support
References: TASK-546, TASK-547
```

- [x] **Step 2: Write failing Chatbook adapter tests**

Test a complete tree export:

```python
def test_chatbook_adapter_serializes_complete_session(explainer_repo):
    payload = build_explainer_chatbook_payload(
        repo=explainer_repo,
        session_id="session-1",
        owner_user_id="7",
    )

    assert payload["type"] == "explainer_session"
    assert payload["structured"]["session"]["id"] == "session-1"
    assert payload["structured"]["nodes"]
    assert payload["rendered"]["markdown"].startswith("#")
    assert payload["structured"]["nodes"][0]["citations"][0]["excerpt"]
```

Also cover:

- Export rejects cross-user session access.
- Export includes clarifying questions and selected answers.
- Export includes generation metadata but no API keys or prompt secrets.
- Import restores an `explainer_session`.
- Import detects `generated_document` with `metadata.subtype = "explainer_session"` and routes it to Explainer restoration.

- [x] **Step 3: Run Chatbook tests and verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer/test_explainer_chatbook_export.py tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py -v
```

Expected: FAIL because Chatbook support is not implemented.

- [x] **Step 4: Add `explainer_session` to Chatbook models**

Add:

```python
class ContentType(str, Enum):
    ...
    EXPLAINER_SESSION = "explainer_session"
```

Add `explainer_sessions: dict[str, Any]` to `ChatbookContent` and include it in `get_all_ids`.

- [x] **Step 5: Implement structured plus rendered export payload**

`chatbook_adapter.py` should output:

```json
{
  "format": "tldw.explainer_session.v1",
  "type": "explainer_session",
  "structured": {
    "session": {},
    "selectedSources": [],
    "nodes": [],
    "citations": []
  },
  "rendered": {
    "markdown": "# Session title\n\n..."
  },
  "metadata": {
    "schemaVersion": 1,
    "exportedAt": "...",
    "sourceBundling": "references_only"
  }
}
```

Keep original linked source documents out of the archive unless a later option explicitly includes them.

- [x] **Step 6: Add Chatbook service collection**

Add `_collect_explainer_sessions` that writes:

```text
content/explainer_sessions/session_{session_id}.json
```

Then append a manifest item:

```python
ContentItem(
    id=session_id,
    type=ContentType.EXPLAINER_SESSION,
    title=payload["structured"]["session"]["title"],
    file_path=f"content/explainer_sessions/session_{session_id}.json",
    metadata={"format": "tldw.explainer_session.v1"},
)
```

- [x] **Step 7: Add Explainer export endpoint**

Add:

```text
POST /api/v1/explainer/sessions/{session_id}/export-chatbook
```

The endpoint re-checks session ownership, then delegates to `ChatbookService.export_chatbook(...)` with:

```python
content_selections={ContentType.EXPLAINER_SESSION: [session_id]}
include_media=False
include_embeddings=False
include_generated_content=True
async_mode=True
```

Return the normal Chatbooks export response shape.

- [x] **Step 8: Implement import restoration**

Import should restore Explainer sessions by creating a new session ID for the importing user, preserving original IDs in metadata for traceability. Cross-user source references remain references only; if the source does not exist for the importing user, mark it `unresolved` in selected-source metadata.

- [x] **Step 9: Run Task 3 tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer/test_explainer_chatbook_export.py tldw_Server_API/tests/Chatbooks -k explainer -v
```

Expected: PASS.

- [x] **Step 10: Commit Task 3**

```bash
git add tldw_Server_API/app/core/Explainer/chatbook_adapter.py \
  tldw_Server_API/app/core/Chatbooks/chatbook_models.py \
  tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py \
  tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
  tldw_Server_API/app/api/v1/endpoints/explainer.py \
  tldw_Server_API/app/api/v1/schemas/explainer.py \
  tldw_Server_API/tests/Explainer/test_explainer_chatbook_export.py \
  tldw_Server_API/tests/Chatbooks/test_explainer_session_content_type.py
git commit -m "feat: export explainer sessions to chatbooks"
```

---

### Task 4: Frontend Route, API Client, And Workspace UI

**Files:**
- Create: `apps/tldw-frontend/pages/explainer.tsx`
- Create: `apps/tldw-frontend/extension/routes/option-explainer.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/route-metadata.ts`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Create: `apps/packages/ui/src/components/Option/Explainer/types.ts`
- Create: `apps/packages/ui/src/components/Option/Explainer/explainerApi.ts`
- Create: `apps/packages/ui/src/components/Option/Explainer/tree.ts`
- Create: `apps/packages/ui/src/components/Option/Explainer/useExplainerQueries.ts`
- Create: `apps/packages/ui/src/components/Option/Explainer/ExplainerWorkspace.tsx`
- Create: `apps/packages/ui/src/components/Option/Explainer/ExplainerModeTabs.tsx`
- Create: `apps/packages/ui/src/components/Option/Explainer/ExplainerGoalComposer.tsx`
- Create: `apps/packages/ui/src/components/Option/Explainer/ExplainerSourcePicker.tsx`
- Create: `apps/packages/ui/src/components/Option/Explainer/ExplainerTree.tsx`
- Create: `apps/packages/ui/src/components/Option/Explainer/ExplainerDetailPanel.tsx`
- Create: `apps/packages/ui/src/components/Option/Explainer/ExplainerChatbookExportButton.tsx`
- Modify: `apps/packages/ui/src/public/_locales/en/option.json`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.explainer.test.ts`
- Test: `apps/packages/ui/src/components/Option/Explainer/__tests__/ExplainerWorkspace.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Explainer/__tests__/explainer-tree.test.ts`

- [x] **Step 1: Create or update Backlog task**

Use a frontend implementation task with the above files and reference `TASK-546`/`TASK-547`.

- [x] **Step 2: Write failing API client tests**

Add tests for each client method:

```ts
it("creates an Explainer session through the typed client", async () => {
  mocks.bgRequest.mockResolvedValue({ id: "session-1" })
  const client = new TldwApiClient()

  await client.createExplainerSession({
    mode: "goal",
    title: "Learn attention",
    outputIntent: "explain",
    grounding: "open",
    depthPreset: "standard",
    prompt: "Explain transformer attention",
    selectedSources: []
  })

  expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({
    path: "/api/v1/explainer/sessions",
    method: "POST"
  }))
})
```

- [x] **Step 3: Run API client tests and verify failure**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/tldw-api-client.explainer.test.ts
```

Expected: FAIL because Explainer client methods and guarded paths do not exist.

- [x] **Step 4: Implement typed client methods and guard paths**

Add methods:

```ts
createExplainerSession(payload, options?)
listExplainerSessions(params?, options?)
getExplainerSession(sessionId, options?)
updateExplainerSession(sessionId, payload, options?)
deleteExplainerSession(sessionId, options?)
createExplainerNode(sessionId, payload, options?)
updateExplainerNode(sessionId, nodeId, payload, options?)
deleteExplainerNode(sessionId, nodeId, options?)
expandExplainerNode(sessionId, nodeId, payload, options?)
answerExplainerQuestion(sessionId, nodeId, payload, options?)
getExplainerJob(jobId, options?)
exportExplainerChatbook(sessionId, payload, options?)
```

- [x] **Step 5: Write failing tree utility tests**

Cover stable ordering, selected-node fallback, deletion pruning, and status/evidence labels:

```ts
expect(flattenExplainerTree(nodes, rootIds).map((node) => node.id)).toEqual([
  "root",
  "child-a",
  "child-b"
])
```

- [x] **Step 6: Implement tree utilities**

Keep tree transforms pure in `tree.ts`; do not bury ordering logic inside React render code.

- [x] **Step 7: Write failing workspace tests**

Cover:

- `/explainer` workspace renders a heading and explicit `Goal`/`Sources` tabs.
- Goal tab creates a persisted session through the client.
- Sources tab searches media/notes and shows selected sources.
- Grounding mode is user configurable.
- Output intent can be `Explain`, `Plan`, or `Both`.
- Tree and detail panel render persisted node data.
- Export button calls the Explainer export endpoint and surfaces queued/completed/failed state.

- [x] **Step 8: Implement route wrapper and nav**

`apps/tldw-frontend/pages/explainer.tsx`:

```ts
import dynamic from "next/dynamic"

export default dynamic(() => import("@/routes/option-explainer"), { ssr: false })
```

`option-explainer.tsx` should follow `option-research-workspace.tsx` with `RouteErrorBoundary` added:

```tsx
<RouteErrorBoundary routeId="explainer" routeLabel="Explainer">
  <OptionLayout>
    <PageShell className="flex h-full min-h-0 w-full flex-1 overflow-hidden" maxWidthClassName="max-w-full">
      <ExplainerWorkspace />
    </PageShell>
  </OptionLayout>
</RouteErrorBoundary>
```

- [x] **Step 9: Implement the workspace UI**

Use a restrained workbench layout:

- Header: title, current session status, export action.
- Setup area: explicit `Goal` and `Sources` tabs.
- Source picker: media/document and notes search, selected list, remove controls.
- Main body: tree rail plus detail panel.
- Right rail/drawer: session settings and source summary.

Do not use nested cards. Keep repeated node rows as compact list items, and reserve cards for modals or individual repeated source rows if needed.

- [x] **Step 10: Implement polling**

Use TanStack Query polling while a node has `queued` or `generating` status. Poll `getExplainerJob(jobId)`, then refresh `getExplainerSession(sessionId)` on terminal status. Stop polling when the document is hidden unless the active job was just created by the current interaction.

- [x] **Step 11: Run frontend unit tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/tldw-api-client.explainer.test.ts apps/packages/ui/src/components/Option/Explainer/__tests__
```

Expected: PASS.

- [x] **Step 12: Commit Task 4**

```bash
git add apps/tldw-frontend/pages/explainer.tsx \
  apps/tldw-frontend/extension/routes/option-explainer.tsx \
  apps/tldw-frontend/extension/routes/route-registry.tsx \
  apps/packages/ui/src/routes/route-metadata.ts \
  apps/packages/ui/src/services/tldw/openapi-guard.ts \
  apps/packages/ui/src/services/tldw/TldwApiClient.ts \
  apps/packages/ui/src/components/Option/Explainer \
  apps/packages/ui/src/public/_locales/en/option.json \
  apps/packages/ui/src/services/__tests__/tldw-api-client.explainer.test.ts
git commit -m "feat: add explainer workspace ui"
```

---

### Task 5: E2E, Accessibility, Security, And Release Verification

**Files:**
- Create: `apps/tldw-frontend/e2e/utils/page-objects/ExplainerPage.ts`
- Create: `apps/tldw-frontend/e2e/explainer.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- Modify: backend/frontend tests from earlier tasks as needed
- Update: related Backlog task(s) with verification notes

- [x] **Step 1: Create or update Backlog task**

Use a final verification task if Tasks 1-4 were separate slices.

- [x] **Step 2: Add Playwright page object**

Page object should expose:

```ts
goto()
createGoalSession(goal: string)
openSourcesTab()
searchSource(query: string)
selectFirstSource()
expandSelectedNode()
exportToChatbook()
expectNodeStatus(status: RegExp)
expectCitation(text: RegExp)
```

- [x] **Step 3: Add mocked E2E tests**

Cover:

- Smoke loads `/explainer`.
- Goal flow creates a session and renders first node.
- Sources flow selects a source and renders citation chips.
- Job polling completion refreshes an expanded node.
- Chatbook export posts to `/api/v1/explainer/sessions/{session_id}/export-chatbook` and displays the returned Chatbooks job/download state.

- [x] **Step 4: Run backend tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Explainer tldw_Server_API/tests/Chatbooks -k "explainer or chatbook" tldw_Server_API/tests/Services/test_router_groups_contract.py -v
```

Expected: PASS.

- [x] **Step 5: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/explainer.py tldw_Server_API/app/api/v1/schemas/explainer.py tldw_Server_API/app/api/v1/API_Deps/Explainer_DB_Deps.py tldw_Server_API/app/core/DB_Management/Explainer_DB.py tldw_Server_API/app/core/Explainer -f json -o /tmp/bandit_explainer.json
```

Expected: PASS or no new findings in touched Explainer code. Record the result in Backlog.

- [x] **Step 6: Run frontend tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/tldw-api-client.explainer.test.ts apps/packages/ui/src/components/Option/Explainer/__tests__
```

Expected: PASS.

- [x] **Step 7: Run E2E tests**

Run:

```bash
npx playwright test apps/tldw-frontend/e2e/explainer.spec.ts --reporter=line
```

Expected: PASS.

- [x] **Step 8: Run browser visual checks**

Start the WebUI dev server with the project-approved command, then use the in-app Browser or Playwright to inspect:

- Desktop 1440px wide.
- Mobile 390px wide.
- Goal tab empty state.
- Sources tab with selected sources.
- Generated tree with a selected detail node.
- Export queued/completed states.

Verify:

- No text overlap.
- Tree controls have accessible names.
- Evidence state is not color-only.
- Mobile drawer traps/restores focus.
- The UI does not look like a marketing page.

- [x] **Step 9: Update Backlog final summaries**

Each implementation task should record:

- Commits.
- Tests run and results.
- Bandit result.
- Any skipped checks and why.
- Known follow-ups.

- [x] **Step 10: Commit Task 5**

```bash
git add apps/tldw-frontend/e2e/utils/page-objects/ExplainerPage.ts \
  apps/tldw-frontend/e2e/explainer.spec.ts \
  apps/tldw-frontend/e2e/smoke/page-inventory.ts \
  backlog/tasks
git commit -m "test: verify explainer workspace"
```

---

## Final Acceptance Criteria

- `/explainer` is reachable from the WebUI and workspace navigation.
- The page has explicit `Goal` and `Sources` tabs.
- Goal sessions and source sessions are persisted server-side.
- The in-page source picker can search and select sources without using Research Workspace.
- Grounding mode is user configurable and enforced by the backend.
- Output intent can be `Explain`, `Plan`, or `Both`.
- Node expansion is Jobs-backed and resumable through persisted session state.
- Source-only insufficient evidence is represented as an `insufficient` node, not hallucinated content.
- Citations store source ID, title, excerpt, location metadata, and snapshot hash when available.
- Chatbook export creates one `explainer_session` item containing the full session.
- Chatbook import restores `explainer_session` and recognizes generated-document subtype fallback.
- Backend unit/integration tests, frontend unit tests, E2E, and Bandit touched-scope checks pass or have documented blockers.

## Implementation Risks To Watch

- Chatbook service is large; keep Explainer-specific serialization in `core/Explainer/chatbook_adapter.py` and call it from the service rather than expanding Chatbook internals with generation logic.
- Jobs payloads must not contain full source excerpts unless strictly necessary. Prefer IDs/settings in the Jobs payload and load context inside the worker.
- `source_only` is the highest-risk behavior. Add tests that fail if outside knowledge is marked as supported.
- Avoid localStorage for session state except ephemeral UI preferences such as collapsed rails. Persist all explainer content on the backend.
- Do not add a broad global Jobs status endpoint. Keep Explainer job status ownership-checked.
- Keep router imports lightweight so minimal tests do not import optional LLM/RAG providers at app startup.
