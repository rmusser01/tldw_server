# Research Workspace Source Preview Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Research Workspace a single authoritative page-context envelope for source readiness/capability status and a bounded source preview endpoint for captured content, evidence snippets, and annotations.

**Architecture:** Add a read-computed workspace context endpoint that composes workspace metadata, source rows, source readiness, capabilities, workspace service status, and partial errors without embedding large source bodies. Add a separate source preview endpoint for bounded content/chunk snippets. Keep existing `/sources`, `/sources/status`, and `/capabilities` endpoints in place for compatibility, but move the WebUI page shell to the new context endpoint.

**Tech Stack:** FastAPI, Pydantic, existing `CharactersRAGDB`, Media DB read APIs, Jobs manager projection, React, Zustand, Vitest, Pytest, Playwright/CDP.

---

## Design Review Decisions

- The singular endpoint should be `GET /api/v1/workspaces/{workspace_id}/context`, not a catch-all data dump.
- The context envelope owns page-shell state: workspace metadata, source list, source readiness, capability gates, MCP/ACP/Sandbox service summary, active Jobs status, and partial read errors.
- Full source text, chunk navigation, citation snippets, and large-source inspection stay in `GET /api/v1/workspaces/{workspace_id}/sources/{source_id}/preview`.
- Context response failures should be partial where possible. Missing Jobs or Media DB status should produce conservative status plus `partial_errors`, not a 500 when the workspace row itself is readable.
- Existing source status/capabilities endpoints remain API-compatible for now; the WebUI should prefer the context endpoint to avoid split-brain readiness state.
- Source annotations become persisted client workspace state in this task. Server-side shared annotations are deferred because they need schema, sync, export/import, and ownership semantics.

## Stage 1: Backend Context Envelope

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_context_api.py`

- [ ] **Step 1: Write failing context endpoint tests**
  - Ready/partially-queryable/missing source status appears inside one context response.
  - MCP/ACP/Sandbox capability service metadata appears in the context envelope.
  - Jobs/media read failures produce conservative `partial_errors` rather than breaking the workspace response.

- [ ] **Step 2: Run focused pytest and verify failures**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_context_api.py -q`
  - Expected: FAIL because schemas/endpoint do not exist.

- [ ] **Step 3: Implement minimal schemas and endpoint**
  - Add `WorkspaceContextResponse`, `WorkspaceContextSources`, and `WorkspaceContextPartialError`.
  - Reuse `_ws_to_response`, `_src_to_response`, `build_source_status_projection`, and `build_workspace_capability_projection`.
  - Return `preview.detail_href` per source when a source can be inspected.

- [ ] **Step 4: Run focused pytest and verify pass**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_context_api.py -q`
  - Expected: PASS.

## Stage 2: Backend Source Preview Endpoint

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/workspaces.py`
- Test: `tldw_Server_API/tests/Workspaces/test_workspace_source_preview_api.py`

- [ ] **Step 1: Write failing source preview tests**
  - Available source returns bounded captured content and chunk snippets.
  - Pending extraction returns a precise unavailable reason.
  - Missing media returns a precise unavailable reason.
  - Large content is capped and reports total size.

- [ ] **Step 2: Run focused pytest and verify failures**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_source_preview_api.py -q`
  - Expected: FAIL because endpoint does not exist.

- [ ] **Step 3: Implement minimal source preview builder**
  - Fetch source from workspace membership, then media by `media_id`.
  - Return bounded content excerpt and first chunk snippets where available.
  - Use status projection semantics for readiness and unavailable reasons.
  - Never return unbounded document content in the page shell.

- [ ] **Step 4: Run focused pytest and verify pass**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspace_source_preview_api.py -q`
  - Expected: PASS.

## Stage 3: WebUI Client, Store, and Modal

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Modify: `apps/packages/ui/src/types/workspace.ts`
- Modify: `apps/packages/ui/src/store/workspace.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx`
- Test: `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage2.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx`

- [ ] **Step 1: Write failing API/client tests**
  - `getWorkspaceContext` calls `/api/v1/workspaces/{id}/context`.
  - `getWorkspaceSourcePreview` calls `/api/v1/workspaces/{id}/sources/{source_id}/preview`.

- [ ] **Step 2: Write failing UI tests**
  - Modal renders captured content and evidence snippets after preview load.
  - Modal renders pending/error/empty preview states.
  - Annotation create/edit/delete persists across modal close/reopen.

- [ ] **Step 3: Implement client and store wiring**
  - Add context and preview TypeScript response types.
  - Add persisted source annotations to workspace store.
  - Prefer context endpoint for page shell source/status/capability reconciliation.

- [ ] **Step 4: Implement modal content layout**
  - Keep source inspection above annotations.
  - Add bounded preview, evidence snippets, loading, retry, empty, pending, and failure states.
  - Keep existing annotation action parity and undo behavior.

- [ ] **Step 5: Run focused Vitest suites**
  - Run: `bunx vitest run src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts src/components/Option/ResearchWorkspace/__tests__/SourcesPane.stage2.test.tsx src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx --maxWorkers=1 --no-file-parallelism`
  - Expected: PASS.

## Stage 4: Verification and Live Validation

**Files:**
- Modify: `backlog/tasks/task-478.9 - Gate-D-improve-source-preview-annotations-and-evidence-inspection.md`

- [ ] **Step 1: Run focused backend tests**
  - Run context and preview pytest files.

- [ ] **Step 2: Run focused frontend tests**
  - Run client and Research Workspace Vitest suites.

- [ ] **Step 3: Run Bandit on touched backend scope**
  - Run: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/workspaces.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py -f json -o /tmp/bandit_task_478_9.json`

- [ ] **Step 4: Live CDP/Playwright validation**
  - Start/verify actual backend and WebUI.
  - Open `/research-workspace`.
  - Open a real source preview.
  - Verify captured content or precise unavailable reason, evidence snippet metadata, annotation persistence, and no console errors.

- [ ] **Step 5: Update Backlog final notes and push**
  - Record tests, live validation, known skips/blockers, and pushed commit hash.
