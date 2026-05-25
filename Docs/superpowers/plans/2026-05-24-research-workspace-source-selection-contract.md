# Research Workspace Source Selection Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and superpowers:systematic-debugging to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make selected workspace sources a single, persisted contract used consistently by the sources pane, RAG chat, Studio, status projection, and server-backed workspace workflows.

**Architecture:** Keep the UI optimistic with `selectedSourceIds` as the local source of interaction truth, then persist that exact set to `/api/v1/workspaces/{workspace_id}/sources/selection`. Server reconciliation must create missing source rows with the current local selection bit and resync existing rows with the local selected IDs so status APIs, agents, and later extension handoffs do not disagree with the WebUI.

**Tech Stack:** React, Zustand, Vitest, Playwright/CDP, FastAPI workspace APIs.

---

### Task 1: Server Reconciliation Selection Contract

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/workspace-server-reconcile.ts`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts`

- [x] **Step 1: Write failing tests**
  Add tests proving missing source rows are created with `selected: true` only when their IDs are in `selectedSourceIds`, and existing source rows are reconciled through the batch selection endpoint.

- [x] **Step 2: Run tests to verify red**
  Run: `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts`
  Expected: FAIL because reconciliation currently always creates sources as selected and has no batch selection client method.

- [x] **Step 3: Implement minimal reconciliation changes**
  Extend the reconcile input/client with `selectedSourceIds` and `updateWorkspaceSourceSelection`, pass selected state into source creation, and call the batch selection endpoint after source row reconciliation.

- [x] **Step 4: Verify green**
  Run the same test and confirm it passes.

### Task 2: WebUI Selection Persistence Wiring

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx`

- [x] **Step 1: Write failing component test**
  Assert that the workspace bootstrap passes the current `selectedSourceIds` into reconciliation and that individual local selection changes cause the batch selection endpoint to receive the same IDs.

- [x] **Step 2: Run test to verify red**
  Run: `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx`
  Expected: FAIL because the API client lacks `updateWorkspaceSourceSelection` and `ResearchWorkspace` does not pass selection into server reconciliation.

- [x] **Step 3: Implement minimal UI/API changes**
  Add `updateWorkspaceSourceSelection()` to the workspace API client and include it in `tldwClient`. Pass `selectedSourceIds` into reconciliation and include selection in the reconcile signature so changes are persisted.

- [x] **Step 4: Verify green**
  Run the same component test and confirm it passes.

### Task 3: Hydration and Consumer Alignment

**Files:**
- Modify: `apps/packages/ui/src/store/workspace-api.ts`
- Test: `apps/packages/ui/src/store/__tests__/workspace-api-first.test.ts`
- Optional test: existing ChatPane/StudioPane selected-source tests if touched.

- [x] **Step 1: Write failing hydration test**
  Assert server-hydrated workspace state exposes selected source IDs from backend source rows so reload/workspace switch can recover persisted selection.

- [x] **Step 2: Run test to verify red**
  Run: `bunx vitest run src/store/__tests__/workspace-api-first.test.ts`
  Expected: FAIL if selected IDs are not returned in the local hydrated state.

- [x] **Step 3: Implement minimal hydration changes**
  Return selected source IDs from server hydration without changing the existing local source shape unnecessarily.

- [x] **Step 4: Verify green**
  Run the hydration test and selected RAG/Studio unit tests if any consumer contracts changed.

### Task 4: Browser/CDP Validation

**Files:**
- Test: `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`
- Optional: update e2e/page object only if the validation reveals a test helper bug.

- [x] **Step 1: Run targeted frontend/unit suite**
  Run the modified Vitest files together.

- [x] **Step 2: Run targeted Playwright/CDP validation**
  Validate `/research-workspace` against a live backend: individual checkbox selection persists through status APIs, RAG uses selected media IDs, and Studio enablement matches the same selected set.
  Result: initial run exposed a status-projection regression where processing status cleared `selectedSourceIds` and synced an empty selection to the backend. Fixed the store status update path so processing preserves direct selection intent while ready-only RAG media IDs remain filtered and terminal errors still clear selection. Re-run showed backend status `summary.selected = 1`, selected row `selected: true`, and RAG `include_media_ids` contained only the selected media ID.

- [x] **Step 3: Record verification in Backlog**
  Update `TASK-478.4` with tests run, skips, blockers, and final behavior.
