# Research Workspace Server Bootstrap Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/research-workspace` upsert the active backend workspace and mirror visible local source rows before fetching trust/status projections.

**Architecture:** Add a small frontend reconciliation helper that uses the existing workspace API client methods. Wire it into the existing trust-panel polling effect so reconciliation is best-effort, deduplicated by source id/media id, and never blocks status/capability fetching.

**Tech Stack:** React, TypeScript, Vitest, Testing Library, existing tldw workspace REST APIs.

---

### Task 1: Reconciliation Helper Contract

**Files:**
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/workspace-server-reconcile.ts`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts`

- [x] **Step 1: Write failing helper tests**
  - Verify the helper upserts the workspace, lists existing backend sources, adds missing valid local sources, skips invalid media IDs, and skips duplicates by source id or media id.
  - Verify source-add failures are reported but later sources are still attempted.

- [x] **Step 2: Run helper tests and confirm RED**
  - Run: `cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts`
  - Expected: FAIL because the helper module does not exist yet.

- [x] **Step 3: Implement helper**
  - Export `reconcileResearchWorkspaceServerState`.
  - Export source-signature helper for stable effect dependencies.
  - Return bounded result metadata: `workspaceReady`, `sourceRowsChecked`, `addedSourceIds`, `skippedSourceIds`, `errors`.
  - Do not throw for expected API failures; collect error text and return.

- [x] **Step 4: Run helper tests and confirm GREEN**
  - Run: `cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts`
  - Expected: PASS.

### Task 2: Research Workspace Effect Wiring

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx`

- [x] **Step 1: Write failing page tests**
  - Verify `ResearchWorkspace` upserts/lists/adds missing local source rows before source-status/capability calls.
  - Verify a reconciliation failure still allows source-status/capability calls and shows a bounded trust warning.

- [x] **Step 2: Run page tests and confirm RED**
  - Run: `cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx`
  - Expected: FAIL because the page does not call the reconciliation helper yet.

- [x] **Step 3: Wire reconciliation into trust refresh**
  - Import helper and compute a stable source signature.
  - Reconcile once per workspace/name/source signature before trust projection fetches.
  - Preserve the existing in-flight guard and mismatched-workspace protections.
  - Keep reconciliation failures best-effort and additive to the trust warning.

- [x] **Step 4: Run focused tests and confirm GREEN**
  - Run: `cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx`
  - Expected: PASS.

### Task 3: Validation And Task Closeout

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/tests/Workspaces/test_workspaces_api.py`
- Modify: `backlog/tasks/task-467 - Bootstrap-server-backed-active-Research-Workspace-and-source-rows.md`

- [x] **Step 1: Write failing backend duplicate-source test**
  - Extend workspace source API coverage so a duplicate `POST /api/v1/workspaces/{id}/sources` for the same source id returns the existing row instead of a 500.

- [x] **Step 2: Run backend duplicate-source test and confirm RED**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py::test_workspace_source_endpoints_happy_path -q`
  - Expected: FAIL because duplicate source add currently raises through as a server error.

- [x] **Step 3: Make DB source add idempotent for existing source ids**
  - Catch SQLite integrity errors in `add_workspace_source`.
  - If the source row already exists for the workspace/source id, return it unchanged.
  - Preserve existing conflict behavior for other integrity failures.

- [x] **Step 4: Run backend duplicate-source test and confirm GREEN**
  - Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Workspaces/test_workspaces_api.py::test_workspace_source_endpoints_happy_path -q`
  - Expected: PASS.

- [x] **Step 5: Run frontend verification**
  - Run focused Vitest tests.
  - Run `git diff --check` on touched files.

- [x] **Step 6: Validate with real backend and CDP**
  - Start the FastAPI backend and Next WebUI.
  - Use Playwright/CDP only, not Computer Control.
  - Confirm `/research-workspace` has no `/workspace-playground` redirects and that workspace/source bootstrap runs before status/capability calls.

- [x] **Step 7: Record closeout**
  - Update Backlog `TASK-467` with touched files, verification commands, Bandit skip reason if frontend-only, and final summary.

### Task 4: Code Review Follow-Up

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/workspace-server-reconcile.ts`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts`

- [x] **Step 1: Request focused code review**
  - Use the requesting-code-review workflow on the TASK-467 scoped files.

- [x] **Step 2: Write failing bounded-error regression test**
  - Verify repeated source-add failures continue attempting later sources but keep returned diagnostic errors bounded.

- [x] **Step 3: Cap reconciliation error metadata**
  - Keep first diagnostic messages bounded and append a single omission summary once the cap is reached.

- [x] **Step 4: Rerun verification**
  - Run focused Vitest tests, backend workspace tests, live CDP validation, Bandit, and diff checks.
