# Research Workspace Sandbox Execution Proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove whether a workspace-linked sandbox run can appear in the Research Workspace sandbox diagnostics envelope without moving sandbox ownership into Research Workspace.

**Architecture:** Keep the source of truth in the sandbox service and diagnostics endpoint. Add focused regression coverage that creates a run through the sandbox API and reads it back through the workspace diagnostics API, then update live UAT evidence honestly based on whether the full live backend exposes sandbox run creation under current route policy.

**Tech Stack:** Python 3.11, FastAPI TestClient, pytest, Playwright E2E, existing sandbox service singleton, Backlog.md task tracking.

---

## Design Review Findings

- The existing API already carries `workspace_id`, `workspace_group_id`, and `scope_snapshot_id` through `SandboxRunCreateRequest` into `RunSpec`; this should be regression-tested before changing production code.
- The diagnostics route is intentionally registered outside the broader sandbox admin route gate, so Research Workspace can inspect readiness even when sandbox admin routes are disabled.
- A normal uvicorn run with `ROUTES_ENABLE=sandbox` did not expose `POST /api/v1/sandbox/runs`; the runtime route policy treats config as authoritative outside pytest. `TEST_MODE=1` cannot be used as a live-server workaround because the startup guard rejects it outside pytest.
- Therefore, the least risky next step is a backend API contract test plus a conditional live E2E that skips with an explicit blocker if sandbox run creation is policy-disabled in the target backend.

## Files

- Modify: `tldw_Server_API/tests/sandbox/test_workspace_diagnostics.py`
- Modify: `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`
- Modify: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Modify: `backlog/tasks/task-478.24 - Gate-F-prove-workspace-linked-sandbox-execution-appears-in-Research-Workspace-diagnostics.md`

### Task 1: Backend Sandbox API To Diagnostics Contract

- [x] **Step 1: Write the failing test**

Add a test to `tldw_Server_API/tests/sandbox/test_workspace_diagnostics.py` that mounts both sandbox routers into one `FastAPI` test app, posts a run with a unique `workspace_id`, then reads `/api/v1/sandbox/workspaces/{workspace_id}/diagnostics`.

- [x] **Step 2: Run the focused test for RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_workspace_diagnostics.py::test_workspace_diagnostics_includes_run_created_through_sandbox_api -q
```

Expected: fail if the route/dependency/test seam is missing or the run is not returned by diagnostics.

- [x] **Step 3: Implement the smallest fix**

If the test fails due to production behavior, pass the existing workspace fields through the sandbox service/diagnostics path. If it fails only because the test app lacks router dependency overrides, fix the test harness without changing production code.

- [x] **Step 4: Verify GREEN**

Run the focused test and then the whole workspace diagnostics test file.

### Task 2: Conditional Live Research Workspace E2E

- [x] **Step 1: Add a Playwright live-backend scenario**

Extend `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts` with a scenario that reads the active workspace ID, tries to create a sandbox run through `/api/v1/sandbox/runs`, then opens Workspace settings > Sandbox diagnostics and asserts the run ID appears when creation is available.

- [x] **Step 2: Preserve honest skip behavior**

If `/api/v1/sandbox/runs` is absent, forbidden, or runtime-blocked in the live backend, skip with the exact HTTP status/reason instead of marking the product as passing.

- [x] **Step 3: Run the focused E2E**

Run the new test against the available live backend/WebUI if those processes are reachable. Record pass/skip/failure evidence in the UAT row and task notes.

### Task 3: Matrix, Task, And Verification

- [x] **Step 1: Update `RW-UAT-023`**

Move the row to Pass only if a real live workspace-linked sandbox run appears in Research Workspace diagnostics. Otherwise keep Partial and record the backend contract evidence plus the live blocker.

- [x] **Step 2: Run security and text guards**

Run focused backend tests, the focused Playwright test when possible, Bandit for touched backend Python files, and `git diff --check`.

- [x] **Step 3: Finalize Backlog and commit**

Completed in TASK-478.24. The backend contract test proves the sandbox-owned API
round-trip when the sandbox route is available. The live backend/WebUI run on
`127.0.0.1:18017` and `127.0.0.1:8081` recorded the current route-policy blocker
instead of promoting `RW-UAT-023` to Pass.

Mark completed acceptance criteria, record blockers/skips, stage only relevant files, and commit with a `TASK-478.24` message.
