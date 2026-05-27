# Research Workspace Sandbox Enabled Runtime Validation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate `RW-UAT-023` against a live backend/WebUI configuration where the sandbox route is enabled and a workspace-linked sandbox run can be created and observed from Research Workspace diagnostics.

**Architecture:** Keep sandbox execution and diagnostics owned by the Sandbox API. Research Workspace only passes the canonical workspace ID and opens the sandbox-owned diagnostics panel. Tighten the diagnostics admission state so it distinguishes route-disabled, execution-disabled, runtime-unavailable, and available states before live validation updates the UAT matrix.

**Tech Stack:** FastAPI sandbox routes, sandbox service runtime discovery, Next.js/React Research Workspace panel, Playwright real-backend E2E, Backlog.md task tracking.

---

### Task 1: Make Workspace Sandbox Admission Truthful

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/sandbox_workspace_diagnostics.py`
- Test: `tldw_Server_API/tests/sandbox/test_workspace_diagnostics.py`

- [x] **Step 1: Write the failing backend test**

Add a unit test where runtime discovery reports a ready runtime, `route_enabled("sandbox")` returns true, but `SANDBOX_ENABLE_EXECUTION=0`. Assert diagnostics returns `admission.state="blocked"` and `reason_code="sandbox_execution_disabled"` instead of "may run."

- [x] **Step 2: Run the focused test to verify RED**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_workspace_diagnostics.py::test_workspace_diagnostics_blocks_admission_when_execution_disabled -q`

Expected: FAIL because execution-disabled admission is not implemented.

- [x] **Step 3: Implement the minimal diagnostics gate**

Add an internal helper that reads `SANDBOX_ENABLE_EXECUTION` with the same env-over-settings precedence used by `SandboxService.start_run_scaffold`. In `_sandbox_workspace_runtime_state()`, after route-policy is allowed and before returning available admission, return blocked admission with reason `sandbox_execution_disabled` when execution is disabled.

- [x] **Step 4: Run focused backend tests**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/sandbox/test_workspace_diagnostics.py -q`

Expected: PASS.

### Task 2: Make Live E2E Fixture Fail Closed When Required

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`

- [x] **Step 1: Write the failing E2E harness assertion path**

Add env-controlled strict mode to the existing workspace sandbox run test:
`TLDW_E2E_REQUIRE_SANDBOX_WORKSPACE_RUN=1` should fail with the create-run reason instead of silently skipping, and `TLDW_E2E_EXPECT_SANDBOX_RUN_PHASE` should assert the returned phase when set.

- [x] **Step 2: Verify the strict path is RED against route-disabled default config**

Run the single Playwright test against a live backend/WebUI without enabling the sandbox route.

Expected: FAIL with the existing `POST /api/v1/sandbox/runs returned HTTP 404` reason.

- [x] **Step 3: Keep default behavior backward-compatible**

The test should still skip when strict mode is unset and the sandbox route/runtime is unavailable.

Verified against a default-route backend on `127.0.0.1:18032`: the focused Playwright test skipped when `TLDW_E2E_REQUIRE_SANDBOX_WORKSPACE_RUN` was unset.

### Task 3: Run Enabled-Route Live Validation

**Files:**
- Update: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Update: `backlog/tasks/task-478.29 - Validate-Sandbox-enabled-runtime-workspace-run-diagnostics-for-Research-Workspace.md`

- [x] **Step 1: Start a live backend with sandbox route and execution enabled**

Use a validation-only environment with `TLDW_TEST_MODE=1`, `ROUTES_ENABLE=sandbox`, `SANDBOX_ENABLE_EXECUTION=1`, and `TLDW_SANDBOX_DOCKER_FAKE_EXEC=1` if the local Docker daemon is unavailable. Record whether this is a real Docker daemon proof or a sandbox fake-execution fixture.

Observed: normal live startup rejects `TLDW_TEST_MODE=1` outside pytest. Validation used a temporary config file at `/private/tmp/tldw_task47829_config.txt` with `sandbox` added to `[API-Routes].enable`, plus `SANDBOX_ENABLE_EXECUTION=1` and `TLDW_SANDBOX_DOCKER_FAKE_EXEC=1`. Docker CLI was installed, but the daemon was unavailable, so this is a fake-execution fixture, not a real Docker daemon proof.

- [x] **Step 2: Run the strict Playwright test**

Run the workspace sandbox diagnostics E2E with `TLDW_E2E_REQUIRE_SANDBOX_WORKSPACE_RUN=1` and, for fake execution, `TLDW_E2E_EXPECT_SANDBOX_RUN_PHASE=completed`.

Expected: PASS. If the environment cannot create the run, leave `RW-UAT-023` Partial and record the blocker.

- [x] **Step 3: Update UAT evidence only as far as verified**

Update `RW-UAT-023` with exact backend/WebUI ports, route/runtime/execution state, run ID evidence, diagnostics response evidence, and whether the runtime was real Docker or fake execution.

### Task 4: Final Verification and Closeout

**Files:**
- Update: `backlog/tasks/task-478.29 - Validate-Sandbox-enabled-runtime-workspace-run-diagnostics-for-Research-Workspace.md`

- [x] **Step 1: Run focused frontend/unit verification**

Run focused Vitest only if the panel test changes; otherwise document why it was not needed.

No panel component was changed, so no focused Vitest was needed. The changed E2E path was verified through strict route-disabled failure, route-disabled non-strict skip, and enabled-route fixture pass.

- [x] **Step 2: Run Bandit on touched backend Python**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/sandbox_workspace_diagnostics.py -f json -o /tmp/bandit_task_478_29.json`

Expected: 0 new findings.

- [x] **Step 3: Run diff hygiene**

Run: `git diff --check`

Expected: PASS.

- [x] **Step 4: Update Backlog final summary and commit**

Mark acceptance criteria and Definition of Done according to verified evidence. Commit the task, tests, docs, and Backlog updates together.
