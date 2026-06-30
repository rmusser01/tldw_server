# Research Workspace Sandbox Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sandbox-owned workspace diagnostics so Research Workspace can hand off its canonical workspace ID to sandbox admission/runtime/run status without owning sandbox state.

**Architecture:** Sandbox owns the new read-only diagnostics envelope and admin run filters. Research Workspace only passes the active canonical workspace ID plus `source_label=research_workspace` and renders the returned user-safe state inside existing remediation/settings surfaces.

**Tech Stack:** FastAPI, Pydantic v2, existing SandboxService/store APIs, React, Vitest, Playwright/CDP.

---

## File Structure

- Modify `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`: add sandbox workspace diagnostics response models.
- Modify `tldw_Server_API/app/api/v1/endpoints/sandbox.py`: expose existing workspace filters on `GET /sandbox/admin/runs`.
- Add `tldw_Server_API/app/api/v1/endpoints/sandbox_service.py`: share the sandbox service singleton across sandbox routers.
- Add `tldw_Server_API/app/api/v1/endpoints/sandbox_workspace_diagnostics.py`: add stable user-safe `GET /sandbox/workspaces/{workspace_id}/diagnostics`.
- Modify `tldw_Server_API/app/api/v1/router_groups/content.py`: register the diagnostics router outside the admin-only sandbox route gate.
- Modify `tldw_Server_API/tests/sandbox/test_admin_list_filters_pagination.py`: add admin run filter coverage for `workspace_id`, `workspace_group_id`, and `scope_snapshot_id`.
- Add `tldw_Server_API/tests/sandbox/test_workspace_diagnostics.py`: cover user-safe workspace diagnostics states and workspace-scoped run filtering.
- Modify `apps/packages/ui/src/services/tldw/TldwApiClient.ts`: add typed sandbox workspace diagnostics method and response types.
- Modify `apps/packages/ui/src/services/tldw/openapi-guard.ts`: add the new sandbox diagnostics path if path guarding requires it.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceCapabilityRemediation.tsx`: expose `View Sandbox Diagnostics` for sandbox remediation with active workspace context.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`: expose stable sandbox diagnostics access from workspace settings so healthy sandboxes remain inspectable.
- Add `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceSandboxDiagnosticsPanel.tsx`: render loading, empty, unavailable, denied, and recent-run states.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceCapabilityRemediation.test.tsx`: assert the sandbox diagnostics action and no new banner/route leakage.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`: assert the workspace settings sandbox diagnostics entry opens the panel.
- Add `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceSandboxDiagnosticsPanel.test.tsx`: cover request path and terminal states.
- Modify or add Playwright coverage under `apps/tldw-frontend/e2e/workflows`: validate live request context and terminal state.
- Modify `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`: update `RW-UAT-023`.
- Modify `backlog/tasks/task-478.23 - Gate-F-validate-Sandbox-handoff-for-Research-Workspace.md`: record notes, verification, and final summary.

## Task 1: Backend Admin Run Workspace Filters

**Files:**
- Add: `tldw_Server_API/app/api/v1/endpoints/sandbox_service.py`
- Add: `tldw_Server_API/app/api/v1/endpoints/sandbox_workspace_diagnostics.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Test: `tldw_Server_API/tests/sandbox/test_admin_list_filters_pagination.py`

- [x] **Step 1: Write failing admin filter test**

Add a test that seeds runs with different `workspace_id`, `workspace_group_id`,
and `scope_snapshot_id`, then asserts `/api/v1/sandbox/admin/runs` filters by
each query param and returns totals aligned with the filtered result.

- [x] **Step 2: Run test to verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_admin_list_filters_pagination.py -k "workspace_filters" -q
```

Expected: fail because the endpoint does not accept or pass workspace filters.

- [x] **Step 3: Implement minimal endpoint filter pass-through**

Add optional query params `workspace_id`, `workspace_group_id`, and
`scope_snapshot_id` to `admin_list_runs()`, then pass them to
`_service._orch.list_runs()` and `_service._orch.count_runs()`.

- [x] **Step 4: Run test to verify GREEN**

Run the same pytest command. Expected: pass.

## Task 2: Sandbox-Owned Workspace Diagnostics Endpoint

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
- Add: `tldw_Server_API/tests/sandbox/test_workspace_diagnostics.py`

- [x] **Step 1: Write failing diagnostics tests**

Cover at least:

- no runtime configured/discovered returns `runtime.state == "not_configured"`
  or `admission.state == "blocked"` with a user-safe message;
- seeded runs are filtered to the requested `workspace_id`;
- unrelated workspace runs are not returned;
- another user's run for the same `workspace_id` is not returned;
- response includes `source_label == "research_workspace"`;
- response never emits `workspace_playground`;
- response does not require admin role for the user-safe route.

- [x] **Step 2: Run diagnostics tests to verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_workspace_diagnostics.py -q
```

Expected: fail because the route and schemas do not exist.

- [x] **Step 3: Add Pydantic response models**

Add small models for:

- `SandboxWorkspaceDiagnosticState`
- `SandboxWorkspaceDiagnosticsRunSummary`
- `SandboxWorkspaceDiagnosticsRunList`
- `SandboxWorkspaceDiagnosticsLinks`
- `SandboxWorkspaceDiagnosticsResponse`

Keep the envelope user-safe and omit raw runtime internals.

- [x] **Step 4: Implement read-only endpoint**

Add `GET /workspaces/{workspace_id}/diagnostics` to a narrow sandbox-owned
diagnostics router. Use the authenticated user, `source_label` defaulting to
`research_workspace`, and store-backed run filters. Derive runtime/admission
state from existing sandbox feature discovery or a defensive unavailable/not
configured fallback. Register this router through the stable user-facing router
group so the endpoint remains live when the broader sandbox admin route is
disabled, and share the sandbox service singleton with the admin router so
memory-backed run state is consistent.

- [x] **Step 5: Run diagnostics tests to verify GREEN**

Run the same pytest command. Expected: pass.

## Task 3: WebUI Client And Diagnostics Panel

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/openapi-guard.ts`
- Add: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceSandboxDiagnosticsPanel.tsx`
- Add: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceSandboxDiagnosticsPanel.test.tsx`

- [x] **Step 1: Write failing panel/client tests**

Assert the panel calls:

```text
/api/v1/sandbox/workspaces/workspace-alpha/diagnostics?source_label=research_workspace
```

Then assert loading, empty runs, forbidden, and unavailable copy.

- [x] **Step 2: Run Vitest to verify RED**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceSandboxDiagnosticsPanel.test.tsx --maxWorkers=1
```

Expected: fail because the client method and panel do not exist.

- [x] **Step 3: Add typed client method and panel**

Add the response types and `getSandboxWorkspaceDiagnostics(workspaceId, options)`
to `TldwApiClient`. Implement the panel with existing product state styling,
compact copy, accessible labels, and no persistent banner.

- [x] **Step 4: Run Vitest to verify GREEN**

Run the same Vitest command. Expected: pass.

## Task 4: Research Workspace Handoff Action

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceCapabilityRemediation.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceCapabilityRemediation.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`

- [x] **Step 1: Write failing remediation and settings tests**

Assert sandbox remediation renders `View Sandbox Diagnostics` with the active
workspace ID, opens the diagnostics panel, and still renders runtime config only
as a secondary management action. Assert workspace settings also exposes
`Sandbox diagnostics` so healthy sandboxes remain inspectable. Assert neither
path renders a new trust bar.

- [x] **Step 2: Run test to verify RED**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceCapabilityRemediation.test.tsx --maxWorkers=1
```

Expected: fail because remediation only links to runtime config.

- [x] **Step 3: Add diagnostics actions**

Thread `workspaceId` into the remediation item. For sandbox only, render an
inline/details action that opens `WorkspaceSandboxDiagnosticsPanel`. Keep the
existing management link as a separate action when present. Add a stable
workspace settings menu entry that opens the same panel in a modal.

- [x] **Step 4: Run test to verify GREEN**

Run the same Vitest command. Expected: pass.

## Task 5: Live Validation, Matrix, And Backlog

**Files:**
- Modify or add: `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`
- Modify: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Modify: `backlog/tasks/task-478.23 - Gate-F-validate-Sandbox-handoff-for-Research-Workspace.md`

- [x] **Step 1: Add live Playwright/CDP assertion**

Validate `/research-workspace` opens the sandbox diagnostics action, the
diagnostics request includes the active workspace ID and
`source_label=research_workspace`, and the UI reaches a truthful terminal state.
After design review, strengthen this assertion to require a `<400` diagnostics
response so an endpoint 404 cannot pass by rendering a generic error state.

- [x] **Step 2: Run live backend/WebUI validation**

Use the project backend and WebUI with CDP/Playwright only. Capture request and
terminal-state evidence.

- [x] **Step 3: Update `RW-UAT-023`**

Move the matrix row only as far as the live evidence supports. Keep it `Partial`
unless a real workspace-scoped sandbox run is created or observed.

- [x] **Step 4: Update Backlog task**

Record implementation notes, verification commands, live evidence, and any
known limits.

## Task 6: Final Verification And Commit

**Files:**
- All touched files

- [x] **Step 1: Run focused backend tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/sandbox/test_admin_list_filters_pagination.py::test_admin_list_filters_by_workspace_scope tldw_Server_API/tests/sandbox/test_workspace_diagnostics.py -q
```

Result: `4 passed, 2 warnings`. The broader
`test_admin_list_filters_pagination.py` file still has an unrelated existing
background-worker teardown hang in `test_admin_list_filter_by_user_and_phase`;
this slice verified the new workspace filter coverage directly.

- [x] **Step 2: Run focused frontend tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceCapabilityRemediation.test.tsx src/components/Option/ResearchWorkspace/__tests__/WorkspaceSandboxDiagnosticsPanel.test.tsx src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx --maxWorkers=1
```

Result: `48 passed`.

- [x] **Step 3: Run route-label guard**

```bash
rg -n "workspace-playground|workspace_playground|Workspace Playground" apps tldw_Server_API Docs/Reviews Docs/Design Docs/superpowers/specs Docs/superpowers/plans
```

Expected: no new active aliases, redirects, labels, or UI copy.

Result: only historical docs, defensive negative tests, and existing legacy
local-storage import references were found; no new active route alias, redirect,
label, or UI copy was added.

- [x] **Step 4: Run Bandit on touched backend Python**

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/sandbox.py tldw_Server_API/app/api/v1/endpoints/sandbox_service.py tldw_Server_API/app/api/v1/endpoints/sandbox_workspace_diagnostics.py tldw_Server_API/app/api/v1/router_groups/content.py tldw_Server_API/app/api/v1/schemas/sandbox_schemas.py -f json -o /tmp/bandit_task47823.json
```

Result: 0 errors, 0 results.

- [x] **Step 5: Run `git diff --check`**

Expected: no whitespace errors.

Result: no whitespace errors.

- [x] **Step 6: Commit**

Commit with a message scoped to `TASK-478.23`.
