# Research Workspace ACP Canonical Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class ACP project filtering by canonical Research Workspace ID and validate Research Workspace to ACP handoff with focused tests and live Playwright evidence.

**Architecture:** ACP remains the owner of execution state. The ACP projects endpoint gets canonical workspace query filters; Research Workspace and Agent Tasks pass those filters and keep client-side guards for older or malformed responses.

**Tech Stack:** FastAPI, Pydantic query params, OrchestrationDB models, React, Vitest, Playwright.

---

## File Structure

- Modify `tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py`: add canonical query params and endpoint-level project filtering.
- Modify `tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py`: add focused backend tests for canonical project filters.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceACPHistoryModal.tsx`: request filtered ACP projects for the current canonical workspace.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`: assert the history modal sends canonical filter params and still handles empty/error states.
- Modify `apps/packages/ui/src/components/Option/AgentTasks/index.tsx`: request filtered ACP projects when workspace filter route params are present.
- Modify `apps/packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx`: assert Agent Tasks uses the canonical filter.
- Modify or add Playwright coverage under `apps/tldw-frontend/e2e/workflows`: validate the live WebUI sends the canonical filter and shows a truthful ACP state.
- Modify `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`: update `RW-UAT-022` only to the evidence level proven by live validation.
- Modify `backlog/tasks/task-478.22 - Gate-F-validate-ACP-canonical-bridge-for-Research-Workspace.md`: record implementation notes, verification, and final summary.

## Task 1: Backend Canonical Project Filter

- [ ] **Step 1: Write failing backend tests**

Add tests proving `list_projects` can filter by `canonical_workspace_id` and `canonical_workspace_source`, including one project linked through ACP workspace metadata and one unrelated project that must not appear.

- [ ] **Step 2: Run backend test and verify RED**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py -k "list_projects and canonical" -q
```

Expected: fail because `list_projects` has no canonical query params yet.

- [ ] **Step 3: Implement minimal endpoint filtering**

Add query params to `list_projects`, enrich projects with workspace canonical links as it already does, and filter rows using a helper that matches `canonical_workspace` or project metadata.

- [ ] **Step 4: Run backend test and verify GREEN**

Run the same focused pytest command and expect pass.

## Task 2: Research Workspace History Uses ACP Filter

- [ ] **Step 1: Write failing Vitest assertion**

Update the existing `WorkspaceHeader.test.tsx` ACP history test so the expected first project request is:

```text
/api/v1/agent-orchestration/projects?canonical_workspace_id=workspace-alpha&canonical_workspace_source=research_workspace
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
bunx vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx --maxWorkers=1
```

Expected: fail because the modal currently fetches all projects.

- [ ] **Step 3: Implement filtered request path**

Build the project query path from the canonical workspace ID and call `fetchJson` with that path.

- [ ] **Step 4: Run the test and verify GREEN**

Run the same Vitest file and expect pass.

## Task 3: Agent Tasks Uses ACP Filter

- [ ] **Step 1: Write failing Vitest assertion**

Update `AgentTasksPage.connection.test.tsx` so route `/agent-tasks?workspace=workspace-alpha` expects the projects request to include canonical filter params.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
bunx vitest run src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx --maxWorkers=1
```

Expected: fail because Agent Tasks currently fetches all projects.

- [ ] **Step 3: Implement filtered request URL**

When `workspaceFilterId` is present, append `canonical_workspace_id` and `canonical_workspace_source=research_workspace` to the projects request. Preserve existing client-side filtering and URL behavior.

- [ ] **Step 4: Run the test and verify GREEN**

Run the same Vitest file and expect pass.

## Task 4: Live WebUI Validation And Matrix

- [ ] **Step 1: Add or update a focused Playwright scenario**

Validate the Research Workspace ACP history entry point against a live backend and WebUI, capturing the canonical-filtered projects request and modal terminal state.

- [ ] **Step 2: Run live backend + WebUI Playwright**

Use a real backend and WebUI. Use Playwright/CDP only.

- [ ] **Step 3: Update `RW-UAT-022`**

Record the live route/request/state evidence. Keep status `Partial` unless real run history and diagnostics are proven live.

## Task 5: Final Verification And Commit

- [ ] **Step 1: Run route-label guard**

Search active code/tests/docs for forbidden route aliases:

```bash
rg -n "workspace-playground|workspace_playground|Workspace Playground" apps tldw_Server_API Docs/Reviews Docs/Design
```

Expected: only legacy/historical references allowed by the active replacement policy; no new aliases or redirects from this slice.

- [ ] **Step 2: Run Bandit for touched backend Python**

Run:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py -f json -o /tmp/bandit_task47822.json
```

- [ ] **Step 3: Run `git diff --check`**

Expected: no whitespace errors.

- [ ] **Step 4: Update Backlog task**

Record verification results, matrix status, known skips, and final summary.

- [ ] **Step 5: Commit**

Commit with a message scoped to TASK-478.22.
