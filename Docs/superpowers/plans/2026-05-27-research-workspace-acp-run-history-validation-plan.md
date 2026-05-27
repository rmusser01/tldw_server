# Research Workspace ACP Run History Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that Research Workspace can hand off to ACP-owned execution state and reopen workspace-scoped run history diagnostics without duplicating ACP storage.

**Architecture:** This slice uses a live fixture-backed Playwright/CDP path. The test creates a canonical Research Workspace, bridges it to an ACP execution workspace, creates a project/task/run through ACP APIs, then verifies the Research Workspace history modal and Agent Tasks diagnostics route consume ACP-owned state filtered by the canonical workspace envelope.

**Tech Stack:** FastAPI workspaces + agent-orchestration APIs, Next.js WebUI, Playwright E2E, Backlog.md task tracking.

---

### Task 1: Establish RED Contract

**Files:**
- Test: `apps/tldw-frontend/e2e/workflows/tier-3-automation/agent-tasks.spec.ts`

- [x] **Step 1: Run targeted Playwright discovery before adding the test**

Run: `npx playwright test e2e/workflows/tier-3-automation/agent-tasks.spec.ts --project=tier-3 --grep "binds a Research Workspace to a real ACP run history and diagnostics path" --list`

Expected: no matching test exists.

### Task 2: Probe Live ACP Feasibility

**Files:**
- Read: `tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py`
- Read: `tldw_Server_API/Config_Files/agents.yaml`

- [x] **Step 1: Start a real backend and WebUI on local test ports**
- [x] **Step 2: Use API calls to create workspace, bridge, project, task, and run**
- [x] **Step 3: Record whether the live runner creates a session-backed run with diagnostics links**

Expected: either a real run with ACP diagnostics links exists, or the blocker is explicit and documented without marking RW-UAT-022 Pass.

### Task 3: Add Live Fixture E2E

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/tier-3-automation/agent-tasks.spec.ts`

- [x] **Step 1: Add a Playwright test that creates the canonical workspace and ACP bridge**
- [x] **Step 2: Assert `/agent-orchestration/projects` filters by `canonical_workspace_id` and `canonical_workspace_source=research_workspace`**
- [x] **Step 3: Open Research Workspace ACP run history and assert ACP-owned run state is surfaced**
- [x] **Step 4: Open diagnostics through the ACP-owned session link**

Expected: the test fails when the bridge/history/diagnostics handoff is broken and skips only when the live server lacks ACP runtime support.

### Task 4: Update Evidence

**Files:**
- Modify: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Modify: `backlog/tasks/task-478.28 - Validate-ACP-workspace-scoped-run-history-and-diagnostics-for-Research-Workspace.md`

- [x] **Step 1: Update RW-UAT-022 with only the verified live outcome**
- [x] **Step 2: Record verification commands and any live ACP runtime blocker in the Backlog task**
- [x] **Step 3: Keep Research Workspace ownership language explicit: ACP stores runs; Research Workspace links and filters**

Expected: matrix status reflects live backend/WebUI/CDP evidence, not inferred behavior.

### Task 5: Verify and Commit

**Files:**
- All files touched by Tasks 1-4

- [x] **Step 1: Run the focused Playwright test**
- [x] **Step 2: Run focused frontend tests or type checks if touched helper logic requires it**
- [x] **Step 3: Run Bandit or document non-code skip for docs/E2E-only changes**
- [x] **Step 4: Review diff, update Backlog final summary, and commit the slice**

Expected: committed work contains only the ACP validation slice and leaves unrelated local files untouched.
