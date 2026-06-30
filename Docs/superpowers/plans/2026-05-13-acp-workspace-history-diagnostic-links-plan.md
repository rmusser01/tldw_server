# ACP Workspace History Diagnostic Links Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make recent ACP run history discoverable from WorkspacePlayground for the active canonical workspace, with links into existing Agent Tasks and ACP Playground diagnostic views.

**Architecture:** Add a small WorkspacePlayground modal that reuses existing Agent Orchestration project, task, and enriched task-detail contracts. The modal filters projects by `canonical_workspace_id`, fetches task detail only for matching workspace tasks, flattens recent runs, and routes users to existing Agent Tasks and ACP Playground views instead of creating a parallel ACP workspace browser.

**Tech Stack:** React, TypeScript, Ant Design, Vitest, React Testing Library, existing ACP auth/request transport helpers.

---

### Task 1: Red Tests For Workspace ACP History

**Files:**
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/__tests__/WorkspaceHeader.test.tsx`

- [x] **Step 1: Add a failing test for successful workspace history rendering**

Add a test that opens Workspace settings, selects `ACP run history`, mocks:
- `GET /api/v1/agent-orchestration/projects`
- `GET /api/v1/agent-orchestration/projects/{project_id}/tasks`
- `GET /api/v1/agent-orchestration/tasks/{task_id}`

Expected assertions:
- The modal renders a recent run for `workspace-alpha`.
- The modal includes project/task context, status, session id, and artifact/diagnostic/audit counts.
- `Open Agent Tasks` navigates to `/agent-tasks?workspace=workspace-alpha`.
- `Open diagnostics` navigates to `/acp-playground?session=sess-alpha&view=diagnostics`.

- [x] **Step 2: Add failing tests for empty and error states**

Add tests proving:
- A workspace with no matching ACP runs shows a non-misleading empty state.
- A backend load error shows an error state instead of setup guidance.

- [x] **Step 3: Run focused tests and verify red**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/WorkspacePlayground/__tests__/WorkspaceHeader.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: new tests fail because the modal/menu item does not exist yet.

### Task 2: Workspace ACP History Modal

**Files:**
- Create: `apps/packages/ui/src/components/Option/WorkspacePlayground/WorkspaceACPHistoryModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/WorkspaceHeader.tsx`

- [x] **Step 1: Implement the modal data loader**

Create a component that:
- Accepts `open`, `workspaceId`, `workspaceName`, `onCancel`, `onOpenAgentTasks`.
- Uses `useCanonicalConnectionConfig`, `resolveBrowserRequestTransport`, and `buildACPAuthHeaders`.
- Fetches projects from `/api/v1/agent-orchestration/projects`.
- Filters to projects whose `canonical_workspace.canonical_workspace_id` or metadata fallback matches the active workspace.
- Fetches tasks for matching projects.
- Fetches enriched task detail for a bounded number of matching tasks.
- Flattens and sorts recent runs by `started_at` or `completed_at`, newest first.

- [x] **Step 2: Render recent run history and diagnostic actions**

Render:
- Loading spinner while fetching.
- Empty state when no matching projects/tasks/runs exist.
- Alert when orchestration is unsupported or load fails.
- Recent run cards with project name, task title, run status, session id, counts, result/failure preview, and buttons for existing detail routes.

- [x] **Step 3: Wire the settings menu entry**

Add an `ACP run history` action to Workspace settings when `workspaceId` exists. Keep it near `Create agent task` because both are ACP workspace actions.

- [x] **Step 4: Run focused tests and verify green**

Run the same focused Vitest command. Expected: all WorkspaceHeader tests pass.

### Task 3: Documentation And Task Record

**Files:**
- Modify: `Docs/Design/ACP_Workspace_Integration_Decision_2026_05.md`
- Modify: `backlog/tasks/task-318 - Implement-ACP-workspace-history-and-diagnostic-links-for-issue-1540.md`

- [x] **Step 1: Update the design doc slice status**

Add a short implementation note under Slice 4 documenting that the first UI pass reuses existing Agent Orchestration and ACP Playground routes.

- [x] **Step 2: Update TASK-318 notes**

Record implementation details, touched files, verification commands, and Bandit rationale if no Python files were changed.

### Task 4: Verification And PR Packaging

**Files:**
- Modify only if verification exposes issues.

- [x] **Step 1: Run targeted verification**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/WorkspacePlayground/__tests__/WorkspaceHeader.test.tsx --maxWorkers=1 --no-file-parallelism
bunx tsc --noEmit -p /private/tmp/acp-workspace-history-tsconfig.json --pretty false
git diff --check
```

If the temporary TypeScript config does not exist, create one scoped to the touched WorkspacePlayground files and existing test ambient globals.

- [x] **Step 2: Commit and open PR**

Commit the focused slice and open a PR against `dev` referencing #1540 and TASK-318.
