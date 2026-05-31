# Research Workspace MCP Hub Deep-Link Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approach C handoff where Research Workspace deep-links to MCP Hub with the active canonical workspace ID, and MCP Hub owns workspace-set binding interpretation.

**Architecture:** Keep the slice frontend-owned. Research Workspace builds a contextual MCP Hub route using `capabilities.workspace_id`; MCP Hub reads the query context and derives existing/no-binding state from `listWorkspaceSetObjects()` plus `listWorkspaceSetMembers()`.

**Tech Stack:** React, React Router, Ant Design, Vitest/jsdom, Playwright/CDP for live validation.

---

### Task 1: Research Workspace Link Contract

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceCapabilityRemediation.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceCapabilityRemediation.test.tsx`

- [x] **Step 1: Write the failing test**

Add a test proving an MCP readiness item links to `/mcp-hub?workflow=workspaces&view=workspace-sets&workspace_id=<encoded>&source=research-workspace`.

- [x] **Step 2: Run test to verify it fails**

Run:
`cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceCapabilityRemediation.test.tsx --maxWorkers=1`

Expected: FAIL because the link still targets `/mcp-hub`.

- [x] **Step 3: Implement minimal link builder**

Use `capabilities.workspace_id` when building the MCP Hub management link. Keep other management links unchanged.

- [x] **Step 4: Run test to verify it passes**

Run the same Vitest command.

Expected: PASS.

### Task 2: MCP Hub Workspace Context Interpretation

**Files:**
- Modify: `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/WorkspaceSetsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/__tests__/WorkspaceSetsTab.test.tsx`

- [x] **Step 1: Write route-context failing tests**

Add tests proving `/mcp-hub?workflow=workspaces&view=workspace-sets&workspace_id=rw-1&source=research-workspace` opens Workspace Sets and passes `rw-1` into the tab.

- [x] **Step 2: Add Workspace Sets state tests**

Mock workspace sets and members. Cover:
- a matching member shows the matching workspace-set names;
- no matching member shows an explicit no-binding message;
- API load failure keeps the existing MCP Hub load error.

- [x] **Step 3: Run tests to verify failures**

Run:
`cd apps/packages/ui && bunx vitest run src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx src/components/Option/MCPHub/__tests__/WorkspaceSetsTab.test.tsx --maxWorkers=1`

Expected: FAIL because query context is ignored.

- [x] **Step 4: Implement MCP Hub context**

Parse `workspace_id` and `source` in `McpHubPage`; pass `focusWorkspaceId` and `focusSource` to `WorkspaceSetsTab`. In `WorkspaceSetsTab`, render a compact contextual callout above the form/list once loading completes.

- [x] **Step 5: Run tests to verify pass**

Run the same MCP Hub Vitest command.

Expected: PASS.

### Task 3: Matrix And Route Regression

**Files:**
- Modify: `Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md`
- Modify if needed: `apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`

- [x] **Step 1: Add or update focused Playwright assertion**

Ensure an e2e route test covers the Research Workspace MCP Hub link route or direct MCP Hub route with workspace query context.

- [x] **Step 2: Run focused frontend tests**

Run focused Vitest suites from Tasks 1 and 2.

- [x] **Step 3: Run live backend + WebUI CDP validation**

Use the current local backend/WebUI pattern from the Research Workspace UAT matrix. Validate:
- `/research-workspace` renders the contextual MCP Hub link when MCP readiness needs setup;
- clicking/opening the link lands on MCP Hub Workspace Sets with the same `workspace_id`;
- MCP Hub shows existing/no-binding state truthfully.

- [x] **Step 4: Update matrix honestly**

Move `RW-UAT-021` only as far as live evidence supports. If no real binding can be created, keep `Partial` and document the no-binding handoff evidence.

### Task 4: Closeout

**Files:**
- Modify: `backlog/tasks/task-478.21 - Gate-F-validate-MCP-workspace-set-binding-for-Research-Workspace.md`

- [x] **Step 1: Run route-name guard search**

Run:
`rg "workspace-playground|workspace_playground|Workspace Playground" apps/packages/ui/src/components/Option/ResearchWorkspace apps/packages/ui/src/components/Option/MCPHub apps/tldw-frontend/e2e/workflows/research-workspace.real-backend.spec.ts`

Expected: no active aliases or user-facing labels introduced.

- [x] **Step 2: Run Bandit only if backend Python changed**

If no Python files changed, record Bandit skip as frontend/docs-only.

- [x] **Step 3: Update TASK-478.21**

Record implementation notes, verification commands, known residuals, and final summary.
