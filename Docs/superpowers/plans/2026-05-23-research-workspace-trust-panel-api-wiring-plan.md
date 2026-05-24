# Research Workspace Trust Panel API Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the first Phase D trust surface in `/research-workspace` from the authoritative workspace source-status and capabilities APIs.

**Architecture:** Add typed frontend API methods for the backend contracts, then keep UI projection logic in a small Research Workspace component. The page fetches source status and capabilities for the active workspace, reconciles the legacy local `processing | ready | error` source state from server lifecycle states, and renders a compact trust panel without adding any legacy `/workspace-playground` path.

**Tech Stack:** React, TypeScript, Vitest, Testing Library, tldw frontend API client, FastAPI backend contracts.

---

### Task 1: API Client Contract

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Test: `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts`

- [x] **Step 1: Write failing tests**

Add tests that call `workspaceApiMethods.getWorkspaceSourcesStatus.call(fakeClient, "ws-1")` and `workspaceApiMethods.getWorkspaceCapabilities.call(fakeClient, "ws-1")`, asserting the paths are:

```text
/api/v1/workspaces/ws-1/sources/status
/api/v1/workspaces/ws-1/capabilities
```

- [x] **Step 2: Run test to verify RED**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts
```

Expected: fail because the methods do not exist.

- [x] **Step 3: Implement API methods and response types**

Add TypeScript interfaces for source readiness, source status summary, capability services, allowed actions, and methods for the two endpoints.

- [x] **Step 4: Run API tests to verify GREEN**

Run the same Vitest command. Expected: pass.

### Task 2: Trust Panel Component

**Files:**
- Create: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceTrustPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceTrustPanel.test.tsx`

- [x] **Step 1: Write failing tests**

Cover:
- queryable/source summary rendering;
- disabled grounded-question reason rendering;
- MCP Hub, ACP, sandbox, and provider service states;
- API error fallback state.

- [x] **Step 2: Run test to verify RED**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceTrustPanel.test.tsx
```

Expected: fail because the component does not exist.

- [x] **Step 3: Implement component**

Use existing `bg`, `surface`, `border`, `text`, and icon conventions. Keep the component compact and scan-friendly. Do not use nested cards or decorative gradients.

- [x] **Step 4: Run component tests to verify GREEN**

Run the same Vitest command. Expected: pass.

### Task 3: Page Wiring And Source Reconciliation

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx`

- [x] **Step 1: Write failing page test**

Add a test that mocks `tldwClient.getWorkspaceSourcesStatus` and `getWorkspaceCapabilities`, renders `ResearchWorkspace`, and asserts:
- the new panel renders source summary and service reasons;
- queryable source maps to local `ready`;
- missing/failed source maps to local `error`;
- processing/indexing source maps to local `processing`.

- [x] **Step 2: Run focused page test to verify RED**

Run:

```bash
cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx
```

Expected: fail because page wiring is missing.

- [x] **Step 3: Implement page wiring**

Fetch both backend projections when the store is hydrated and `workspaceId` exists. Refresh on the same existing status poll interval. Fail gracefully with a compact warning in the trust panel. Reconcile local source status only by `source.id` or `media_id`, with fail-closed mapping:
- `queryable` -> `ready`
- `queued`, `ingesting`, `extracting`, `chunking`, `indexing`, `retrying`, `partially_queryable` -> `processing`
- `failed`, `missing_media`, `blocked_by_permissions` -> `error`

- [x] **Step 4: Run focused page test to verify GREEN**

Run the same Vitest command. Expected: pass.

### Task 4: Verification

**Files:**
- Update: `backlog/tasks/task-466 - Wire-Research-Workspace-trust-panel-to-source-status-and-capabilities-API.md`

- [x] **Step 1: Run focused Vitest suite**

```bash
cd apps/packages/ui && bunx vitest run \
  src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts \
  src/components/Option/ResearchWorkspace/__tests__/WorkspaceTrustPanel.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx
```

- [x] **Step 2: Run diff check**

```bash
git diff --check -- \
  apps/packages/ui/src/services/tldw/domains/workspace-api.ts \
  apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.status-capabilities.test.ts \
  apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceTrustPanel.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceTrustPanel.test.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx
```

- [x] **Step 3: CDP/browser validation**

Start backend and WebUI when feasible, then use CDP-based browser automation to inspect `/research-workspace`. Do not use Computer Control. If the dev server cannot start because of existing repo-wide frontend churn, document the blocker and rely on focused Vitest plus backend HTTP validation.

- [x] **Step 4: Finalize Backlog**

Record verification, known caveats, and completion status in `TASK-466`.
