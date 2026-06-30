# MCP Hub Setup Polish Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete `TASK-223.2` by closing the remaining MCP Hub setup diagnostics gap: deployment diagnostics must expose the WebUI readiness health URL, last health status, and last HTTP status code, while preserving the already-landed no-auth, catalog recovery, setup isolation, and toy MCP smoke behavior.

**Architecture:** Keep MCP Hub as the owner of setup diagnostics. Extend the existing `tldw:server-readiness-state` event emitted by `ServerReadinessGate` with bounded health-check details and cache the latest detail on `window` so diagnostics panels mounted after app readiness can still read it. Consume that snapshot in `DeploymentDiagnosticsPanel` without introducing a backend endpoint.

**Tech Stack:** React, TypeScript, Vitest/jsdom, Playwright E2E, Backlog.md.

---

### Task 1: Extend Server Readiness Diagnostics

**Files:**
- Modify: `apps/tldw-frontend/components/networking/ServerReadinessGate.tsx`
- Test: `apps/tldw-frontend/components/networking/__tests__/ServerReadinessGate.test.tsx`

- [x] **Step 1: Write the failing readiness diagnostics test**

Add a test that stubs advanced mode health to return HTTP `206` with `{ status: "degraded" }`, listens for `tldw:server-readiness-state`, and asserts the event detail includes:

```ts
{
  state: "degraded",
  healthUrl: "http://127.0.0.1:8000/api/v1/health",
  httpStatus: 206,
  healthStatus: "degraded"
}
```

Also assert `(window as any).__tldwServerReadinessState` receives the same bounded detail.

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run components/networking/__tests__/ServerReadinessGate.test.tsx
```

Expected: FAIL because the event/global snapshot does not yet expose `healthUrl`, `httpStatus`, or `healthStatus`.

- [x] **Step 3: Implement minimal readiness diagnostics**

In `ServerReadinessGate.tsx`, extend the health result type with:

```ts
healthUrl: string
httpStatus?: number
healthStatus?: string
errorMessage?: string
checkedAt: string
```

Populate those fields inside `checkHealth()`. For response bodies, parse `status` when JSON is available; for fetch/parse failures, return a blocked result with `errorMessage` and no `httpStatus`. Update `emitServerReadinessState()` to include the diagnostic fields and assign the same object to:

```ts
(window as any).__tldwServerReadinessState = detail
```

before dispatching the event.

- [x] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run components/networking/__tests__/ServerReadinessGate.test.tsx
```

Expected: PASS.

### Task 2: Surface Last Readiness Result In MCP Hub Diagnostics

**Files:**
- Modify: `apps/packages/ui/src/components/Option/MCPHub/DeploymentDiagnosticsPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/MCPHub/__tests__/DeploymentDiagnosticsPanel.test.tsx`

- [x] **Step 1: Write the failing deployment panel test**

Set `(window as any).__tldwServerReadinessState` before render and assert the panel shows:

- `Last health status`
- `degraded`
- `Last status code`
- `206`
- `Health URL`
- the cached readiness health URL

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd apps/packages/ui
bun run test src/components/Option/MCPHub/__tests__/DeploymentDiagnosticsPanel.test.tsx
```

Expected: FAIL because the panel currently shows computed Health URL and MCP health only, not the readiness snapshot fields.

- [x] **Step 3: Implement minimal panel state**

In `DeploymentDiagnosticsPanel.tsx`, read the cached value from `(window as any).__tldwServerReadinessState` on mount and subscribe to `tldw:server-readiness-state`. Render compact description rows:

- `Last health status`: `healthStatus ?? state ?? "unknown"`
- `Last status code`: `String(httpStatus)`, or `not recorded`
- `Last checked`: `checkedAt`, or `not recorded`

Keep the existing deployment mode, request mode, API origin, computed Health URL, and MCP health rows.

- [x] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
cd apps/packages/ui
bun run test src/components/Option/MCPHub/__tests__/DeploymentDiagnosticsPanel.test.tsx
```

Expected: PASS.

### Task 3: Re-run Existing Setup Polish Coverage

**Files:**
- Test only:
  - `apps/packages/ui/src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx`
  - `apps/packages/ui/src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx`
  - `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx`
  - `apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts`
  - `Docs/MCP/mcp_hub_management.md`

- [x] **Step 1: Verify already-landed PR 2 UI states**

Run:

```bash
cd apps/packages/ui
bun run test src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx
```

Expected: PASS. These tests cover no-auth stdio state, legacy fallback gating, catalog empty/stale guidance, and MCP Hub placement of deployment diagnostics.

- [x] **Step 2: Verify setup isolation documentation exists**

Check `Docs/MCP/mcp_hub_management.md` includes the disposable-path recipe for `USER_DB_BASE_DIR`, `DATABASE_URL`, and `MCP_DATABASE_URL`, and that the toy E2E uses a temporary stdio server path.

- [ ] **Step 3: Run the toy MCP E2E smoke when feasible**

Not run in this turn because no local backend was listening on `127.0.0.1:8000`; existing Playwright coverage and disposable-path documentation were verified by inspection.

Run:

```bash
cd apps/tldw-frontend
npx playwright test e2e/workflows/tier-2-features/mcp-hub.spec.ts --grep "Toy MCP walkthrough smoke" --reporter=line
```

Expected: PASS against a live backend that supports MCP Hub mutations, or SKIP with the existing guarded reason when the live API cannot mutate external servers or the temp stdio runtime is not executable from the API process.

### Task 4: Final Verification And Backlog Closeout

**Files:**
- Modify: `backlog/tasks/task-223.2 - PR-2-MCP-Hub-setup-polish-and-diagnostics.md`

- [x] **Step 1: Run final focused verification**

Run:

```bash
cd apps/packages/ui
bun run test src/components/Option/MCPHub/__tests__/DeploymentDiagnosticsPanel.test.tsx src/components/Option/MCPHub/__tests__/ExternalServersTab.test.tsx src/components/Option/MCPHub/__tests__/ToolCatalogsTab.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx
```

Run:

```bash
cd apps/tldw-frontend
bunx vitest run components/networking/__tests__/ServerReadinessGate.test.tsx
```

- [x] **Step 2: Run security/static hygiene checks**

Run:

```bash
git diff --check
```

Bandit is not required for this slice unless Python files are touched.

- [x] **Step 3: Update Backlog**

Mark `TASK-223.2` acceptance criteria and DoD complete, record the focused verification commands, and summarize that PR 2 now has no-auth setup, catalog recovery, deployment diagnostics with last readiness health status/code, setup isolation docs, and toy MCP smoke coverage.

- [ ] **Step 4: Commit**

Run:

```bash
git add apps/tldw-frontend/components/networking/ServerReadinessGate.tsx apps/tldw-frontend/components/networking/__tests__/ServerReadinessGate.test.tsx apps/packages/ui/src/components/Option/MCPHub/DeploymentDiagnosticsPanel.tsx apps/packages/ui/src/components/Option/MCPHub/__tests__/DeploymentDiagnosticsPanel.test.tsx docs/superpowers/plans/2026-05-28-mcp-hub-setup-polish-diagnostics-plan.md "backlog/tasks/task-223.1 - PR-1-MCP-Hub-live-discovery-and-chat-payload-correctness.md" "backlog/tasks/task-223.2 - PR-2-MCP-Hub-setup-polish-and-diagnostics.md"
git commit -m "feat: complete MCP Hub setup diagnostics"
```
