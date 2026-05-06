# Sandbox Host-Local Warnings UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface sandbox host-local isolation warnings in the existing admin monitoring UI so operators can see that `seatbelt` and `worktree` are weaker than VM-grade isolation.

**Architecture:** Keep the backend unchanged and consume the existing `GET /api/v1/sandbox/admin/runtime-diagnostics` endpoint from the shared UI client. Add a read-only sandbox diagnostics card to `MonitoringDashboardPage` because it already aggregates admin health signals and has focused Vitest coverage. Render warning badges from API metadata instead of hard-coding runtime names as the source of truth.

**Tech Stack:** React 18, Ant Design, shared `@tldw/ui` package, `bgRequest`, Vitest with Testing Library.

---

### Task 1: Add Sandbox Diagnostics Client Contract

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`

- [x] **Step 1: Add typed response interfaces**

Add lightweight interfaces for `SandboxAdminRuntimeDiagnosticsResponse`, summary, runtime item, and startup warning summary near the existing admin diagnostics types. Include only fields used by the UI while allowing optional extra metadata.

- [x] **Step 2: Add the API method**

Add `getSandboxRuntimeDiagnostics()` to `TldwApiClientBase` and call:

```ts
return await bgRequest<SandboxAdminRuntimeDiagnosticsResponse>({
  path: "/api/v1/sandbox/admin/runtime-diagnostics",
  method: "GET"
})
```

### Task 2: Write Failing Monitoring UI Test

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Admin/__tests__/MonitoringDashboardPage.test.tsx`

- [x] **Step 1: Extend the client mock**

Add a `getSandboxRuntimeDiagnostics` mock and default response with no warning runtimes.

- [x] **Step 2: Add host-local warning test**

Create a test that returns `seatbelt` and `worktree` runtime rows with `isolation_warnings`, renders `MonitoringDashboardPage`, and expects:

```ts
expect(screen.getByText("Sandbox Runtime Isolation")).toBeTruthy()
expect(screen.getByText("Host-local sandbox runtimes require operator review")).toBeTruthy()
expect(screen.getByText("seatbelt")).toBeTruthy()
expect(screen.getByText("worktree")).toBeTruthy()
expect(screen.getByText(/not VM-grade isolation/i)).toBeTruthy()
```

- [x] **Step 3: Verify red**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Admin/__tests__/MonitoringDashboardPage.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: fail because the card is not implemented yet.

### Task 3: Render Sandbox Runtime Isolation Card

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Admin/MonitoringDashboardPage.tsx`

- [x] **Step 1: Add loading and diagnostics state**

Add `sandboxDiagnostics` and `sandboxDiagnosticsLoading` state plus a `loadSandboxDiagnostics` callback. Mark admin guard errors the same way as other admin calls.

- [x] **Step 2: Include diagnostics in refresh paths**

Call `loadSandboxDiagnostics()` during initial load and `refreshAll()`.

- [x] **Step 3: Render the card**

Add a `Card` below System Overview titled `Sandbox Runtime Isolation`. Include:
- summary counts for ready, unavailable, host-gated, and scaffold runtimes
- an `Alert` when `summary.host_local_warning_runtimes` is non-empty
- a small table with runtime, readiness, boundary, VM-grade status, untrusted eligibility, warning badges, and recommended action

- [x] **Step 4: Verify green**

Run the same focused Vitest command and confirm the new test passes.

### Task 4: Final Verification And Task Closeout

**Files:**
- Modify: `backlog/tasks/task-94 - Show-sandbox-host-local-runtime-warnings-in-admin-UI.md`

- [x] **Step 1: Run touched-file checks**

Run:

```bash
git diff --check
bunx vitest run apps/packages/ui/src/components/Option/Admin/__tests__/MonitoringDashboardPage.test.tsx --maxWorkers=1 --no-file-parallelism
```

- [x] **Step 2: Record Backlog completion**

Update TASK-94 acceptance criteria, implementation notes, final summary, and Bandit skip rationale. Bandit is not applicable if the final diff is frontend/docs/backlog only.

- [x] **Step 3: Commit**

```bash
git add Docs/superpowers/plans/2026-05-06-sandbox-host-local-warnings-ui.md apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/packages/ui/src/components/Option/Admin/MonitoringDashboardPage.tsx apps/packages/ui/src/components/Option/Admin/__tests__/MonitoringDashboardPage.test.tsx "backlog/tasks/task-94 - Show-sandbox-host-local-runtime-warnings-in-admin-UI.md"
git commit -m "feat(admin): show sandbox isolation warnings"
```
