# Workspace Frontend Server Context Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add shared frontend Workspace context contracts that unify Research Workspace and ACP Playground on the server Workspace model for issue #1993.

**Architecture:** Keep `workspace-api.ts` as the raw server DTO layer. Add a focused `services/workspace-context` contract layer with pure normalizers, recovery-copy helpers, action eligibility helpers, and thin React hooks. Pilot the contract in Research Workspace and ACP Playground without creating #1994 resource index/activity UI.

**Tech Stack:** TypeScript, React 18, Vitest, Testing Library, existing Tldw API client, existing Research Workspace and ACP session stores.

---

## File Map

- Create `apps/packages/ui/src/services/workspace-context/contracts.ts`
  - Export normalized Workspace summary, active context, eligibility, recovery, membership label, and ACP session context types.
- Create `apps/packages/ui/src/services/workspace-context/normalizers.ts`
  - Pure helpers that derive frontend display/action contracts from `WorkspaceApiResponse`, `WorkspaceContextResponse`, `WorkspaceCapabilitiesResponse`, and `WorkspaceAllowedAction`.
- Create `apps/packages/ui/src/services/workspace-context/hooks.tsx`
  - Thin hooks for resolving active server Workspace context and capability-derived action state.
- Create `apps/packages/ui/src/services/workspace-context/index.ts`
  - Barrel exports for the shared contract.
- Create `apps/packages/ui/src/services/workspace-context/__tests__/normalizers.test.ts`
  - Pure contract tests.
- Create `apps/packages/ui/src/services/workspace-context/__tests__/hooks.test.tsx`
  - Hook behavior tests with a mocked Workspace API client.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
  - Render a compact server-authoritative Workspace context indicator and shared recovery copy.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`
  - Pilot rendering tests.
- Modify `apps/packages/ui/src/components/Option/ACPPlayground/ACPWorkspacePanel.tsx`
  - Render ACP session/server Workspace alignment and mismatch copy.
- Modify `apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPWorkspacePanel.test.tsx`
  - ACP pilot tests.
- Modify or add one global browse/list guard test near the existing global Workspace/search tests:
  - Prefer `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/workspace-global-search.test.ts` if it already covers the behavior.
  - Otherwise add `apps/packages/ui/src/services/workspace-context/__tests__/global-visibility.guard.test.ts`.
- Modify `Docs/superpowers/specs/2026-06-18-workspace-frontend-server-context-contract-design.md` only if implementation uncovers an API gap.
- Modify `backlog/tasks/task-2386 - Implement-Workspace-Phase-2-frontend-context-contracts-pilot.md`
  - Track plan path, touched files, verification, known skips, and final summary.

## Task 1: Pure Server Contract Normalizers

**Files:**
- Create: `apps/packages/ui/src/services/workspace-context/contracts.ts`
- Create: `apps/packages/ui/src/services/workspace-context/normalizers.ts`
- Create: `apps/packages/ui/src/services/workspace-context/index.ts`
- Test: `apps/packages/ui/src/services/workspace-context/__tests__/normalizers.test.ts`

- [x] **Step 1: Write failing normalizer tests**

Add tests for:

```ts
import {
  compareACPWorkspaceContext,
  normalizeActiveWorkspaceContext,
  normalizeWorkspaceSummary,
  resolveWorkspaceActionEligibility,
  resolveWorkspaceRecovery
} from "../normalizers"

it("keeps server workspace identity authoritative", () => {
  const summary = normalizeWorkspaceSummary({
    id: "ws-server-1",
    name: null,
    archived: false,
    study_materials_policy: "workspace",
    workspace_profile: "project",
    deleted: false,
    banner_title: null,
    banner_subtitle: null,
    banner_color: null,
    audio_provider: null,
    audio_model: null,
    audio_voice: null,
    audio_speed: null,
    created_at: "2026-06-18T00:00:00Z",
    last_modified: "2026-06-18T00:10:00Z",
    version: 7
  })

  expect(summary.id).toBe("ws-server-1")
  expect(summary.label).toBe("Workspace ws-server-1")
  expect(summary.profile).toBe("project")
})

it("maps partial workspace context to visible recovery copy", () => {
  const context = normalizeActiveWorkspaceContext({
    workspace_id: "ws-1",
    workspace_profile: "project",
    workspace_kind: "project",
    schema_version: 1,
    generated_at: "2026-06-18T00:00:00Z",
    workspace: workspaceFixture({ id: "ws-1", name: "Server Workspace" }),
    attention_state: "needs_attention",
    resolution: {
      status: "partial",
      partial_errors: [{ scope: "sources", code: "source_status_unavailable", message: "Source status unavailable" }]
    },
    project_root: projectRootFixture({ state: "attached" }),
    sources: { items: [], summary: sourceSummaryFixture({ total: 2 }) },
    capabilities: capabilitiesFixture({ workspace_id: "ws-1" }),
    services: {},
    allowed_actions: {},
    active_jobs: [],
    active_operations: [],
    partial_errors: [{ scope: "sources", code: "source_status_unavailable", message: "Source status unavailable" }]
  })

  expect(context.state).toBe("partial")
  expect(context.recovery.reasonCode).toBe("partial_context")
})

it("uses server action reason codes for eligibility recovery", () => {
  const decision = resolveWorkspaceActionEligibility("open_terminal", {
    allowed: false,
    reason_code: "workspace_project_root_missing"
  })

  expect(decision.allowed).toBe(false)
  expect(decision.recovery.nextStepHref).toBe("#/workspaces")
})

it("detects ACP session and active workspace mismatch without mutating either side", () => {
  const context = compareACPWorkspaceContext({
    sessionWorkspaceId: "ws-session",
    activeWorkspaceId: "ws-active"
  })

  expect(context.state).toBe("mismatch")
  expect(context.recovery.reasonCode).toBe("workspace_mismatch")
})
```

- [x] **Step 2: Run test to verify RED**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/workspace-context-contract/apps/packages/ui
npx vitest run src/services/workspace-context/__tests__/normalizers.test.ts --maxWorkers=1
```

Expected: fail because `services/workspace-context` does not exist.

- [x] **Step 3: Implement minimal contract types and normalizers**

Implement:

- `normalizeWorkspaceSummary(response)`
- `normalizeActiveWorkspaceContext(response | null, options?)`
- `resolveWorkspaceRecovery(reasonCode, options?)`
- `resolveWorkspaceActionEligibility(action, allowedAction?)`
- `compareACPWorkspaceContext({ sessionWorkspaceId, activeWorkspaceId, sessionWorkspaceLabel?, activeWorkspaceLabel? })`

Keep server enum values visible in the returned contracts. Add display labels only as derived fields.

- [x] **Step 4: Run test to verify GREEN**

Run the same focused Vitest command. Expected: pass.

- [x] **Step 5: Refactor**

Remove duplicated copy branches inside `normalizers.ts`. Do not add hooks or UI in this task.

## Task 2: Active Workspace Context Hooks

**Files:**
- Create: `apps/packages/ui/src/services/workspace-context/hooks.tsx`
- Test: `apps/packages/ui/src/services/workspace-context/__tests__/hooks.test.tsx`
- Modify: `apps/packages/ui/src/services/workspace-context/index.ts`

- [x] **Step 1: Write failing hook tests**

Add tests that mount a small component using `useActiveWorkspaceContext` with a mocked client:

```tsx
it("does not fetch when no active server workspace id exists", async () => {
  const getWorkspaceContext = vi.fn()
  render(<Probe workspaceId={null} client={{ getWorkspaceContext }} />)

  expect(screen.getByTestId("state")).toHaveTextContent("none")
  expect(getWorkspaceContext).not.toHaveBeenCalled()
})

it("normalizes fetched server context", async () => {
  const getWorkspaceContext = vi.fn(async () => workspaceContextFixture({ workspace_id: "ws-1" }))
  render(<Probe workspaceId="ws-1" client={{ getWorkspaceContext }} />)

  expect(await screen.findByTestId("state")).toHaveTextContent("ready")
  expect(getWorkspaceContext).toHaveBeenCalledWith("ws-1")
})

it("surfaces failed server resolution as degraded context", async () => {
  const getWorkspaceContext = vi.fn(async () => {
    throw new Error("network down")
  })
  render(<Probe workspaceId="ws-1" client={{ getWorkspaceContext }} />)

  expect(await screen.findByTestId("state")).toHaveTextContent("error")
})
```

- [x] **Step 2: Run test to verify RED**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/workspace-context-contract/apps/packages/ui
npx vitest run src/services/workspace-context/__tests__/hooks.test.tsx --maxWorkers=1
```

Expected: fail because hook file does not exist.

- [x] **Step 3: Implement minimal hook**

Implement `useActiveWorkspaceContext({ workspaceId, client })`.

Rules:

- `workspaceId` is the server Workspace ID.
- `null` or blank ID returns `state: "none"` and does not fetch.
- Fetch `client.getWorkspaceContext(workspaceId)` on ID change.
- Use an ignore flag in `useEffect` cleanup to avoid stale updates.
- Return `{ context, loading, error, refresh }`.
- Default the client to the existing Tldw API client only after checking the local import pattern.

- [x] **Step 4: Run hook tests**

Expected: pass.

## Task 3: Research Workspace Pilot

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`

- [x] **Step 1: Write failing rendering tests**

Add tests for:

- Ready server Workspace context renders a compact server Workspace label.
- Missing/error context renders shared recovery copy and a canonical Workspaces link.
- Existing local workspace title editing still works.

Example expected UI assertions:

```tsx
expect(await screen.findByText(/Server Workspace/i)).toBeInTheDocument()
expect(screen.getByText(/Server context ready/i)).toBeInTheDocument()
expect(screen.getByRole("link", { name: /open Workspaces/i })).toHaveAttribute("href", "#/workspaces")
```

- [x] **Step 2: Run test to verify RED**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/workspace-context-contract/apps/packages/ui
npx vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx --maxWorkers=1
```

Expected: fail because the header does not render the server-context contract.

- [x] **Step 3: Implement compact server context indicator**

Use `useWorkspaceStore((s) => s.workspaceId)` as the server Workspace ID candidate, then call `useActiveWorkspaceContext`.

Render a small unframed inline indicator near existing header status controls:

- Ready: server Workspace label and state.
- Partial: label plus recovery message.
- Missing/error: recovery message and `#/workspaces` link.

Keep copy concise. Do not add a new Workspace browser or index.

- [x] **Step 4: Run Research Workspace test**

Expected: pass.

## Task 4: ACP Playground Pilot

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ACPPlayground/ACPWorkspacePanel.tsx`
- Test: `apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPWorkspacePanel.test.tsx`

- [x] **Step 1: Write failing ACP panel tests**

Add tests for:

- No-session state still links to canonical Workspaces.
- Aligned ACP session Workspace displays aligned context.
- Mismatch state displays both Workspace references and recovery copy.
- Terminal unavailable copy remains visible when `sshWsUrl` is missing.

Use the existing ACP session store test setup and add `workspaceId` to the active session fixture.

- [x] **Step 2: Run test to verify RED**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/workspace-context-contract/apps/packages/ui
npx vitest run src/components/Option/ACPPlayground/__tests__/ACPWorkspacePanel.test.tsx --maxWorkers=1
```

Expected: fail because the ACP panel does not render alignment/mismatch contract copy.

- [x] **Step 3: Implement ACP session comparison**

Use `compareACPWorkspaceContext` with:

- `sessionWorkspaceId` from `activeSession.workspaceId` if present.
- `activeWorkspaceId` from the active server Workspace context hook or local Workspace store ID.

Render a compact notice in the panel header or empty-state copy. Do not mutate either store and do not auto-switch Workspaces.

- [x] **Step 4: Run ACP panel test**

Expected: pass.

## Task 5: Global Visibility Guard, Docs, Backlog, And Verification

**Files:**
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/workspace-global-search.test.ts` or `apps/packages/ui/src/services/workspace-context/__tests__/global-visibility.guard.test.ts`
- Modify: `backlog/tasks/task-2386 - Implement-Workspace-Phase-2-frontend-context-contracts-pilot.md`
- Modify: `Docs/superpowers/specs/2026-06-18-workspace-frontend-server-context-contract-design.md` only if needed

- [x] **Step 1: Write global visibility guard test**

Add a focused test proving active Workspace context does not filter global browse/search/list rows. The test can be pure if the existing global search utility is pure:

```ts
it("does not filter global rows by active workspace context unless an explicit workspace filter is passed", () => {
  const rows = [
    { id: "note-1", workspaceId: "ws-active", title: "Active" },
    { id: "note-2", workspaceId: "ws-other", title: "Other" }
  ]

  const visible = applyGlobalWorkspaceVisibility(rows, { activeWorkspaceId: "ws-active" })

  expect(visible.map((row) => row.id)).toEqual(["note-1", "note-2"])
})
```

If no shared utility exists, assert the invariant through the closest existing global search/list rendering test and do not introduce a fake product API.

- [x] **Step 2: Run guard test to verify RED or baseline**

If the behavior already exists, the test may pass on first run. In that case, note it as a characterization guard in the Backlog task.

- [x] **Step 3: Implement only if needed**

If the guard exposes accidental filtering, fix only that filtering path. Do not build Workspace filter UI in this task.

- [x] **Step 4: Run focused verification**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/workspace-context-contract/apps/packages/ui
npx vitest run \
  src/services/workspace-context/__tests__/normalizers.test.ts \
  src/services/workspace-context/__tests__/hooks.test.tsx \
  src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx \
  src/components/Option/ACPPlayground/__tests__/ACPWorkspacePanel.test.tsx \
  --maxWorkers=1
```

Also run the global guard test if it is in a separate file.

- [x] **Step 5: Record Bandit skip**

Because this slice is frontend/docs only, record in `TASK-2386` that Bandit is not applicable unless Python files were touched.

- [x] **Step 6: Finalize Backlog task**

Update `TASK-2386` with:

- plan path;
- touched files;
- verification commands and results;
- known skips or blockers;
- final summary.

- [x] **Step 7: Commit**

Run:

```bash
git status --short
git add Docs/superpowers/specs/2026-06-18-workspace-frontend-server-context-contract-design.md \
  Docs/superpowers/plans/2026-06-18-workspace-frontend-server-context-contract.md \
  "backlog/tasks/task-2386 - Implement-Workspace-Phase-2-frontend-context-contracts-pilot.md" \
  apps/packages/ui/src/services/workspace-context \
  apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx \
  apps/packages/ui/src/components/Option/ACPPlayground/ACPWorkspacePanel.tsx \
  apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPWorkspacePanel.test.tsx
git commit -m "feat(ui): add server workspace context contracts"
```

Add any global guard test path if separate.
