# WebUI Capability Error State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace raw endpoint and capability failures on WebUI/extension routes with reusable, user-language capability states and diagnostics.

**Architecture:** Reuse the existing design-system state primitives first, then add a small pure capability-state adapter only if route code needs a shared mapping layer. Adopt the shared states in `/sources`, `/scheduled-tasks`, and `/integrations` before expanding to other capability-sensitive routes. Keep raw endpoint details available as diagnostics, never as the primary route message.

**Tech Stack:** React, TypeScript, shared `@tldw/ui` components, existing design-system state tokens, TanStack Query route data states, Vitest, Testing Library, Playwright, Backlog.md task tracking.

---

## Source Documents

- Parent plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- Source spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Source audit: `Docs/Reviews/WEBUI_EXTENSION_UX_HCI_AUDIT_2026_05_17.md`
- Planning Backlog task: `TASK-420`
- Parent planning task: `TASK-418`

## Scope

### Findings Closed Or Partially Closed

- F4: `/sources` exposes raw technical 404 as the main page state.
- F5 support: model settings need configured/usable provider states before full catalog work.
- F9: capability and unsupported-state handling is inconsistent.
- F18 support: hosted, beta, unsupported, and debug routes need explicit visibility/capability language.

### Primary Route Rows

This slice owns the shared state model and first route adoption for:

`/sources`, `/scheduled-tasks`, `/integrations`, `/admin`, `/agents`,
`/agent-tasks`, `/acp-playground`, `/settings/model`, `/evaluations`,
`/mcp-hub`, `/skills`, `/tts`, `/speech`, `/data-tables`.

First implementation adopters:

`/sources`, `/scheduled-tasks`, `/integrations`.

Later route-family adopters must be handled in their own child implementation
tasks after the shared mapping is stable.

### Out Of Scope

- No backend module implementation.
- No broad page redesign.
- No provider/model settings restructure beyond shared state contracts.
- No new design system. Use existing `StatePanel`, `RecoveryCallout`,
  `DiagnosticRow`, `SetupRequiredPanel`, `FeatureEmptyState`, and
  `RouteErrorBoundary` patterns first.
- No hiding diagnostics from operators. Move raw details behind diagnostics.

## Capability Vocabulary

Create or document a vocabulary that maps product situations to existing
design-system states:

| Capability situation | Design state | Primary user-language meaning | Diagnostic examples |
|---|---|---|---|
| No data | `empty` | The feature is available, but there is nothing here yet. | Count, current filter, scope. |
| Unavailable server capability | `unavailable` | This server does not support this capability or endpoint. | Method, path, status, server URL. |
| Missing worker | `setup_required` or `degraded` | A background worker or service is not running. | Worker name, queue, status probe. |
| Authentication required | `auth_required` | The user needs to connect or sign in before this feature can load. | Status 401, auth mode, server URL. |
| Missing permission | `permission_denied` | The signed-in user cannot perform this action. | Status 403, role/scope when safe. |
| Not configured | `setup_required` | Required provider, API key, server URL, or feature setting is missing. | Provider id, config key name, route. |
| Degraded | `degraded` | Some data loaded but part of the feature is limited. | Partial errors, failed providers. |
| Unsupported server version | `unavailable` | The connected server is older or lacks this route. | Version, endpoint, expected capability. |
| Network failure | `unavailable` | The frontend cannot reach the configured server. | Request path, status, raw message. |

Do not introduce new top-level design-system state keys in this slice unless
the existing canonical keys cannot express the capability situation.

## File Map

### Shared State Foundation

- Modify: `apps/packages/ui/src/components/ui/state/StatePanel.tsx`
  - Preserve current API.
  - Add test hooks or diagnostics affordances only if needed.

- Modify: `apps/packages/ui/src/components/ui/state/RecoveryCallout.tsx`
  - Keep it a thin wrapper over `StatePanel`.
  - Extend only if capability callers need a typed recovery state.

- Modify: `apps/packages/ui/src/components/ui/state/DiagnosticRow.tsx`
  - Preserve code wrapping and copy-label behavior.
  - Add copy actions only if implementation task includes tests.

- Create only if needed: `apps/packages/ui/src/components/ui/state/capability-state.ts`
  - Pure mapping helpers from route/query errors to state props.
  - No React hooks.
  - No network calls.

- Modify: `apps/packages/ui/src/components/ui/state/index.ts`
  - Export shared capability helpers if created.

- Test: `apps/packages/ui/src/components/ui/state/__tests__/state-primitives.test.tsx`
- Test if helper is created: `apps/packages/ui/src/components/ui/state/__tests__/capability-state.test.ts`

### Existing Recovery/Error Foundation

- Modify: `apps/packages/ui/src/components/Common/RouteErrorBoundary.tsx`
- Modify: `apps/packages/ui/src/components/Common/BackendUnavailableRecovery.tsx`
- Reference: `apps/packages/ui/src/components/Common/QuickIngest/ErrorClassification.ts`
- Test: `apps/packages/ui/src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx`
- Test: `apps/packages/ui/src/components/Common/__tests__/BackendUnavailableRecovery.test.tsx`

### First Route Adopters

- Modify: `apps/packages/ui/src/components/Option/Sources/SourcesWorkspacePage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Sources/SourcesAvailabilityGate.tsx`
- Test: `apps/packages/ui/src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/option-sources-route-guards.test.tsx`

- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
- Test: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx`

- Modify: `apps/packages/ui/src/components/Option/Integrations/IntegrationManagementPage.tsx`
- Modify only if needed: `apps/packages/ui/src/components/Option/Integrations/IntegrationPolicyPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/integrations-route.test.tsx`

### Later Route Adopters

These routes should not be changed in the first implementation PR unless the
change is a shared helper with test coverage:

- `apps/packages/ui/src/components/Option/AgentRegistry/index.tsx`
- `apps/packages/ui/src/components/Option/AgentTasks/index.tsx`
- `apps/packages/ui/src/components/Option/ACPPlayground/index.tsx`
- `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`
- `apps/packages/ui/src/components/Option/Models/index.tsx`
- `apps/packages/ui/src/components/Option/TTS/TtsPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/DataTables/DataTablesPage.tsx`
- `apps/packages/ui/src/components/Option/Skills/SkillsWorkspace.tsx`

## Implementation Tasks

### Task 0: Baseline And Backlog Setup

**Files:**
- Reference: `Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md`
- Reference: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Backlog: create or update an implementation task before product code edits.

- [ ] **Step 1: Verify branch and dirty worktree**

Run:

```bash
git branch --show-current
git status --short
```

Expected:
- You know the active branch.
- Existing unrelated dirty files are left untouched.

- [ ] **Step 2: Create implementation Backlog task**

Create a task named `Implement WebUI capability and error states`.

Expected:
- The task links this plan, parent plan, source spec, and audit.
- The task lists F4, F9, and support coverage for F5 and F18.

- [ ] **Step 3: Capture current route evidence**

Use browser or Playwright evidence for:
- `/sources` with unavailable sources endpoint.
- `/scheduled-tasks` with unavailable scheduled task endpoint.
- `/integrations` with unsupported personal integrations.

Expected:
- Baseline evidence is linked from the Backlog task.

### Task 1: Lock Shared State Primitive Expectations

**Files:**
- Modify: `apps/packages/ui/src/components/ui/state/__tests__/state-primitives.test.tsx`
- Modify only if needed: `apps/packages/ui/src/components/ui/state/StatePanel.tsx`
- Modify only if needed: `apps/packages/ui/src/components/ui/state/RecoveryCallout.tsx`
- Modify only if needed: `apps/packages/ui/src/components/ui/state/DiagnosticRow.tsx`

- [ ] **Step 1: Add failing tests for route capability requirements**

Extend `state-primitives.test.tsx` so `StatePanel` and `RecoveryCallout` prove:
- state label is visible
- user-language title is visible
- primary action is visible when provided
- diagnostics are rendered as a separate diagnostics region
- raw endpoint examples can be rendered only inside diagnostics

Use a test case with a raw endpoint value such as `/api/v1/sources`.

- [ ] **Step 2: Run test to verify failure or current coverage**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/ui/state/__tests__/state-primitives.test.tsx
```

Expected:
- Either fails on missing behavior or documents that existing primitives already
  meet the shared-state requirements.

- [ ] **Step 3: Implement minimal primitive changes if needed**

Only change primitives if a requirement cannot be met through current props.
Do not add route-specific wording to shared primitives.

- [ ] **Step 4: Run test to verify pass**

Run:

```bash
bunx vitest run src/components/ui/state/__tests__/state-primitives.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/ui/state apps/packages/ui/src/components/ui/state/__tests__/state-primitives.test.tsx
git commit -m "test: lock shared capability state primitives"
```

### Task 2: Add Pure Capability State Mapping If Needed

**Files:**
- Create only if needed: `apps/packages/ui/src/components/ui/state/capability-state.ts`
- Create only if needed: `apps/packages/ui/src/components/ui/state/__tests__/capability-state.test.ts`
- Modify if helper is created: `apps/packages/ui/src/components/ui/state/index.ts`

- [ ] **Step 1: Decide whether a helper is necessary**

Do not create a helper if route components can pass `RecoveryCallout` or
`StatePanel` props directly without duplication.

Create a helper only if at least two first-adopter routes would otherwise
duplicate error-to-state mapping.

- [ ] **Step 2: Write failing helper tests if helper is needed**

Test mappings for:
- HTTP 404 endpoint missing -> `unavailable`
- HTTP 401 -> `auth_required`
- HTTP 403 -> `permission_denied`
- fetch failed or connection refused -> `unavailable`
- partial response with errors -> `degraded`
- unsupported capability flag -> `unavailable`
- missing config/provider -> `setup_required`

- [ ] **Step 3: Run test to verify failure**

Run:

```bash
bunx vitest run src/components/ui/state/__tests__/capability-state.test.ts
```

Expected:
- Fails before helper exists.

- [ ] **Step 4: Implement pure helper**

Suggested exports:

```ts
export type CapabilityStateKind =
  | "empty"
  | "unavailable"
  | "missing_worker"
  | "auth_required"
  | "permission_denied"
  | "not_configured"
  | "degraded"
  | "unsupported_version"
  | "network_failure"

export type CapabilityStateDescriptor = {
  kind: CapabilityStateKind
  state: DesignSystemStateKey
  title: string
  message: string
  diagnostics?: StatePanelDiagnostic[]
  primaryAction?: StateAction
  secondaryActions?: StateAction[]
}
```

Keep route-specific titles/messages in route code unless the helper receives
feature labels as arguments.

- [ ] **Step 5: Run test to verify pass**

Run:

```bash
bunx vitest run src/components/ui/state/__tests__/capability-state.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/ui/state/capability-state.ts apps/packages/ui/src/components/ui/state/__tests__/capability-state.test.ts apps/packages/ui/src/components/ui/state/index.ts
git commit -m "feat: add capability state mapping helper"
```

Skip this commit if the helper is not needed.

### Task 3: Replace `/sources` Raw Route States

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Sources/SourcesWorkspacePage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Sources/SourcesAvailabilityGate.tsx`
- Modify: `apps/packages/ui/src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/option-sources-route-guards.test.tsx`

- [ ] **Step 1: Write failing sources tests**

Test these states:
- unsupported ingestion sources capability uses shared state UI
- list endpoint 404 does not show raw `Not Found (GET /api/v1/sources)` as primary text
- endpoint method/path/status appear only in diagnostics
- empty list uses an empty state with a creation action
- offline/setup state still routes through the connection gate

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
bunx vitest run src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx src/routes/__tests__/option-sources-route-guards.test.tsx
```

Expected:
- Fails where current `Alert` or `FeatureEmptyState` does not meet shared-state
  expectations.

- [ ] **Step 3: Adopt shared state components**

Replace primary error `Alert` usage with `RecoveryCallout` or `StatePanel`.
Keep Ant Design table/form controls unchanged.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
bunx vitest run src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx src/routes/__tests__/option-sources-route-guards.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/Sources apps/packages/ui/src/routes/__tests__/option-sources-route-guards.test.tsx
git commit -m "fix: use shared capability states for sources"
```

### Task 4: Replace `/scheduled-tasks` Raw Route States

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx`

- [ ] **Step 1: Write failing scheduled-task tests**

Test these states:
- unsupported scheduled task endpoint uses shared unavailable state
- query error does not show raw endpoint text as primary UI
- partial data errors use `degraded`
- diagnostics include endpoint/status details when available
- existing reminder create/edit/delete flows still render when supported

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/routes/__tests__/scheduled-tasks-route.test.tsx
```

Expected:
- Fails until unsupported/error/partial states use shared state language.

- [ ] **Step 3: Adopt shared state components**

Use `RecoveryCallout` for unavailable/error states and `StatePanel` with
`degraded` for partial data. Keep the table/editor unchanged.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/routes/__tests__/scheduled-tasks-route.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx
git commit -m "fix: use shared capability states for scheduled tasks"
```

### Task 5: Replace `/integrations` Unsupported And Error States

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Integrations/IntegrationManagementPage.tsx`
- Modify only if needed: `apps/packages/ui/src/components/Option/Integrations/IntegrationPolicyPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/integrations-route.test.tsx`

- [ ] **Step 1: Write failing integrations tests**

Test these states:
- personal integrations unsupported uses shared unavailable state
- overview query failure does not show raw endpoint text as primary UI
- provider policy failures remain scoped to provider cards or diagnostics
- admin/workspace integrations route keeps existing route boundary behavior

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
bunx vitest run src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx src/routes/__tests__/integrations-route.test.tsx
```

Expected:
- Fails until unsupported and error states use shared capability language.

- [ ] **Step 3: Adopt shared state components**

Replace top-level unsupported/error `Alert` states with `RecoveryCallout` or
`StatePanel`. Leave lower-level provider-card alerts alone unless they leak raw
endpoint text into primary route state.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
bunx vitest run src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx src/routes/__tests__/integrations-route.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/Integrations apps/packages/ui/src/routes/__tests__/integrations-route.test.tsx
git commit -m "fix: use shared capability states for integrations"
```

### Task 6: Keep Route Error Boundary Diagnostics Behind Disclosure

**Files:**
- Modify: `apps/packages/ui/src/components/Common/RouteErrorBoundary.tsx`
- Modify: `apps/packages/ui/src/components/Common/BackendUnavailableRecovery.tsx`
- Modify: `apps/packages/ui/src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx`
- Modify: `apps/packages/ui/src/components/Common/__tests__/BackendUnavailableRecovery.test.tsx`

- [ ] **Step 1: Write or extend recovery tests**

Test that backend-unreachable recovery:
- shows user-language title/message
- exposes retry/reload/health/settings actions
- places method/path/status/raw message in diagnostics
- does not put raw endpoint details in the primary title/message

- [ ] **Step 2: Run tests to verify failure or current coverage**

Run:

```bash
bunx vitest run src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx src/components/Common/__tests__/BackendUnavailableRecovery.test.tsx
```

Expected:
- Fails only if current recovery UI leaks raw details into primary copy.

- [ ] **Step 3: Patch boundary/recovery only if needed**

Keep existing backend classification behavior. Move raw details into diagnostics
only where tests prove leakage.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
bunx vitest run src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx src/components/Common/__tests__/BackendUnavailableRecovery.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Common/RouteErrorBoundary.tsx apps/packages/ui/src/components/Common/BackendUnavailableRecovery.tsx apps/packages/ui/src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx apps/packages/ui/src/components/Common/__tests__/BackendUnavailableRecovery.test.tsx
git commit -m "fix: keep backend recovery diagnostics out of primary copy"
```

Skip this commit if existing coverage already passes.

### Task 7: Browser QA For First Adopters

**Files:**
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/sources.spec.ts`
- Test if available or added later: scheduled tasks and integrations workflow specs.
- Backlog: update active implementation task.

- [ ] **Step 1: Run route-focused browser checks**

From `apps/tldw-frontend`, run:

```bash
bunx playwright test e2e/workflows/tier-2-features/sources.spec.ts --reporter=line
```

Expected:
- `/sources` loads and uses the shared state for unavailable/error/empty states.

- [ ] **Step 2: Capture manual browser evidence for scheduled tasks and integrations**

If there is no focused workflow spec for these routes, capture browser evidence
manually for:
- `/scheduled-tasks`
- `/integrations`

Expected:
- Evidence shows no primary raw endpoint text.
- Diagnostics remain available.

- [ ] **Step 3: Record remaining route adopters**

Update the Backlog task with later adopters that still need this shared state:
- `/admin`
- `/agents`
- `/agent-tasks`
- `/acp-playground`
- `/settings/model`
- `/evaluations`
- `/mcp-hub`
- `/skills`
- `/tts`
- `/speech`
- `/data-tables`

### Task 8: Final Capability State Gate

**Files:**
- Reference all files changed in Tasks 1-7.
- Backlog: update the active implementation task.

- [ ] **Step 1: Run unit and route tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/ui/state/__tests__/state-primitives.test.tsx src/components/Common/__tests__/RouteErrorBoundary.backend-recovery.test.tsx src/components/Common/__tests__/BackendUnavailableRecovery.test.tsx src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx src/routes/__tests__/option-sources-route-guards.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/routes/__tests__/scheduled-tasks-route.test.tsx src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx src/routes/__tests__/integrations-route.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run browser smoke for changed routes**

From `apps/tldw-frontend`, run:

```bash
bunx playwright test e2e/workflows/tier-2-features/sources.spec.ts --reporter=line
```

Expected:
- PASS or environment-specific skip documented with evidence.

- [ ] **Step 3: Run diff check**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Update Backlog task**

Record:
- findings closed or partially closed
- first adopter route evidence
- tests run
- browser evidence paths
- known skips
- remaining route-family adopters
- whether a backend capability-map dependency is needed

- [ ] **Step 5: Commit final task notes if needed**

```bash
git add backlog/tasks/<task-file>.md
git commit -m "docs: record capability state verification"
```

## Review Checklist

Before opening an implementation PR:

- [ ] Shared state primitives are reused before new primitives are added.
- [ ] `/sources`, `/scheduled-tasks`, and `/integrations` no longer show raw
  endpoint text as primary route state.
- [ ] Raw method/path/status/message details remain available in diagnostics.
- [ ] Empty, unsupported, unauthenticated, unauthorized, degraded, and network
  failure states have user-language copy and next actions.
- [ ] Existing tables, forms, provider cards, and dense controls are preserved.
- [ ] Later adopters are listed instead of silently skipped.
- [ ] Browser evidence is attached for changed visual routes.
- [ ] No backend API was changed unless separately justified.

## Planning Verification

After editing this plan, run:

```bash
rg -n 'T[O]DO|T[B]D|FIX[M]E|\\.\\.\\.|\\bmaybe\\b|\\bprobably\\b|\\bshould consider\\b' Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md 'backlog/tasks/task-420 - Plan-WebUI-capability-and-error-state-implementation.md'
rg -n '[[:blank:]]$|[^\\x00-\\x7F]' Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md 'backlog/tasks/task-420 - Plan-WebUI-capability-and-error-state-implementation.md'
git diff --check -- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md 'backlog/tasks/task-420 - Plan-WebUI-capability-and-error-state-implementation.md'
```

Expected:
- Placeholder scan exits 1 with no output.
- ASCII/trailing whitespace scan exits 1 with no output.
- `git diff --check` exits 0.
