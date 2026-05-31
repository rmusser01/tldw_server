# WebUI Operations Integrations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make operations, automation, integration, admin, MCP, sources, watchlists, workflow, and skills routes status-first and capability-aware without building missing backend systems.

**Architecture:** Add a small operations route-job contract, then adopt the WP2 shared capability-state vocabulary across the scoped route shells and route-owned components. Keep operator diagnostics available behind disclosure, keep `/connectors` as an honest placeholder family unless a backend exists, and treat new backend capability-map work as a separate contract gate.

**Tech Stack:** React, Next.js pages, shared `apps/packages/ui` route shells, route registry metadata, TanStack Query, existing design-system state primitives, Vitest, React Testing Library, Playwright.

---

## Source Documents

- Backlog task: `TASK-418.7`
- Parent implementation plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- UX remediation spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Dependency plans:
  - `Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md`
  - `Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md`
  - `Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md`

## Audit Findings Addressed

- `F4`: Raw endpoint or unsupported-module errors become the primary UI on operations routes.
- `F9`: Capability, unsupported, unavailable, unauthorized, degraded, and not-configured states are inconsistent across routes.
- `F12 support`: Connector and integration routes need truthful current-state routing and setup guidance.
- `F17 support`: Operator routes need status summaries before module detail and diagnostics.
- `F18 support`: Hosted, beta, placeholder, unsupported, and debug states need explicit visibility language.

## Route Inventory And Ownership

| Route | Operator goal | Current ownership | Primary workflows | UX contract |
| --- | --- | --- | --- | --- |
| `/admin` | Reach operations overview before module drill-down | `apps/tldw-frontend/pages/admin/index.tsx` currently redirects to `/admin/server` | See system health, admin modules, module statuses, next actions | Replace blind redirect with a WebUI overview or an explicit overview-to-server handoff. Do not invent extension-only admin root unless route registry work adds it. |
| `/admin/server` | Inspect server status, users, roles, and media budget diagnostics | `apps/packages/ui/src/routes/option-admin-server.tsx`, `ServerAdminPage` | Refresh stats, inspect users and roles, diagnose permission or timeout states | Keep as a module drill-down. Add cross-link from admin overview and make failure states diagnostic-first. |
| `/admin/integrations` | Manage workspace-level integration policy | `option-admin-integrations.tsx`, `IntegrationManagementPage scope="workspace"` | Review Slack, Discord, Telegram policy, linked actors, bot setup | Keep workspace policy distinct from personal `/integrations`. |
| `/admin/sources` | Manage sources as an admin | `option-admin-sources.tsx`, `SourcesWorkspacePage mode="admin"` | View, create, sync, inspect source config | Share source capability states with `/sources` and keep admin scope visible. |
| `/admin/monitoring` | Inspect monitoring and operations metrics | `option-admin-monitoring.tsx`, `MonitoringDashboardPage` | View status, refresh, inspect degraded data | Include as admin overview target but do not make it the only status summary. |
| `/mcp-hub` | Configure external MCP servers and governance workflows | `option-mcp-hub.tsx`, `McpHubPage` | Workflows for servers, credentials, policies, approvals, workspaces, audit | Keep workflow-first setup. Add clear status summary and diagnostic disclosure. |
| `/sources` | Manage ingestion sources that sync into notes or media | `option-sources.tsx`, `SourcesWorkspacePage mode="user"` | New source, list sources, sync, inspect unsupported/offline state | Use capability states before route data. Keep raw endpoint details out of primary UI. |
| `/connectors` | Explain connector route availability and current alternatives | `apps/tldw-frontend/pages/connectors/index.tsx` placeholder | Reach settings, integrations, sources, or docs depending on current support | Keep placeholder truthful. Do not build connector management without backend support. |
| `/connectors/browse` | Planned connector catalog entry | `apps/tldw-frontend/pages/connectors/browse.tsx` placeholder | Return to connector hub or settings | Keep as planned route unless backend support exists. |
| `/connectors/jobs` | Planned connector job orchestration entry | `apps/tldw-frontend/pages/connectors/jobs.tsx` placeholder | Return to connector hub or scheduled tasks when relevant | Do not imply jobs exist if they are not implemented. |
| `/connectors/sources` | Planned connector-source workflow entry | `apps/tldw-frontend/pages/connectors/sources.tsx` placeholder | Return to connector hub or sources when relevant | Connect to `/sources` only as current supported alternative. |
| `/integrations` | Manage personal Slack and Discord connections | `option-integrations.tsx`, `IntegrationManagementPage scope="personal"` | Refresh, inspect provider cards, connect, reconnect, disable, remove | Show provider status, unsupported state, auth state, and recovery actions in user language. |
| `/scheduled-tasks` | Manage reminder tasks and see scheduled task availability | `option-scheduled-tasks.tsx`, `ScheduledTasksPage` | Create reminder, edit, delete, refresh, see unsupported endpoint | Use WP2 capability states for unavailable and partial data states. |
| `/watchlists` | Operate feed collection, monitor jobs, activity, articles, reports, templates, settings | `option-watchlists.tsx`, `WatchlistsPlaygroundPage` | Add feeds, configure monitors, inspect runs, review items, generate outputs | Expose monitor/feed health and repeat-user controls without hiding advanced tabs. |
| `/workflow-editor` | Build and run visual workflows | `option-workflow-editor.tsx`, `WorkflowEditor` | Add nodes, configure steps, validate, save, import, export, run | Surface step-type availability, validation, save state, and run state before raw editor internals. |
| `/skills` | Manage skills when the server supports Skills API | `option-skills.tsx`, `SkillsWorkspace` | View skills, create/import/edit where supported | Keep capability gate and empty state explicit, with diagnostics behind disclosure. |

## Frontend-Only Versus Backend Capability-Map Work

This WP10 plan must separate user-facing cleanup from backend contract work.

### Frontend-Only Work

Use frontend-only changes when the route can already derive state from:

- `useServerCapabilities`.
- Existing TanStack Query loading, error, partial, and data states.
- Existing OpenAPI probing used by `ScheduledTasksPage` and `IntegrationManagementPage`.
- Existing route placeholder metadata.
- Existing admin, watchlists, workflow editor, MCP, and skills component state.

Frontend-only changes include:

- Route labels and headings.
- Empty, loading, unsupported, unavailable, unauthorized, degraded, partial, and not-configured state presentation.
- Diagnostics disclosure for raw endpoint, status code, request path, and server URL details.
- Local route metadata tests.
- Browser QA and Playwright assertions.

### Backend Capability-Map Gate

Create a separate backend contract task before implementation if a route needs state that is not available through current frontend inputs.

Backend-gated examples:

- A cross-module operations health map that requires server aggregation.
- A single `/api/v1/operations/status` endpoint.
- Connector inventory that is not represented by existing integrations, sources, or scheduled task APIs.
- Watchlist worker health that cannot be inferred from current watchlists endpoints.
- MCP server health that current MCP Hub queries cannot expose.

Do not add backend API changes inside a WP10 UI implementation PR unless the Backlog task explicitly broadens scope and the plan is updated first.

## Non-Goals

- Do not build missing connector management backends.
- Do not build missing integration providers.
- Do not build a new scheduler backend.
- Do not create a new design system.
- Do not hide operator diagnostics; move them behind disclosure.
- Do not replace existing Watchlists, MCP Hub, Workflow Editor, Sources, Integrations, or Skills runtimes.
- Do not rename current route paths.
- Do not add broad admin module work outside the scoped route surfaces.

## File Structure

### New Files

- `apps/packages/ui/src/routes/operations-route-jobs.ts`
  - Owns the WP10 route-job and capability-state metadata.
  - Keeps `/admin` and connector placeholder routes explicit even when they are Next pages rather than shared route registry entries.
- `apps/packages/ui/src/routes/__tests__/operations-route-jobs.test.ts`
  - Verifies route coverage, concepts, route ownership, and backend-gate flags.
- `apps/packages/ui/src/routes/__tests__/operations-route-boundaries.test.tsx`
  - Verifies the shared route wrappers use `OptionLayout`, route error boundaries where appropriate, and route-owned components.
- `apps/packages/ui/src/components/Option/Admin/AdminOperationsOverviewPage.tsx`
  - Create only if the implementation changes `/admin` from redirect to a real overview.
- `apps/packages/ui/src/components/Option/Admin/__tests__/AdminOperationsOverviewPage.test.tsx`
  - Required if `AdminOperationsOverviewPage.tsx` is created.

### Modified Files

- `apps/packages/ui/src/routes/option-sources.tsx`
- `apps/packages/ui/src/routes/option-integrations.tsx`
- `apps/packages/ui/src/routes/option-scheduled-tasks.tsx`
- `apps/packages/ui/src/routes/option-watchlists.tsx`
- `apps/packages/ui/src/routes/option-workflow-editor.tsx`
- `apps/packages/ui/src/routes/option-mcp-hub.tsx`
- `apps/packages/ui/src/routes/option-skills.tsx`
- `apps/packages/ui/src/routes/option-admin-server.tsx`
- `apps/packages/ui/src/routes/option-admin-integrations.tsx`
- `apps/packages/ui/src/routes/option-admin-sources.tsx`
- `apps/packages/ui/src/routes/option-admin-monitoring.tsx`
- `apps/tldw-frontend/pages/admin/index.tsx`
- `apps/tldw-frontend/pages/connectors/index.tsx`
- `apps/tldw-frontend/pages/connectors/browse.tsx`
- `apps/tldw-frontend/pages/connectors/jobs.tsx`
- `apps/tldw-frontend/pages/connectors/sources.tsx`
- `apps/packages/ui/src/components/Option/Sources/SourcesWorkspacePage.tsx`
- `apps/packages/ui/src/components/Option/Sources/SourcesAvailabilityGate.tsx`
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
- `apps/packages/ui/src/components/Option/Integrations/IntegrationManagementPage.tsx`
- `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`
- `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
- `apps/packages/ui/src/components/WorkflowEditor/WorkflowEditor.tsx`
- `apps/packages/ui/src/components/Option/Skills/SkillsWorkspace.tsx`
- `apps/packages/ui/src/components/Option/Admin/ServerAdminPage.tsx`

### Existing Tests To Extend

- `apps/packages/ui/src/routes/__tests__/option-sources-route-guards.test.tsx`
- `apps/packages/ui/src/routes/__tests__/integrations-route.test.tsx`
- `apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx`
- `apps/packages/ui/src/routes/__tests__/mcp-hub-route.test.tsx`
- `apps/packages/ui/src/routes/__tests__/option-watchlists.route-state.test.tsx`
- `apps/packages/ui/src/routes/__tests__/chat-workflows-route.test.tsx`
- `apps/packages/ui/src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx`
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`
- `apps/packages/ui/src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx`
- `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx`
- `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx`
- `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.orientation-guidance.test.tsx`
- `apps/packages/ui/src/components/WorkflowEditor/__tests__/WorkflowEditor.responsive.test.tsx`
- `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/sources.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-3-automation/chat-workflows.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-3-automation/workflow-editor.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-server.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts`
- `apps/tldw-frontend/e2e/workflows/route-placeholder-settings.spec.ts`
- `apps/tldw-frontend/e2e/workflows/hosted-placeholder-routes.spec.ts`

### New Tests If Needed

- `apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-overview.spec.ts`
  - Required if `/admin` stops redirecting and renders an overview page.
- `apps/tldw-frontend/e2e/workflows/tier-2-features/integrations.spec.ts`
  - Add if existing route-level tests do not cover personal integration unsupported, unavailable, and empty states.
- `apps/tldw-frontend/e2e/workflows/tier-3-automation/scheduled-tasks.spec.ts`
  - Add if existing tests do not cover scheduled task route states.
- `apps/tldw-frontend/e2e/workflows/tier-3-automation/watchlists.spec.ts`
  - Add if `watchlists-items.spec.ts` and journey tests do not cover health overview and repeat-user controls.

## Route Job Contract

Create a route metadata file that is small enough to test and useful enough to prevent drift:

```ts
export type OperationsRouteConcept =
  | "admin"
  | "mcp"
  | "source"
  | "connector"
  | "integration"
  | "schedule"
  | "watchlist"
  | "workflow"
  | "skill"

export type OperationsCapabilityMode =
  | "frontend_state"
  | "existing_probe"
  | "placeholder"
  | "backend_gate"

export type OperationsRouteJob = {
  route: string
  concept: OperationsRouteConcept
  label: string
  primaryJob: string
  primaryActionLabel: string
  capabilityMode: OperationsCapabilityMode
  diagnosticsPolicy: "disclosed" | "not_applicable"
  implementationOwner: "shared_route" | "next_page"
  relatedRoutes?: string[]
}
```

Initial inventory:

```ts
export const OPERATIONS_ROUTE_JOBS: OperationsRouteJob[] = [
  {
    route: "/admin",
    concept: "admin",
    label: "Admin",
    primaryJob: "Review operations status and choose an admin module",
    primaryActionLabel: "Open server admin",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "next_page",
    relatedRoutes: [
      "/admin/server",
      "/admin/integrations",
      "/admin/sources",
      "/admin/monitoring"
    ]
  },
  {
    route: "/mcp-hub",
    concept: "mcp",
    label: "MCP Hub",
    primaryJob: "Manage external tool servers and governance workflows",
    primaryActionLabel: "Check servers",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route"
  },
  {
    route: "/sources",
    concept: "source",
    label: "Sources",
    primaryJob: "Manage ingestion sources and sync status",
    primaryActionLabel: "New source",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route"
  },
  {
    route: "/connectors",
    concept: "connector",
    label: "Connectors",
    primaryJob: "Understand connector availability and current alternatives",
    primaryActionLabel: "Open settings",
    capabilityMode: "placeholder",
    diagnosticsPolicy: "not_applicable",
    implementationOwner: "next_page",
    relatedRoutes: [
      "/connectors/browse",
      "/connectors/jobs",
      "/connectors/sources"
    ]
  },
  {
    route: "/integrations",
    concept: "integration",
    label: "Personal Integrations",
    primaryJob: "Manage personal Slack and Discord connections",
    primaryActionLabel: "Refresh all",
    capabilityMode: "existing_probe",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route",
    relatedRoutes: ["/admin/integrations"]
  },
  {
    route: "/scheduled-tasks",
    concept: "schedule",
    label: "Scheduled Tasks",
    primaryJob: "Manage reminder tasks and endpoint availability",
    primaryActionLabel: "Create reminder",
    capabilityMode: "existing_probe",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route"
  },
  {
    route: "/watchlists",
    concept: "watchlist",
    label: "Watchlists",
    primaryJob: "Monitor feeds, jobs, runs, articles, reports, and templates",
    primaryActionLabel: "Set up feeds",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route"
  },
  {
    route: "/workflow-editor",
    concept: "workflow",
    label: "Workflow Editor",
    primaryJob: "Build and validate visual workflows",
    primaryActionLabel: "Add node",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route"
  },
  {
    route: "/skills",
    concept: "skill",
    label: "Skills",
    primaryJob: "Manage server-backed skill definitions",
    primaryActionLabel: "Open skills",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route"
  }
]
```

If WP1 selected different canonical labels, use the WP1 labels and update these tests to point at that source of truth.

## Task 1: Add Operations Route Job Tests

**Files:**
- Create: `apps/packages/ui/src/routes/operations-route-jobs.ts`
- Create: `apps/packages/ui/src/routes/__tests__/operations-route-jobs.test.ts`
- Test: `apps/packages/ui/src/routes/__tests__/operations-route-boundaries.test.tsx`

- [ ] **Step 1: Write failing route coverage test**

```ts
import { OPERATIONS_ROUTE_JOBS } from "../operations-route-jobs"

const requiredRoutes = [
  "/admin",
  "/mcp-hub",
  "/sources",
  "/connectors",
  "/integrations",
  "/scheduled-tasks",
  "/watchlists",
  "/workflow-editor",
  "/skills"
]

it("defines every WP10 root route job", () => {
  const routes = new Set(OPERATIONS_ROUTE_JOBS.map((job) => job.route))

  for (const route of requiredRoutes) {
    expect(routes.has(route)).toBe(true)
  }
})
```

- [ ] **Step 2: Write failing backend gate test**

```ts
it("distinguishes frontend state cleanup from backend-gated work", () => {
  const connectors = OPERATIONS_ROUTE_JOBS.find((job) => job.route === "/connectors")
  const scheduledTasks = OPERATIONS_ROUTE_JOBS.find((job) => job.route === "/scheduled-tasks")

  expect(connectors).toMatchObject({
    capabilityMode: "placeholder",
    diagnosticsPolicy: "not_applicable"
  })
  expect(scheduledTasks).toMatchObject({
    capabilityMode: "existing_probe",
    diagnosticsPolicy: "disclosed"
  })
})
```

- [ ] **Step 3: Write failing admin route relation test**

```ts
it("treats admin root as an overview with module drill-down routes", () => {
  const admin = OPERATIONS_ROUTE_JOBS.find((job) => job.route === "/admin")

  expect(admin?.relatedRoutes).toEqual(
    expect.arrayContaining([
      "/admin/server",
      "/admin/integrations",
      "/admin/sources",
      "/admin/monitoring"
    ])
  )
})
```

- [ ] **Step 4: Run the failing tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/operations-route-jobs.test.ts
```

Expected: FAIL because `operations-route-jobs.ts` does not exist.

- [ ] **Step 5: Add the route-job module**

Create `apps/packages/ui/src/routes/operations-route-jobs.ts` with the type and route inventory from this plan.

- [ ] **Step 6: Run route-job tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/operations-route-jobs.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add apps/packages/ui/src/routes/operations-route-jobs.ts apps/packages/ui/src/routes/__tests__/operations-route-jobs.test.ts
git commit -m "test: add operations route job contract"
```

## Task 2: Adopt Capability States For Sources And Scheduled Tasks

**Files:**
- Modify: `apps/packages/ui/src/routes/option-sources.tsx`
- Modify: `apps/packages/ui/src/routes/option-scheduled-tasks.tsx`
- Modify: `apps/packages/ui/src/components/Option/Sources/SourcesWorkspacePage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Sources/SourcesAvailabilityGate.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/option-sources-route-guards.test.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/sources.spec.ts`

- [ ] **Step 1: Write failing sources state tests**

Assert:

- Offline state uses user-language setup or recovery copy.
- Unsupported ingestion sources state does not show raw endpoint text as the primary message.
- Query error state has primary recovery action and optional diagnostics.
- Empty state names the next action, `New source`.
- Admin mode visibly labels the admin scope without changing the user mode workflow.

- [ ] **Step 2: Write failing scheduled task state tests**

Assert:

- Endpoint unavailable state uses a shared capability or recovery state.
- Partial task load state keeps loaded data visible and shows partial diagnostics behind disclosure.
- Query error state has a retry action.
- Empty task state still exposes the create reminder action.
- Watchlist jobs remain clearly owned by Watchlists.

- [ ] **Step 3: Run failing component tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
```

Expected: FAIL on new state requirements.

- [ ] **Step 4: Apply minimal frontend-only state changes**

Use WP2 state primitives where available:

- `StatePanel` for empty and degraded route states.
- `RecoveryCallout` for unavailable, network, and unsupported states.
- `PermissionNotice` for 403 states when the error classification exposes permission denial.
- `DiagnosticRow` for endpoint, status, and raw error details.

Keep current data fetching and API clients.

- [ ] **Step 5: Re-run route and component tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/option-sources-route-guards.test.tsx src/routes/__tests__/scheduled-tasks-route.test.tsx src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Run sources E2E**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/tier-2-features/sources.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add apps/packages/ui/src/routes/option-sources.tsx apps/packages/ui/src/routes/option-scheduled-tasks.tsx apps/packages/ui/src/components/Option/Sources apps/packages/ui/src/components/Option/ScheduledTasks apps/tldw-frontend/e2e/workflows/tier-2-features/sources.spec.ts
git commit -m "fix: clarify source and scheduled task states"
```

## Task 3: Clarify Integrations And Connector Placeholder Routes

**Files:**
- Modify: `apps/packages/ui/src/routes/option-integrations.tsx`
- Modify: `apps/packages/ui/src/routes/option-admin-integrations.tsx`
- Modify: `apps/packages/ui/src/components/Option/Integrations/IntegrationManagementPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Integrations/IntegrationProviderCard.tsx`
- Modify only if needed: `apps/packages/ui/src/components/Option/Integrations/IntegrationPolicyPanel.tsx`
- Modify: `apps/tldw-frontend/pages/connectors/index.tsx`
- Modify: `apps/tldw-frontend/pages/connectors/browse.tsx`
- Modify: `apps/tldw-frontend/pages/connectors/jobs.tsx`
- Modify: `apps/tldw-frontend/pages/connectors/sources.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/integrations-route.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/route-placeholder-settings.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/hosted-placeholder-routes.spec.ts`

- [ ] **Step 1: Write failing integration route tests**

Assert:

- `/integrations` has route error boundary parity with `/admin/integrations`.
- Personal scope labels Slack and Discord as personal connections.
- Workspace scope labels Slack, Discord, and Telegram as workspace policy.
- Unsupported personal integrations show a capability state and not a raw path.
- Provider cards expose connected, disconnected, disabled, and action-pending status.

- [ ] **Step 2: Write failing connector placeholder tests**

Extend placeholder tests so connector routes:

- Identify themselves as connector placeholders.
- Link users to current supported alternatives: Settings, Integrations, Sources, or Scheduled Tasks.
- Do not imply connector catalog, connector jobs, or connector source workflows are already implemented.
- Render a single primary action per placeholder route.

- [ ] **Step 3: Run failing tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/integrations-route.test.tsx src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx
```

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/route-placeholder-settings.spec.ts e2e/workflows/hosted-placeholder-routes.spec.ts --reporter=line
```

Expected: FAIL on new state, boundary, or placeholder assertions.

- [ ] **Step 4: Apply frontend-only route cleanup**

- Add `RouteErrorBoundary` to `option-integrations.tsx` if missing.
- Keep `option-admin-integrations.tsx` boundary intact.
- Keep personal and workspace scopes in `IntegrationManagementPage`.
- Use WP2 capability states for unsupported overview and top-level load errors.
- Keep lower-level policy/provider diagnostics visible behind local panels.
- Update connector placeholders to be honest route-state pages, not product marketing.

- [ ] **Step 5: Re-run tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/integrations-route.test.tsx src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx
```

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/route-placeholder-settings.spec.ts e2e/workflows/hosted-placeholder-routes.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add apps/packages/ui/src/routes/option-integrations.tsx apps/packages/ui/src/routes/option-admin-integrations.tsx apps/packages/ui/src/components/Option/Integrations apps/tldw-frontend/pages/connectors apps/tldw-frontend/e2e/workflows/route-placeholder-settings.spec.ts apps/tldw-frontend/e2e/workflows/hosted-placeholder-routes.spec.ts
git commit -m "fix: clarify integrations and connector placeholders"
```

## Task 4: Add Admin Overview And Module Drill-Down Status

**Files:**
- Modify: `apps/tldw-frontend/pages/admin/index.tsx`
- Create if needed: `apps/packages/ui/src/components/Option/Admin/AdminOperationsOverviewPage.tsx`
- Create if needed: `apps/packages/ui/src/components/Option/Admin/__tests__/AdminOperationsOverviewPage.test.tsx`
- Modify: `apps/packages/ui/src/routes/option-admin-server.tsx`
- Modify: `apps/packages/ui/src/routes/option-admin-sources.tsx`
- Modify: `apps/packages/ui/src/routes/option-admin-monitoring.tsx`
- Modify: `apps/packages/ui/src/components/Option/Admin/ServerAdminPage.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-server.spec.ts`
- Create if needed: `apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-overview.spec.ts`

- [ ] **Step 1: Write failing admin overview test**

If `/admin` remains a redirect by WP1 policy, update the test to assert an explicit redirect rationale and stop here. Otherwise, assert:

- `/admin` renders an operations overview.
- It links to `/admin/server`, `/admin/integrations`, `/admin/sources`, and `/admin/monitoring`.
- It shows available, degraded, unavailable, or not configured status for each module using frontend-derived state.
- It keeps module diagnostics behind disclosure.

- [ ] **Step 2: Write failing admin server state test**

Assert:

- Timeout and permission errors use recovery state copy.
- Users, roles, storage, sessions, and media budget sections remain module detail, not overview clutter.
- The page keeps a refresh action close to the failed state.

- [ ] **Step 3: Run failing admin tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Admin/__tests__/AdminOperationsOverviewPage.test.tsx src/components/Option/Admin/__tests__/ServerAdminPage.design-system.test.tsx
```

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/tier-4-admin/admin-server.spec.ts e2e/workflows/tier-4-admin/admin-overview.spec.ts --reporter=line
```

Expected: FAIL until overview or explicit redirect policy exists.

- [ ] **Step 4: Implement overview or explicit policy**

If creating an overview:

- Replace `apps/tldw-frontend/pages/admin/index.tsx` redirect with a dynamic import of `AdminOperationsOverviewPage`.
- Build overview cards from current frontend state and static module definitions.
- Do not call new backend endpoints.
- Use existing admin module links.

If retaining redirect:

- Update tests and copy so `/admin` is an explicit alias to `/admin/server`.
- Record why the spec acceptance was intentionally narrowed in the implementation task.

- [ ] **Step 5: Re-run admin tests**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/tier-4-admin/admin-server.spec.ts e2e/workflows/tier-4-admin/admin-overview.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add apps/tldw-frontend/pages/admin/index.tsx apps/packages/ui/src/components/Option/Admin apps/packages/ui/src/routes/option-admin-server.tsx apps/packages/ui/src/routes/option-admin-sources.tsx apps/packages/ui/src/routes/option-admin-monitoring.tsx apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-server.spec.ts apps/tldw-frontend/e2e/workflows/tier-4-admin/admin-overview.spec.ts
git commit -m "fix: clarify admin operations entry"
```

## Task 5: Add Status-First MCP Hub, Workflow Editor, And Skills States

**Files:**
- Modify: `apps/packages/ui/src/routes/option-mcp-hub.tsx`
- Modify: `apps/packages/ui/src/routes/option-workflow-editor.tsx`
- Modify: `apps/packages/ui/src/routes/option-skills.tsx`
- Modify: `apps/packages/ui/src/components/Option/MCPHub/McpHubPage.tsx`
- Modify only if needed: `apps/packages/ui/src/components/Option/MCPHub/mcpHubWorkflowConfig.ts`
- Modify: `apps/packages/ui/src/components/WorkflowEditor/WorkflowEditor.tsx`
- Modify: `apps/packages/ui/src/components/Option/Skills/SkillsWorkspace.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/mcp-hub-route.test.tsx`
- Test: `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx`
- Test: `apps/packages/ui/src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx`
- Test: `apps/packages/ui/src/components/WorkflowEditor/__tests__/WorkflowEditor.responsive.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-3-automation/workflow-editor.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts`

- [ ] **Step 1: Write failing MCP Hub status tests**

Assert:

- The page keeps workflow buttons visible.
- A status summary identifies servers, credentials, policy assignments, approvals, and audit state using available data.
- First-time explainer remains dismissible.
- Diagnostics do not replace workflow-first setup.

- [ ] **Step 2: Write failing workflow editor status tests**

Assert:

- Step-type loading and failure states are visible in the editor shell.
- Validation error count is visible and named.
- Save dirty state, import, export, and run state remain visible.
- Mobile panel access remains reachable.

- [ ] **Step 3: Write failing skills capability tests**

Assert:

- Loading, unsupported, no connection, and supported states use route-appropriate capability language.
- Skills manager is not rendered before capability is known.
- Unsupported state includes a recovery action or version/update hint.

- [ ] **Step 4: Run failing tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/mcp-hub-route.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx src/components/WorkflowEditor/__tests__/WorkflowEditor.responsive.test.tsx src/components/Option/Skills/__tests__/Manager.test.tsx
```

Expected: FAIL on new status-first assertions.

- [ ] **Step 5: Apply frontend-only route changes**

- Add route error boundaries if wrappers do not have them and the project pattern supports them.
- Add a compact MCP Hub status summary based on current tab data or current workflow config.
- Keep MCP Hub raw governance details inside tab panels.
- Add visible Workflow Editor step-type state and validation summary without changing graph logic.
- Keep Skills capability gate and improve unsupported state using existing state primitives.

- [ ] **Step 6: Re-run tests and E2E**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/mcp-hub-route.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.ftux.test.tsx src/components/WorkflowEditor/__tests__/WorkflowEditor.responsive.test.tsx src/components/Option/Skills/__tests__/Manager.test.tsx
```

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/tier-2-features/mcp-hub.spec.ts e2e/workflows/tier-3-automation/workflow-editor.spec.ts e2e/workflows/tier-5-specialized/skills.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add apps/packages/ui/src/routes/option-mcp-hub.tsx apps/packages/ui/src/routes/option-workflow-editor.tsx apps/packages/ui/src/routes/option-skills.tsx apps/packages/ui/src/components/Option/MCPHub apps/packages/ui/src/components/WorkflowEditor apps/packages/ui/src/components/Option/Skills apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts apps/tldw-frontend/e2e/workflows/tier-3-automation/workflow-editor.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/skills.spec.ts
git commit -m "fix: add status first operations tooling"
```

## Task 6: Clarify Watchlists Health And Repeat-User Controls

**Files:**
- Modify: `apps/packages/ui/src/routes/option-watchlists.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
- Modify only if needed: `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsHealthBar.tsx`
- Modify only if needed: `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsCommandPalette.tsx`
- Modify only if needed: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/option-watchlists.route-state.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.orientation-guidance.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlists-first-run-copy-contract.test.ts`
- Test: `apps/tldw-frontend/e2e/workflows/journeys/watchlist-ingest-notify.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/watchlists-items.spec.ts`

- [ ] **Step 1: Write failing watchlists route tests**

Assert:

- Health summary is visible before deep tabs.
- Feed, monitor, activity, article, report, and template terms remain consistent with current terminology contracts.
- Repeat-user command palette remains reachable.
- Query parameters for tab, source, job, run, output, item, and smart filters continue to hand off into store state.

- [ ] **Step 2: Write failing watchlists browser assertions**

Assert:

- New user can tell the next action is to set up feeds or monitors.
- Returning user can jump to activity, articles, or reports without restarting the guided path.
- Failed or stalled runs surface next actions next to run state.
- Mobile tabs or progressive sections remain reachable.

- [ ] **Step 3: Run failing tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/option-watchlists.route-state.test.tsx src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.orientation-guidance.test.tsx src/components/Option/Watchlists/__tests__/watchlists-first-run-copy-contract.test.ts
```

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/journeys/watchlist-ingest-notify.spec.ts e2e/workflows/watchlists-items.spec.ts --reporter=line
```

Expected: FAIL on new health or repeat-user assertions.

- [ ] **Step 4: Apply minimal Watchlists changes**

- Keep existing progressive tab layout.
- Keep command palette and keyboard shortcuts.
- Make `WatchlistsHealthBar` the first operational status surface.
- Keep advanced settings and templates discoverable but not first-time blockers.
- Preserve URL query handoff behavior.

- [ ] **Step 5: Re-run tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/option-watchlists.route-state.test.tsx src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.orientation-guidance.test.tsx src/components/Option/Watchlists/__tests__/watchlists-first-run-copy-contract.test.ts
```

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/journeys/watchlist-ingest-notify.spec.ts e2e/workflows/watchlists-items.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

```bash
git add apps/packages/ui/src/routes/option-watchlists.tsx apps/packages/ui/src/components/Option/Watchlists apps/tldw-frontend/e2e/workflows/journeys/watchlist-ingest-notify.spec.ts apps/tldw-frontend/e2e/workflows/watchlists-items.spec.ts
git commit -m "fix: clarify watchlists operations health"
```

## Task 7: Browser QA And Final Verification

**Files:**
- Verify only unless browser QA exposes a defect in scoped files.

- [ ] **Step 1: Run route contract tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/operations-route-jobs.test.ts src/routes/__tests__/operations-route-boundaries.test.tsx src/routes/__tests__/option-sources-route-guards.test.tsx src/routes/__tests__/integrations-route.test.tsx src/routes/__tests__/scheduled-tasks-route.test.tsx src/routes/__tests__/mcp-hub-route.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run component state tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx src/components/WorkflowEditor/__tests__/WorkflowEditor.responsive.test.tsx src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.orientation-guidance.test.tsx
```

Expected: PASS.

- [ ] **Step 3: Run parent-required E2E command**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/tier-2-features/sources.spec.ts e2e/workflows/tier-2-features/mcp-hub.spec.ts e2e/workflows/tier-3-automation/chat-workflows.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 4: Run expanded WP10 E2E**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/tier-3-automation/workflow-editor.spec.ts e2e/workflows/tier-4-admin/admin-server.spec.ts e2e/workflows/tier-5-specialized/skills.spec.ts e2e/workflows/route-placeholder-settings.spec.ts e2e/workflows/hosted-placeholder-routes.spec.ts e2e/workflows/journeys/watchlist-ingest-notify.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 5: Browser QA first-time path**

Use a local dev server and inspect:

- `/admin`: operations overview or explicit redirect policy.
- `/admin/server`: status, permission, timeout, users, roles, media budget.
- `/mcp-hub`: workflow setup and status summary.
- `/sources`: unsupported, empty, loading, error, and list states.
- `/connectors`: placeholder state and supported alternatives.
- `/integrations`: personal provider cards and unsupported state.
- `/scheduled-tasks`: unsupported, partial, empty, and task list states.
- `/watchlists`: health bar, guided path, feed and monitor setup.
- `/workflow-editor`: step-type state, validation, save, import, export, run.
- `/skills`: loading, unsupported, and supported states.

Expected: Each route explains availability and next action without raw endpoint details as the primary UI.

- [ ] **Step 6: Browser QA power-user path**

At desktop and mobile widths, inspect:

- Diagnostics are available behind disclosure.
- Refresh, retry, create, edit, save, import, export, and command palette actions remain reachable.
- Operators can recover from partial data without leaving the page.
- Advanced controls do not hide the main status summary.
- No primary controls overlap on mobile.

Expected: Returning users can complete repeated operations quickly.

- [ ] **Step 7: Final commit**

```bash
git status --short
git add apps/packages/ui/src/routes apps/packages/ui/src/components/Option/Sources apps/packages/ui/src/components/Option/ScheduledTasks apps/packages/ui/src/components/Option/Integrations apps/packages/ui/src/components/Option/MCPHub apps/packages/ui/src/components/Option/Watchlists apps/packages/ui/src/components/WorkflowEditor apps/packages/ui/src/components/Option/Skills apps/packages/ui/src/components/Option/Admin apps/tldw-frontend/pages/admin/index.tsx apps/tldw-frontend/pages/connectors apps/tldw-frontend/e2e
git commit -m "fix: clarify operations and integrations routes"
```

Expected: Commit contains only WP10 scoped files.

## Acceptance Criteria

- `/sources` and `/scheduled-tasks` do not show raw endpoint errors as the main UI.
- `/admin` either renders an operations overview before module drill-down or has an explicit tested redirect policy.
- `/mcp-hub` keeps workflow-first setup and adds a clearer status summary.
- Watchlists expose monitor/feed health and repeat-user controls.
- `/connectors` remains an honest placeholder family unless backend support exists.
- `/integrations` distinguishes personal connection management from workspace policy.
- `/workflow-editor` exposes step-type, validation, dirty, save, import, export, and run state.
- `/skills` uses capability-aware loading, unsupported, and supported states.
- Operator diagnostics remain available behind disclosure.
- No backend API changes are introduced without a separate backend capability-map task.

## Verification Commands

Run route and component tests:

```bash
cd apps/packages/ui
bunx vitest run src/routes/__tests__/operations-route-jobs.test.ts src/routes/__tests__/operations-route-boundaries.test.tsx src/routes/__tests__/option-sources-route-guards.test.tsx src/routes/__tests__/integrations-route.test.tsx src/routes/__tests__/scheduled-tasks-route.test.tsx src/routes/__tests__/mcp-hub-route.test.tsx
```

Run parent-required E2E:

```bash
cd apps/tldw-frontend
bunx playwright test e2e/workflows/tier-2-features/sources.spec.ts e2e/workflows/tier-2-features/mcp-hub.spec.ts e2e/workflows/tier-3-automation/chat-workflows.spec.ts --reporter=line
```

Run expanded WP10 E2E:

```bash
cd apps/tldw-frontend
bunx playwright test e2e/workflows/tier-3-automation/workflow-editor.spec.ts e2e/workflows/tier-4-admin/admin-server.spec.ts e2e/workflows/tier-5-specialized/skills.spec.ts e2e/workflows/route-placeholder-settings.spec.ts e2e/workflows/hosted-placeholder-routes.spec.ts e2e/workflows/journeys/watchlist-ingest-notify.spec.ts --reporter=line
```

## Rollback Plan

- Revert route-job metadata first if it conflicts with WP1 route metadata.
- Revert admin overview separately from route state cleanup.
- Revert connector placeholder copy separately from integrations changes.
- Revert each route-family slice independently: sources and scheduled tasks, integrations and connectors, admin, MCP and workflow and skills, watchlists.
- Do not roll back WP2 shared state primitives while isolating WP10 route defects.

## Handoff Notes

- Start with route-job tests so the implementation cannot drift into unrelated admin or automation modules.
- Use WP2 shared state primitives before adding any new UI state component.
- Treat backend capability-map work as gated, not assumed.
- Keep operator diagnostics reachable but not first in the visual hierarchy.
- Preserve existing route paths, stores, query keys, and API clients.
- Capture browser evidence for unavailable, degraded, empty, and supported states before final review.
