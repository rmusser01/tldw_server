# Scheduled Tasks Automation Workbench Phase 2A Create Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the frontend-only Phase 2A `/scheduled-tasks` Create framework: URL-addressable Overview/Tasks/Create tabs, deterministic intent template matching, Reminder creation as the only fully available template, and honest handoff/planned-state panels for non-reminder automation intents.

**Architecture:** Keep Phase 2A in the existing ScheduledTasks feature folder and reuse the current scheduled-tasks control-plane API. Add pure helper modules for URL state and template capability state, then wrap the existing Phase 1 overview/table/detail/reminder editor with a tabbed workbench shell and a Create panel. Do not add backend contracts, Watchlists adapters, RAG schedule primitives, ACP integration, run-now, dry-run, bulk actions, or Home result surfacing in this slice.

**Tech Stack:** React, TypeScript, Ant Design, React Router search params, TanStack Query, Vitest, Testing Library, existing `scheduled-tasks-control-plane` service, existing Watchlists deep links.

---

## Source Spec

- `Docs/superpowers/specs/2026-06-08-scheduled-tasks-automation-workbench-phase2-creation-design.md`
- Backlog planning task: `TASK-2321`
- Prior implementation task: `TASK-498`
- Phase 1 base commit: PR #2217 merged into `origin/dev`

## Scope

In scope:

- `/scheduled-tasks` tab state: Overview, Tasks, Create.
- URL behavior for `tab`, `template`, and `task_id`.
- Static Phase 2A template registry with capability states.
- Deterministic "Find a template" matcher.
- Create tab with template cards, filters, matched suggestions, selected template panel.
- Reminder template path using existing reminder API and schedule controls.
- Handoff-only panels for Watch for new items, Ingest new content, and Advanced task.
- Planned panels for Recurring question and Agent task.
- Conservative reminder success copy and created-task detail navigation.
- URL privacy safeguards for visible handoff summaries.
- Extension-sized responsive behavior covered by component tests where feasible.

Out of scope:

- Backend changes or new capability APIs.
- New Watchlists create/prefill adapters.
- RAG recurring query scheduling.
- ACP/Agent task scheduling.
- Home automation inbox.
- New cross-task Runs or Results tabs.
- Run now, dry run, duplicate, saved views, bulk actions, export.

## Pre-Implementation Requirement

Before Task 1 product-code edits, create a new implementation Backlog.md task such as `Implement Scheduled Tasks Automation Workbench Phase 2A create framework`.

Link that implementation task to:

- `TASK-2321`
- `TASK-2320`
- `Docs/superpowers/plans/2026-06-08-scheduled-tasks-automation-workbench-phase2a-create-framework-implementation-plan.md`
- `Docs/superpowers/specs/2026-06-08-scheduled-tasks-automation-workbench-phase2-creation-design.md`

Record modified files, verification commands, known skips, and final summary on the implementation task. `TASK-2321` tracks this implementation plan only.

## File Structure

### New Files

- `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-route-state.ts`
  - Pure helpers for parsing and serializing tab/template/task URL state.
- `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-templates.ts`
  - Static Phase 2A template registry, filter model, deterministic matcher, handoff summary sanitization helpers.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx`
  - Create tab UI: finder, filters, template cards, selected template content, Reminder editor handoff, planned/handoff panels.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts`
  - Unit coverage for valid and invalid URL state.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts`
  - Unit coverage for template registry, filters, matcher, and privacy-safe handoff summaries.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx`
  - Component coverage for Create tab cards, matcher, handoff panels, planned states, and reminder editor rendering.

### Modified Files

- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
  - Add tab shell, URL state wiring, task-detail deep link handling, invalid route messages, create panel integration, reminder success navigation.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskTable.tsx`
  - Keep existing table behavior; ensure the create action opens the Create tab instead of directly opening the old reminder editor when called from the table empty/action area.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ReminderTaskEditor.tsx`
  - Reuse for create/edit. Add optional `submitLabel` and `heading` props only if needed by `ScheduledTaskCreatePanel`; avoid duplicating reminder form logic.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`
  - Update existing expectations for tabs, Create entry, reminder success detail navigation, invalid tab/template/task states, and no fake Watch/Ingest creation.
- `apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx`
  - Keep route registration coverage; add a static assertion only if route shell changes.
- `apps/extension/tests/e2e/integrations-and-scheduled-tasks.spec.ts`
  - Update smoke copy if the Phase 2A page header/description changes.
- Implementation Backlog task file.

### Files To Avoid Changing

- `apps/packages/ui/src/components/Option/Watchlists/**`
  - Do not change Watchlists setup, sources, jobs, runs, outputs, reports, or templates for Phase 2A.
- `tldw_Server_API/**`
  - No backend changes in Phase 2A.
- Home, RAG, ACP, Jobs, Scheduler, Notifications.
  - Link/handoff only if existing stable routes are already known.

## Existing Contracts To Preserve

Scheduled task rows currently use:

```ts
interface ScheduledTask {
  id: string
  primitive: "reminder_task" | "watchlist_job"
  title: string
  description?: string | null
  status: string
  enabled: boolean
  schedule_summary?: string | null
  timezone?: string | null
  next_run_at?: string | null
  last_run_at?: string | null
  edit_mode: "native" | "external"
  manage_url?: string | null
  source_ref: Record<string, unknown>
}
```

Do not add new primitives or assume new fields in Phase 2A.

Existing reminder creation returns a `ScheduledTask`:

```ts
const createdTask = await createScheduledTaskReminder(payload)
```

Use `createdTask.id` for `?tab=tasks&task_id=<id>` navigation. Keep the returned `createdTask` as a temporary selected-task fallback until the refreshed list contains the same ID, so success detail navigation does not depend on list-refresh timing.

Existing Watchlists deep links are already built by `buildWatchlistTaskLinks`; do not duplicate that logic in the Create panel.

## Task 1: Add URL State Helpers

**Files:**

- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-route-state.ts`
- Test: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts`

- [ ] **Step 1: Write failing tests for tab parsing**

Add tests:

```ts
import { describe, expect, it } from "vitest"
import {
  SCHEDULED_TASK_TABS,
  buildScheduledTaskSearch,
  parseScheduledTaskRouteState
} from "../scheduled-task-route-state"

describe("scheduled task route state", () => {
  it("defaults to overview when no tab is provided", () => {
    expect(parseScheduledTaskRouteState(new URLSearchParams())).toMatchObject({
      tab: "overview",
      invalidTab: null
    })
  })

  it("accepts tasks and create tabs", () => {
    expect(parseScheduledTaskRouteState(new URLSearchParams("tab=tasks")).tab).toBe("tasks")
    expect(parseScheduledTaskRouteState(new URLSearchParams("tab=create")).tab).toBe("create")
  })

  it("falls back to overview for invalid tabs", () => {
    expect(parseScheduledTaskRouteState(new URLSearchParams("tab=runs"))).toMatchObject({
      tab: "overview",
      invalidTab: "runs"
    })
  })

  it("keeps valid template and task ids", () => {
    expect(
      parseScheduledTaskRouteState(new URLSearchParams("tab=create&template=watch"))
    ).toMatchObject({ tab: "create", templateId: "watch" })
    expect(
      parseScheduledTaskRouteState(new URLSearchParams("tab=tasks&task_id=reminder_task%3A2"))
    ).toMatchObject({ tab: "tasks", taskId: "reminder_task:2" })
  })

  it("builds search strings without dropping existing valid state", () => {
    expect(buildScheduledTaskSearch({ tab: "tasks", taskId: "reminder_task:2" })).toBe(
      "?tab=tasks&task_id=reminder_task%3A2"
    )
  })

  it("exposes exactly the Phase 2A tabs", () => {
    expect(SCHEDULED_TASK_TABS.map((tab) => tab.id)).toEqual([
      "overview",
      "tasks",
      "create"
    ])
  })
})
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: fail because `scheduled-task-route-state.ts` does not exist.

- [ ] **Step 3: Implement the route helper**

Create:

```ts
export type ScheduledTaskTabId = "overview" | "tasks" | "create"

export interface ScheduledTaskTabDefinition {
  id: ScheduledTaskTabId
  label: string
}

export interface ScheduledTaskRouteState {
  tab: ScheduledTaskTabId
  invalidTab: string | null
  templateId: string | null
  taskId: string | null
}

export const SCHEDULED_TASK_TABS: readonly ScheduledTaskTabDefinition[] = [
  { id: "overview", label: "Overview" },
  { id: "tasks", label: "Tasks" },
  { id: "create", label: "Create" }
] as const

const SCHEDULED_TASK_TAB_IDS = new Set<ScheduledTaskTabId>(
  SCHEDULED_TASK_TABS.map((tab) => tab.id)
)

const normalizeNullableParam = (value: string | null): string | null => {
  const trimmed = String(value ?? "").trim()
  return trimmed || null
}

export const parseScheduledTaskRouteState = (
  params: URLSearchParams
): ScheduledTaskRouteState => {
  const rawTab = normalizeNullableParam(params.get("tab"))
  const tab =
    rawTab && SCHEDULED_TASK_TAB_IDS.has(rawTab as ScheduledTaskTabId)
      ? (rawTab as ScheduledTaskTabId)
      : "overview"

  return {
    tab,
    invalidTab: rawTab && tab === "overview" && rawTab !== "overview" ? rawTab : null,
    templateId: normalizeNullableParam(params.get("template")),
    taskId: normalizeNullableParam(params.get("task_id"))
  }
}

export const buildScheduledTaskSearch = ({
  tab,
  templateId,
  taskId
}: {
  tab: ScheduledTaskTabId
  templateId?: string | null
  taskId?: string | null
}): string => {
  const params = new URLSearchParams()
  if (tab !== "overview") params.set("tab", tab)
  if (templateId && tab === "create") params.set("template", templateId)
  if (taskId && tab === "tasks") params.set("task_id", taskId)
  const serialized = params.toString()
  return serialized ? `?${serialized}` : ""
}
```

- [ ] **Step 4: Run route helper tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-route-state.ts \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts \
  backlog/tasks/<implementation-task-file>.md
git commit -m "feat: add scheduled task route state helpers"
```

## Task 2: Add Template Registry And Matcher

**Files:**

- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-templates.ts`
- Test: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts`

- [ ] **Step 1: Write failing tests for template capability states**

Add tests:

```ts
import { describe, expect, it } from "vitest"
import {
  SCHEDULED_TASK_TEMPLATE_FILTERS,
  SCHEDULED_TASK_TEMPLATES,
  filterScheduledTaskTemplates,
  findScheduledTaskTemplates,
  getScheduledTaskTemplate,
  toSafeHandoffSourceText
} from "../scheduled-task-templates"

describe("scheduled task templates", () => {
  it("keeps Reminder as the only available Phase 2A creation template", () => {
    expect(SCHEDULED_TASK_TEMPLATES.filter((template) => template.state === "available").map((template) => template.id)).toEqual(["reminder"])
  })

  it("marks Watch, Ingest, and Advanced as handoff-only", () => {
    expect(getScheduledTaskTemplate("watch")?.state).toBe("handoff_only")
    expect(getScheduledTaskTemplate("ingest")?.state).toBe("handoff_only")
    expect(getScheduledTaskTemplate("advanced")?.state).toBe("handoff_only")
  })

  it("marks Recurring Question and Agent Task as planned", () => {
    expect(getScheduledTaskTemplate("recurring_question")?.state).toBe("planned")
    expect(getScheduledTaskTemplate("agent_task")?.state).toBe("planned")
  })

  it("matches prompt text deterministically without inferring config", () => {
    expect(findScheduledTaskTemplates("keep this channel searchable").map((template) => template.id)).toContain("ingest")
    expect(findScheduledTaskTemplates("send this prompt to an agent tomorrow").map((template) => template.id)).toContain("agent_task")
    expect(findScheduledTaskTemplates("watch new issues").map((template) => template.id)).toContain("watch")
  })

  it("filters templates by availability and category", () => {
    expect(filterScheduledTaskTemplates("available_now").map((template) => template.id)).toEqual(["reminder"])
    expect(filterScheduledTaskTemplates("agent").map((template) => template.id)).toEqual(["agent_task"])
  })

  it("sanitizes unsafe handoff source text", () => {
    expect(toSafeHandoffSourceText("https://example.com/feed?token=secret")).toBe(null)
    expect(toSafeHandoffSourceText("https://example.com/feed")).toBe("https://example.com/feed")
    expect(toSafeHandoffSourceText("  ")).toBe(null)
  })

  it("exposes the expected filter list", () => {
    expect(SCHEDULED_TASK_TEMPLATE_FILTERS.map((filter) => filter.id)).toEqual([
      "all",
      "available_now",
      "watch",
      "ingest",
      "research",
      "agent",
      "advanced"
    ])
  })
})
```

- [ ] **Step 2: Run failing tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: fail because template helper does not exist.

- [ ] **Step 3: Implement the template model**

Create a static model with no backend capability fetch:

```ts
export type ScheduledTaskTemplateId =
  | "reminder"
  | "watch"
  | "ingest"
  | "recurring_question"
  | "agent_task"
  | "advanced"

export type ScheduledTaskTemplateState =
  | "available"
  | "handoff_only"
  | "needs_setup"
  | "managed_in_watchlists"
  | "planned"
  | "unavailable"

export type ScheduledTaskTemplateCategory =
  | "reminder"
  | "watch"
  | "ingest"
  | "research"
  | "agent"
  | "advanced"

export interface ScheduledTaskTemplate {
  id: ScheduledTaskTemplateId
  category: ScheduledTaskTemplateCategory
  title: string
  intent: string
  description: string
  state: ScheduledTaskTemplateState
  primaryActionLabel: string
  secondaryActionLabel?: string
  examples?: string[]
  keywords: string[]
}

export const SCHEDULED_TASK_TEMPLATES: readonly ScheduledTaskTemplate[] = [
  {
    id: "reminder",
    category: "reminder",
    title: "Reminder",
    intent: "Remind me later or repeatedly",
    description: "Schedule a one-time or recurring reminder.",
    state: "available",
    primaryActionLabel: "Create reminder",
    secondaryActionLabel: "Create another",
    keywords: ["remind", "reminder", "later", "daily", "weekly", "monthly"]
  },
  {
    id: "watch",
    category: "watch",
    title: "Watch for new items",
    intent: "Tell me when something new appears",
    description: "Get notified when a source has new matching items.",
    state: "handoff_only",
    primaryActionLabel: "Continue in Watchlists",
    secondaryActionLabel: "Copy setup summary",
    examples: ["repository issues", "RSS feeds", "forums", "vendor advisories"],
    keywords: ["watch", "monitor", "new", "changes", "alert", "notify"]
  }
  // Add ingest, recurring_question, agent_task, advanced using spec copy.
]
```

Implement:

- `getScheduledTaskTemplate(id)`
- `filterScheduledTaskTemplates(filterId)`
- `findScheduledTaskTemplates(query)`
- `getScheduledTaskTemplateStateLabel(state)`
- `toSafeHandoffSourceText(value)`

Privacy rule for `toSafeHandoffSourceText`:

```ts
const SENSITIVE_URL_PARAM_PATTERN = /(^|[?&#])(token|api[_-]?key|key|secret|session|sid|auth|code|invite)=/i
```

Return `null` when the string includes the pattern, contains a URL fragment, or is not visible text after trimming.

- [ ] **Step 4: Run template tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-templates.ts \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts \
  backlog/tasks/<implementation-task-file>.md
git commit -m "feat: add scheduled task template registry"
```

## Task 3: Build The Create Panel

**Files:**

- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx`
- Test: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx`
- Modify only if needed: `apps/packages/ui/src/components/Option/ScheduledTasks/ReminderTaskEditor.tsx`

- [ ] **Step 1: Write failing component tests for template cards and matcher**

Use mocked callbacks and render `ScheduledTaskCreatePanel` directly.

Test cases:

```ts
it("renders intent templates by state without source-vendor IA", () => {
  render(<ScheduledTaskCreatePanel selectedTemplateId={null} onSelectTemplate={vi.fn()} ... />)

  expect(screen.getByRole("heading", { name: "Choose what you want to automate" })).toBeInTheDocument()
  expect(screen.getByRole("button", { name: /Create reminder/i })).toBeInTheDocument()
  expect(screen.getByText("Handoff only")).toBeInTheDocument()
  expect(screen.getByText("Planned capability")).toBeInTheDocument()
  expect(screen.queryByRole("heading", { name: /GitHub monitor/i })).not.toBeInTheDocument()
  expect(screen.queryByRole("heading", { name: /YouTube ingest/i })).not.toBeInTheDocument()
})

it("suggests templates from deterministic finder text", async () => {
  const user = userEvent.setup()
  render(<ScheduledTaskCreatePanel selectedTemplateId={null} onSelectTemplate={vi.fn()} ... />)

  await user.type(screen.getByRole("textbox", { name: "Find a template" }), "watch new advisories")

  expect(screen.getByText("Best match: Watch for new items")).toBeInTheDocument()
})

it("shows handoff panel copy without creation language", async () => {
  render(<ScheduledTaskCreatePanel selectedTemplateId="watch" onSelectTemplate={vi.fn()} ... />)

  expect(screen.getByText("Setup continues in Watchlists.")).toBeInTheDocument()
  expect(screen.getByText("No scheduled task has been created yet.")).toBeInTheDocument()
  expect(screen.getByRole("link", { name: "Open Watchlists setup" })).toHaveAttribute("href", "/watchlists")
})
```

- [ ] **Step 2: Run failing create-panel tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: fail because component does not exist.

- [ ] **Step 3: Implement `ScheduledTaskCreatePanel`**

Props:

```ts
export interface ScheduledTaskCreatePanelProps {
  selectedTemplateId: ScheduledTaskTemplateId | null
  onSelectTemplate: (templateId: ScheduledTaskTemplateId | null) => void
  onCreateReminder: (payload: CreateScheduledTaskReminderPayload) => Promise<void> | void
  savingReminder?: boolean
}
```

Implementation notes:

- Use `Input.Search` or `Input` with `aria-label="Find a template"`.
- Use `Segmented` or `Select` for filters with accessible labels.
- Use compact Ant Design `Card` or flat panels; no nested cards.
- Cards must show title, intent, state label, description, primary CTA.
- Reminder selected state renders `ReminderTaskEditor` with `task={null}` and `open`.
- Handoff-only selected state renders a panel with:
  - template title and intent;
  - owner copy;
  - "Setup continues in Watchlists." for Watch/Ingest;
  - "No scheduled task has been created yet.";
  - optional copyable setup summary;
  - `Open Watchlists setup` link for Watch/Ingest;
  - `Choose destination` domain links for Advanced.
- Planned selected state renders:
  - `Planned capability`;
  - no create CTA;
  - optional domain link only if stable.
- Use polite live region for match result:

```tsx
<div role="status" aria-live="polite">
  {bestMatch ? `Best match: ${bestMatch.title}` : null}
</div>
```

- [ ] **Step 4: Add privacy and planned-state tests**

Test:

```ts
it("does not include sensitive URL text in handoff summary", async () => {
  const user = userEvent.setup()
  render(<ScheduledTaskCreatePanel selectedTemplateId="watch" ... />)

  await user.type(screen.getByRole("textbox", { name: "Optional source or setup note" }), "https://example.com/feed?token=secret")

  expect(screen.getByText(/This source contains private-looking values/)).toBeInTheDocument()
  expect(screen.queryByText("https://example.com/feed?token=secret")).not.toBeInTheDocument()
})

it("renders planned Recurring question without create controls", () => {
  render(<ScheduledTaskCreatePanel selectedTemplateId="recurring_question" ... />)

  expect(screen.getByText("Planned capability")).toBeInTheDocument()
  expect(screen.queryByRole("button", { name: /Create/i })).not.toBeInTheDocument()
})
```

- [ ] **Step 5: Run create-panel tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/ReminderTaskEditor.tsx \
  backlog/tasks/<implementation-task-file>.md
git commit -m "feat: add scheduled task create panel"
```

## Task 4: Integrate Tabs, URL State, And Task Detail Deep Links

**Files:**

- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
- Test: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`

- [ ] **Step 1: Update test render helper for initial URL entries**

Change the helper to:

```ts
const renderWithQueryClient = (
  ui: React.ReactElement,
  initialEntry = "/scheduled-tasks"
) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })

  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>
    </MemoryRouter>
  )
}
```

- [ ] **Step 2: Write failing tests for tab URLs**

Add tests:

```ts
it("opens the Create tab from the tab URL", async () => {
  mocks.listScheduledTasks.mockResolvedValue({ items: [], total: 0, partial: false, errors: [] })

  renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=create")

  expect(await screen.findByRole("tab", { name: "Create", selected: true })).toBeInTheDocument()
  expect(screen.getByRole("heading", { name: "Choose what you want to automate" })).toBeInTheDocument()
})

it("opens task detail from task_id URL after task data loads", async () => {
  mocks.listScheduledTasks.mockResolvedValue({
    items: [{ id: "reminder_task:1", primitive: "reminder_task", title: "Review notes", status: "scheduled", enabled: true, edit_mode: "native", source_ref: { task_id: "1" } }],
    total: 1,
    partial: false,
    errors: []
  })

  renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks&task_id=reminder_task%3A1")

  expect(await screen.findByRole("dialog", { name: /Review notes/i })).toBeInTheDocument()
})

it("shows non-blocking invalid tab and invalid task messages", async () => {
  mocks.listScheduledTasks.mockResolvedValue({ items: [], total: 0, partial: false, errors: [] })

  renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=runs&task_id=missing")

  expect(await screen.findByText("That tab is not available. Showing Overview.")).toBeInTheDocument()
})
```

- [ ] **Step 3: Run failing page tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: fail because tabs and route state are not integrated.

- [ ] **Step 4: Integrate route state in `ScheduledTasksPage`**

Implementation notes:

- Import `Tabs` and `Alert`/`Typography` as needed.
- Use `useSearchParams` from `react-router-dom`.
- Parse route state with `parseScheduledTaskRouteState(searchParams)`.
- Use AntD `Tabs` with `items` for Overview, Tasks, Create.
- On tab change, call `setSearchParams` with `buildScheduledTaskSearch({ tab })`.
- Overview tab contains existing overview cards plus partial/error/watchlists preservation messages.
- Tasks tab contains existing table, empty state, and detail drawer.
- Create tab renders `ScheduledTaskCreatePanel`.
- For `task_id`:
  - after `hasLoadedTasks`, if matching task exists, set `selectedTaskId`;
  - if not, show non-blocking "Task not found." message and leave Tasks tab visible.
- Invalid tab message should be non-blocking and not prevent task loading.

Pseudo-code:

```tsx
const [searchParams, setSearchParams] = useSearchParams()
const routeState = React.useMemo(
  () => parseScheduledTaskRouteState(searchParams),
  [searchParams]
)

const updateRoute = (next: { tab: ScheduledTaskTabId; templateId?: string | null; taskId?: string | null }) => {
  setSearchParams(buildScheduledTaskSearch(next))
}
```

When a row is inspected:

```ts
const openTaskDetail = (task: ScheduledTask) => {
  setSelectedTaskId(task.id)
  setSearchParams(buildScheduledTaskSearch({ tab: "tasks", taskId: task.id }))
}
```

- [ ] **Step 5: Keep existing Phase 1 tests passing**

Existing tests that call `Create scheduled task` should be updated:

- If the button now opens Create tab, click Reminder's `Create reminder` CTA before filling fields.
- Existing table/detail/Watchlists link assertions should still pass inside Tasks tab.

- [ ] **Step 6: Run page tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: pass.

- [ ] **Step 7: Commit Task 4**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx \
  backlog/tasks/<implementation-task-file>.md
git commit -m "feat: add scheduled task workbench tabs"
```

## Task 5: Wire Reminder Creation Success To Task Detail

**Files:**

- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
- Test: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`

- [ ] **Step 1: Write failing success-navigation test**

Add or update reminder creation test:

```ts
it("opens the created reminder detail after successful creation", async () => {
  const user = userEvent.setup()

  mocks.listScheduledTasks
    .mockResolvedValueOnce({ items: [], total: 0, partial: false, errors: [] })
    .mockResolvedValueOnce({
      items: [{
        id: "reminder_task:2",
        primitive: "reminder_task",
        title: "Daily review",
        status: "scheduled",
        enabled: true,
        edit_mode: "native",
        source_ref: { task_id: "2" }
      }],
      total: 1,
      partial: false,
      errors: []
    })

  mocks.createScheduledTaskReminder.mockResolvedValue({
    id: "reminder_task:2",
    primitive: "reminder_task",
    title: "Daily review",
    status: "scheduled",
    enabled: true,
    edit_mode: "native",
    source_ref: { task_id: "2" }
  })

  renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=create&template=reminder")

  await user.type(await screen.findByRole("textbox", { name: "Title" }), "Daily review")
  fireEvent.change(screen.getByLabelText("Run once at"), { target: { value: "2026-03-21T10:00" } })
  await user.click(screen.getByRole("button", { name: "Save reminder" }))

  expect(await screen.findByRole("dialog", { name: /Daily review/i })).toBeInTheDocument()
})
```

- [ ] **Step 2: Run failing test**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: fail until success path uses returned task ID and route update.

- [ ] **Step 3: Implement success behavior**

In `handleSubmit`:

- Store the returned task from `createScheduledTaskReminder`.
- On create:
  - close editor/create wizard state;
  - store the returned task as a temporary selected-task fallback;
  - navigate to `?tab=tasks&task_id=<created.id>`;
  - set `selectedTaskId(created.id)`;
  - refetch tasks;
  - show message: `Reminder scheduled. Status appears in Tasks.`
- In selected-task computation, prefer the task from the refreshed list and fall back to the temporary created task only when its ID matches `selectedTaskId`.
- Clear the temporary created-task fallback when:
  - the refreshed list contains the created ID;
  - the selected task changes to another ID;
  - the detail drawer closes.
- On edit:
  - preserve current detail/edit behavior.

Do not use "Results and status" copy for reminders.

- [ ] **Step 4: Run focused page tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx \
  backlog/tasks/<implementation-task-file>.md
git commit -m "feat: open created reminder detail"
```

## Task 6: Extension And Route Parity Checks

**Files:**

- Modify if needed: `apps/extension/tests/e2e/integrations-and-scheduled-tasks.spec.ts`
- Modify if needed: `apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx`
- Test existing route shells:
  - `apps/packages/ui/src/routes/option-scheduled-tasks.tsx`
  - `apps/tldw-frontend/extension/routes/option-scheduled-tasks.tsx`

- [ ] **Step 1: Update route/extension tests for new Create copy if needed**

If the Scheduled Tasks description changes, update:

```ts
await expect(page.getByText(/Choose what you want to automate|Track reminders/i)).toBeVisible()
```

Do not require live backend data in extension E2E.

- [ ] **Step 2: Add route test assertion for shared page component if the route shell changes**

If route files remain unchanged, no modification is required.

- [ ] **Step 3: Run route tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/routes/__tests__/scheduled-tasks-route.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: pass.

- [ ] **Step 4: Run focused ScheduledTasks test group**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts \
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts \
  src/components/Option/ScheduledTasks/__tests__/reminder-schedule-utils.test.ts \
  src/routes/__tests__/scheduled-tasks-route.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add apps/extension/tests/e2e/integrations-and-scheduled-tasks.spec.ts \
  apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx \
  backlog/tasks/<implementation-task-file>.md
git commit -m "test: update scheduled tasks route parity"
```

If neither file changes, skip this commit and record the skip on the implementation Backlog task.

## Task 7: Final Verification And Closeout

**Files:**

- Implementation Backlog task file.
- No product files unless verification reveals issues.

- [ ] **Step 1: Run full focused frontend verification**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts \
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts \
  src/components/Option/ScheduledTasks/__tests__/reminder-schedule-utils.test.ts \
  src/routes/__tests__/scheduled-tasks-route.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: all tests pass.

- [ ] **Step 2: Run extension E2E only if prerequisites are available**

Run if the extension build/server prerequisites are already available:

```bash
TLDW_E2E_SERVER_URL=127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY \
  bunx playwright test apps/extension/tests/e2e/integrations-and-scheduled-tasks.spec.ts --reporter=line
```

If prerequisites are not available, document the skip. Existing test helper may skip when the built extension is missing.

- [ ] **Step 3: Run repository whitespace check**

Run:

```bash
git diff --check
```

Expected: no output, exit 0.

- [ ] **Step 4: Run Bandit only if backend Python changed**

Phase 2A should not change backend Python. If no backend Python changed, record:

```text
Bandit skipped because Phase 2A touched frontend TypeScript/tests and Backlog markdown only.
```

If backend Python changed unexpectedly, stop and reassess scope before continuing. Do not silently ship backend changes in Phase 2A.

- [ ] **Step 5: Manual browser verification if a dev server is already available**

If a WebUI dev server is already running or can be started safely, verify:

- `/scheduled-tasks`
- `/scheduled-tasks?tab=tasks`
- `/scheduled-tasks?tab=create`
- `/scheduled-tasks?tab=create&template=watch`
- `/scheduled-tasks?tab=create&template=recurring_question`
- `/scheduled-tasks?tab=tasks&task_id=missing`

Expected:

- No console errors.
- Overview/Tasks/Create tabs are keyboard reachable.
- Create cards stack at extension-like width.
- Watch/Ingest/Advanced do not claim a task was created.
- Reminder create flow opens task detail when the mocked or real API returns a created row.

- [ ] **Step 6: Update implementation Backlog task**

Record:

- completed acceptance criteria;
- modified files;
- exact verification commands and results;
- extension/browser skips, if any;
- Bandit skip rationale;
- final summary.

- [ ] **Step 7: Final commit**

Commit any final Backlog or test-stabilization changes:

```bash
git add backlog/tasks/<implementation-task-file>.md
git commit -m "chore: close scheduled tasks phase 2a task"
```

Skip this commit if there are no changes.

## Review Checklist Before PR

- [ ] Reminder remains the only fully createable Phase 2A template.
- [ ] Watch and Ingest are intent-based, not GitHub/YouTube-specific.
- [ ] Watchlists remains the deep setup workspace.
- [ ] Handoff panels use "No scheduled task has been created yet" where applicable.
- [ ] Free text only finds templates; it does not infer config.
- [ ] URLs with tokens/secrets are not copied into setup summaries.
- [ ] Task detail URL opens the drawer when the task exists.
- [ ] Invalid tab/template/task states are non-blocking.
- [ ] Existing Phase 1 table/detail/Watchlists links still work.
- [ ] Extension route still renders the shared page.
- [ ] No backend files changed.

## Expected Commit Sequence

1. `feat: add scheduled task route state helpers`
2. `feat: add scheduled task template registry`
3. `feat: add scheduled task create panel`
4. `feat: add scheduled task workbench tabs`
5. `feat: open created reminder detail`
6. `test: update scheduled tasks route parity` (only if needed)
7. `chore: close scheduled tasks phase 2a task` (only if needed)

## Known Risks

- `ScheduledTasksPage.test.tsx` is already large. Keep new pure logic in helper tests and only test integration behavior in the page test.
- Ant Design tabs and cards can become cramped in extension width. Prefer wrapping, single-column panels, and existing compact controls.
- The current reminder editor is a card, not a multi-step wizard. Reuse it for Phase 2A unless adding steps is necessary to satisfy the tests; do not duplicate reminder form logic.
- The current route uses fallback i18n strings. Avoid a broad locale migration in this slice unless the implementation naturally touches localized resources.
- Watchlists prefill is not in scope. Use links and copyable summaries only.
