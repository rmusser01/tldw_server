# Scheduled Tasks Automation Workbench Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first shippable `/scheduled-tasks` Automation Workbench slice: unified visibility, clearer status, better task table/detail, Watchlists run/result deep links, and safer reminder scheduling.

**Architecture:** Keep Phase 1 frontend-heavy and contract-preserving. Reuse the existing scheduled-tasks control-plane response for reminders and Watchlists jobs, derive product-level display state in shared UI helpers, and deep-link to Watchlists for domain configuration, activity, and reports instead of moving Watchlists functionality into `/scheduled-tasks`.

**Tech Stack:** React, TypeScript, Ant Design, TanStack Query, React Router, Vitest, Testing Library, existing tldw shared UI and FastAPI control-plane contracts.

---

## Scope

This plan implements Phase 1 from `Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md`.

In scope:

- `/scheduled-tasks` page copy and IA shell.
- Overview cards for task counts and attention states.
- Product status mapping for reminders and Watchlists jobs.
- Enhanced task table with task type, status, schedule, last run, next run, and common actions.
- Detail drawer for an existing scheduled-task row.
- Watchlists deep links for monitor settings, activity, and reports.
- Safer reminder scheduling controls using recognition-friendly inputs and next-run preview copy.
- Focused component/unit tests and route parity checks.

Out of scope for this plan:

- GitHub, YouTube, RAG, ACP, and extension context-aware creation templates.
- New backend normalized run/result inbox APIs.
- Home automation inbox.
- Bulk actions and saved views.
- Any reduction or replacement of existing Watchlists UX.

## File Structure

### New Files

- `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-status.ts`
  - Pure helpers for product status, task type labels, date formatting, and Watchlists deep-link construction.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskOverview.tsx`
  - Top summary cards for total tasks, running/attention states, upcoming runs, and last completed run.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx`
  - Drawer for task summary, schedule, last/next run, source reference, and domain deep links.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ReminderScheduleControls.tsx`
  - Safer reminder schedule form controls used by `ReminderTaskEditor`.
- `apps/packages/ui/src/components/Option/ScheduledTasks/reminder-schedule-utils.ts`
  - Pure helpers for native datetime-local conversion, timezone defaulting, cron presets, validation, and preview text.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts`
  - Unit coverage for status mapping and Watchlists link builders.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/reminder-schedule-utils.test.ts`
  - Unit coverage for reminder schedule conversion and validation.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx`
  - Component coverage for detail drawer content and links.

### Modified Files

- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
  - Add workbench intro, overview, selected task state, drawer wiring, and updated create/edit labels.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskTable.tsx`
  - Add search/filter controls, richer columns, product status labels, detail action, and Watchlists action links.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ReminderTaskEditor.tsx`
  - Replace raw schedule fields with `ReminderScheduleControls`, keep emitted payload compatible with current API.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`
  - Update copy assertions and add overview/table/detail/reminder-control coverage.
- `apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx`
  - Confirm route remains registered for WebUI and extension after copy/component changes.
- `apps/extension/tests/e2e/integrations-and-scheduled-tasks.spec.ts`
  - Update smoke expectations for new workbench copy if this E2E is run in the implementation slice.
- Backlog task file for the implementation branch.
  - Before Task 1, create a new implementation Backlog.md task such as `Implement Scheduled Tasks Automation Workbench Phase 1`. Use that new implementation task for product-code edits and verification notes. `TASK-496` tracks this planning document only.

### Files To Avoid Changing In Phase 1

- `apps/packages/ui/src/components/Option/Watchlists/**`
  - Do not move Watchlists configuration into `/scheduled-tasks`.
- `tldw_Server_API/app/services/scheduled_tasks_control_plane_service.py`
  - Avoid backend changes unless a frontend test proves the existing fields are insufficient.
- ACP, RAG, Home, and extension context-detection surfaces.

## Existing Contracts To Reuse

Scheduled task rows already include:

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

Watchlists route deep links already support:

- `/watchlists?tab=jobs`
- `/watchlists?tab=runs&job_id=<id>`
- `/watchlists?tab=outputs&job_id=<id>`
- `/watchlists?tab=runs&run_id=<id>&open_run=1`
- `/watchlists?tab=outputs&output_id=<id>&open_output=1`

Phase 1 should derive Watchlists job links from `task.source_ref.job_id` and keep `task.manage_url` as the monitor settings fallback. If `source_ref` includes a run id or output id from the backend now or later, Phase 1 helpers should also build exact run/result links.

Accepted run id keys:

- `run_id`
- `runId`
- `last_run_id`
- `lastRunId`
- `latest_run_id`
- `latestRunId`

Accepted output id keys:

- `output_id`
- `outputId`
- `last_output_id`
- `lastOutputId`
- `latest_output_id`
- `latestOutputId`

## Pre-Implementation Requirement

Before Task 1 edits product files, create a new Backlog.md task for the implementation work. Use a title such as `Implement Scheduled Tasks Automation Workbench Phase 1`. Link it to `TASK-496` and the PRD. Record all implementation files, verification commands, skips, and final summary on that implementation task. `TASK-496` is only the planning-task record.

## Task 1: Add Scheduled Task Display Helpers

**Files:**

- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-status.ts`
- Test: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts`

- [ ] **Step 1: Write failing tests for status mapping**

Add tests for:

```ts
import {
  buildWatchlistTaskLinks,
  getScheduledTaskProductStatus,
  getScheduledTaskTypeLabel
} from "../scheduled-task-status"

it("maps disabled tasks to Disabled before backend status text", () => {
  expect(
    getScheduledTaskProductStatus({
      id: "watchlist_job:1",
      primitive: "watchlist_job",
      title: "Monitor",
      status: "scheduled",
      enabled: false,
      edit_mode: "external",
      source_ref: { job_id: 1 }
    })
  ).toMatchObject({
    label: "Disabled",
    tone: "default"
  })
})

it("maps failed-like statuses to Needs attention", () => {
  expect(
    getScheduledTaskProductStatus({
      id: "reminder_task:1",
      primitive: "reminder_task",
      title: "Reminder",
      status: "failed",
      enabled: true,
      edit_mode: "native",
      source_ref: {}
    })
  ).toMatchObject({
    label: "Needs attention",
    tone: "error"
  })
})

it("distinguishes blocked, found-results, and draft states", () => {
  expect(
    getScheduledTaskProductStatus({
      id: "watchlist_job:2",
      primitive: "watchlist_job",
      title: "Blocked monitor",
      status: "blocked",
      enabled: true,
      edit_mode: "external",
      source_ref: {}
    }).label
  ).toBe("Blocked")

  expect(
    getScheduledTaskProductStatus({
      id: "watchlist_job:3",
      primitive: "watchlist_job",
      title: "Monitor with outputs",
      status: "scheduled",
      enabled: true,
      edit_mode: "external",
      source_ref: { result_count: 2 }
    }).label
  ).toBe("Found results")

  expect(
    getScheduledTaskProductStatus({
      id: "reminder_task:4",
      primitive: "reminder_task",
      title: "Draft reminder",
      status: "draft",
      enabled: true,
      edit_mode: "native",
      source_ref: {}
    }).label
  ).toBe("Draft")
})

it("builds Watchlists deep links from source_ref.job_id", () => {
  expect(
    buildWatchlistTaskLinks({
      id: "watchlist_job:42",
      primitive: "watchlist_job",
      title: "Morning brief",
      status: "scheduled",
      enabled: true,
      edit_mode: "external",
      manage_url: "/watchlists?tab=jobs",
      source_ref: { job_id: 42 }
    })
  ).toMatchObject({
    settingsUrl: "/watchlists?tab=jobs",
    activityUrl: "/watchlists?tab=runs&job_id=42",
    reportsUrl: "/watchlists?tab=outputs&job_id=42"
  })
})

it("builds exact Watchlists run and output links when ids are available", () => {
  expect(
    buildWatchlistTaskLinks({
      id: "watchlist_job:42",
      primitive: "watchlist_job",
      title: "Morning brief",
      status: "scheduled",
      enabled: true,
      edit_mode: "external",
      source_ref: { job_id: 42, latest_run_id: 101, latest_output_id: 202 }
    })
  ).toMatchObject({
    latestRunUrl: "/watchlists?tab=runs&run_id=101&open_run=1",
    latestOutputUrl: "/watchlists?tab=outputs&output_id=202&open_output=1"
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts
```

Expected: FAIL because the helper module does not exist.

- [ ] **Step 3: Implement minimal helper module**

Implement exported helpers:

```ts
export type ScheduledTaskStatusTone = "success" | "processing" | "warning" | "error" | "default"

export interface ScheduledTaskProductStatus {
  label: string
  tone: ScheduledTaskStatusTone
  description: string
}

export interface WatchlistTaskLinks {
  settingsUrl: string | null
  activityUrl: string | null
  reportsUrl: string | null
  latestRunUrl: string | null
  latestOutputUrl: string | null
}
```

Mapping rules:

- `enabled === false` -> `Disabled`.
- status contains `draft` -> `Draft`.
- status contains `running`, `active`, `processing`, or `in_progress` -> `Running now`.
- status contains `blocked`, `auth`, `permission`, `unavailable`, or `dependency` -> `Blocked`.
- status contains `found`, `match`, `matched`, `result`, or `output` -> `Found results`.
- any positive `source_ref.result_count`, `source_ref.results_count`, `source_ref.output_count`, `source_ref.outputs_count`, `source_ref.latest_output_id`, or `source_ref.output_id` -> `Found results`.
- status contains `fail`, `error`, or `missed` -> `Needs attention`.
- If one status contains both result-like and failure-like tokens, failure-like tokens take precedence so an errored output state does not appear successful.
- status contains `paused` -> `Paused`.
- status contains `complete`, `success`, `done`, or `finished` -> `Completed last run`.
- otherwise enabled scheduled task -> `Waiting for next run`.

Task type labels:

- `reminder_task` -> `Reminder`.
- `watchlist_job` -> `Watchlist monitor`.
- fallback -> `Scheduled task`.

Link builder:

- If `primitive !== "watchlist_job"`, return all `null` except settings if `manage_url` exists.
- If `source_ref.job_id` is a positive integer, build job-filtered activity and reports URLs.
- If an accepted run id key exists and is a positive integer, build `/watchlists?tab=runs&run_id=<id>&open_run=1`.
- If an accepted output id key exists and is a positive integer, build `/watchlists?tab=outputs&output_id=<id>&open_output=1`.
- Always prefer `task.manage_url` for settings URL, falling back to `/watchlists?tab=jobs`.

- [ ] **Step 4: Run helper tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-status.ts apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts
git commit -m "feat: add scheduled task display helpers"
```

## Task 2: Add Workbench Overview And Core Page States

**Files:**

- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskOverview.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
- Test: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`

- [ ] **Step 1: Write failing page tests**

Add or update tests to assert:

- page subtitle says scheduled tasks are for reminders and recurring automation visibility;
- overview shows total tasks;
- overview shows needs-attention count;
- overview shows next upcoming run when `next_run_at` exists;
- empty state says "No scheduled tasks yet" and includes "Create scheduled task";
- loading state says "Loading tasks and latest run state";
- partial state still shows loaded rows and explains that some task families are unavailable;
- load-error and unsupported states keep recovery actions and diagnostics;
- Watchlists preservation copy is visible when a Watchlists job exists.

Example assertion:

```ts
expect(
  await screen.findByText(/Track reminders, Watchlist monitors, and recurring automation from one place/i)
).toBeInTheDocument()
expect(screen.getByText("2 scheduled tasks")).toBeInTheDocument()
expect(screen.getByText("1 needs attention")).toBeInTheDocument()
expect(screen.getByText(/Watchlists remains the full workspace/i)).toBeInTheDocument()
```

- [ ] **Step 2: Run test to verify it fails**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
```

Expected: FAIL on missing overview/copy.

- [ ] **Step 3: Implement `ScheduledTaskOverview`**

Use existing Ant Design primitives already used on the route. Keep the component pure:

```ts
interface ScheduledTaskOverviewProps {
  tasks: ScheduledTask[]
  partial: boolean
}
```

Render four compact panels:

- Total scheduled tasks.
- Needs attention.
- Running now.
- Next upcoming run.

Use `getScheduledTaskProductStatus` from Task 1 for status counts.

- [ ] **Step 4: Wire overview into `ScheduledTasksPage`**

Change page intro copy to:

```text
Track reminders, Watchlist monitors, and recurring automation from one place. Use domain workspaces like Watchlists for deep source and output configuration.
```

Render `ScheduledTaskOverview` above the table when the list is loaded. When loaded and empty, render a clear empty state before the table title:

```text
No scheduled tasks yet.
Create a reminder now. Automation templates for GitHub, YouTube, RAG, and agents are planned follow-up phases.
```

Replace the bare `Spin` loading UI with a small loading state that includes the exact text:

```text
Loading tasks and latest run state
```

Keep unsupported, load-error, and partial recovery callouts, but make sure tests assert that:

- unsupported state includes the endpoint diagnostics and health action;
- load-error state includes retry and diagnostics actions;
- partial state still renders any loaded rows and clearly says that some scheduled-task data loaded while one dependency could not be reached.

- [ ] **Step 5: Run page tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskOverview.tsx apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
git commit -m "feat: add scheduled tasks workbench overview"
```

## Task 3: Upgrade The Task Table

**Files:**

- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskTable.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
- Test: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`

- [ ] **Step 1: Write failing tests for table behavior**

Add assertions for:

- CTA label is `Create scheduled task`.
- type labels are `Reminder` and `Watchlist monitor`.
- mode labels are `Managed here` and `Managed in Watchlists`.
- status label is product status, not raw backend `scheduled`.
- table includes `Last run` and `Next run`.
- Watchlists row has actions for `Open monitor settings`, `Open activity`, and `Open reports`.
- selecting `Needs attention` filter hides healthy rows.
- search filters by task title.

- [ ] **Step 2: Run test to verify it fails**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
```

Expected: FAIL on old table labels and missing filters.

- [ ] **Step 3: Update table props**

Change props to:

```ts
export interface ScheduledTaskTableProps {
  tasks: ScheduledTask[]
  onCreateReminder: () => void
  onInspectTask: (task: ScheduledTask) => void
  onEditReminder: (task: ScheduledTask) => void
  onDeleteReminder: (task: ScheduledTask) => void
}
```

Keep `onCreateReminder` for Phase 1 even though the user-facing label becomes `Create scheduled task`.

- [ ] **Step 4: Add table local filtering**

Add local `useState` for:

- search text;
- status filter: `all | needs_attention | running | waiting | found_results | blocked | paused | disabled | draft | completed`;
- type filter: `all | reminder_task | watchlist_job`.

Do not introduce URL state in Phase 1.

- [ ] **Step 5: Update columns**

Columns:

- Task: title, description, type tag.
- Status: product status label and description.
- Schedule: schedule summary, timezone.
- Last run: formatted timestamp or `No completed runs yet`.
- Next run: formatted timestamp or `No upcoming run`.
- Management: `Managed here` or `Managed in Watchlists`.
- Actions: inspect, edit/delete for reminders, and Watchlists links for watchlist jobs.
- If exact Watchlists run/result links exist, show them as `Open latest run` and `Open latest report` actions.

Use the helper link builder from Task 1 for Watchlists actions.

- [ ] **Step 6: Wire inspect handler**

In `ScheduledTasksPage`, add:

```ts
const [selectedTask, setSelectedTask] = useState<ScheduledTask | null>(null)
const openTaskDetail = (task: ScheduledTask) => setSelectedTask(task)
const closeTaskDetail = () => setSelectedTask(null)
```

Pass `openTaskDetail` to the table. The detail drawer is added in Task 4; for this task it is acceptable to set state without rendering it, or render a temporary no-op behind Task 4. Prefer wiring it with Task 4 if the same worker is continuing immediately.

- [ ] **Step 7: Run focused tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskTable.tsx apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
git commit -m "feat: upgrade scheduled tasks table"
```

## Task 4: Add Scheduled Task Detail Drawer

**Files:**

- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx`
- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskTable.tsx`

- [ ] **Step 1: Write failing drawer tests**

Test reminder detail:

```ts
render(
  <ScheduledTaskDetailDrawer
    open
    task={reminderTask}
    onClose={vi.fn()}
    onEditReminder={vi.fn()}
    onDeleteReminder={vi.fn()}
  />
)

expect(screen.getByRole("dialog", { name: /Review notes/i })).toBeInTheDocument()
expect(screen.getByText("Reminder")).toBeInTheDocument()
expect(screen.getByRole("button", { name: "Edit reminder" })).toBeInTheDocument()
```

Test Watchlists detail:

```ts
expect(screen.getByRole("link", { name: "Open monitor settings" })).toHaveAttribute(
  "href",
  "/watchlists?tab=jobs"
)
expect(screen.getByRole("link", { name: "Open activity" })).toHaveAttribute(
  "href",
  "/watchlists?tab=runs&job_id=42"
)
expect(screen.getByRole("link", { name: "Open reports" })).toHaveAttribute(
  "href",
  "/watchlists?tab=outputs&job_id=42"
)
expect(screen.getByRole("link", { name: "Open latest run" })).toHaveAttribute(
  "href",
  "/watchlists?tab=runs&run_id=101&open_run=1"
)
expect(screen.getByRole("link", { name: "Open latest report" })).toHaveAttribute(
  "href",
  "/watchlists?tab=outputs&output_id=202&open_output=1"
)
expect(screen.getByText(/Watchlists remains the full workspace/i)).toBeInTheDocument()
```

- [ ] **Step 2: Run drawer tests to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx
```

Expected: FAIL because component does not exist.

- [ ] **Step 3: Implement drawer**

Use Ant Design `Drawer`, `Descriptions`, `Space`, `Tag`, `Button`, and `Typography`.

Props:

```ts
interface ScheduledTaskDetailDrawerProps {
  open: boolean
  task: ScheduledTask | null
  onClose: () => void
  onEditReminder: (task: ScheduledTask) => void
  onDeleteReminder: (task: ScheduledTask) => void
}
```

Content:

- Product status.
- Task type.
- Management owner.
- Schedule summary.
- Timezone.
- Last run.
- Next run.
- Source reference summary:
  - reminder task id and link fields when present;
  - Watchlists job id and scope when present.
- Actions:
  - reminders: `Edit reminder`, `Delete reminder`;
  - Watchlists: `Open monitor settings`, `Open activity`, `Open reports`;
  - Watchlists with exact ids: also show `Open latest run` and `Open latest report` when helper URLs exist.

Do not call Watchlists APIs from this drawer in Phase 1.

- [ ] **Step 4: Wire drawer into page**

In `ScheduledTasksPage`, render:

```tsx
<ScheduledTaskDetailDrawer
  open={Boolean(selectedTask)}
  task={selectedTask}
  onClose={closeTaskDetail}
  onEditReminder={openEditReminder}
  onDeleteReminder={handleDeleteReminder}
/>
```

When the user clicks edit from the drawer, close the drawer first or after opening the editor. Pick one behavior and cover it in tests.

- [ ] **Step 5: Add inspect action to table**

Every row should have an `Inspect` action. For keyboard accessibility, use a real `Button`.

- [ ] **Step 6: Run focused tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskTable.tsx apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
git commit -m "feat: add scheduled task detail drawer"
```

## Task 5: Replace Raw Reminder Schedule Inputs

**Files:**

- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/reminder-schedule-utils.ts`
- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/ReminderScheduleControls.tsx`
- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/reminder-schedule-utils.test.ts`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ReminderTaskEditor.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`

- [ ] **Step 1: Write failing utility tests**

Cover:

- `getDefaultReminderTimezone()` returns an IANA timezone or `UTC`.
- one-time `datetime-local` string converts to an ISO string.
- recurring presets produce expected cron:
  - daily 09:00 -> `0 9 * * *`;
  - weekly Monday 09:00 -> `0 9 * * MON`;
- invalid custom cron returns a field-count validation error.

Do not import `dayjs`; shared UI has tests preventing direct dayjs imports.

- [ ] **Step 2: Run utility tests to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/reminder-schedule-utils.test.ts
```

Expected: FAIL because utilities do not exist.

- [ ] **Step 3: Implement schedule utilities**

Use browser-native `Date` and `Intl`.

Exports:

```ts
export type ReminderRecurrencePreset = "daily" | "weekly" | "custom"

export const getDefaultReminderTimezone = (): string => {
  return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"
}

export const buildDailyCron = (hour: number, minute: number): string => `${minute} ${hour} * * *`
export const buildWeeklyCron = (weekday: string, hour: number, minute: number): string =>
  `${minute} ${hour} * * ${weekday}`
```

Validation:

- one-time requires a valid datetime-local value;
- recurring requires timezone;
- custom cron requires exactly 5 fields and allowed token characters.

- [ ] **Step 4: Implement `ReminderScheduleControls`**

Use Ant Design `Segmented` or `Select`, `Input`, `InputNumber`, and `Switch` as appropriate.

Controls:

- Schedule kind: `Run once` or `Repeat`.
- Run once:
  - `Input` with `type="datetime-local"`;
  - timezone display/helper;
  - next-run preview.
- Repeat:
  - preset select: daily, weekly, custom schedule;
  - hour/minute controls for daily/weekly;
  - weekday select for weekly;
  - custom cron input behind the custom preset;
  - timezone select/input defaulting to local timezone;
  - next-run preview copy based on selected preset.

Keep payload compatibility by writing to the existing form fields:

- `schedule_kind`;
- `run_at`;
- `cron`;
- `timezone`.

- [ ] **Step 5: Replace raw fields in `ReminderTaskEditor`**

Remove the separate raw `Run at`, `Cron`, and `Timezone` `Form.Item`s from the editor and render `ReminderScheduleControls`.

Update copy:

- Card title: `Create reminder` or `Edit reminder`.
- Save button: `Save reminder`.
- Enabled label: `Task is active`.

Keep current `onSubmit` payload shape.

- [ ] **Step 6: Update page tests**

Update create reminder test to use the new controls. For example:

```ts
await user.click(await screen.findByRole("button", { name: "Create scheduled task" }))
await user.type(await screen.findByRole("textbox", { name: "Title" }), "Daily review")
fireEvent.change(screen.getByLabelText("Run once at"), {
  target: { value: "2026-03-21T10:00" }
})
await user.click(screen.getByRole("button", { name: "Save reminder" }))
```

Assert the API payload still includes `schedule_kind: "one_time"` and an ISO-like `run_at` string.

- [ ] **Step 7: Run focused tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/reminder-schedule-utils.test.ts src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/reminder-schedule-utils.ts apps/packages/ui/src/components/Option/ScheduledTasks/ReminderScheduleControls.tsx apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/reminder-schedule-utils.test.ts apps/packages/ui/src/components/Option/ScheduledTasks/ReminderTaskEditor.tsx apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
git commit -m "feat: improve reminder scheduling controls"
```

## Task 6: Route Parity, Extension Smoke, And Copy Contracts

**Files:**

- Modify: `apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx`
- Modify if needed: `apps/extension/tests/e2e/integrations-and-scheduled-tasks.spec.ts`
- Test: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`

- [ ] **Step 1: Check route tests for changed copy**

Open `apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx` and confirm it does not assert old "Review reminder tasks here" copy. If it does, update it to the workbench copy.

- [ ] **Step 2: Check extension E2E smoke copy**

Open `apps/extension/tests/e2e/integrations-and-scheduled-tasks.spec.ts`. If it asserts `Watchlist jobs remain managed from Watchlists`, update to a stable assertion such as:

```ts
await expect(page.getByRole("heading", { name: /scheduled tasks/i })).toBeVisible()
await expect(page.getByText(/Track reminders, Watchlist monitors/i)).toBeVisible()
```

- [ ] **Step 3: Run route and component tests**

Before running tests, add or update at least one assertion that Watchlists remains read-only from `/scheduled-tasks`:

```ts
expect(screen.getByRole("link", { name: "Open monitor settings" })).toHaveAttribute(
  "href",
  "/watchlists?tab=jobs"
)
expect(screen.queryByRole("button", { name: "Edit watchlist job" })).not.toBeInTheDocument()
```

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/scheduled-tasks-route.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
```

Expected: PASS.

- [ ] **Step 4: Optionally run extension E2E only if the implementation branch already has extension test prerequisites running**

Run from repo root only if local E2E setup is available:

```bash
bunx playwright test apps/extension/tests/e2e/integrations-and-scheduled-tasks.spec.ts
```

Expected: PASS. If unavailable, document the skip and rely on shared route/component coverage.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/routes/__tests__/scheduled-tasks-route.test.tsx apps/extension/tests/e2e/integrations-and-scheduled-tasks.spec.ts apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
git commit -m "test: update scheduled tasks route parity coverage"
```

If no files changed in this task, do not make an empty commit. Record the no-op in the Backlog task notes.

## Task 7: Final Verification And Closeout

**Files:**

- Modify: the Backlog.md task created for the Phase 1 implementation branch before Task 1 starts. The implementer must create that task before any product-code edits and record the concrete path in their branch notes. Do not use `TASK-496` for implementation progress; it tracks this plan document only.
- No product code changes unless verification finds a defect.

- [ ] **Step 1: Run frontend focused tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-status.test.ts src/components/Option/ScheduledTasks/__tests__/reminder-schedule-utils.test.ts src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/routes/__tests__/scheduled-tasks-route.test.tsx src/services/__tests__/scheduled-tasks-control-plane.test.ts
```

Expected: PASS.

- [ ] **Step 2: Run backend contract tests**

Run from repo root:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_tasks_control_plane.py -q
```

Expected: PASS. If Phase 1 has no backend changes and the environment cannot run backend tests, document the exact environment failure.

- [ ] **Step 3: Run Bandit on touched backend scope if backend files changed**

If no backend Python files changed, record:

```text
Bandit skipped: Phase 1 implementation changed only frontend TypeScript/tests.
```

If backend Python files changed, run from repo root:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/services/scheduled_tasks_control_plane_service.py tldw_Server_API/app/api/v1/schemas/scheduled_tasks_control_plane_schemas.py -f json -o /tmp/bandit_scheduled_tasks_phase1.json
```

Expected: no new findings in touched code.

- [ ] **Step 4: Run whitespace check**

Run from repo root:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Manual browser verification**

Start or reuse the WebUI dev server. Open `/scheduled-tasks` in the Browser plugin or Playwright and verify:

- empty state;
- loaded reminder row;
- loaded Watchlists row;
- overview counts;
- filters/search;
- detail drawer;
- reminder create/edit schedule controls;
- Watchlists links open the expected Watchlists tab/filter.

If no seeded backend is available, document that live data verification was skipped and include component test coverage as evidence.

- [ ] **Step 6: Update Backlog task**

Record:

- files changed;
- tests run and results;
- Bandit result or documented skip;
- manual verification result or documented skip;
- known follow-up phases not implemented.

- [ ] **Step 7: Final implementation commit**

If Task 7 changed only Backlog task notes, stage the concrete Backlog task path returned by `mcp__backlog.task_create` for the implementation branch and commit with:

```bash
git commit -m "chore: record scheduled tasks phase 1 verification"
```

If no files changed, do not commit.

## Required Verification Summary

Before calling Phase 1 complete, the implementer must provide:

- focused frontend test command and result;
- backend contract test command and result or documented no-backend-change skip;
- Bandit result or documented no-backend-change skip;
- `git diff --check` result;
- manual browser verification result or documented environment limitation;
- confirmation that Watchlists configuration UX remains in Watchlists.

## Follow-Up Plans

Create separate implementation plans after Phase 1 lands:

1. Phase 2: template creation for GitHub issue monitor, YouTube channel ingest, and improved reminder creation entry.
2. Phase 3: scheduled results inbox and Home automation surfacing.
3. Phase 4: recurring RAG query and ACP/agent schedule integration.
4. Phase 5: extension context-aware creation and power-user bulk management.

## Plan Review

Reviewed by plan-document-reviewer subagents on 2026-06-01.

Status: Approved after three iterations.

Resolved review blockers:

- Added the full Phase 1 status taxonomy, including blocked, found results, and draft.
- Added exact Watchlists run and output deep-link handling when ids are available.
- Added loading, partial, unsupported, and load-error state requirements and tests.
- Clarified that implementation requires a new Backlog.md task before product-code edits; `TASK-496` tracks this plan only.
- Fixed the `WatchlistTaskLinks.settingsUrl` contract so it is nullable for non-Watchlists tasks.

Advisory recommendation:

- Keep Phase 1 scoped to existing reminders and Watchlists control-plane data; defer Home/results inbox/templates/ACP/RAG integration to later phases.
