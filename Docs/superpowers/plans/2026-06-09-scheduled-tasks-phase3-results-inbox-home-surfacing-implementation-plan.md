# Scheduled Tasks Phase 3 Results Inbox And Home Surfacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Phase 3 scheduled-task results discovery experience so users can see automation outcomes on Home and inside `/scheduled-tasks`, inspect exact task/run/result provenance, triage failures, and follow deep links back to the owning workspace without changing Watchlists or other domain-specific setup flows.

**Architecture:** Add a frontend-owned scheduled-task result projection layer that can consume the existing `/api/v1/scheduled-tasks` task list now and later swap to a normalized result-inbox API. Extend the existing `/scheduled-tasks` query-state tab model with a Results tab, result/run deep-link parameters, result filters, and a detail drawer. Add `/scheduled-tasks/results` as a compatibility alias that opens the same Results tab state so PRD, Home, and notification links have a durable route. Add an automation-specific Home module that can render status, owner, and provenance compactly instead of flattening automation outcomes into generic Companion cards. Add notification-to-result link normalization and dedupe helpers with an explicit notification data source. Backend changes remain dependencies for this phase unless a normalized result endpoint already exists when implementation starts.

**Tech Stack:** React, TypeScript, Ant Design, Tailwind utility classes in Companion Home, React Router search params, TanStack Query, existing `bgRequest` service layer, existing ScheduledTasks and CompanionHome components, Vitest, Testing Library, existing browser/extension route shells.

---

## Source Inputs

- Product/design spec: `Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md`
- Phase 1 plan: `Docs/superpowers/plans/2026-06-01-scheduled-tasks-automation-workbench-phase1-implementation-plan.md`
- Phase 2A plan: `Docs/superpowers/plans/2026-06-08-scheduled-tasks-automation-workbench-phase2a-create-framework-implementation-plan.md`
- Phase 2B plan: `Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase2b-capability-aware-frontend-shell-implementation-plan.md`
- Backlog references: `TASK-494`, `TASK-496`, `TASK-498`
- Planning task: `TASK-2331`
- Plan amendment task: `TASK-2332`

## Current Evidence

- `/scheduled-tasks` is a single options route with query-state tabs in `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-route-state.ts`.
- Existing tabs are `overview`, `tasks`, and `create`; unsupported tabs fall back to Overview with non-blocking copy.
- `ScheduledTasksPage.tsx` loads `/api/v1/scheduled-tasks`, shows partial dependency failures, and renders task overview, table, creation panel, and a task detail drawer.
- `scheduled-task-status.ts` already maps raw task state into product statuses and builds safe Watchlists deep links for settings, activity, reports, latest run, and latest output.
- Watchlists jobs are intentionally read-only from Scheduled Tasks and retain “Managed in Watchlists” copy.
- Companion Home has fixed system cards for `Inbox Preview` and `Needs Attention`, but its current data model is companion-personalization centered: `canonical_inbox`, `goal`, `reading`, and `note`.
- Companion Home returns an empty snapshot when personalization is unavailable, so scheduled-task outcomes need a non-personalized path to Home.
- `apps/packages/ui/src/services/notifications.ts` already exposes notification fields such as `link_type`, `link_id`, `link_url`, `source_task_id`, `source_task_run_id`, `source_job_id`, and source metadata that can support exact scheduled-task result links once normalized.
- The extension and hosted WebUI share the same UI package components.

## Product Decisions

- Scheduled Tasks is the cross-automation control and monitoring surface.
- Watchlists remains the owner for monitor setup, source tuning, run activity, and reports.
- GitHub and YouTube are examples only; Phase 3 must use source-agnostic result language such as “source”, “match”, “ingested item”, “run”, “output”, and “owning workspace”.
- Home should surface “something happened” and “something needs attention”; detailed configuration and domain output review stays behind deep links.
- Phase 3 should not implement recurring RAG query creation or ACP/agent schedule creation. Those remain Phase 4 work.
- Phase 3 can show result discovery for task families already represented by the Scheduled Tasks control plane, including reminders and Watchlist-backed monitors, but it must not imply unavailable creation adapters exist.

## Scope

In scope:

- Add `/scheduled-tasks?tab=results` to the existing tab model.
- Add `/scheduled-tasks/results` as a route alias that resolves to the Results tab.
- Add result/run deep-link parameters, with support for `task_id`, `run_id`, and `result_id`.
- Add a source-agnostic result inbox list inside `/scheduled-tasks`.
- Add a result detail drawer with task state, run state, output summary, provenance, owning workspace, and safe actions.
- Add Home surfacing through an automation-specific module/card that can show result/failure state, owner, and exact deep links.
- Add notification deep-link normalization for scheduled-task task/run/result targets.
- Add dedupe keys so the same run/result does not appear twice across Home inbox and notification-derived items.
- Preserve Companion Home personalization behavior while allowing scheduled-task signals to render without personalization.
- Add empty, loading, partial, unsupported, success, no-new-results, running, failure, and review-complete states.
- Add copy that explains where results came from and where to continue.
- Add accessibility and extension-width requirements to implementation tests.
- Add analytics-free state logic; do not add telemetry.

Out of scope:

- Moving Watchlists setup, source tuning, run logs, or report builders into `/scheduled-tasks`.
- Source-specific first-class UI for GitHub, YouTube, or any specific provider.
- Creating Watch/Ingest/RAG/ACP/agent scheduled tasks if creation adapters are not already available.
- Building backend normalized result storage in this Phase 3 frontend plan.
- Adding push notifications beyond linking to existing notification/event surfaces.
- Replacing the existing Notifications service or Watchlists run notification logic.
- Bulk management across many results beyond filters and supported review actions.
- Reworking Companion Home layout customization beyond adding scheduled-task-aware system signals.

## Backend Dependencies

The frontend implementation must be designed around these contracts, but this planning branch does not implement them:

- `GET /api/v1/scheduled-tasks/results`
  - Returns normalized result inbox items across task primitives.
  - Supports filters: status, review state, task id, owning workspace, time range, result type.
  - Returns partial dependency metadata with recoverable source errors.
- `GET /api/v1/scheduled-tasks/results/{result_id}`
  - Returns detail fields, provenance, run summary, source references, result links, and safe action availability.
- `GET /api/v1/scheduled-tasks/runs/{run_id}`
  - Returns run state, timestamps, logs/events summary, failure hints, retry eligibility, and output ids.
- `POST /api/v1/scheduled-tasks/results/{result_id}/review`
  - Marks reviewed, snoozed, dismissed, or important.
- `POST /api/v1/scheduled-tasks/runs/{run_id}/retry`
  - Retries eligible failed runs without changing task configuration.
- Notification payloads include exact `task_id`, `run_id`, `result_id`, owning workspace, and `href`.

Until those endpoints exist, implementation should use a local projection adapter from `ScheduledTask[]` and `source_ref` fields, with clear capability copy and deep-link-only actions where direct review/retry cannot be performed.

## Capability Modes

The Results experience must expose only the behavior supported by the current data source. Do not show fake review state or mutation actions when the UI is only projecting task-list signals.

| Mode | Data Source | User-Facing Label | Review State | Retry/Review Actions | Default Copy |
| --- | --- | --- | --- | --- | --- |
| `projected_signals` | `GET /api/v1/scheduled-tasks` only | Latest signals | Hidden | Hidden | “These signals are inferred from task status until result history is available.” |
| `normalized_results_read` | `GET /api/v1/scheduled-tasks/results` | Results | Visible, read-only | Hidden unless mutation paths exist | “Review state comes from the scheduled-task results API.” |
| `normalized_results_mutation` | Results API plus review/retry endpoints | Results | Visible and actionable | Shown only when item-level availability is true | “You can review results and retry eligible runs from here.” |

Mode detection:

- Use the existing OpenAPI support probe pattern to detect `/api/v1/scheduled-tasks/results`, `/api/v1/scheduled-tasks/results/{result_id}`, `POST /api/v1/scheduled-tasks/results/{result_id}/review`, and `POST /api/v1/scheduled-tasks/runs/{run_id}/retry`.
- Default to `projected_signals` if the normalized result paths are absent or the probe fails open.
- In `projected_signals`, hide the Review state filter, hide `Mark reviewed`, hide `Retry run`, and label the primary tab content “Latest automation signals” in supporting copy.
- In `normalized_results_read`, show review state as information but hide mutation actions.
- In `normalized_results_mutation`, show mutation actions only when the result item sets `reviewAvailable` or `retryAvailable`.

## Information Architecture

`/scheduled-tasks` tabs:

- `Overview`
  - Cross-automation counts, needs attention, running now, next run, newest results, and short links to Tasks/Results.
- `Results`
  - Primary Phase 3 inbox for outcomes.
  - Defaults to newest actionable signal/result first.
  - Filters: Signal/result state, task type, owning workspace, time range.
  - Adds Review state filter only outside `projected_signals` mode.
  - Detail drawer opens from row/card or deep link.
- `Tasks`
  - Existing task management table and task detail drawer.
  - Adds a per-row “Results” action only when the task has result signals.
- `Create`
  - Existing creation framework. No Phase 3 creation expansion.

Home:

- Keep existing Companion Home Inbox Preview and Needs Attention behavior.
- Add a fixed, automation-specific Home module titled `Automation Inbox` near the existing Home system cards.
- `Automation Inbox` shows recent results, failed/stalled/blocked signals, owner, status, and exact deep links.
- Do not add a movable Customize Home card in Phase 3; keep layout customization unchanged.
- Home links point to `/scheduled-tasks?tab=results&result_id=...` or `/scheduled-tasks?tab=results&run_id=...`.
- Items that belong to Watchlists also provide secondary links from the result detail drawer to Watchlists Activity/Reports.

Notifications:

- Home loads recent notifications through `listNotifications({ limit: 50 })` when the notifications endpoint is available, and treats notification loading failure as non-blocking.
- Existing notification payloads normalize to the same scheduled-task result target shape used by Home.
- Notification clicks should prefer exact `result_id`, then exact `run_id`, then task-scoped Results tab.
- Home and notification-derived result items should share a deterministic dedupe key: `result:<result_id>`, `run:<run_id>`, or `task:<task_id>:state:<state>:time:<occurredAt>`.
- Watchlists run notifications remain Watchlists-owned, but when they are represented inside Scheduled Tasks they should link through the Scheduled Tasks result drawer with secondary Watchlists links.

## Result Item Model

Create a frontend model that can be produced from current `ScheduledTask[]` and later from a normalized API:

```ts
export type ScheduledTaskResultState =
  | "new"
  | "reviewed"
  | "running"
  | "completed_no_results"
  | "failed"
  | "blocked"
  | "paused"

export type ScheduledTaskResultSeverity = "info" | "success" | "warning" | "error"

export type ScheduledTaskResultOwner =
  | "scheduled_tasks"
  | "watchlists"
  | "reminders"
  | "external_workspace"

export type ScheduledTaskResultsCapabilityMode =
  | "projected_signals"
  | "normalized_results_read"
  | "normalized_results_mutation"

export type ScheduledTaskResultSignalKind =
  | "result"
  | "failure"
  | "running"
  | "completed_no_results"

export interface ScheduledTaskResultItem {
  id: string
  capabilityMode: ScheduledTaskResultsCapabilityMode
  signalKind: ScheduledTaskResultSignalKind
  taskId: string
  runId: string | null
  resultId: string | null
  taskTitle: string
  resultKind: string
  title: string
  summary: string
  matchReason: string | null
  matchedRuleLabel: string | null
  outputLabel: string | null
  state: ScheduledTaskResultState
  severity: ScheduledTaskResultSeverity
  reviewed: boolean
  owner: ScheduledTaskResultOwner
  ownerLabel: string
  occurredAt: string | null
  sourceLabel: string | null
  provenance: Array<{ label: string; value: string }>
  primaryHref: string
  sourceHref: string | null
  taskHref: string
  runHref: string | null
  resultHref: string | null
  domainHref: string | null
  notificationIds: number[]
  dedupeKey: string
  retryAvailable: boolean
  reviewAvailable: boolean
}
```

Projection rules:

- `found_results` tasks produce `new` success items when `source_ref` has a result/output count or latest output id.
- `needs_attention` and `blocked` tasks produce attention items with failure-oriented recovery copy.
- `running` tasks produce running items only in `/scheduled-tasks`; Home should not surface normal running state unless the run appears stalled.
- `completed` tasks with no result signal are visible behind a “Completed/no results” filter but are not Home inbox items.
- Disabled, paused, and draft tasks appear only when filters request them.
- Watchlists result items must include Watchlists deep links from `buildWatchlistTaskLinks`; Scheduled Tasks remains the triage surface.
- Multiple events for the same run/result collapse into one visible Home result item, with notification ids retained for future read/dismiss actions.

Mixed-state rules:

- Never reduce a task to one result item only because its task-level product status has one label.
- If one task has both failure tokens and output/result ids, create a failure signal and a result signal. The failure signal appears in Needs attention; the result signal appears in Results/Automation Inbox if it is otherwise visible.
- If a latest run failed after producing an output, the detail drawer must say both: “Run needs attention” and “Output was produced.”
- Exact result ids dedupe only result signals. Exact run ids can dedupe run-level signals only when `signalKind` is the same. Fallback dedupe keys include `signalKind` so failures do not erase results.
- A task-level `found_results` signal without an output id uses `task:<taskId>:state:result:time:<lastRunAt>` as a fallback key.
- A task-level failure without a run id uses `task:<taskId>:state:failure:time:<lastRunAt>` as a fallback key.
- Review state is durable only in `normalized_results_read` and `normalized_results_mutation`; projected result signals are treated as not reviewable.

## UX Requirements

First-time user:

- Can explain what the Results tab is from the empty state and header copy.
- Sees where results will appear on Home.
- Can distinguish “no tasks yet”, “tasks exist but no results yet”, “results found”, and “task failed”.
- Understands when a result is managed elsewhere and why the primary setup action opens another workspace.
- Can inspect provenance without reading raw ids first.

Power user:

- Can filter by state, task type, owner, and review state when durable review data exists.
- Can deep-link directly to a run/result drawer.
- Can open the owning workspace in one click.
- Can scan many results without detail drawers.
- Can mark items reviewed where supported and retry eligible failures where supported.
- Does not see unsupported mutation buttons in row actions.
- Can preserve fast task management in the Tasks tab without result cards making the table slower to scan.

## Empty, Loading, Error, Running, And Success States

Scheduled Tasks Results tab:

- Empty when no tasks exist: “No scheduled tasks yet” with Create action and copy explaining results appear after an automation runs.
- Empty when tasks exist but no results: “No results to review” with next-run summary and link to Tasks.
- Empty after filters: “No results match these filters” with clear-filters action.
- Loading: “Loading results and latest run state” with polite live region.
- Partial: reuse `RecoveryCallout`, preserve visible results, and show dependency diagnostics.
- Unsupported normalized result API: show current task-derived signals and a capability note, not a dead-end error.
- Running: show “Running now” tag, started time if available, and no retry/review actions.
- Failed: show failure hint, owning workspace, last attempt time, retry action only when item and capability mode allow it, otherwise recovery link.
- Success: show result count/output summary, reviewed state only when durable, and primary action “Open result” or “Review result” depending on capability mode.

Home:

- Empty Automation Inbox: “Automation results and failures will appear here after scheduled tasks run.”
- Loading Home: do not block the whole Home page on scheduled-task result loading; render cards with available companion items and update counts when scheduled-task items resolve.
- Partial Home: show companion items and scheduled-task items independently; one degraded source must not erase the other.
- Success Home: scheduled-task items include title, short summary, visible status text, owner, and deep link to the exact result/run.

## Copy Recommendations

Use these strings unless implementation discovers stronger local wording:

- Results tab label: `Results`
- Results page title: `Scheduled task results`
- Results page subtitle: `Review outputs, failures, and run state from recurring automations. Source-specific setup stays in the owning workspace.`
- Projected-mode subtitle: `Latest signals inferred from task status. Durable review state appears when the results API is available.`
- Filter label, normalized result modes only: `Review state`
- Filter values, normalized result modes only: `Unreviewed`, `Reviewed`, `All`
- Filter label: `Result state`
- Filter values: `Found results`, `Needs attention`, `Running`, `Completed/no results`, `Paused/disabled`
- Primary action in projected mode: `Open signal`
- Primary action with durable results: `Review result`
- Failure action: `Inspect failure`
- Retry action: `Retry run`
- Review action: `Mark reviewed`
- Unsupported-action detail note: `Review and retry actions appear when this server supports them for the selected result.`
- Watchlists owner note: `This monitor is configured in Watchlists. Scheduled Tasks shows status and links to the exact run or report.`
- Home module title: `Automation Inbox`
- Home result item prefix: `Automation result`
- Home attention item prefix: `Automation needs attention`
- Result detail provenance heading: `Why this is here`
- Result detail action heading: `Continue in`
- Completed/no results copy: `The latest run completed and did not produce new results.`

Avoid:

- Source-vendor-first labels such as “GitHub results” or “YouTube ingest results” in generic Scheduled Tasks UI.
- “Notification sent” unless the notification system confirms delivery.
- “Searchable” unless ingest/search indexing has confirmed completion.
- “Managed here” for external tasks.

## Accessibility And Usability Requirements

- Results tab is keyboard reachable through the existing Ant Design tablist.
- Result cards/rows expose one clear accessible name: task title plus result state.
- Detail drawer has an accessible title and focus returns to the opening result row/card.
- Status is represented by text and tag tone, never color alone.
- Automation Home module exposes status and owner as text, not only color or position.
- Running/loading states use `role="status"` and `aria-live="polite"`.
- Partial/error states preserve diagnostics labels already used by `RecoveryCallout`.
- Filter controls have visible text labels and `aria-label` where visible labels are not programmatically associated.
- Extension-width layout at 360px must keep filters, cards, and action buttons from overflowing.
- Desktop table view must keep stable columns and use horizontal scroll where needed.
- Result summaries truncate only after preserving full text in `title` or drawer detail.
- Deep-link-only actions use links; mutation actions use buttons.

## File Structure

### New Files

- `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-results.ts`
  - Result item types.
  - Projection from `ScheduledTask[]`.
  - Sort/filter helpers.
  - Result detail lookup by `result_id`, `run_id`, or `task_id`.
  - Home item adapter.
- `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-result-links.ts`
  - Pure URL and dedupe-key helpers for task, run, result, Home, and notification targets.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskResultsPanel.tsx`
  - Results list/table, filters, counts, empty states, partial capability note, and open-detail callbacks.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskResultDetailDrawer.tsx`
  - Result/run detail, provenance, owner links, retry/review actions only when available.
- `apps/packages/ui/src/components/Option/CompanionHome/cards/AutomationInboxCard.tsx`
  - Automation-specific Home module with compact status, owner, deep-link, and failure/result treatment.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts`
  - Pure projection, filtering, deep-link, and Home adapter coverage.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx`
  - Component coverage for list, filters, empty states, loading, failure, and action availability.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultDetailDrawer.test.tsx`
  - Drawer provenance, owner links, capability-aware actions, and accessibility coverage.

### Modified Files

- `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-route-state.ts`
  - Add `results` tab.
  - Parse/build `result_id` and `run_id`.
  - Preserve `task_id` for both Tasks and Results where relevant.
- `apps/tldw-frontend/extension/routes/route-registry.tsx`
  - Add `/scheduled-tasks/results` alias to the scheduled-tasks route element.
- `apps/tldw-frontend/pages/scheduled-tasks/results.tsx`
  - Add hosted WebUI alias that loads the existing scheduled-tasks route.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
  - Add Results tab.
  - Build projected result items from task data.
  - Route result deep links to detail drawer.
  - Add missing-result non-blocking alert.
  - Keep existing task detail route behavior intact.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskOverview.tsx`
  - Add newest result/needs-review summary or link to Results.
  - Keep existing total/attention/running/next-run cards readable.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskTable.tsx`
  - Add row action to view task results only when result signals exist.
  - Keep edit/delete behavior limited to native reminders.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskDetailDrawer.tsx`
  - Add latest result/run link when available.
  - Preserve Watchlists owner copy.
- `apps/packages/ui/src/components/Option/CompanionHome/CompanionHomePage.tsx`
  - Load scheduled-task Home signals independently from Companion personalization.
  - Render `AutomationInboxCard` near existing Home system cards.
  - Keep Home usable if scheduled-task result loading fails.
- `apps/packages/ui/src/components/Option/CompanionHome/hooks.ts`
  - Add a hook or adapter for non-personalized scheduled-task Home signals.
  - Avoid blocking `fetchCompanionHomeSnapshot`.
- `apps/packages/ui/src/services/companion-home.ts`
  - Extend `CompanionHomeSource` with a scheduled-task source.
  - Extend `CompanionHomeEntityType` with a scheduled-task result entity.
  - Keep existing Companion cards unchanged; automation rendering lives in `AutomationInboxCard`.
- `apps/packages/ui/src/services/notifications.ts`
  - Add pure helpers only if needed to normalize scheduled-task notification link targets.
  - Do not change notification stream subscription behavior.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts`
  - Update tab contract and deep-link builder assertions.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`
  - Add Results tab, result deep-link, missing result, partial state, and Watchlists-preservation tests.
- `apps/packages/ui/src/components/Option/CompanionHome/__tests__/CompanionHomePage.test.tsx`
  - Add Automation Inbox rendering, personalization-off behavior, partial failure, and link tests.
- `apps/packages/ui/src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx`
  - Add status, owner, result/failure, empty, and exact-link coverage.
- `apps/packages/ui/src/components/Option/CompanionHome/__tests__/CardShell.test.tsx`
  - Update only if scheduled-task status metadata changes card rendering.
- `apps/packages/ui/src/services/__tests__/notifications.test.ts`
  - Add scheduled-task link normalization coverage if `notifications.ts` gains helper functions.

### Files To Avoid Changing

- `apps/packages/ui/src/components/Option/Watchlists/**`
  - Exception: only add tests if a broken deep link requires proving an existing Watchlists query parameter contract.
- `apps/packages/ui/src/services/watchlists*.ts`
- `tldw_Server_API/**`
- Browser extension background networking unless a route/path allowlist rejects the new result URLs.

## Implementation Stages

## Stage 1: Result Projection Contract

**Goal:** Create a source-agnostic frontend result model that works with current task-list data and can later map to a normalized backend result API.

**Success Criteria:**

- `ScheduledTask[]` can project to stable `ScheduledTaskResultItem[]`.
- Watchlists result links reuse existing `buildWatchlistTaskLinks`.
- Result items have exact Home and `/scheduled-tasks` deep links.
- Projection never exposes raw unknown objects directly in UI copy.
- Capability mode is explicit and defaults to `projected_signals`.
- Mixed failure-plus-output states produce separate failure and result signals.

**Tests:**

- `scheduled-task-results.test.ts`
  - Projects found-results task into a new success signal in projected mode.
  - Projects failed task into an attention signal with recovery summary.
  - Projects Watchlists latest output/run links.
  - Excludes normal waiting tasks from default Home items.
  - Includes completed/no-results only under matching filters.
  - Sanitizes unsafe or empty source reference values.
  - Hides durable review semantics in `projected_signals`.
  - Produces both failure and result signals for a task with failure status and output ids.
  - Keeps fallback dedupe keys separate by `signalKind`.

**Tasks:**

- [x] Add `scheduled-task-results.ts` types and helpers.
- [x] Add `scheduled-task-result-links.ts` URL and dedupe helpers.
- [x] Add capability-mode detection helpers that can accept OpenAPI path availability.
- [x] Add result item sorting by newest `occurredAt`, then severity, then title.
- [x] Add `buildScheduledTaskResultHref` for `result_id`, `run_id`, and `task_id`.
- [x] Add `findScheduledTaskResultByRouteState`.
- [x] Add Home adapter returning automation inbox items with status and owner metadata.
- [x] Add mixed-state projection rules before React rendering work starts.
- [x] Run `bunx vitest run apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts`.

## Stage 2: Results Route State And Results Tab

**Goal:** Extend `/scheduled-tasks` IA with a Results tab and deep-linkable result drawer while preserving existing tab fallbacks.

**Success Criteria:**

- `?tab=results` is a first-class tab.
- `/scheduled-tasks/results` opens the same Results tab experience.
- `result_id`, `run_id`, and `task_id` are parsed and serialized safely.
- Invalid tabs still fall back to Overview.
- Existing `?tab=tasks&task_id=...` drawer behavior remains unchanged.

**Tests:**

- `scheduled-task-route-state.test.ts`
  - Accepts `results`.
  - Builds `?tab=results&result_id=...`.
  - Builds `?tab=results&run_id=...`.
  - Preserves task links for task-scoped result views.
  - Keeps invalid-tab fallback behavior.
- `ScheduledTasksPage.test.tsx`
  - Opens Results tab from URL.
  - Opens result drawer from result deep link after task data loads.
  - Shows missing-result alert without leaving Results.
- Browser smoke or route-level coverage:
  - `/scheduled-tasks/results` loads the scheduled-tasks page and selects Results.

**Tasks:**

- [x] Update `ScheduledTaskTabId` and `SCHEDULED_TASK_TABS`.
- [x] Extend `ScheduledTaskRouteState` with `resultId` and `runId`.
- [x] Update `buildScheduledTaskSearch`.
- [x] Add extension route alias for `/scheduled-tasks/results`.
- [x] Add hosted WebUI alias page for `/scheduled-tasks/results`.
- [x] Add Results tab rendering to `ScheduledTasksPage.tsx`.
- [x] Keep task-detail state separate from result-detail state.
- [x] Run the route-state and page tests.

## Stage 3: Results Panel And Detail Drawer

**Goal:** Build the main Phase 3 inspection workflow inside Scheduled Tasks.

**Success Criteria:**

- Users can scan result state, task, owner, last run/result time, and primary action.
- Filters support result/signal state, owner, and task type.
- Review-state filters appear only in normalized result modes.
- Drawer explains what happened, why it is shown, where it came from, and what can be done next.
- Retry/review actions are hidden unless capability mode and item availability support them.
- The drawer uses a short capability note instead of disabled mutation buttons when actions are unsupported.

**Tests:**

- `ScheduledTaskResultsPanel.test.tsx`
  - Renders success, failure, running, and completed/no-results items.
  - Filters by needs attention, task type, and owner in projected mode.
  - Filters by unreviewed/reviewed only in normalized result modes.
  - Hides Review state filter in `projected_signals`.
  - Shows all three empty states: no tasks, no results, no filter matches.
  - Emits accessible action names.
- `ScheduledTaskResultDetailDrawer.test.tsx`
  - Shows provenance and owner.
  - Shows match reason, matched rule label, output label, and domain deep link when present.
  - Shows Watchlists deep links for Watchlist-owned results.
  - Hides unsupported retry/review buttons and shows one capability note instead.
  - Shows retry/review buttons when capability mode and item availability allow them.
  - Uses role/dialog title that includes the result title.

**Tasks:**

- [x] Implement `ScheduledTaskResultsPanel.tsx`.
- [x] Implement `ScheduledTaskResultDetailDrawer.tsx`.
- [x] Implement capability-aware filter/action visibility.
- [x] Add list/table responsive behavior for extension width.
- [x] Add result action wiring in `ScheduledTasksPage.tsx`.
- [x] Add missing-result and partial-dependency alerts.
- [x] Run component tests.

## Stage 4: Notification Links And Dedupe

**Goal:** Normalize scheduled-task notification targets and prevent duplicate Home/notification result cards for the same run or result.

**Success Criteria:**

- Notification payloads with `link_url`, `source_task_id`, `source_task_run_id`, or source job metadata resolve to Scheduled Tasks result deep links when possible.
- Home notification-derived automation items come from `listNotifications({ limit: 50 })`.
- Home result items and notification-derived items share deterministic dedupe keys.
- Exact result ids win over run ids, and run ids win over task-scoped fallback links.
- Existing notification stream, mark-read, dismiss, and snooze behavior is unchanged.
- Notification loading failure does not block Scheduled Tasks or Home rendering.

**Tests:**

- `scheduled-task-results.test.ts`
  - Builds the same dedupe key for task projection and notification-derived result for the same run.
  - Keeps separate keys for separate runs from the same task.
  - Keeps separate fallback keys for failure and result signals from the same task/run when no exact result id exists.
- `notifications.test.ts`
  - Covers scheduled-task notification link helper if implemented in `notifications.ts`.
- `CompanionHomePage.test.tsx`
  - Keeps Automation Inbox rendered from task projection when notification loading fails.

**Tasks:**

- [x] Add notification target normalization to `scheduled-task-result-links.ts` or `notifications.ts`.
- [x] Add dedupe-key generation and merge helpers.
- [x] Add non-blocking recent notification load for Home automation signals.
- [x] Preserve existing notification service behavior.
- [x] Run scheduled-task result and notification service tests.

## Stage 5: Overview And Task Cross-Links

**Goal:** Make results discoverable from existing Scheduled Tasks views without slowing down task management.

**Success Criteria:**

- Overview shows newest results or signals and a needs-review count only when durable review state exists.
- Tasks table exposes a `Results` action only when result signals exist.
- Task drawer links to latest run/result when known.
- Watchlists copy remains clear that configuration is managed in Watchlists.

**Tests:**

- `ScheduledTasksPage.test.tsx`
  - Overview shows result summary and link to Results.
  - Tasks row with result signal has `View results`.
  - Watchlists row still has Watchlists settings/activity/reports links.
  - Reminder row still has native edit/delete.
- `ScheduledTaskDetailDrawer.test.tsx`
  - Latest result link appears when projection provides it.
  - Watchlists owner copy is unchanged.

**Tasks:**

- [x] Update `ScheduledTaskOverview.tsx`.
- [x] Update `ScheduledTaskTable.tsx`.
- [x] Update `ScheduledTaskDetailDrawer.tsx`.
- [x] Confirm no Watchlists-owned edit controls are introduced.
- [x] Run ScheduledTasks tests.

## Stage 6: Home Surfacing

**Goal:** Surface scheduled-task results and failures on Home without requiring Companion personalization and without overloading generic Companion cards.

**Success Criteria:**

- Home renders a dedicated `Automation Inbox` module with result and failure items.
- The automation module shows status text, owner, timestamp, summary, and exact deep link.
- Scheduled-task Home signals load independently of Companion personalization.
- Personalization-off Home can still show scheduled-task automation signals.
- If scheduled-task load fails, Companion Home still renders and shows available companion items.
- Home items deep-link to exact `/scheduled-tasks?tab=results...` targets.
- Existing Companion Inbox Preview and Needs Attention cards keep their current companion-centered behavior.

**Tests:**

- `CompanionHomePage.test.tsx`
  - Renders Automation Inbox with scheduled-task result item.
  - Renders Automation Inbox with scheduled-task failure item.
  - Renders scheduled-task items when `hasPersonalization` is false.
  - Keeps companion items when scheduled-task load fails.
  - Home links open exact scheduled-task result routes.
- `AutomationInboxCard.test.tsx`
  - Renders empty, loading, partial, result, failure, and mixed-state items.
  - Exposes status and owner as visible text.
- Pure helper tests in `scheduled-task-results.test.ts`
  - Maps result item to `CompanionHomeItem` using the scheduled-task source/entity type extensions.
  - Dedupes scheduled-task Home items against notification-derived items for the same run/result.

**Tasks:**

- [x] Add scheduled-task Home signal hook/adapter.
- [x] Extend `CompanionHomeSource` and `CompanionHomeEntityType` for scheduled-task result items.
- [x] Implement `AutomationInboxCard.tsx`.
- [x] Render `AutomationInboxCard` after `WhatsNextCard` and before generic Companion inbox/attention cards.
- [x] Merge task-projected and notification-derived automation items for the module.
- [x] Keep existing layout and Customize Home behavior unchanged.
- [x] Add loading and partial-failure behavior that does not block Home.
- [x] Run CompanionHome tests.

## Stage 7: UX Polish, Responsive Checks, And Copy Review

**Goal:** Verify the experience against the first-time and power-user workflows before implementation is considered complete.

**Success Criteria:**

- First-time flow explains no tasks, no results, and where future results appear.
- Power-user flow supports scanning, filtering, deep-linking, and owner handoff.
- Extension-width layout works at 360px and desktop layout works at 1280px+.
- Empty/error/loading/running/success states use final copy from this plan or stronger local copy.
- No generic source-vendor copy appears in generic Scheduled Tasks UI.
- Projected mode does not use durable-review language.
- Unsupported mutation actions are not visible in row actions.

**Tests And Checks:**

- Component tests above.
- Browser smoke after implementation if a dev server is practical:
  - `/scheduled-tasks`
  - `/scheduled-tasks/results`
  - `/scheduled-tasks?tab=results`
  - `/scheduled-tasks?tab=results&task_id=...`
  - Home route used by the options UI.
- Manual checks:
  - Keyboard tab through Results filters, result list, drawer actions.
  - 360px width overflow check.
  - Color-independent status comprehension.

**Tasks:**

- [ ] Review all new visible copy against copy recommendations.
- [ ] Search for source-specific examples in generic UI.
- [ ] Search for unsupported review/retry buttons in projected-mode rendering.
- [ ] Verify drawer focus and accessible labels.
- [ ] Verify no nested cards or decorative layout drift in Home.
- [ ] Record screenshots or browser observations in the implementation Backlog task.

## Stage 8: Documentation, Backlog, And Verification

**Goal:** Finish with a clean, reviewable implementation branch.

**Success Criteria:**

- Implementation Backlog task links this plan and records touched files.
- Tests pass for touched ScheduledTasks and CompanionHome areas.
- Bandit is run for touched backend scope if backend files are touched; otherwise record docs/frontend-only skip rationale.
- Final summary explains what changed and why.

**Commands:**

```bash
bunx vitest run apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts
bunx vitest run apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts
bunx vitest run apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx
bunx vitest run apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultDetailDrawer.test.tsx
bunx vitest run apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
bunx vitest run apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx
bunx vitest run apps/packages/ui/src/components/Option/CompanionHome/__tests__/CompanionHomePage.test.tsx
bunx vitest run apps/packages/ui/src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx
bunx vitest run apps/packages/ui/src/services/__tests__/notifications.test.ts
git diff --check
```

Bandit:

```bash
source .venv/bin/activate
python -m bandit -r <backend_touched_paths> -f json -o /tmp/bandit_scheduled_tasks_phase3.json
```

If implementation touches only frontend and docs, record: `Bandit skipped because no Python/backend files were changed.`

## Implementation Notes For Future Workers

- Start with pure projection tests before rendering. It is easier to stabilize UX states when result semantics are tested outside React.
- Prefer additive route-state changes. Do not replace existing `task_id` behavior for the Tasks tab.
- Keep task state and result state separate. A task can be waiting while its latest result is unreviewed, or failed while an older result still exists.
- Keep Home loading independent. Scheduled-task results should improve Home, not make it fragile.
- Use the query-tab URL shape internally, but ship `/scheduled-tasks/results` as an alias so PRD, Home, and notification links stay stable.
- Use Watchlists deep links for source-owned details; do not duplicate Watchlists reports or run logs.
- Deduplicate by exact result id first, exact run id second, and task/time/state fallback last.
- Treat backend retry/review endpoints as capability-gated actions. Hide unsupported mutation buttons in lists; use one concise capability note in details instead of disabled-button clutter.
- Keep source names generic in default UI and let task titles/source labels carry domain specificity.

## PR Review Checklist

- [ ] `/scheduled-tasks?tab=results` is discoverable from Overview, Tasks, Home, and direct URL.
- [ ] `/scheduled-tasks/results` aliases to the same Results experience.
- [ ] Result drawer answers: what happened, when, where from, owning workspace, next action.
- [ ] Home shows an Automation Inbox even when Companion personalization is not enabled.
- [ ] Watchlists UX remains intact and domain configuration is not moved.
- [ ] Result and failure states are understandable without color.
- [ ] Extension-width layout has no horizontal overflow except intentional tables.
- [ ] No backend-only promise appears in frontend copy before the backend supports it.
- [ ] Projected mode does not claim durable review state or expose unsupported retry/review buttons.
- [ ] Mixed failure-plus-output states preserve both signals.
- [ ] Tests cover missing deep links and partial dependency failures.
- [ ] Backlog task records verification, known skips, and final summary.
