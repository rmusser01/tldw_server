# Scheduled Tasks Automation Workbench PRD Design

Date: 2026-06-01
Status: Ready for user review
Owner: Codex brainstorming session
Backlog: TASK-494

## Summary

Create a full target UX for `/scheduled-tasks` as an Automation Workbench: a cross-domain hub where users can create, understand, monitor, debug, and act on recurring or deferred automations across reminders, Watchlists jobs, recurring RAG queries, ACP/agent schedules, and browser-extension-created page watches.

The workbench should not replace existing domain workspaces. Watchlists remains the first-class workspace for source curation, scraping, ingestion, monitor configuration, outputs, digests, and reports. Agent Tasks/ACP remains the deep workspace for agent sessions, artifacts, governance, and diagnostics. RAG and knowledge surfaces remain the deep workspace for retrieval configuration and source inspection. `/scheduled-tasks` owns the shared automation lifecycle and results-discovery layer.

Home should become a lightweight automation inbox that surfaces recent results, failures, running work, and simple recovery actions. The browser extension should support context-aware scheduled-task creation from the current page and then hand off to the shared creation flow.

This PRD is product/UX focused. It intentionally references backend, API, and data-model work only as dependencies.

## Product Decision

Use a Unified Automation Workbench model.

`/scheduled-tasks` should be the canonical product place for:

1. Discovering what can be automated.
2. Creating common automations from templates.
3. Creating advanced automations from a power-user path.
4. Seeing all scheduled automations and their current state.
5. Inspecting run history, failures, logs, and outputs.
6. Acting on results and recovering from failures.

Domain surfaces still own deep editing:

| Surface | Role after this PRD |
| --- | --- |
| `/scheduled-tasks` | Unified automation creation, status, run history, result discovery, and common actions |
| Home | Lightweight automation inbox and status surfacing |
| Watchlists | Full source, monitor, scrape, ingest, output, report, digest, and delivery workspace |
| Agent Tasks / ACP | Full agent task, session, artifact, governance, and diagnostics workspace |
| RAG / Knowledge surfaces | Full retrieval scope, source, citation, and answer-inspection workspace |
| Browser extension | Context-aware creation from current page plus shared workbench access |

## Current Context

Source review found these implementation facts:

- `/scheduled-tasks` currently lists normalized reminder tasks and Watchlists jobs.
- Reminder tasks are editable from `/scheduled-tasks`; Watchlists jobs are shown as externally managed and link to Watchlists.
- The current create path is a reminder-only form with raw ISO datetime and cron inputs.
- The scheduled-tasks table does not expose a full task detail, run history, result inbox, live running state, logs, deep result links, bulk actions, duplicate, dry run, or run-now actions.
- Watchlists already contains stronger schedule, preview, run, activity, item, and output concepts than `/scheduled-tasks`.
- ACP schedules exist in backend/API documentation, but are not visibly integrated into `/scheduled-tasks`.
- Companion Home and notifications surface job/reminder activity, but scheduled-task results are not first-class home content with exact run/result deep links.
- The extension route reuses the same scheduled-tasks page, but does not currently provide context-aware creation from the current page.

## Problem

The current `/scheduled-tasks` page does not match user expectations for a scheduled automation hub.

First-time users do not know:

- what scheduled tasks are;
- what they can automate;
- whether GitHub, YouTube, RAG, and agent tasks are supported;
- how to create a task safely without knowing cron or product internals;
- where results will appear after a task runs.

Power users cannot efficiently:

- inspect many automations;
- filter by status, task type, source, owner, or next run;
- pause/resume, duplicate, dry run, run now, or bulk manage tasks;
- debug failed runs from the same place;
- jump directly from a notification or Home result to the exact run/output.

The product also has a cross-surface IA problem: Watchlists, reminders, notifications, Agent Tasks, RAG, Home, and extension capture each expose parts of the automation story, but the shared lifecycle is not visible in one place.

## Goals

1. Make `/scheduled-tasks` the user's reliable answer to "what is running later or repeatedly?"
2. Let first-time users create useful automations from plain-language templates without knowing cron, Watchlists internals, ACP schedules, or RAG configuration.
3. Let power users manage many automations quickly with dense tables, filters, bulk actions, duplication, run-now, dry-run, and detailed run inspection.
4. Preserve Watchlists as a separate first-class workspace and persona/job fit.
5. Surface scheduled-task results and failures on Home without turning Home into the full management console.
6. Support browser-extension context-aware creation from current pages such as GitHub repos, YouTube channels, RSS feeds, and articles.
7. Make every task state, run, result, failure, and output traceable enough to support trust and recovery.
8. Harmonize terminology across scheduled tasks, watchlist jobs, reminders, ACP schedules, runs, results, outputs, and notifications.

## Non-Goals

- Do not replace, remove, or limit Watchlists functionality.
- Do not collapse Watchlists into `/scheduled-tasks`.
- Do not deeply redesign Agent Tasks, ACP Playground, RAG, Knowledge, or Watchlists in this PRD.
- Do not fully specify backend schemas, table migrations, or endpoint payloads.
- Do not require all automation types to ship at once.
- Do not hide existing specialized entry points.
- Do not build a generic workflow builder in this PRD.

## Personas

### First-Time Automation User

Wants a practical recurring outcome, such as "tell me when a non-bot GitHub issue appears" or "ingest new videos from this channel." They need templates, preview, schedule help, visible results, and safe defaults.

### Researcher / Analyst

Uses scheduled tasks to watch sources, run unanswered questions across newly ingested material, and receive evidence-grounded alerts. They need provenance, citations, confidence, review state, and "mark solved" behavior.

### Watchlists Power User

Already uses Watchlists for sources, monitors, reports, digests, and outputs. They need `/scheduled-tasks` to summarize and operate on schedule/run state without weakening the deeper Watchlists workflow.

### Agent / ACP Power User

Schedules recurring agent prompts or deferred agent actions. They need agent/workspace selection, action preview, governance visibility, run transcripts, artifacts, and failure recovery.

### Operator

Owns reliability. They need to know what is queued, running, failed, blocked, stale, or disabled, and which dependency caused the problem.

## Product Principles

1. **One automation lifecycle, many domain owners.** `/scheduled-tasks` owns common lifecycle and visibility; domain workspaces own deep configuration.
2. **Create from intent, not implementation.** First-time users should choose "GitHub issue monitor," not "Watchlist RSS job with filters."
3. **Every result explains why it exists.** Results must show task, run, source, matched rule/query, timestamp, and output link.
4. **Preserve expert surfaces.** Watchlists and ACP are not simplified away.
5. **Status is a product object.** Users should not infer state from color, timestamps, or raw backend labels.
6. **Preview before automation.** Risky or noisy recurring work should be previewable before save.
7. **Home is for triage, not configuration.** Home should surface and lightly act; `/scheduled-tasks` should inspect and manage.

## Target Information Architecture

`/scheduled-tasks` should use a workbench IA:

| Area | Purpose |
| --- | --- |
| Overview | Operational summary: running now, needs attention, latest results, upcoming important runs |
| Tasks | Unified searchable table of all automations with filters, sorting, and bulk actions |
| Create | Template gallery plus advanced task option |
| Runs | Cross-task execution history, logs, retries, cancellation, export |
| Results | Scheduled-task result inbox with review state, provenance, and exact output links |
| Templates | Saved task recipes and recommended starter automations |
| Settings | Notification defaults, timezone, retention, concurrency, integration health |

Task detail should be reachable from any task row, run, result, Home card, or notification.

Task detail sections:

| Section | Content |
| --- | --- |
| Summary | Task name, type, status, owner, schedule, next run, last run, current action |
| Inputs | Source/repo/channel/query/agent/workspace and relevant filters |
| Schedule | Plain-language cadence, timezone, next-run preview, pause/resume |
| Results | Latest outputs, matches, ingested items, agent artifacts, review state |
| Runs | Run history with status, duration, counts, logs, retry/cancel where safe |
| Notifications | Current notification policy and last delivered alert |
| Advanced | Deep links to Watchlists, ACP, RAG, or raw domain configuration |

## Creation Model

The workbench should support dual entry:

1. **Template gallery** for common jobs.
2. **Advanced task** for power users who know the domain and schedule model.

Required templates:

| Template | Primary user intent | First-time path | Advanced controls |
| --- | --- | --- | --- |
| GitHub issue monitor | "Tell me when relevant non-bot issues appear" | Repo URL, non-bot default, preview, schedule, notification | labels, authors, bot heuristics, title/body filters, threshold, delivery |
| YouTube channel ingest | "Keep this channel searchable" | Channel URL, latest-video preview, ingest/search options, schedule | max videos, transcript policy, duplicate handling, media DB/RAG indexing, failure policy |
| Recurring RAG query | "Keep looking for an answer as new data arrives" | Question, source scope, confidence threshold, schedule, notify rule | query variants, retrieval profile, reranker, min confidence, mark-solved policy |
| Scheduled agent message | "Send this prompt to this agent later or repeatedly" | agent/workspace, message, action preview, schedule | model, token budget, sandbox, concurrency, persona, governance, artifacts |
| Reminder | "Remind me later or repeatedly" | title/body, date/time or recurrence, notification | cron, timezone, expiration, snooze |

Each template should include:

- purpose statement;
- required setup/dependency health;
- input validation;
- preview;
- schedule with next-run preview;
- notification/result destination;
- review screen;
- create and run-now option when safe.

## Workflow Requirements

### GitHub Issue Monitor

The target UX should let a user paste a GitHub repo or issues URL, preview recent issues, exclude bot users by default, optionally filter by label/author/title/body, and choose notification behavior.

The resulting task should show:

- repo;
- filters;
- last checked time;
- new matching issues;
- ignored bot/system issues when available;
- failures such as rate limit, private repo, auth missing, or invalid URL.

Dependency notes:

- Product can be implemented through Watchlists, a GitHub connector, or a dedicated backend monitor later.
- PRD does not mandate the backend path.

### YouTube Channel Ingest

The target UX should let a user paste a YouTube channel, playlist, or feed URL and see a resolved channel preview before scheduling.

The resulting task should show:

- latest videos checked;
- videos ingested;
- videos skipped as duplicates;
- transcript/download/indexing failures;
- links to ingested media and search/RAG surfaces.

Dependency notes:

- Current Watchlists feed behavior can be reused where practical.
- The UX should not require users to know canonical YouTube RSS feed formats.

### Recurring RAG Query

The target UX should let a user schedule a question that has not yet been answered, run it repeatedly against newly ingested or selected data, and notify when promising matches appear.

The resulting task should show:

- question;
- source scope;
- retrieval profile;
- latest match summary;
- citations and source links;
- confidence/relevance signal;
- review state: new, reviewed, dismissed, solved.

Required UX behavior:

- Provide "mark solved" to pause or complete the task.
- Provide "keep watching" after dismissing weak matches.
- Explain why a match was surfaced.

Dependency notes:

- Requires a product-level recurring RAG task primitive or a domain schedule integration.

### Scheduled Chat / Agentic Task

The target UX should let a user schedule a specific message to a specific API/ACP agent at a specific time or cadence.

The creation flow should include:

- agent selection;
- workspace/context selection;
- message composer;
- optional model/persona/token budget;
- sandbox/governance visibility;
- action preview;
- schedule;
- result destination.

The resulting task should show:

- run transcript or summary;
- artifacts;
- session links;
- failure reason;
- retry/fork/deep-diagnose links into ACP/Agent Tasks.

Dependency notes:

- Backend ACP schedules exist; `/scheduled-tasks` needs product integration and normalized visibility.

## Home Surfacing Model

Home should include a lightweight automation inbox. It should not replace `/scheduled-tasks`.

Home modules:

| Module | Shows | Lightweight actions |
| --- | --- | --- |
| Needs attention | Failed runs, blocked tasks, paused-by-error tasks, auth problems | Retry, View logs, Edit task |
| Latest automation results | GitHub issues, ingested videos, RAG matches, agent outputs, reminders | Review, Dismiss, Open result |
| Running now | Active tasks with current step and elapsed time | Open run, Cancel if safe |
| Upcoming | Pinned or important next runs | Open task, Pause |

Home cards should deep-link to exact task runs/results, not only broad destinations such as Watchlists.

Home should dedupe notifications and result cards so users are not forced to triage the same event twice.

## Results Model

Every result should carry product-level provenance:

| Field | Purpose |
| --- | --- |
| Task name and type | Explains why the result exists |
| Run ID and timestamp | Supports debugging and auditability |
| Source/query/agent | Identifies the origin |
| Matched rule/filter/query | Explains why it surfaced |
| Summary | Makes triage fast |
| Output/artifact links | Lets the user act |
| Review state | Prevents repeated review burden |
| Domain deep link | Opens Watchlists, RAG, ACP, or media detail when deeper inspection is needed |

Review states:

- new;
- reviewed;
- dismissed;
- solved;
- failed;
- archived.

## Status Model

Use plain-language statuses across the workbench:

| Status | Meaning |
| --- | --- |
| Waiting for next run | Scheduled and healthy |
| Running now | Work is active and has progress |
| Completed last run | Last run finished successfully without reviewable result |
| Found results | Last run produced reviewable output |
| Needs attention | Last run failed or requires user action |
| Paused | User paused the task |
| Disabled | Not scheduled to run |
| Blocked | Missing auth, source unavailable, dependency not ready, or policy denied |
| Draft | Created but not scheduled |

Status UI requirements:

- never rely on color alone;
- show last run and next run where applicable;
- show dependency health when blocked;
- distinguish task status from run status;
- make "paused" user-controlled and "blocked" system/dependency-controlled.

## Extension UX

The browser extension should support context-aware task creation from the current page.

Supported target contexts:

| Current page | Suggested task |
| --- | --- |
| GitHub repository or issues page | GitHub issue monitor |
| YouTube channel or playlist | YouTube channel ingest |
| RSS/feed-like page | Source monitor or Watchlists-backed monitor |
| Article or site homepage | Site monitor |
| Search/results/research context | Recurring RAG query |
| Agent/workspace context | Scheduled agent message where applicable |

Extension behavior:

- show "Create scheduled task from this page" when the page context is recognized;
- prefill source URL and inferred task type;
- show confidence when the inference is uncertain;
- hand off to the shared scheduled-task creation wizard;
- keep advanced editing in the WebUI or appropriate domain workspace;
- preserve parity with the WebUI workbench for listing, inspecting, and opening tasks.

The extension should not duplicate full Watchlists, RAG, or ACP workspaces.

## Preservation Rules

| Existing surface | Rule |
| --- | --- |
| Watchlists | Preserve full current UX and capability. `/scheduled-tasks` may summarize, run, pause, inspect, and deep-link, but deep source/output editing remains in Watchlists. |
| Notifications | Keep notification preferences, but exact event links should point to task run/result details where available. |
| Agent Tasks | Preserve deep agent run/session workflows. `/scheduled-tasks` surfaces scheduled state and links to session diagnostics. |
| RAG / Knowledge | Preserve deep retrieval/source workflows. `/scheduled-tasks` surfaces scheduled query state and links to citations/source detail. |
| Reminders | Keep reminders as a simple automation type, but upgrade creation to safer date/time and recurrence controls. |

## Error Prevention And Recovery

| Risk | UX requirement |
| --- | --- |
| Bad schedule or timezone | Show next-run preview, timezone, and minimum interval warnings before save |
| Noisy automation | Provide preview, result thresholds, notification tuning, and dismiss/solve controls |
| Missing credentials | Show dependency health before save and exact recovery action |
| Partial domain outage | Preserve available task types and show domain-specific unavailable states |
| Failed run | Show failure reason, failed step, logs, retry, edit, and disable/pause actions |
| Duplicate ingestion | Show duplicate/skipped counts and explain duplicate policy |
| Agent action surprise | Show action preview, workspace, sandbox/governance, and result destination before scheduling |

## Empty, Loading, Running, Success, And Error States

| State | Required UX |
| --- | --- |
| Empty | Explain what scheduled tasks automate; show template cards and "Advanced task" |
| Loading | Skeleton overview/table with "Loading tasks and latest run state" |
| Partial | Explain which task families loaded and which did not |
| Running | Show current step, elapsed time, heartbeat/last update, and safe cancel if supported |
| Success without results | "Last run completed. No new results." Include last checked time |
| Success with results | "Found X new results." Link to results and domain detail |
| Failure | Plain-language cause, failed step, retry, edit, logs, and dependency recovery |
| Blocked | Missing auth/source/provider/policy with direct recovery link |

## UX Copy Recommendations

Preferred labels:

| Avoid | Use |
| --- | --- |
| Create Reminder Task | Create scheduled task |
| Native | Managed here |
| External managed | Managed in Watchlists |
| Cron | Custom schedule |
| Run at | Run once at |
| Enabled | Task is active |
| scheduled | Waiting for next run |
| disabled | Disabled |
| failed | Needs attention |
| Manage in Watchlists | Open monitor settings |

Required helper copy:

- "Results will appear on Home and in this task's Results tab."
- "Next run: [date/time/timezone]."
- "This task is managed in Watchlists. You can inspect runs and results here, or open Watchlists for source and output settings."
- "Preview shows what would have matched recently. It does not create results."
- "Mark solved pauses this recurring search unless you choose to keep watching."

## Accessibility And Usability Requirements

- All task, run, and result actions must be keyboard operable.
- Status must use text and icon, not color alone.
- Icon-only controls require accessible names and tooltips.
- Running updates should use polite live regions.
- Focus should move predictably after create, edit, retry, pause, delete, or dismiss.
- Destructive actions need confirmation or undo.
- Tables need responsive alternatives for extension-sized windows.
- Date/time and recurrence controls need validation that is announced to assistive technology.
- Timezone should default to the user's locale and remain visible in previews.
- Reduced-motion preferences should be respected for live progress and transitions.

## Success Metrics

Track first-time and power-user success separately.

### First-Time Metrics

- Percentage of first-time users who create a task from a template without opening advanced settings.
- Template creation completion rate.
- Preview-to-create conversion rate.
- Percentage of created tasks with a successful first run.
- Percentage of users who can find the first result from Home or `/scheduled-tasks`.
- Reduction in raw cron/ISO datetime validation errors.

### Power-User Metrics

- Median time to find a failed task and open its logs.
- Median time to duplicate and modify an existing task.
- Bulk pause/resume usage.
- Run-now/dry-run usage.
- Percentage of notifications/Home cards that deep-link to exact run/result detail.
- Number of active automations managed per user without increased failure-triage time.

### Trust And Recovery Metrics

- Failed runs with actionable reason.
- Blocked tasks with explicit recovery action.
- Results with complete provenance fields.
- Retry success after recoverable failure.
- Dismissed/solved recurring RAG results that do not resurface unnecessarily.

## Backend And API Dependencies

This PRD does not specify backend implementation details, but the target UX depends on product-facing capabilities:

- normalized scheduled-task list covering reminders, Watchlists jobs, recurring RAG queries, ACP/agent schedules, and extension-created watches;
- normalized task detail with current state, schedule, last run, next run, and domain deep links;
- normalized run history and run detail with logs, counts, failure reason, and artifacts;
- normalized result inbox model with provenance and review state;
- domain capability reporting so unavailable task families show recoverable states;
- notification and Home event links that preserve exact task/run/result targets;
- safe actions where supported: pause, resume, run now, dry run, duplicate, retry, cancel, dismiss, mark solved;
- extension context detection and prefilled creation handoff;
- consistent auth and permission behavior across WebUI and extension.

Backend implementation may reuse existing Watchlists, Reminders, Jobs, Scheduler, ACP, RAG, Notifications, and Companion Home systems. The UX does not require a single storage model if the product-facing contracts are coherent.

## Phased Delivery

### Phase 1: Unified Visibility And IA

Outcome: `/scheduled-tasks` becomes a trustworthy overview for existing task families.

Scope:

- workbench IA shell;
- overview cards;
- improved unified task table;
- status taxonomy;
- next/last run columns;
- task detail for reminders and Watchlists jobs;
- exact links to Watchlists run/result pages where available;
- safer reminder schedule controls;
- empty, partial, loading, and failure states.

### Phase 2: Template Creation For Existing Capabilities

Outcome: first-time users can create useful automations without implementation knowledge.

Scope:

- template gallery;
- reminder template;
- GitHub issue monitor template where backend support exists or can map to Watchlists/connector dependency;
- YouTube channel ingest template where backend support exists or can map to Watchlists;
- advanced task entry;
- preview/review/next-run pattern.

### Phase 3: Results Inbox And Home Surfacing

Outcome: users can find and triage scheduled-task outcomes from Home and `/scheduled-tasks`.

Scope:

- `/scheduled-tasks/results`;
- Home automation inbox modules;
- deep links from notifications/Home to exact runs/results;
- result review state;
- provenance display;
- failure and retry actions.

### Phase 4: Recurring RAG And ACP/Agent Schedule Integration

Outcome: RAG queries and ACP schedules become visible, creatable, and inspectable in the workbench.

Scope:

- recurring RAG query template;
- scheduled agent message template;
- ACP schedule visibility and deep links;
- RAG result review states including mark solved;
- agent run transcript/artifact links;
- capability states for missing providers, agents, or source scopes.

### Phase 5: Extension Context-Aware Creation And Power-User Management

Outcome: extension and dense management workflows support advanced users.

Scope:

- extension page-context detection;
- "Create scheduled task from this page";
- prefilled creation handoff;
- task table saved views;
- bulk actions;
- duplicate;
- dry run;
- export;
- keyboard-friendly dense management.

## Acceptance Criteria

- `/scheduled-tasks` clearly explains what scheduled automations are and what users can create.
- A first-time user can start from templates for GitHub issues, YouTube ingest, recurring RAG query, scheduled agent message, and reminder.
- Watchlists remains a full independent workspace; the PRD does not remove or limit existing Watchlists capabilities.
- Watchlists jobs shown in `/scheduled-tasks` deep-link to Watchlists for domain configuration.
- Task detail shows current state, schedule, next run, last run, run history, failures, logs, and result links where available.
- Home surfaces scheduled-task results and failures with lightweight actions and exact deep links.
- Extension supports context-aware creation from current supported pages.
- Status labels distinguish waiting, running, completed, found results, needs attention, paused, disabled, blocked, and draft.
- Creation flows include preview, schedule next-run preview, destination/result explanation, and review before save.
- Power users have search, filters, sort, duplicate, run now, dry run, pause/resume, retry, and bulk management in target UX.
- Accessibility requirements cover keyboard operation, status semantics, focus management, screen-reader announcements, and responsive extension-sized layouts.

## Open Questions For Implementation Planning

1. Which backend primitive should own GitHub issue monitoring first: Watchlists, a GitHub connector, or a dedicated monitor?
2. Which YouTube ingest path should be preferred: Watchlists feed resolution, media ingestion jobs, or a dedicated scheduled ingest primitive?
3. Should recurring RAG query results live in a new result inbox model, an existing notification/activity model, or a domain-specific RAG watch table with a normalized projection?
4. Which ACP schedule actions are safe from `/scheduled-tasks` versus requiring deep navigation to ACP/Agent Tasks?
5. What retention policy should apply to scheduled-task results and run logs across task families?
6. Which extension page contexts can be recognized reliably without extra permissions?

## Verification Plan For Future Implementation

This PRD is documentation-only. Future implementation plans should include:

- unit tests for task status mapping, copy, filters, and capability states;
- component tests for empty/loading/partial/error/result states;
- service contract tests for normalized task, run, and result payloads;
- E2E flows for GitHub issue monitor, YouTube ingest, recurring RAG query, scheduled agent message, reminder creation, Home result review, and extension context creation;
- accessibility checks for keyboard navigation, focus, labels, status semantics, and reduced-motion behavior;
- live-backend manual verification with seeded tasks and failure states.

## Spec Review

Reviewed by a spec-document-reviewer subagent on 2026-06-01.

Status: Approved.

Blocking issues: None.

Advisory recommendations for implementation planning:

- Treat the open questions as explicit planning decisions.
- Plan by phase rather than as one release.
- Define phase-specific acceptance criteria so the global acceptance criteria do not imply all automation types must ship together.
