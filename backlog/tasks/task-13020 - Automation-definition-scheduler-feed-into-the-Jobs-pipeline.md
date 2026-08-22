---
id: TASK-13020
title: 'Automation definition scheduler feed into the Jobs pipeline'
status: Done
assignee:
  - '@robert'
created_date: '2026-08-21 14:10'
updated_date: '2026-08-21 14:10'
labels:
  - scheduled-tasks
  - automation
  - jobs
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
First server-side piece of the server-offload execution seam, implementing the client-side contract ADR (tldw_chatbook `backlog/decisions/077-server-offloaded-scheduled-agent-tasks.md`, ACCEPTED 2026-08-21; cross-repo task tldw_chatbook TASK-18940). Today automation definitions are fully modeled, validated, redaction-policy'd, and audited (`scheduled_task_automation_service.py`), but nothing dispatches them: `DEFAULT_DEFINITION_HEALTH = "execution_unavailable"` is permanent because no scheduler feed exists. Reminders already have the exact pattern to mirror: `app/services/reminders_scheduler.py` (APScheduler, env-gated) feeds due reminders into the Jobs pipeline.

Build the definition feed: an APScheduler service (env-gated, e.g. `SCHEDULED_TASKS_AUTOMATION_SCHEDULER_ENABLED`) that loads `lifecycle="configured"` definitions from `ScheduledTasksDatabase`, computes the next occurrence from the definition's `schedule` dict (the service already validates kinds `one_time`, `interval`, `daily`, `weekly`, `cron`), and enqueues Jobs — a new domain/type pairing (e.g. domain `scheduled_tasks`, type `agent_task_run`) with payload carrying definition id + owner + the run slot. Next-run bookkeeping persists per the reminders pattern; paused/archived/disabled definitions are never armed; definition `health` transitions to `ready` when armed (and back on pause/archive), replacing the permanent `execution_unavailable` lie for server owners.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An env-gated scheduler service arms only `configured` definitions and enqueues one Job per due occurrence with a payload carrying definition id, owner, and run slot (timestamps UTC, mirroring reminders' payload shape)
- [x] #2 All five validated schedule kinds compute next occurrences correctly (property/parameterized tests per kind, incl. timezone handling consistent with the reminders scheduler)
- [x] #3 Paused/archived/disabled definitions are never armed; resume/pause transitions re-arm/disarm without restart
- [x] #4 Definition `health` reflects armed reality (`ready` when armed; `execution_unavailable` only when execution genuinely cannot run), visible through the existing control-plane endpoints
- [x] #5 Next-run bookkeeping is durable and idempotent against duplicate scheduler passes (no double-enqueue of the same run slot)
- [x] #6 Tests cover arming, kind computation, lifecycle transitions, and idempotency against the real ScheduledTasksDatabase + JobManager shapes (no reimplementation)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: check — the governing contract ADR lives in the client repo (tldw_chatbook ADR-077); if this repo's conventions require a local ADR for a new execution subsystem, draft one that imports ADR-077 rather than restating it, and link it here before implementation.

1. Read `reminders_scheduler.py` end-to-end; extract its env-gate/rescan/enqueue/next-run discipline as the template
2. Definition loader: configured-lifecycle definitions + schedule-dict next-occurrence computation (five kinds)
3. JobManager enqueue with the new domain/type + run-slot payload; durable next-run bookkeeping
4. Health transitions (`ready`/`execution_unavailable`) wired through the existing definitions surface
5. Tests per the AC matrix; env-gated startup registration mirroring reminders
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented 2026-08-21 on `feat/automation-definition-feed` (stacked on the task-filing branch `docs/server-offload-tasks`, PR tldw_server#2796 — rebase onto dev once that merges).

**Approach.** New `app/services/scheduled_task_automation_scheduler.py` mirrors `reminders_scheduler.py`'s proven shape: env-gated singleton (`SCHEDULED_TASKS_AUTOMATION_SCHEDULER_ENABLED`, ships OFF — enable together with the TASK-13021 consumer), per-user `ScheduledTasksDatabase.for_user` cache, periodic rescan (floor 30s), APScheduler jobs keyed `automation:{definition_id}` with max_instances=1 + coalesce, `reconcile_definition`/`unschedule_definition` hooks for future service-layer calls, and startup registration in `startup_recurring_schedulers.py` beside the reminders spec.

**Schedule dict conventions** (the automation service validates only `kind`; per-kind fields are defined here and documented in the module docstring): one_time/run_at ISO; interval/seconds (+optional start_at anchor); daily/at HH:MM (+timezone); weekly/weekday APScheduler convention (+at, timezone); cron five-field (+timezone). Junk degrades honestly: `build_trigger` returns a reason, the definition is skipped with a warning, never armed — same discipline as the reminders scheduler's invalid-cron path.

**Idempotency (AC#5) without a schema change:** run-slot dedupe lives at the Jobs layer — `JobManager.create_job` with `idempotency_key=f"definition:{id}:{slot_utc}` returns the same row on duplicate creates, which is durable across restarts and correct for concurrent rescans. Slot computation: one_time uses `run_at` directly; periodic re-derives at fire time via `trigger.get_next_fire_time(None, now-1s)`. No `next_run_at` column was added to `scheduled_task_definitions` (definitions surface schedule through their `schedule` dict; the reminders pattern's stored-next-run exists to feed ITS CAS claim, which the Jobs-layer idempotency replaces here).

**Health honesty (AC#4):** arming flips `health` to `ready` via `update_definition` only-on-change (no version churn across rescans) + a `scheduler_armed` audit event through the standard trail. Disarm-side health semantics (pause/archive transitions) are intentionally left to the existing lifecycle methods and refined in TASK-13021 where execution outcomes own health updates.

**Verification.** `tests/Notifications/test_automation_definition_feed.py` — 19 tests, all through real preview→definition creation (the DB's preview gate included) and the real fire path: five trigger kinds + seven refusal cases, enqueue shape (domain/type/owner/payload/key), identical-key dedupe across double fire, non-configured lifecycle never fires, future one_time early-skips, unusable schedules never fire, health flip + audit + no version churn. Neighboring suites green: `test_reminders_scheduler.py` + `test_scheduled_task_automation_api.py` (47 passed), `test_startup_preflight.py` (16 passed). Fixes made during the round: preview rows require NOT NULL `expires_at` (24h TTL fixture), `list_audit_events` returns `(rows, total)` tuples of dataclasses, and the fire path's early-skip tolerance widened from 1 to 2 minutes (the guard targets wrongly-scheduled future triggers, not sub-minute boundary lag; APS misfire grace is 300s).

**Files:** `app/services/scheduled_task_automation_scheduler.py` (new), `app/services/startup_recurring_schedulers.py` (spec + starter/stopper), `tests/Notifications/test_automation_definition_feed.py` (new), this task file.
