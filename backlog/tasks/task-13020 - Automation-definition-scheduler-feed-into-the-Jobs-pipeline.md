---
id: TASK-13020
title: 'Automation definition scheduler feed into the Jobs pipeline'
status: To Do
assignee: []
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
- [ ] #1 An env-gated scheduler service arms only `configured` definitions and enqueues one Job per due occurrence with a payload carrying definition id, owner, and run slot (timestamps UTC, mirroring reminders' payload shape)
- [ ] #2 All five validated schedule kinds compute next occurrences correctly (property/parameterized tests per kind, incl. timezone handling consistent with the reminders scheduler)
- [ ] #3 Paused/archived/disabled definitions are never armed; resume/pause transitions re-arm/disarm without restart
- [ ] #4 Definition `health` reflects armed reality (`ready` when armed; `execution_unavailable` only when execution genuinely cannot run), visible through the existing control-plane endpoints
- [ ] #5 Next-run bookkeeping is durable and idempotent against duplicate scheduler passes (no double-enqueue of the same run slot)
- [ ] #6 Tests cover arming, kind computation, lifecycle transitions, and idempotency against the real ScheduledTasksDatabase + JobManager shapes (no reimplementation)
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
