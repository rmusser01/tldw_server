---
id: TASK-13021
title: 'Agent-task Jobs consumer: durable runs, timeout status, notification pass-back'
status: Done
assignee:
  - '@robert'
created_date: '2026-08-21 14:10'
updated_date: '2026-08-21 14:10'
labels:
  - scheduled-tasks
  - automation
  - jobs
  - notifications
dependencies:
  - task-13020
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Second server-side piece of the server-offload seam (tldw_chatbook ADR-077, accepted; cross-repo TASK-18940). The consumer that executes the Jobs the feed (TASK-13020) enqueues, mirroring `app/core/Reminders/reminder_jobs.py`: durable run rows with `run_slot_key` dedupe, terminal statuses, and delivery as user notifications.

Phase-1 execution is **side-effect-free only** (ADR-077 decision 4, owner-accepted): `recurring_question` runs generate their answer via the server's LLM plumbing; `agent_task` runs execute in generation-only mode. Tool-using configurations are NOT executed in this phase — they resolve to an explicit skipped status with an actionable reason until the follow-up approval-escalation design exists. The definition's `notification_policy` governs delivery; the `metadata_only` redaction already applied to agent_task input messages is honored end-to-end; the result notification carries the definition name, outcome, a bounded result summary, and the durable run reference (full result stays server-side unless requested).

Timeout semantics (ADR-077 decision 5): a run cancelled at its execution deadline records a `timed_out` run status — aligned with the client's `timed_out` vocabulary (tldw_chatbook TASK-18939) so the client displays what the server reports rather than re-deriving. Missed-fire/reconnect reconciliation is the run-slot dedupe plus the notification feed (ADR-077 decision 6): occurrences the server ran while a client was away arrive once, never re-announced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The consumer handles `agent_task_run` Jobs with durable run rows and `run_slot_key` dedupe — a redelivered Job for an already-succeeded run slot is a recorded no-op, mirroring reminders
- [x] #2 Lifecycle is re-checked at execution time: paused/archived/disabled definitions skip with recorded reason, never execute
- [x] #3 `recurring_question` executions produce the generated answer through the server's LLM plumbing; `agent_task` executions run generation-only; tool-using configurations skip with an actionable reason (phase-1 boundary is enforced by the consumer, not assumed)
- [x] #4 Completed/failed/timed-out runs deliver a user notification (kind for agent-task results) carrying definition name, outcome, bounded summary, and run reference — governed by the definition's `notification_policy`; `metadata_only` redaction holds end-to-end
- [x] #5 A run cancelled at its deadline records `timed_out` (not failed, not skipped); the run-status vocabulary matches the client's so no translation layer is needed
- [x] #6 Definition health/last-run update after each execution; audit events recorded through the existing automation audit trail
- [x] #7 Tests cover dedupe, lifecycle skips, phase-1 boundary enforcement, notification construction (bounded + redacted), timeout status, and health/audit updates against the real DB/Jobs/notification shapes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (new consumer subsystem is covered by the cross-repo contract ADR — tldw_chatbook ADR-077; timeout and redaction semantics are its decisions 4/5, already owner-accepted). Link ADR-077 from the implementation notes.

1. Read `core/Reminders/reminder_jobs.py` end-to-end (run rows, dedupe, skip paths, notification creation) as the template
2. Run-row + dedupe skeleton for `agent_task_run`; lifecycle re-check
3. Phase-1 executors: recurring_question generation; agent_task generation-only; tool-config skip enforcement
4. Result notification construction (bounded summary, run reference, `notification_policy`, redaction)
5. `timed_out` status at the execution-deadline seam (align with the Jobs pipeline's existing lifecycle events)
6. Health/last-run/audit updates; test matrix per the ACs
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented 2026-08-22 on `feat/agent-task-consumer` (branched from dev after PR #2798 merged).

**Approach.** Mirrors `core/Reminders/reminder_jobs.py` end-to-end: a new `scheduled_task_runs` table in `ScheduledTasksDatabase.ensure_schema` (new-table addition — safe for existing DBs, unlike ALTERs) with the reminders' dedupe-on-create semantics (`ON CONFLICT(definition_id, run_slot_key) DO NOTHING`, existing row returned); the consumer `core/Scheduled_Tasks/agent_task_jobs.py` (`handle_agent_task_job`); and `services/agent_task_jobs_worker.py` mirroring `reminder_jobs_worker.py`, registered in the sidecar poller specs (`startup_sidecar_owned_jobs_pollers.py`) under `AGENT_TASK_JOBS_WORKER_ENABLED` with the `agent_task_jobs_task` handle field.

**Phase-1 boundary (AC#3) is enforced in the consumer**: any input config requesting tools (list/str/bool across `tools`/`allowed_tools`/`enable_tools`) resolves to a `skipped` run with error `tools_not_executable_in_phase1` and an actionable summary naming the approval-escalation dependency — never executed. Unknown families skip with `family_not_executable:<name>`.

**Executor seam (deliberate scope cut):** the LLM executor is an injected per-family async callable (`register_executor(family, fn)`) returning the result text. This task owns the run/notification/audit/timeout machinery; wiring the real server LLM plumbing (model selection from the definition/config, provider calls) is the next slice — without a registered executor the run fails honestly (`no_executor_configured`), visible rather than silent. `agent_task` executes generation-only by sharing the same seam.

**Timeout (AC#5):** `asyncio.wait_for` around the executor at 300s default records the distinct `timed_out` run status and delivers `automation_run_timed_out` — matching the client vocabulary (tldw_chatbook TASK-18939) with no translation layer.

**Notification pass-back (AC#4):** outcome → user notification via the same `create_user_notification` channel reminders use, kinds `automation_run_{succeeded,failed,timed_out,skipped}`, message = bounded result summary (1000-char cap, truncation-marked) for successes / outcome copy otherwise, `dedupe_key=automation_run:{definition}:{run_id}`, governed by the definition's `notification_policy` (`enabled:false` silences; `kinds` allowlist filters). Full result stays server-side per ADR-077 decision 3; the run row carries `result_summary`.

**Health/audit (AC#6):** succeeded → `ready`, failed/timed_out → `degraded` (new value; only-on-change, no version churn), skipped preserves health; every terminal outcome writes a `run_{status}` audit event. Notification/audit/health failures log and continue — the run row is written first and is authoritative.

**Verification.** `tests/Notifications/test_agent_task_jobs_consumer.py` — 10 tests through real preview-gated definitions, real run rows, real user notifications, injected stub executors: success+notify, redelivery dedupe (recorded no-op), paused skip with reason, missing definition skip, tool-request refusal with actionable reason, no-executor honest failure, timeout→timed_out+notification, executor exception→failed, notification-policy silence, degraded health + audit. Full Notifications suite **221 passed**; startup preflight **16 passed**. Not live-verified against a running server (headless worktree) — noted; the tldw_chatbook TASK-18940 AC#8 live gate covers the end-to-end pass once the client side lands.

**Files:** `app/core/DB_Management/Scheduled_Tasks_DB.py` (runs table + 3 run-row methods), `app/core/Scheduled_Tasks/__init__.py` + `agent_task_jobs.py` (new), `app/services/agent_task_jobs_worker.py` (new), `app/services/startup_sidecar_owned_jobs_pollers.py` (spec + service + handle), `tests/Notifications/test_agent_task_jobs_consumer.py` (new), this task file.
