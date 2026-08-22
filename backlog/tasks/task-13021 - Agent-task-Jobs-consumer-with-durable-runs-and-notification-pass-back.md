---
id: TASK-13021
title: 'Agent-task Jobs consumer: durable runs, timeout status, notification pass-back'
status: To Do
assignee: []
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
- [ ] #1 The consumer handles `agent_task_run` Jobs with durable run rows and `run_slot_key` dedupe — a redelivered Job for an already-succeeded run slot is a recorded no-op, mirroring reminders
- [ ] #2 Lifecycle is re-checked at execution time: paused/archived/disabled definitions skip with recorded reason, never execute
- [ ] #3 `recurring_question` executions produce the generated answer through the server's LLM plumbing; `agent_task` executions run generation-only; tool-using configurations skip with an actionable reason (phase-1 boundary is enforced by the consumer, not assumed)
- [ ] #4 Completed/failed/timed-out runs deliver a user notification (kind for agent-task results) carrying definition name, outcome, bounded summary, and run reference — governed by the definition's `notification_policy`; `metadata_only` redaction holds end-to-end
- [ ] #5 A run cancelled at its deadline records `timed_out` (not failed, not skipped); the run-status vocabulary matches the client's so no translation layer is needed
- [ ] #6 Definition health/last-run update after each execution; audit events recorded through the existing automation audit trail
- [ ] #7 Tests cover dedupe, lifecycle skips, phase-1 boundary enforcement, notification construction (bounded + redacted), timeout status, and health/audit updates against the real DB/Jobs/notification shapes
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
