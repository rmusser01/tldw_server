---
id: TASK-13022
title: 'Definition run-now endpoint and capabilities honesty'
status: Done
assignee:
  - '@robert'
created_date: '2026-08-21 14:10'
updated_date: '2026-08-21 14:10'
labels:
  - scheduled-tasks
  - automation
  - api
dependencies:
  - task-13021
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Third server-side piece of the server-offload seam (tldw_chatbook ADR-077, accepted; cross-repo TASK-18940). Two surfaces close the loop with the client:

**Run-now endpoint** (ADR-077 decision 7): `POST` on the definitions resource (e.g. `/scheduled-tasks/definitions/{definition_id}/run`) that enqueues an immediate execution through the same Jobs path TASK-13020/13021 use — a real dispatch with the same run-slot dedupe, NOT a bypass — for manual triggering from chatbook's Run-now action (tldw_chatbook TASK-18938 parity: until this endpoint exists, the client honestly refuses Run-now on server-scoped definitions). Honors lifecycle (paused/archived/disabled refuse with the existing error codes), idempotency key, and the control plane's permission model (`TASKS_CONTROL`).

**Capabilities honesty**: the capabilities endpoint (`GET /capabilities` and per-family availability) reflects real execution availability per family — `recurring_question` executable, `agent_task` generation-only executable with tools explicitly not-yet-executable — instead of a blanket unavailable. This is what lets the client stop rendering permanent `execution_unavailable` for server owners.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `POST .../definitions/{id}/run` enqueues an immediate execution through the standard Jobs path with run-slot dedupe (a manual run colliding with a scheduled run of the same slot dedupes exactly like a redelivered Job)
- [x] #2 Paused/archived/disabled definitions refuse with the existing lifecycle error codes; permissions follow the control plane's `TASKS_CONTROL` model; idempotency keys behave like the other mutating definition endpoints
- [x] #3 The capabilities surface reports per-family execution truth: recurring_question executable; agent_task generation-only; tools not executable (with the phase-1 reason string), consistent with the consumer's enforcement
- [x] #4 Response shape carries the created run reference so a manual trigger can be correlated with its result notification
- [x] #5 Tests cover endpoint authorization, lifecycle refusals, dedupe behavior, and capability truth against the real service shapes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no — API surface addition within the existing control plane; the contract (manual run = real dispatch, dedupe, refusal semantics) is ADR-077 decision 7, already owner-accepted.

1. Endpoint + schema on the existing scheduled-tasks router group, following the pause/resume handler shapes
2. Enqueue-through-the-standard-path implementation (no consumer bypass)
3. Capabilities service truth update (per-family, phase-1 reason strings)
4. Tests per the AC matrix
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented 2026-08-22 on `feat/definition-run-now` (branched from dev after PR #2801 merged).

**Run-now endpoint (AC#1/#2/#4):** `POST /scheduled-tasks/definitions/{id}/run` on the existing control-plane router, following the pause/resume handler shapes (rbac rate limit, `RequirePermission(TASKS_CONTROL)`, idempotency/request ids from headers). The service's `run_now` is a REAL dispatch: it enqueues through `JobManager.create_job` with the same domain (`scheduled_tasks`), type (`agent_task_run`), payload shape, and idempotency-key format (`definition:{id}:{slot}`) the feed uses — a manual run colliding with a scheduled run of the same slot dedupes exactly like a redelivered Job (the Jobs layer's duplicate-create-returns-same-row semantics). The manual slot is second-truncated UTC "now"; the payload carries `manual: true`. Response (`ScheduledTaskRunNowResponse`) returns definition id, run slot, job id, and a `deduped` flag for correlation with the eventual result notification. An audit event (`definition.run_now`) records every trigger; audit failure logs and continues (never fails the trigger).

**Lifecycle refusals (AC#2):** archived → `definition_archived`; admin/security-locked disabled → `definition_disabled_locked`; paused → `definition_paused` (new error mapping added: 409 `scheduled_task_lifecycle_transition_invalid`); unlocked disabled → `definition_disabled`. A manual trigger never silently resurrects a definition the owner paused. All reuse the existing error envelope.

**Capabilities honesty (AC#3):** per-family truth replacing the blanket `execution_not_implemented`: `execute` = available with reason `phase1_generation_only` on both families; new `run_now` action = available; new `execute_tools` = planned with the phase-1 reason (tools not executable until the approval-escalation design lands). This matches what the consumer (TASK-13021) actually enforces.

**Verification.** Automation API suite **42 passed** — including the updated capabilities assertions (execute available/run_now available/execute_tools planned), the real-dispatch test (captures `create_job` kwargs and pins domain/type/key-shape/manual flag), paused refusal (create_job monkeypatched to fail if called; existing 409 envelope), and unknown-definition 404. Full Notifications suite **224 passed**. Not live-verified against a running server (headless worktree); the chatbook TASK-18940 AC#8 live gate covers end-to-end once the client side lands.

**Files:** `app/api/v1/schemas/scheduled_tasks_automation_schemas.py` (ScheduledTaskRunNowResponse), `app/services/scheduled_task_automation_service.py` (run_now + capabilities), `app/api/v1/endpoints/scheduled_tasks_control_plane.py` (endpoint + definition_paused mapping), `tests/Notifications/test_scheduled_task_automation_api.py`, this task file.
