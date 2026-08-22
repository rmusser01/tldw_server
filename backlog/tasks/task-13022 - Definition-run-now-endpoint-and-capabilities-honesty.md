---
id: TASK-13022
title: 'Definition run-now endpoint and capabilities honesty'
status: To Do
assignee: []
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
- [ ] #1 `POST .../definitions/{id}/run` enqueues an immediate execution through the standard Jobs path with run-slot dedupe (a manual run colliding with a scheduled run of the same slot dedupes exactly like a redelivered Job)
- [ ] #2 Paused/archived/disabled definitions refuse with the existing lifecycle error codes; permissions follow the control plane's `TASKS_CONTROL` model; idempotency keys behave like the other mutating definition endpoints
- [ ] #3 The capabilities surface reports per-family execution truth: recurring_question executable; agent_task generation-only; tools not executable (with the phase-1 reason string), consistent with the consumer's enforcement
- [ ] #4 Response shape carries the created run reference so a manual trigger can be correlated with its result notification
- [ ] #5 Tests cover endpoint authorization, lifecycle refusals, dedupe behavior, and capability truth against the real service shapes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no — API surface addition within the existing control plane; the contract (manual run = real dispatch, dedupe, refusal semantics) is ADR-077 decision 7, already owner-accepted.

1. Endpoint + schema on the existing scheduled-tasks router group, following the pause/resume handler shapes
2. Enqueue-through-the-standard-path implementation (no consumer bypass)
3. Capabilities service truth update (per-family, phase-1 reason strings)
4. Tests per the AC matrix
<!-- SECTION:PLAN:END -->
