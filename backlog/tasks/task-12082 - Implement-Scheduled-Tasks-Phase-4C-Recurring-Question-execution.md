---
id: TASK-12082
title: Implement Scheduled Tasks Phase 4C Recurring Question execution
status: In Progress
labels:
- scheduled-tasks
- phase-4c
- implementation
- api-first
priority: high
references:
- Docs/superpowers/specs/2026-06-30-scheduled-tasks-phase4c-recurring-question-execution-design.md
documentation:
- Docs/superpowers/plans/2026-07-01-scheduled-tasks-phase4c-recurring-question-execution-implementation-plan.md
modified_files:
- backlog/tasks/task-12082 - Implement-Scheduled-Tasks-Phase-4C-Recurring-Question-execution.md
- Docs/superpowers/plans/2026-07-01-scheduled-tasks-phase4c-recurring-question-execution-implementation-plan.md
- tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py
- tldw_Server_API/app/core/Scheduled_Tasks/__init__.py
- tldw_Server_API/app/core/Scheduled_Tasks/recurring_question_models.py
- tldw_Server_API/app/core/Scheduled_Tasks/recurring_question_scope.py
- tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py
- tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py
- tldw_Server_API/app/services/scheduled_task_automation_service.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py
- tldw_Server_API/tests/Notifications/test_scheduled_task_recurring_question_scope.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved API-first Scheduled Tasks Phase 4C Recurring Question execution plan. Scope includes storage, schemas, preview/admission, manual runs, run/result APIs, Jobs worker, APScheduler bridge, Home surfacing, WebUI/extension reference-client behavior, retention/privacy hardening, tests, Bandit, and review checkpoints. Preserve Watchlists as a separate UX/job and do not introduce source-specific GitHub/YouTube monitor assumptions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Recurring Question definitions support preview, create/update, inspect, run now, scheduled execution, mark solved, reopen, review state, retention metadata, and durable run/result history through API-first contracts.
- [ ] #2 Every execution attempt creates a run record; surfaced findings and attention-worthy failures create normalized result records; routine no-match runs remain discoverable in run history without polluting Home.
- [ ] #3 Execution uses Jobs for queueing/worker lifecycle and APScheduler only for due schedule claims, with reconciliation for divergent Jobs/run state.
- [ ] #4 WebUI and extension behave as reference/main enterprise API clients, including Home surfacing and `/scheduled-tasks` monitoring, without defining product boundaries in UI-only logic.
- [ ] #5 Watchlists functionality and UX remain unchanged as a separate persona/job; GitHub and YouTube are treated only as examples, not privileged source assumptions.
- [ ] #6 Storage, API, service, worker, scheduler, frontend client, UI, retention/privacy, accessibility, and Watchlists compatibility tests cover the implemented behavior.
- [ ] #7 Bandit and targeted backend/frontend verification are run before completion, with any skips documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-01-scheduled-tasks-phase4c-recurring-question-execution-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation started from plan `Docs/superpowers/plans/2026-07-01-scheduled-tasks-phase4c-recurring-question-execution-implementation-plan.md` after rebasing the planning branch onto `origin/dev` at merge `85ab688b34`.

Stage 1 storage/schema slice added durable recurring-question definition resolution defaults, owner-scoped run/result persistence, result review state, owner+dedupe uniqueness, canonical JSON snapshots, and redacted source-ref guards. Verification: focused scheduled task DB pytest passed (31 passed, 3 warnings); Bandit on touched production files reported zero findings.

Stage 1 spec-review fix mapped persisted definition resolution/policy fields into service responses and added recursive private-key validation before run/result JSON persistence. Verification: focused scheduled task DB+service pytest passed (68 passed, 4 warnings); git diff --check passed; Bandit on touched production files reported zero findings.

Stage 1 local code-quality review tightened private payload key detection for common variants such as `rawText`, `openai_api_key`, and `access_token`. Verification after hardening: focused scheduled task DB+service pytest passed (71 passed, 4 warnings); git diff --check passed; Bandit on touched production files reported zero findings.

Stage 2 preview/readiness slice added source-agnostic Recurring Question scope normalization, finding/retention/generation preview normalization, 4C capability actions without action-level degraded status, `mark-solved` and `reopen` service/API routes, transition enforcement for archived/disabled/unsolved definitions, and audit coverage. Verification: focused Stage 2 pytest passed (31 passed, 11 warnings); full scheduled task automation service/API/scope pytest passed (82 passed, 14 warnings); git diff --check passed; Bandit on touched production files reported zero findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
