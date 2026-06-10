---
id: TASK-2351
title: Implement Scheduled Tasks Phase 4B backend API foundation
status: In Progress
labels:
- scheduled-tasks
- api
- backend
- frontend
priority: high
references:
- TASK-2350
- TASK-2349
documentation:
- Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-implementation-plan.md
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the reviewed Scheduled Tasks Phase 4B implementation plan: durable automation definitions, previews, lifecycle, audit, idempotency, control-plane projection, and reference WebUI client without execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Capability, preview, definition, lifecycle, audit, idempotency, and projection backend APIs are implemented without execution.
- [ ] #2 Agent Task raw message text is redacted from preview, definition, list/detail, audit, and persisted JSON by default.
- [ ] #3 `/api/v1/scheduled-tasks` projects `automation_definition` rows without breaking reminder or Watchlists behavior.
- [ ] #4 WebUI reference client can preview, create, inspect, edit, pause/resume, archive, and duplicate definitions using the API.
- [ ] #5 Focused backend and frontend tests pass for the touched scope.
- [ ] #6 Bandit passes for touched backend scope or any findings are fixed.
- [ ] #7 No Jobs enqueueing, Scheduler integration, RAG execution, ACP dispatch, notifications, fake runs, fake results, or fake Home items are implemented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation will follow `Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-implementation-plan.md` using subagent-driven development with TDD and review checkpoints per task.
Task 1 completed in commit 8e2dd7b1fc: added automation schemas, route skeletons, capability service, primitive extension, and route/capability tests.
Task 1 verification: focused pytest for `test_scheduled_task_automation_api.py` and `test_scheduled_tasks_control_plane.py` passed with 6 tests; touched-scope Bandit reported zero findings; git diff --check was clean.
Task 1 review: spec-compliance reviewer approved; code-quality reviewer approved with one minor suggestion to add extra 501 skeleton assertions if useful before later endpoint wiring.
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
