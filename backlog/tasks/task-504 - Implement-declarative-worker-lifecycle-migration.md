---
id: TASK-504
title: Implement declarative worker lifecycle migration
status: In Progress
labels:
- backend
- implementation
- lifecycle
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved declarative worker lifecycle implementation plan using subagent-driven development. Replace managed-worker task/stop-event field threading with WorkerSpec definitions, lifecycle session/engine ownership, diagnostics compatibility, parity checks, and final verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All tasks in Docs/superpowers/plans/2026-06-02-declarative-worker-lifecycle-implementation-plan.md are completed or blockers documented.
- [ ] #2 Each task receives spec compliance and code quality review before moving to the next task.
- [ ] #3 Focused lifecycle tests, compileall on app/services, and Bandit on app/services run with results recorded.
- [ ] #4 Final summary and touched files are recorded in Backlog.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 1 completed at commit 3dfcf77f4e0a561bd399b9d640f90d9fa0be3f1f. Added lifecycle worker spec types, graph validation, post-predicate dependency validation, POST_WORKER_SHUTDOWN phase, and focused spec tests. Verification: focused pytest 36 passed/6 warnings; git diff --check passed; Bandit on lifecycle_worker_specs.py reported 0 findings. Reviews: spec compliance passed; code quality passed with one minor non-blocking note about future string-field validation.

Task 2 completed at commit 5d18e330639689b1880b1031078f43a0f62246ea. Added WorkerLifecycleSession diagnostics and stop-state tracking, preserved inventory compatibility through publish_worker_inventory(), mapped stopped-name app-state fields, and fixed callback-only diagnostics to publish the spec task_name with has_stop_event=false. Verification: focused pytest 19 passed/6 warnings; git diff --check passed; Bandit on lifecycle_worker_session.py reported 0 findings. Reviews: spec compliance passed after the callback-only task_name fix; code quality passed with no blocking issues.

Task 3 completed at commit fb189903fbd2ef2b6cfc1ded46d843a5f63716fa. Added LifecycleWorkerEngine startup/shutdown orchestration, dependency-aware startup/reverse shutdown, failure-policy handling, startup abort cleanup, immediate task failure/cancellation classification, diagnostic-name publication, and focused engine tests. Verification: lifecycle pytest suite 47 passed/6 warnings; Ruff touched-file check passed; git diff --check and git diff --cached --check passed; Bandit on engine/session scope reported 0 findings. Reviews: spec compliance passed; code quality passed after startup cleanup, cancellation/diagnostic-name, and lint fixes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
