---
id: TASK-2350
title: Plan Scheduled Tasks Phase 4B backend API foundation implementation
status: Done
labels:
- scheduled-tasks
- api
- planning
priority: high
references:
- TASK-2349
documentation:
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-design.md
- Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase4b-backend-api-foundation-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the Scheduled Tasks Phase 4B implementation plan for durable definitions, previews, lifecycle, audit, control-plane projection, and WebUI reference-client wiring from the approved backend/API foundation design spec.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan covers approved Phase 4B scope without adding execution, Jobs enqueueing, Scheduler integration, RAG execution, ACP dispatch, notifications, or fake results.
- [x] #2 Plan identifies concrete backend, storage, service, API, control-plane, frontend, and test files.
- [x] #3 Plan includes task-by-task TDD steps with commands and expected outcomes.
- [x] #4 Plan review findings are addressed or surfaced for user review.
- [x] #5 Plan document and Backlog task are committed on the isolated worktree branch.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Drafted implementation plan from the approved Phase 4B backend/API foundation spec.
Plan review pass 1 found gaps in Agent Task redaction coverage, idempotency coverage, disabled duplicate data-model support, and error-envelope mapping.
Revised the plan and spec to add redaction sentinel tests, idempotency route coverage, `disabled_lock_kind`/`disabled_reason`, duplicate lock tests, and complete error-envelope requirements including `correlation_id`.
Plan review pass 2 approved the revised plan and spec with no remaining blocking issues.
Verification: git diff --check passed before commit; no code tests or Bandit were run because this task touched only docs and Backlog metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Scheduled Tasks Phase 4B implementation plan and addressed plan-review findings. The plan is ready for execution handoff.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
