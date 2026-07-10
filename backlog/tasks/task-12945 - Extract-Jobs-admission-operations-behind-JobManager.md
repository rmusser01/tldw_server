---
id: TASK-12945
title: Extract Jobs admission operations behind JobManager
status: Done
assignee: []
created_date: '2026-07-10 05:16'
updated_date: '2026-07-10 05:16'
labels:
  - jobs
  - implementation
  - refactor
dependencies: []
references:
  - TASK-12015
  - TASK-12016
  - TASK-12017
  - Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md
  - >-
    Docs/superpowers/plans/2026-06-24-jobs-backend-parity-refactor-implementation-plan.md
  - 'https://github.com/rmusser01/tldw_server/pull/2527'
  - 'https://github.com/rmusser01/tldw_server/pull/2611'
documentation:
  - Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extract the Jobs create/admission transaction path behind backend-specific operation modules while preserving JobManager.create_job as the public facade. Scope includes SQLite/Postgres admission operation modules, typed CreateJobCommand/AdmissionResult use, idempotency, quota/fair-share/queue policy behavior, durable event facts, and parity/API contract verification. Lifecycle operations remain out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 JobManager.create_job remains the public caller-facing API and preserves existing row/REST-compatible behavior.
- [x] #2 SQLite and Postgres admission transaction logic is routed through backend-specific operation modules under app/core/Jobs/operations/ without those modules importing JobManager.
- [x] #3 Admission parity tests cover idempotent create, queue controls, quota/fair-share rejection, durable event facts, and public facade mapping for touched behavior.
- [x] #4 No lifecycle methods are extracted in this slice.
- [x] #5 Focused Jobs/Chatbooks parity and contract tests pass; Postgres-specific tests either pass or record an explicit fixture skip.
- [x] #6 Bandit and diff hygiene checks pass for the touched implementation scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm baseline Jobs safety-net tests pass or record existing skips. 2. Write a focused implementation plan for admission extraction. 3. Add failing admission-operation tests for the new backend operation boundary. 4. Implement the minimal SQLite and Postgres admission operation modules and route JobManager.create_job through them. 5. Preserve public response mapping and post-commit side effects in JobManager. 6. Run focused parity/API/security verification and update this task with results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-07-04-jobs-admission-operations-extraction-plan.md
PR: https://github.com/rmusser01/tldw_server/pull/2611
Original implementation extracted SQLite/Postgres admission operation modules while keeping JobManager.create_job as facade for validation, policy, metrics, audit/fanout, and public row mapping. Review feedback addressed request_id/trace_id propagation for SQLite idempotent replay in-process events and Postgres non-idempotent durable job.created events.
Latest rebase: rebased codex/jobs-admission-operations-extraction onto origin/dev 20d96055e8a4fbe99a0394ca11015977167e1f26. Verification after rebase: focused Jobs/Chatbooks matrix passed with 64 passed, 13 skipped, 246 warnings; skips were explicit local Postgres fixture-unavailable skips. Operation import-boundary scan had no JobManager references in app/core/Jobs/operations. git diff --check passed. py_compile passed for Jobs manager/contracts/sqlite admission/postgres admission. Bandit exited 0 on touched scope with only existing #nosec B608 warnings in manager.py.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2611 onto latest dev and kept the Jobs admission extraction valid. The branch preserves JobManager.create_job as the facade, keeps backend-specific admission SQL in operation modules, and includes regression coverage for the reviewed request/trace propagation issues.
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
