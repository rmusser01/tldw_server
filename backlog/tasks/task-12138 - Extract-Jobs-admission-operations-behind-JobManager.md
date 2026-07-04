---
id: TASK-12138
title: Extract Jobs admission operations behind JobManager
status: Done
assignee: []
created_date: '2026-07-04 01:28'
updated_date: '2026-07-04 02:07'
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
Baseline before implementation: focused Jobs matrix passed with 55 passed, 424 warnings.
Red/green evidence: added failing AdmissionResult.existing durable-event contract test, then updated the contract and verified test_jobs_operation_contracts.py passed. Added direct SQLite admission operation tests; they initially failed on missing module, then passed after implementation.
Implementation: added SQLite/Postgres admission operation modules, kept JobManager.create_job as facade for validation, policy, metrics, audit/fanout, and public row mapping. Operation modules own transactional quota checks, idempotent insert/select, counter updates, and durable job.created rows. Lifecycle methods were not extracted.
Verification: focused matrix passed on final code with 62 passed, 13 skipped, 243 warnings. Skips were explicit Postgres-unreachable fixture skips in Postgres idempotency/quota/parity tests. Additional affected tests passed: SQLite admission operation tests and fake Postgres manager tests, 11 passed.
Static checks: operation import-boundary scan returned no matches; git diff --check passed; py_compile passed for manager/contracts/sqlite/postgres admission modules; Bandit exited 0 on touched scope with only pre-existing #nosec warnings in manager.py.

PR: https://github.com/rmusser01/tldw_server/pull/2611
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extracted Jobs create/admission transactions behind backend-specific SQLite and Postgres operation modules while preserving JobManager.create_job as the public facade. Added durable-event support for idempotent existing admission results, direct SQLite admission operation tests, and shared parity coverage proving idempotent replay writes a current-request job.created event while preserving the original returned row request/trace IDs. Verified focused Jobs/Chatbooks tests and security/static checks; real Postgres parity was skipped because the local Postgres fixture was unreachable.
<!-- SECTION:FINAL_SUMMARY:END -->

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
