---
id: TASK-12147
title: Extract Jobs admission operations behind JobManager
status: Done
assignee: []
created_date: 2026-07-04 01:28
updated_date: 2026-07-05 00:28
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
- Docs/superpowers/plans/2026-06-24-jobs-backend-parity-refactor-implementation-plan.md
- https://github.com/rmusser01/tldw_server/pull/2527
- https://github.com/rmusser01/tldw_server/pull/2611
documentation:
- Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md
priority: medium
modified_files:
- Docs/superpowers/plans/2026-07-04-jobs-admission-operations-extraction-plan.md
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/app/core/Jobs/operations/contracts.py
- tldw_Server_API/app/core/Jobs/operations/postgres/admission.py
- tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py
- tldw_Server_API/tests/Jobs/parity/scenarios.py
- tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py
- tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py
- tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py
- tldw_Server_API/tests/Jobs/test_jobs_fault_injection_sqlite.py
- tldw_Server_API/tests/Jobs/test_jobs_manager_pg_fakes_extended.py
- tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py
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
Original verification: focused matrix passed on final code with 62 passed, 13 skipped, 243 warnings. Skips were explicit Postgres-unreachable fixture skips in Postgres idempotency/quota/parity tests. Additional affected tests passed: SQLite admission operation tests and fake Postgres manager tests, 11 passed. Static checks: operation import-boundary scan returned no matches; git diff --check passed; py_compile passed for manager/contracts/sqlite/postgres admission modules; Bandit exited 0 on touched scope with only pre-existing #nosec warnings in manager.py.
PR: https://github.com/rmusser01/tldw_server/pull/2611
Pre-rebase validation refresh (2026-07-04): branch was stale at merge-base ded7b5158f676263fa4224fd721ac22c3a51c2f1, then rebased cleanly onto origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5. Post-rebase merge-base matched origin/dev at that time. Addressed PR review feedback by ensuring SQLite idempotent replay in-process emit_job_event uses the durable event current request/trace context when the outbox is disabled, and by ensuring Postgres non-idempotent admission job.created durable events use CreateJobCommand.request_id/trace_id instead of row fields that can be absent in mocked/fallback rows. Added focused regressions: test_idempotent_replay_in_process_event_uses_current_context_sqlite and test_pg_non_idempotent_create_uses_command_context_for_job_created_event. Passed targeted review tests (2 passed). Passed focused jobs matrix with 31 passed, 7 skipped. Passed Bandit over touched production Jobs scope with 0 findings in /tmp/bandit_jobs_admission_latest_dev.json. Passed git diff --check. Postgres parity file was invoked directly and skipped all 7 tests because the local Postgres fixture was unavailable in this environment.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extracted Jobs admission operations and contract coverage. Final refresh validated against origin/dev 6b727b221e55646eba663a03571e38302f7fafc2 with focused Jobs tests passing, Bandit reporting zero findings on touched production scope, and whitespace check clean.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Post-rebase validation on current origin/dev 4c1ca5d8358bff2a5a7fb5c75d60d1bd6728e702: rebased codex/jobs-admission-operations-extraction so merge-base equals current origin/dev. Fresh verification after rebase: focused Jobs admission/operation suite passed (31 passed, 14 skipped, 70 warnings); Bandit over touched Jobs production files scanned 7294 LOC and reported 0 findings in /tmp/bandit_jobs_admission_rebased_dev.json, with existing skipped # nosec B608 notices only; git diff --check HEAD~1..HEAD passed.
2026-07-04 current-dev refresh: rebased `codex/jobs-admission-operations-extraction` onto `origin/dev` `09d9ec901e1d4548f7924f1c6bcefa963fadd9bd`; merge-base matches `origin/dev`. Validation: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Jobs/test_jobs_admission_operations_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py tldw_Server_API/tests/Jobs/test_jobs_fault_injection_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_manager_pg_fakes_extended.py tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py -q` passed with 31 tests and 14 expected parity skips; `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/core/Jobs/operations/contracts.py tldw_Server_API/app/core/Jobs/operations/postgres/admission.py tldw_Server_API/app/core/Jobs/operations/sqlite/admission.py -f json -o /tmp/bandit_jobs_admission_origin_dev_09d9ec.json` reported 0 findings over 7294 LOC with existing skipped `# nosec B608` warnings; `git diff --check HEAD~1..HEAD` passed.
2026-07-04 latest-dev refresh: rebased and validated PR #2611 on origin/dev 6b727b221e55646eba663a03571e38302f7fafc2. Tested head a34d7bf9f3fc. Verification: focused Jobs pytest suite => 31 passed, 14 skipped, 70 warnings; Bandit over Jobs manager/admission operation scope => 0 findings over 7294 LOC with existing skipped #nosec B608 warnings only; git diff --check HEAD~1..HEAD => clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
