---
id: TASK-13215
title: Fix Jobs completion missing-row concurrent-insert atomicity
status: In Progress
created_date: 2026-09-07 19:39
labels:
- jobs
- defect
- atomicity
- concurrency
priority: High
references:
- codex/jobs-completion-foundation@877c86e7bb
- TASK-13216
- TASK-13217
- https://github.com/rmusser01/tldw_server/pull/3092
- TASK-13421
documentation:
- Docs/superpowers/specs/2026-09-07-jobs-completion-row-identity-atomicity-design.md
- Docs/superpowers/plans/2026-09-07-jobs-completion-row-identity-atomicity-implementation-plan.md
- Docs/ADR/058-jobs-completion-row-identity.md
modified_files:
- Docs/ADR/058-jobs-completion-row-identity.md
- Docs/ADR/README.md
- Docs/superpowers/specs/2026-09-07-jobs-completion-row-identity-atomicity-design.md
- Docs/superpowers/plans/2026-09-07-jobs-completion-row-identity-atomicity-implementation-plan.md
- backlog/tasks/task-13215 - Fix-Jobs-completion-missing-row-concurrent-insert-atomicity.md
- backlog/tasks/task-13216 - Migrate-acquired-job-completion-callers-to-UUID-preconditions.md
- backlog/tasks/task-13217 - Assess-historical-Jobs-completion-bookkeeping-drift.md
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/app/core/Jobs/worker_sdk.py
- tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py
- tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py
- tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py
- tldw_Server_API/tests/Jobs/test_worker_sdk.py
updated_date: 2026-10-03 01:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Blocking remediation required before strict complete_job extraction. A completion attempt whose initial row lookup returns no row can race with a concurrent insert of the same numeric id. The later guarded update can complete that new row while base facts remain absent, so complete_job returns True without completion counters, job.completed outbox persistence, metrics, or observers. Design and implement an atomic row-identity/fact boundary before resuming the completion extraction stream.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A deterministic SQLite regression reproduces the initial-miss/concurrent-insert race before the fix.
- [x] #2 A real-PostgreSQL regression validates whether the same race is possible under the configured isolation and RLS cursor path.
- [x] #3 No completion can commit successfully unless the operation has authoritative facts for the exact durable row identity established by its locked lookup.
- [x] #4 An applied completion updates or reconciles the correct lifecycle counter and writes job.completed atomically when the outbox is enabled.
- [x] #5 After the authoritative locked lookup begins, concurrent insert, delete/reinsert, and row-visibility changes cannot redirect completion to another row incarnation selected only by a reused numeric id; pre-call protection is asserted only when a usable expected_uuid is supplied.
- [x] #6 Normal processing completion, permitted queued completion, token replay, missing-row, failure precedence, and RLS behavior are explicitly characterized and intentionally updated where remediation requires it.
- [x] #7 Focused SQLite and required real-PostgreSQL tests, full relevant Jobs regressions, formatting/lint, and scoped Bandit pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute `Docs/superpowers/plans/2026-09-07-jobs-completion-row-identity-atomicity-implementation-plan.md` in five stages: refresh and baseline latest dev; add deterministic red SQLite/PostgreSQL/RLS and contract tests; implement both backend lock/identity paths as one green manager change; bind WorkerSDK ordinary completion to the acquired UUID; then run focused cross-backend verification, Ruff/Black/compile/Bandit, code review, and PR-readiness checks. TASK-13216 and TASK-13217 remain excluded.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Validated on SQLite and real PostgreSQL 18 under READ COMMITTED, including forced RLS with a visible chatbooks/u1 row: after an initial missing SELECT, a concurrent queued insert with the same numeric id can be completed while only job.created exists and ready_count remains 1. The approved remediation loads the authoritative row under SELECT FOR UPDATE on PostgreSQL or BEGIN IMMEDIATE on SQLite, returns False on a locked miss, captures the raw stored UUID, and guards every mutation/replay query by id plus null-safe stored UUID. expected_uuid is optional and WorkerSDK supplies the acquired UUID; legacy null/empty UUIDs receive only in-operation replacement protection. Completion outbox and lifecycle counter bookkeeping are mandatory and atomic when enabled. SLA attachment/event statement failures remain best-effort only after a savepoint is established; savepoint-control failures propagate. Metrics and observers remain post-commit. The slides-specific terminal-result operation is not a suitable reuse target because it does not preserve general result or queued-completion semantics. Strict completion extraction stays paused until this blocker merges. The provisional TASK-13112.3 was not carried forward because current dev already assigns TASK-13112 to unrelated work. Second design review added stable database error-code/class assertions, bounded concurrency teardown, exact same-token side-effect coverage, explicit SQLite contention behavior, and tracked follow-ups TASK-13216 (direct-caller UUID adoption) and TASK-13217 (historical bookkeeping drift).
Implementation plan drafted and self-reviewed on 2026-09-07. The plan keeps both backend behavior changes in one green commit, requires red evidence for the validated races, uses zero-timeout SQLite error codes and PostgreSQL LockNotAvailable rather than sleeps/messages, checks counter and outbox snapshots, exercises forced RLS and legacy UUID replay, preserves savepoint-control failure rollback, and requires fresh verification after the final dev rebase.
Final plan review verified the live remote dev tip remains e3174f1ad9f6dd0b11e4ecb20d48c1c4090d3bfe after correcting a locally rewritten origin/dev ref. No requirement or test-coverage gaps remain. The review made the forced-RLS cursor wrapper/import explicit and scoped Black to the new test module because all five existing touched files already fail whole-file Black on the unchanged baseline; Ruff currently passes those files, and the plan retains Ruff, syntax, diff, and changed-hunk formatting checks across the full touched scope.
2026-10-02 execution resumed using subagent-driven development. Rebased the three planning commits cleanly onto live dev 9958110df2a9011e19f48b0eae821353e19d4af8. Focused SQLite baseline: 127 passed, 55 deselected, 82 warnings. Latest dev changes RLS to transaction-local fail-closed context; tests retain the original cursor path. ADR assessment: required yes because the optional acquired-row identity precondition is a durable completion API rule; record ADR-058 from the already-approved design, with no renewed decision or scope change.
2026-10-02 red/green evidence: two SQLite race failures (miss completed concurrent row; replacement-uuid completed), two PostgreSQL race failures (miss completed concurrent row; missing initial row lock); 13 SQLite UUID contract TypeErrors. Forced-RLS miss returned True/new row completed; strict WorkerSDK spy had no call because expected_uuid was absent. Fix implemented both backend locked lookups, immediate false on miss/stale UUID, captured raw UUID guards, direct state branches, authoritative bookkeeping, ordinary WorkerSDK UUID forwarding. Green expanded matrices: 242 SQLite passed / 65 deselected; 83 PostgreSQL passed / 71 deselected with 2 opt-in SSE skips. Explicit RUN_PG_JOBS_TESTS=1 and JOBS_SSE_TEST_MAX_SECONDS=0.5 rerun: both SSE tests passed. Worker suites 132 passed; new atomicity module 24 passed. Black new module, Ruff all touched Python files, py_compile runtime, diff check passed. Raw Bandit had one unchanged B608 warning in canonical webhook pruning; archived dev baseline comparison exit zero with no new results/errors at /tmp/bandit_task_13215_delta.json. Initial spec review findings fixed by full completion-field RLS snapshots and concurrent processing counter 2->1 with a retained sentinel. Final spec/quality review and final dev rebase remain in progress.
Final integrated review/verification: spec re-review confirmed forced-RLS NOWAIT lock proof, missing-counter reconciliation real SQL failure rollback, optional SLA attachment/outbox failures, and creation/release/rollback-to savepoint-control failures all close the identified coverage gaps. Runtime/worker quality review reported no actionable findings; final lifecycle-test quality pass is pending. Fresh matrices: SQLite 247 passed / 70 deselected / 1212 warnings; required PostgreSQL 90 passed / 76 deselected / 358 warnings with no skips, including opt-in outbox tests using RUN_PG_JOBS_TESTS=1 and bounded SSE. Ruff all six touched Python files, Black new module, runtime py_compile, diff check, production Bandit delta passed. Full touched test-scope Bandit (B101 excluded for assertions) initially flagged two new literal fixture tokens; reused existing acquired UUID/seeded lease instead, delta comparison using normalized archived filenames now exit zero with no findings/errors at /tmp/bandit_task_13215_tests_delta.json. Final rebase/push/draft PR pending; no merge authorized in this task.
Final code-quality review (including lifecycle test additions/token cleanup) has no actionable findings. Both implementation commits and the ADR/verification documentation checkpoint were committed with hooks enabled, then all six branch commits rebased cleanly onto explicitly verified live dev 86e287fee7bfa1a1588639232e35db3666851ded. Upstream changes were unrelated MCP tests only. Post-rebase: atomicity/strict worker SQLite 16 passed / 9 deselected / 51 warnings; required PostgreSQL atomicity/RLS completion 12 passed / 15 deselected / 36 warnings. Initial PG collection had a mistyped node; corrected actual test name passed. Ruff and full branch diff check passed. Branch pushed; draft PR #3092 opened against dev and attached to the chat: https://github.com/rmusser01/tldw_server/pull/3092. gh initially resolved the unrelated upstream repo, so explicit --repo rmusser01/tldw_server was used. Jobs Suite queued automatically via pull_request; mandatory repository gates and Jobs CI remain pending. Human-authored Change summary must be added by requester before merge. No merge performed; retain worktree for review, and keep strict extraction/TASK-13216/TASK-13217 excluded.
2026-10-02 Qodo completed review of 59f8c5582651a1bfce6ecce29b17566aa5792e34: zero bugs, four rule findings. Validated missing documentation on newly added test/helper definitions and missing Iterator return annotations on two RLS context managers; narrow documentation/type-only correction is in progress. The requested completion SQL relocation conflicts with the approved focused fix/extraction non-goals and is tracked separately. The fixture finding is invalid for this scope: the new module requests existing Jobs jobs_pg_dsn, which delegates to shared pg_temp_db; AuthNZ isolated_test_environment is local to the AuthNZ suite, not a Jobs fixture. Required hosted CI remains queued behind the frontend-license audit; no merge gate is waived.
2026-10-02 Qodo follow-up verification: all newly added definitions now have docstrings; both RLS context managers return Iterator[_CompletionReadCursor]. Pre/post AST audits passed, and AST comparison proves review changes are documentation/annotation-only. Fresh SQLite 247 passed / 70 deselected (1212 warnings), required PostgreSQL 90 passed / 76 deselected (358 warnings), no PG skips including SSE/outbox. Ruff six touched Python files, Black new module, runtime/test py_compile and whitespace checks passed. Production/test Bandit baseline deltas have no findings/errors at /tmp/bandit_task_13215_qodo_delta.json and /tmp/bandit_task_13215_qodo_tests_delta.json; test B101 excluded, inherited warnings not waived. XML records /tmp/task13215_qodo_sqlite.xml and /tmp/task13215_qodo_postgres.xml. Existing completion SQL ownership review is tracked under TASK-13421, not expanded into this focused fix. Live dev remains 86e287fee7bfa1a1588639232e35db3666851ded and remote PR head remains 59f8c5582651a1bfce6ecce29b17566aa5792e34 before publishing. Requester summary preserved and conditional merge authorized; required hosted checks are still queued.
2026-10-02 published Qodo corrections at 83a68c7203978d9bb0f36a57699d74e637928040. Replied in all four original inline threads with verified fixes or evidence-backed disposition; all four threads are resolved. SQL ownership is separately tracked under TASK-13421, the fixture recommendation is explained as AuthNZ-specific and inapplicable to existing unified Jobs isolation. PR body preserves the requester verbatim Change summary and now correctly states ready-for-review and conditional merge authorization. Current-head frontend-license audit and all required/Jobs gate workflows remain queued; GitHub mergeStateStatus is BLOCKED. No merge, bypass or unrelated CI cancellation performed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the approved completion row-identity atomicity fix in the existing JobManager facade and ordinary WorkerSDK success path. PostgreSQL FOR UPDATE / SQLite BEGIN IMMEDIATE establish authoritative identity; a locked miss or stale expected UUID rejects completion; all mutations/replay use null-safe raw UUID guards. Deterministic race, RLS, legacy UUID, exact replay counter, and real rollback coverage included. Final focused matrices: SQLite 247 passed and required PostgreSQL 90 passed (no skips); post-rebase 16 SQLite/worker and 12 PostgreSQL/RLS passed. Ruff, scoped Black, syntax, whitespace and no-new-finding production/test Bandit comparisons passed; all validated review gaps addressed. Draft PR #3092 against dev is open. Implementation and PR preparation complete; task remains In Progress until remote CI, human-written Change summary, and separately authorized integration are completed. No broad caller migration, historical repair, or completion extraction performed.
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
