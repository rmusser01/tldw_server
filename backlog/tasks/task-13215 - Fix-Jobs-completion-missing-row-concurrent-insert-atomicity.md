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
documentation:
- Docs/superpowers/specs/2026-09-07-jobs-completion-row-identity-atomicity-design.md
- Docs/superpowers/plans/2026-09-07-jobs-completion-row-identity-atomicity-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-09-07-jobs-completion-row-identity-atomicity-design.md
- Docs/superpowers/plans/2026-09-07-jobs-completion-row-identity-atomicity-implementation-plan.md
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/app/core/Jobs/worker_sdk.py
- tldw_Server_API/tests/Jobs
updated_date: 2026-09-07 19:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Blocking remediation required before strict complete_job extraction. A completion attempt whose initial row lookup returns no row can race with a concurrent insert of the same numeric id. The later guarded update can complete that new row while base facts remain absent, so complete_job returns True without completion counters, job.completed outbox persistence, metrics, or observers. Design and implement an atomic row-identity/fact boundary before resuming the completion extraction stream.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A deterministic SQLite regression reproduces the initial-miss/concurrent-insert race before the fix.
- [ ] #2 A real-PostgreSQL regression validates whether the same race is possible under the configured isolation and RLS cursor path.
- [ ] #3 No completion can commit successfully unless the operation has authoritative facts for the exact durable row identity it transitions.
- [ ] #4 An applied completion updates or reconciles the correct lifecycle counter and writes job.completed atomically when the outbox is enabled.
- [ ] #5 Concurrent insert, delete/reinsert, and row-visibility changes cannot cause completion of a different row incarnation selected only by a reused numeric id.
- [ ] #6 Normal processing completion, permitted queued completion, token replay, missing-row, failure precedence, and RLS behavior are explicitly characterized and intentionally updated where remediation requires it.
- [ ] #7 Focused SQLite and required real-PostgreSQL tests, full relevant Jobs regressions, formatting/lint, and scoped Bandit pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Validated on SQLite and real PostgreSQL 18 under READ COMMITTED, including the forced-RLS visible-row path: after an initial missing SELECT, a concurrent queued insert with the same numeric id can be completed while only job.created exists and ready_count remains 1. The approved remediation locks and loads the target at transaction start, returns False on initial miss, captures the stored UUID as row-incarnation identity, and guards every transition/replay query by id plus null-safe stored UUID. PostgreSQL uses SELECT FOR UPDATE; SQLite uses BEGIN IMMEDIATE. expected_uuid is optional and WorkerSDK supplies it. Completion outbox and lifecycle counter bookkeeping remain mandatory and atomic; SLA persistence remains best-effort under its existing savepoint; metrics and observers remain post-commit. Strict completion extraction stays paused until this blocker merges. The previous provisional TASK-13112.3 was not carried forward because current dev already assigns TASK-13112 to unrelated work.
Design written and re-audited on 2026-09-07 against origin/dev e3174f1ad9f6. The audit clarified that null/empty legacy UUIDs cannot provide pre-call incarnation distinction, even when WorkerSDK forwards the exposed empty UUID; those rows receive only the transaction's in-operation replacement protection. It also records why the slides-specific terminal-result operation is not a suitable reuse target for ordinary completion.
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
