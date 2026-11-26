---
id: TASK-15669
title: Remove staged media uploads cancelled before worker claim
status: Done
labels:
- media-ingest
- jobs
references:
- https://github.com/rmusser01/tldw_server/pull/3645
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Quick Ingest ordinary single and batch cancellation can leave staged multipart files indefinitely when Jobs cancels a queued upload before a worker runs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both Quick Ingest cancellation routes release the cancelled queued upload by acknowledgement.
- [x] #2 Worker claims, unrelated uploads and untrusted file paths remain safe through repeated cancellation and fresh submission.
- [x] #3 Verify real multipart, Jobs state and filesystem timing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Previous live evidence from PR 3641 and later inspection showed three original queued files still present more than eight hours after cancellation. ADR required: no. ADR-003 and ADR-058 govern the existing Jobs and locked identity transitions. No public contract or schema change is intended.
Plan: reproduce using real multipart + Jobs SQLite, implement safe cleanup at the queued transition, then replay single, batch, repeated cancel and worker claim boundaries and run scoped quality checks.
Local live-tier Uvicorn + SQLite and separate real ingest worker: two trials per route in the final run. Single cancel acknowledgements took 12.8 and 20.9 ms, batch took 29.4 and 36.1 ms. All four individual sources and their empty directories were gone by acknowledgement. A fifth independent upload remained intact, processed successfully and was then cleaned by the worker. Repeated batch cancellations counted the rows already terminal.
155 focused media/SQLite Jobs tests and 3 PostgreSQL integration cases passed. Initial queued single, queued batch, pre-handler, admission rejection and concurrent interrupted-cleanup tests all reproduced on pre-fix code. Safety coverage includes an active successor attempt, expired unclaimed prepare, foreign or copied payloads, replaced directory links, surviving siblings and recovery on repeat cancel. Ruff and Bandit match the dev baseline (Bandit excludes pytest assertion B101 in test files only). No public API schema, migrations or new configuration. Authenticated participants use their configured shared OS temporary directory.
Upgrading: a queued upload created by an older server can still be processed by a worker as before. Its legacy random directory has no authoritative binding to the Jobs UUID. Automatic cancellation cleanup deliberately leaves legacy files in place. For a historical cancellation an operator can reconcile the exact job and source with trusted admission evidence, confirm that no live jobs/workers share the path and remove that individual file and empty directory. There is no historical directory sweep.
Review follow-up verified the previously expired SDK handoff, concurrent interrupted-cancellation recovery and admission-rejection cleanup cases. The standalone implementation is in PR #3645.
CI follow-up: Jobs (SQLite) on e5e24c9d4a failed seven test teardowns because fake PostgreSQL setup allowed the real DSN negotiation to resolve host fake. Fix the shared fake fixture and rerun with the network guard enabled.
Jobs Suite CI had 2,549 passing tests, 4 skipped and seven live-DNS teardown errors in test_jobs_manager_pg_fakes_extended.py. The shared fake now stubs negotiate_pg_dsn before JobManager construction, matching the existing recovery-regressions fixture. Locally reproduced all seven teardown errors before the change, then ran both modules with the normal network guard: 13 passed. Scoped Bandit had zero findings beyond pytest assertions.
The backend-required lint gate on 87561f26ef found admit_job now takes 20 parameters while the recorded ceiling is 19. The additional optional caller-generated Jobs UUID is how multipart staging is bound to its immutable row before publication. Record this bounded, reviewable security rationale in the existing baseline.
Backend CI follow-up: reproduced the function-size ratchet at 20 parameters versus the established 19 on JobManager.admit_job. Recorded a single TASK-15669 security exception for the optional server-issued UUID with its 464-line ceiling unchanged. Narrowed optional JobManager results in new media tests for the advisory type check. Replayed both commands in the backend-required CI contracts and code ratchets step locally under Python 3.12: 1,854 passed, 2 skipped, then 198 passed. The media endpoint suite passed all 42 tests, and the focused function-size module passed all 13. Bandit reported only the pre-existing test credential at line 466.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Quick Ingest removes the UUID-bound multipart source when its queued cancellation commits, before acknowledgement. A worker publishes prepare before touching the source, so an SDK pre-handler cancellation can release a file without deleting input in use by another attempt. Admission failures relinquish only their own unpublished staging. A repeated cancellation recovers an interrupted, never-claimed unlink.
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
