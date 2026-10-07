---
id: TASK-13412
title: >-
  ChaCha Postgres HTTP operation lifecycle tests flake (connection ACTIVE after
  owner close; probe 500)
status: Done
assignee: []
created_date: '2026-10-01 17:53'
updated_date: '2026-10-02 21:20'
labels:
  - bug
  - database
  - postgres
  - testing
  - flaky
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tests/DB_Management/test_chacha_postgres_http_operation_lifecycle.py fails intermittently. Locally on macOS about 15-19 of 87 cases fail on both FastAPI 0.141.1 and 0.142.1 (probe asserts raw.info.transaction_status == IDLE after owner.close() but sees ACTIVE, so the probe returns 500). In CI (db-management-a-l, #3063 run 36693239728, 2026-09-30) test_http_caller_keeps_pending_write_until_explicit_decision[rollback-raw-begin] returned 500. The shard passed on #3053.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Determine whether a returned HTTP checkout can stay in an open transaction (product bug) or the tests race the background worker; the file passes reliably
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Two causes, both confirmed locally. Docker Desktop was wedged (hung docker rm/ps), so tests ran against PostgreSQL 16 from pgserver binaries on :55433 using POSTGRES_TEST_DSN and TLDW_TEST_NO_DOCKER=1. (1) Product bug, fixed in PR #3082. Inside a request, a nested get_chacha_db_for_user_id/_for_owner call re-probed the cached instance, and that probe has a 1s deadline. When the probe was slow, the cache evicted the instance the request was already using. The nested lookup then got a new instance and a second checkout, which could not see the caller's pending write. CI run 36693239728 shows this: the dependency was called at 48.24s and the instance was evicted at 49.69s, which is the first probe plus the 1s nested deadline. The rebuild landing on SQLite was already fixed by 72c4e12da6. Forcing the nested probe to report unhealthy makes all 10 caller cases return 500 on dev, and all 10 pass with the fix. Fix: _get_or_init_db_instance skips the re-probe when the current operation already holds a connection to that instance (new helper current_operation_holds_connection). The first resolution still probes with the same deadline. SQLite and auth are unchanged. (2) The ACTIVE-after-close failures were not a leak. psycopg_pool resets a returned connection asynchronously on a pool worker (rollback, then RESET ROLE/SESSION AUTHORIZATION/ALL and COMMIT), and the old test read raw.info from another thread during that reset. The old file at 6110d2ae43 had 24 failures, 18 of them ACTIVE/INTRANS. With the pool reset forced inline, those 18 go away; the 6 that remain are that old file's unrelated 'request did not reach a database connection' failures. The pool republishes a connection only after verifying it is IDLE, and dev already snapshots at publication (2546cf484e). Forcing the first probe of every test to report unhealthy on dev gives 55 passes; the only failure is the test that stubs the probe itself. A latency-proxy run at about 9ms RTT was stopped after 25 of 25 passed.

Qodo on #3082 (merged 2026-10-02 21:18Z): type hints, docstrings and the current_operation_holds_connection Args/Returns contract were addressed in the follow-up PR chore/followups-13410-13416. The probe-count assertion was kept deliberately: it verifies the nested lookup does not re-probe, which is the fix. Re-verified with Docker Postgres: the regression test plus test_chacha_operation_scope.py, 20 passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
No transaction leak: the pool republishes a returned checkout only after verifying it is IDLE. The ACTIVE readings came from the test racing psycopg_pool's async reset, which dev already fixed. The remaining flake was a product bug. A nested accessor's 1s health re-probe could evict the instance a request was already using, splitting the request across two connections, which caused the hosted 500. Fixed by not re-probing an instance that the current operation already holds a connection to. Added a regression test that fails without the fix. Verified: lifecycle file 3 runs, 57/57 passed each time; 239 neighboring ChaCha/PG lifecycle tests passed; ruff clean; Bandit found no issues. Known skip: Docker Desktop was wedged, so tests ran on local PG16 from pgserver rather than the postgres:18 container; hosted CI on PR #3082 still needs to pass.
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
