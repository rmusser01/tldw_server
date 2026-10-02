---
id: TASK-13412
title: >-
  ChaCha Postgres HTTP operation lifecycle tests flake (connection ACTIVE after
  owner close; probe 500)
status: To Do
assignee: []
created_date: '2026-10-01 17:53'
updated_date: '2026-10-01 17:55'
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
- [ ] #1 Determine whether a returned HTTP checkout can stay in an open transaction (product bug) or the tests race the background worker; the file passes reliably
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
