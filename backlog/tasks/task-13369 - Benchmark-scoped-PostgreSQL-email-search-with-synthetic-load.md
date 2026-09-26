---
id: TASK-13369
title: Benchmark scoped PostgreSQL email search with synthetic load
status: In Progress
assignee: []
created_date: 2026-09-26 00:55
labels: []
dependencies: []
updated_date: 2026-09-26 01:44
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the existing reproducible email search benchmark to run against an explicitly selected PostgreSQL content backend under a numeric user RLS scope. Measure a staged synthetic fixture without Gmail, external model calls, or real mail; record backend-specific profile and limits against the 1M target.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CLI rejects PostgreSQL mode without a numeric scope user and reports the actual backend without exposing credentials
- [x] #2 Scoped PostgreSQL fixture and search benchmark run on isolated local test database with non-superuser forced RLS
- [x] #3 Benchmark JSON and operations report record dataset, hardware, query mix, measured results, and remaining scale/deployment limits
- [ ] #4 Disposable benchmark PostgreSQL databases and role are confirmed removed after stable host and Docker storage recovery
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented explicit backend selection, positive numeric PostgreSQL scope, backend mismatch rejection, credential-free backend reporting, ISO date bounds, and an explicit NFR size/operator gate. Design: Docs/Design/email-search-postgresql-benchmark.md. Plan: IMPLEMENTATION_PLAN_email_postgres_benchmark_13369.md. Final combined focused tests: 32 passed, 2 integration tests deselected after separate real PostgreSQL verification; Ruff check/format passed on benchmark files and Bandit implementation scopes had 0 findings/errors. SQLite 100-message smoke also passed. Final PostgreSQL dataset: 10,000 messages, 2,487 attachment rows, 23 labels. Default mix: 150 warm samples p50 33.39 ms/p95 106.86 ms. Required six operators: 90 warm samples p50 29.90 ms/p95 96.85 ms, all positive matches; NFR gate false at 1% of required mailbox scale. Direct RLS: role superuser=false/bypass=false, media RLS enabled/forced, owner rows/search 10000, other user rows/search 0 even with explicit owner tenant. Both final socket guards recorded 0 non-loopback attempts. Reports saved under Docs/Operations; no personal mail or Gmail was accessed. First 10k query pass failed on WAL fsync after host disk reached 119 MiB free; Docker recovery allowed valid query-only reruns after TASK-13370. Storage became unstable again while saving evidence, and final cleanup failed at Docker inspection before connecting. Keep task In Progress until cleanup confirmed. Private manifest /tmp/email_pg_manifest_13364.json and cleanup script /tmp/email_pg_databases_13364.py remain outside repo. Do not start another large fixture until host/Docker storage is stable. Report: Docs/Operations/Email_PostgreSQL_Search_Performance_2026-09-25.md.
Final self-review removed duplicate schema initialization because the Media factory already initializes each handle; the regression rejects a second initialization call. Final verification after that change: 32 passed, 2 deselected; Ruff check/format passed; benchmark Bandit 0 findings/errors. Both saved 10k JSON files exactly match their successful run artifacts and contain no disposable database password. Cleanup remains unconfirmed; task remains In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Scoped PostgreSQL benchmark CLI and 10k performance/RLS evidence are implemented and verified. 1M, archive throughput and production topology gates remain open. Disposable database/role cleanup is pending recurring host/Docker storage failure, so this task remains In Progress.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
