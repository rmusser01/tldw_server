---
id: TASK-13369
title: Benchmark scoped PostgreSQL email search with synthetic load
status: Done
assignee: []
created_date: '2026-09-26 00:55'
updated_date: '2026-09-26 02:08'
labels: []
dependencies: []
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
- [x] #4 Disposable benchmark PostgreSQL databases and role are confirmed removed after stable host and Docker storage recovery
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scoped PostgreSQL benchmark and 10k search/RLS evidence are committed in 638066dd0b; conditional FTS refresh in 6204632094. Focused checks: 32 passed and 2 live PostgreSQL cases separately passed; Ruff/Bandit clean. Dataset 10k messages/2487 attachments/23 labels; six-operator warm p50 29.90 ms/p95 96.85 ms, forced RLS owner 10000/other 0, zero model/external attempts. These bounded results do not certify 1M scale. Cleanup initially failed on host/Docker I/O errors. Follow-up confirmed 237 GiB free, empty Docker container/volume inventories, prior container absent and old port closed: previous disposable store and resources no longer exist. Old credential manifest was removed. Task closed; own completed plan removed. Report Docs/Operations/Email_PostgreSQL_Search_Performance_2026-09-25.md. Later archive measurements/fix are separate TASK-13371/TASK-13372.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
10k scoped PostgreSQL benchmark and RLS evidence committed; previous disposable resources confirmed absent after Docker test-store removal, obsolete credential manifest removed. 1M search scale, throughput and production topology remain open.
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
