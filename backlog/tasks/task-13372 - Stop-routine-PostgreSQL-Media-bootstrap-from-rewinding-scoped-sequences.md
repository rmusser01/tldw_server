---
id: TASK-13372
title: Stop routine PostgreSQL Media bootstrap from rewinding scoped sequences
status: Done
assignee: []
created_date: '2026-09-26 01:58'
updated_date: '2026-09-26 02:13'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Synthetic authenticated 100-message MBOX ingestion stored only the first message; 99 inserts hit media_pkey duplicates. Routine handle bootstrap invokes sequence sync before request scope, so forced RLS hides existing rows and rewinds Media sequence. Remove routine post-core sequence synchronization, retain explicit v18 migration maintenance, and validate multi-message archive ingestion with non-superuser forced RLS.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Routine post-core bootstrap never calls destructive sequence maintenance
- [x] #2 Existing v18 sequence migration remains invoked and its tests pass
- [x] #3 Real PostgreSQL repeated handle and archive writes persist distinct IDs under forced RLS
- [x] #4 Focused regression tests, formatting, Bandit, review and evidence are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red: routine bootstrap unit boundary failed on sequence call; real PostgreSQL repeated-handle test failed on second insert. Removed routine post-core sequence synchronization and obsolete protocol requirement; retained explicit v18 migration maintenance. Green: 75 schema unit tests and 3 sequence/FTS cases (2 live PostgreSQL, 1 unit) passed. Full authenticated MBOX probes persisted all 300 IDs, preserved IDs on 100-message retry, passed cross-user API isolation and direct forced-RLS owner 300/other 0 with a non-superuser/non-bypass role, zero model/external attempts. Ruff and Bandit clean across implementation/new test and touched existing test (B101 excluded only for test assertions). Review confirms v18 maintenance unchanged. Commit 41684adbe3. Design Docs/Design/email-postgres-bootstrap-sequence-safety.md; completed own plan removed. Throughput evidence/follow-up separate in TASK-13371.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Routine PostgreSQL Media bootstrap no longer rewinds global sequences from RLS-limited row maxima; explicit v18 migration repair retained. 75 schema unit tests plus 3 sequence/FTS cases (2 live PostgreSQL) passed. Full authenticated 300-message MBOX persistence, retry and isolation passed. Commit 41684adbe3; throughput remains separate.
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
