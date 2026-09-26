---
id: TASK-13372
title: Stop routine PostgreSQL Media bootstrap from rewinding scoped sequences
status: Done
assignee: []
created_date: '2026-09-26 01:58'
updated_date: '2026-09-26 02:07'
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
TDD red: routine bootstrap regression failed on sequence call; new real PostgreSQL test failed on second scoped insert after unscoped handle construction. Green after removing routine sync: 75 schema unit tests passed (21 integration deselected), 3 real PostgreSQL sequence/FTS tests passed. v18 migration sequence test remains green. Full authenticated archive rerun is running on fresh isolated databases. Design Docs/Design/email-postgres-bootstrap-sequence-safety.md; plan IMPLEMENTATION_PLAN_email_postgres_sequences_13372.md.

Full loopback PostgreSQL archive probe passed after the fix: 300 distinct messages, 100-message rerun preserves IDs, owner native search/detail pass, other user search 0/detail 404, direct forced RLS owner rows 300/other 0, role superuser=false/bypass=false, zero model/external attempts. Throughput is still below target (aggregate 5.03 messages/sec); tracked separately in TASK-13371. Ruff check/format passed on touched implementation and new integration test; implementation Bandit 0 findings/errors, new test Bandit 0 with B101 excluded for assertions. Review confirms explicit v18 migration path unchanged. Plan completed and removed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed sequence synchronization from routine PostgreSQL post-core bootstrap, preserving allocated IDs when unscoped handles cannot see existing rows under forced RLS. Explicit v18 migration sequence repair retained. 75 unit tests and 3 live PostgreSQL tests passed; authenticated 300-message MBOX upload, retry identity and cross-user RLS passed. Throughput optimization remains separate.
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
