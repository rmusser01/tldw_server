---
id: TASK-13370
title: Avoid rewriting unchanged PostgreSQL FTS vectors during bootstrap
status: Done
created_date: 2026-09-26 01:18
updated_date: 2026-09-26 01:38
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every PostgreSQL Media handle runs FTS setup, whose refresh UPDATE currently rewrites every visible source row even when its tsvector is already current. Add a null-safe difference predicate so missing or stale vectors are repaired while unchanged rows do not generate writes. This reduces unnecessary WAL; it does not establish the cause of the host disk exhaustion observed during TASK-13369.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FTS refresh SQL skips unchanged vectors while retaining null/stale-vector repair and existing trigger/index behavior
- [x] #2 Focused unit regression rejects an unconditional refresh and existing PostgreSQL backend tests pass
- [x] #3 Real isolated PostgreSQL verification confirms repeated setup leaves unchanged rows untouched after the local storage failure is resolved
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented a null-safe IS DISTINCT FROM predicate in PostgreSQLBackend.create_fts_table so only missing/stale tsvectors are refreshed. TDD unit regression failed on the prior unconditional UPDATE, then passed. Related non-integration suite: 24 passed. Real standard pg_database_config fixtures: 2 passed, covering null/stale repair, unchanged xmin row versions on repeated setup, and existing FTS search mapping. Final combined benchmark/backend suite: 32 passed, 2 integration tests deselected because they had already passed separately. Bandit implementation scope: 0 findings/errors. Fatal Ruff on backend and full Ruff/format on new test passed; pre-existing broad legacy file formatting was not changed. Evidence: Docs/Operations/Email_PostgreSQL_Search_Performance_2026-09-25.md. TASK-13369 retains the separate disposable-resource cleanup blocker after recurring host/Docker storage failures.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
FTS bootstrap repairs missing/stale vectors without rewriting unchanged rows. Unit and real PostgreSQL regression coverage passed, along with existing FTS search behavior. This avoids unnecessary row writes/WAL but does not establish the cause of host disk exhaustion or certify the 1M scale target.
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
