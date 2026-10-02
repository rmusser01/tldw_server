---
id: TASK-13403
title: Close ChaChaNotes SQLite/Postgres schema parity gap (v68 vs v72)
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch 3 Stages 1-2. Plan: Docs/plans/2026-10-01-db-parity-structural-debt-implementation-plan.md. ChaChaNotes_DB.py:755-756 (_CURRENT_SCHEMA_VERSION=68, _POSTGRES_SCHEMA_VERSION=72); PG-only path gated >=71 at :24902. Audit doc first (port/descope per migration), then execute with parity tests mirroring test_chacha_postgres_migration_* pattern. No bulk refactor of the 45k-line file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
