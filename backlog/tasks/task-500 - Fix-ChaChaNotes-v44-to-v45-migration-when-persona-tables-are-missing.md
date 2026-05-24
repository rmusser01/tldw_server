---
id: TASK-500
title: Fix ChaChaNotes v44 to v45 migration when persona tables are missing
status: Done
labels:
- bug
- database
- migration
- chachanotes
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix a user-reported ChaChaNotes DB migration failure from schema v44 to v45. Reproduced a v44-marked SQLite DB missing persona persistence tables that fails during post-migration persona schema healing with `no such table: persona_profiles`. Add defensive migration repair and focused regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A regression test covers migrating a v44-marked ChaChaNotes SQLite database that is missing persona persistence tables.
- [x] #2 The migration repairs required persona persistence base tables before v45 visual tables and post-migration persona schema healing run.
- [x] #3 Focused ChaChaNotes migration/persona persistence tests pass.
- [x] #4 Bandit is run for touched backend scope and results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reproduced the reported class of failure with a ChaChaNotes SQLite database marked as schema v44 while missing persona persistence tables; migration reached latest schema cleanup and failed with `sqlite3.OperationalError: no such table: persona_profiles`.
- Added `test_migration_v44_to_latest_repairs_missing_persona_tables` to cover the drifted v44 marker case and verify the persona persistence and v45 visual tables exist after migration to the current schema.
- Added SQLite repair logic before the v44->v45 visual migration and before recent persona schema healing so drifted databases recreate base persona persistence tables and backfill later persona memory/profile columns before latest-column checks run.
- Focused verification passed: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py -q --tb=short` (`16 passed, 5 warnings`).
- Hygiene checks passed: `py_compile` for the touched Python files and `git diff --check`.
- Bandit touched-scope scan completed. Full touched scope initially reported only pytest `B101` assert findings in `test_persona_visuals_db.py`; rerun with pytest asserts skipped had zero findings. Production DB module scan had zero findings. Reports: `/tmp/bandit_chachanotes_v44_v45_migration_prod.json` and `/tmp/bandit_chachanotes_v44_v45_migration_skip_b101.json`.
- Broader adjacent sweep found an unrelated stale-date failure in `test_persona_setup_and_live_voice_analytics_roundtrip`: a hard-coded `2026-04-19T12:00:00Z` session is now outside the test's 30-day filter window on 2026-05-24. The migration-focused suites above passed.
- Reopened for PR review follow-up: Qodo flagged the new regression setup for direct SQLite DDL/DML outside `app/core/DB_Management`.
- Addressed PR review follow-up by moving schema-drift fixture setup into a private `CharactersRAGDB` DB_Management-owned helper and updating persona visual migration tests to call that helper instead of opening SQLite directly.
- Review follow-up verification passed: focused persona visual/persistence tests (`16 passed, 5 warnings`), `py_compile`, `git diff --check`, Bandit production scan (`0 findings`), and Bandit touched-scope scan with pytest asserts skipped (`0 findings`). Reports: `/tmp/bandit_chachanotes_v44_v45_migration_review_prod.json` and `/tmp/bandit_chachanotes_v44_v45_migration_review_skip_b101.json`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Fixed the v44->v45 ChaChaNotes migration path so a database with a collided v44 schema marker and missing persona persistence tables repairs those base persona tables before applying the v45 persona visual schema.
- Added regression coverage for the user-reported failure mode and verified the focused migration/persona persistence suites, syntax, whitespace, and Bandit checks.
- Addressed PR review feedback by routing migration drift setup through DB_Management instead of direct SQLite setup in the regression test.
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
