---
id: TASK-351
title: Implement Stage 1A Watchlists DB container and migration default
status: Done
assignee: []
created_date: '2026-05-15 01:29'
updated_date: '2026-05-15 01:42'
labels:
  - watchlists
  - backend
  - stage1a
dependencies:
  - TASK-350
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the backend persistence slice from the Stage 1 first-class Watchlists plan. Scope is limited to Watchlists_DB.py plus DB contract tests: add the Watchlist container schema, default imported Watchlist, source membership join table, job watchlist_id migration/backfill, CRUD helpers, source/job/run/item filters, and compatibility-preserving default behavior. Preserve existing endpoint semantics and do not start API route or frontend work in this task. Known later route-conflict risk: /api/v1/watchlists/{watchlist_id}/clusters currently uses a job id and must be preserved by Stage 1B API tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DB contract tests cover default Watchlist creation, idempotent backfill, source membership, job watchlist_id assignment, CRUD lifecycle, restore behavior, and unchanged source URL uniqueness.
- [x] #2 Watchlists_DB.py adds SQLite/Postgres schema support for watchlists, watchlist_sources, scrape_jobs.watchlist_id, and required indexes using existing migration patterns.
- [x] #3 Existing create/list helpers remain backward compatible when watchlist_id is omitted and support watchlist_id filtering where planned.
- [x] #4 Focused Watchlists DB tests pass and relevant baseline failures, if any, are documented.
- [x] #5 Bandit is run on touched backend code or any skip is explicitly documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline captured before Stage 1A code edits. Route/helper scan confirmed current list_sources, list_jobs, list_outputs, and legacy /{watchlist_id}/clusters locations. Focused backend baseline passed: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_api.py tldw_Server_API/tests/Watchlists/test_runs_list_global.py tldw_Server_API/tests/Watchlists/test_watchlist_clusters_api.py -q => 30 passed, 5 warnings in 57.91s.

Implemented Stage 1A persistence in Watchlists_DB.py. Added WatchlistRow, watchlists/watchlist_sources schema for SQLite and Postgres, lazy default Imported Watchlist creation, idempotent source/job backfill, CRUD lifecycle helpers, source membership helpers, watchlist_id on JobRow, and scoped list filters for sources, jobs, runs, items, and smart counts.

Verification: new DB contract tests first failed as expected for missing watchlist_id/create_watchlist/schema support, then passed after implementation. Final focused backend run: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_first_class_watchlists_db.py tldw_Server_API/tests/Watchlists/test_watchlists_db_user_scope.py tldw_Server_API/tests/Watchlists/test_watchlists_api.py tldw_Server_API/tests/Watchlists/test_runs_list_global.py tldw_Server_API/tests/Watchlists/test_watchlist_clusters_api.py -q => 39 passed, 5 warnings in 54.68s.

Bandit: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/Watchlists_DB.py -f json -o /tmp/bandit_watchlists_stage1a.json => 0 results.

Postgres integration command was attempted: python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_postgres_integration.py -q => 4 skipped because PostgreSQL was unavailable; the new DB contract test includes a captured Postgres DDL/index assertion for the container schema path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 1A of first-class Watchlists at the persistence layer. This adds the Watchlist container tables, source membership join table, job watchlist_id scope, lazy default Imported Watchlist migration/backfill behavior, CRUD lifecycle helpers, restore support, and scoped source/job/run/item list helpers while preserving existing unscoped API behavior through default attachment.

Tests: focused backend Watchlists suite passed with 39 passed and 5 warnings. Bandit on Watchlists_DB.py reported 0 results. Postgres integration tests were attempted but skipped because PostgreSQL was unavailable; Postgres DDL/index coverage is included in the new DB contract tests.
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
