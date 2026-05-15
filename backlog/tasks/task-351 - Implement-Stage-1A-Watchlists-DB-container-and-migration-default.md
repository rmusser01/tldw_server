---
id: TASK-351
title: Implement Stage 1A Watchlists DB container and migration default
status: In Progress
assignee: []
created_date: '2026-05-15 01:29'
updated_date: '2026-05-15 01:30'
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
- [ ] #1 DB contract tests cover default Watchlist creation, idempotent backfill, source membership, job watchlist_id assignment, CRUD lifecycle, restore behavior, and unchanged source URL uniqueness.
- [ ] #2 Watchlists_DB.py adds SQLite/Postgres schema support for watchlists, watchlist_sources, scrape_jobs.watchlist_id, and required indexes using existing migration patterns.
- [ ] #3 Existing create/list helpers remain backward compatible when watchlist_id is omitted and support watchlist_id filtering where planned.
- [ ] #4 Focused Watchlists DB tests pass and relevant baseline failures, if any, are documented.
- [ ] #5 Bandit is run on touched backend code or any skip is explicitly documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline captured before Stage 1A code edits. Route/helper scan confirmed current list_sources, list_jobs, list_outputs, and legacy /{watchlist_id}/clusters locations. Focused backend baseline passed: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_api.py tldw_Server_API/tests/Watchlists/test_runs_list_global.py tldw_Server_API/tests/Watchlists/test_watchlist_clusters_api.py -q => 30 passed, 5 warnings in 57.91s.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
