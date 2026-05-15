---
id: TASK-352
title: Implement Stage 1B Watchlist CRUD API and child scoping
status: Done
assignee: []
created_date: '2026-05-15 01:43'
updated_date: '2026-05-15 01:55'
labels:
  - watchlists
  - backend
  - api
  - stage1b
dependencies:
  - TASK-351
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the API slice from the Stage 1 first-class Watchlists plan. Scope is limited to Pydantic schemas, watchlists router endpoints, and backend calls needed for CRUD plus watchlist_id scoping on sources/jobs/runs/items while preserving existing unscoped behavior and legacy /{watchlist_id}/clusters route semantics. Do not start frontend work in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 API tests cover Watchlist CRUD, patch lifecycle fields, delete/restore, default scope for omitted watchlist_id, explicit watchlist_id filtering, and legacy /{watchlist_id}/clusters stability.
- [x] #2 watchlists_schemas.py exposes Watchlist create/update/container/list/delete schemas and backward-compatible watchlist_id fields on source/job contracts.
- [x] #3 watchlists.py adds root CRUD endpoints without shadowing static child routes and wires watchlist_id to source/job/run/item create/list paths.
- [x] #4 Focused Watchlists API and DB tests pass after implementation.
- [x] #5 Bandit is run on touched backend API/schema/router code or any skip is explicitly documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Stage 1B in isolated worktree .worktrees/watchlists-stage1a on branch codex/watchlists-stage1a. Scope is limited to Watchlist CRUD API schemas/router tests and child endpoint watchlist_id scoping; frontend work remains out of scope.

Added red/green API coverage in tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py for Watchlist CRUD, delete/restore, default source/job scoping, explicit watchlist_id filters, smart counts, and legacy cluster route stability.

Implemented Watchlist CRUD schemas/endpoints plus watchlist_id/watchlist_ids source and job contract fields. Static child routes remain ahead of dynamic root Watchlist routes; dynamic get/patch/delete/restore routes are defined after existing static routes.

Wired watchlist_id to source/job create/list, bulk/OPML source import, global runs, scraped items, and smart counts. Fixed default Watchlist creation so repeated ensure_default_watchlist() calls do not reattach explicitly scoped sources to the default Watchlist; explicit backfill_default_watchlist_scope() remains available for migration repair.

Verification: python -m pytest tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py -q -> 3 passed, 5 warnings. Focused regression: python -m pytest tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py tldw_Server_API/tests/Watchlists/test_first_class_watchlists_db.py tldw_Server_API/tests/Watchlists/test_watchlists_db_user_scope.py tldw_Server_API/tests/Watchlists/test_watchlists_api.py tldw_Server_API/tests/Watchlists/test_runs_list_global.py tldw_Server_API/tests/Watchlists/test_watchlist_clusters_api.py -q -> 42 passed, 5 warnings. Bandit: python -m bandit -r touched backend files -f json -o /tmp/bandit_watchlists_stage1b.json -> results 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 1B exposes first-class Watchlist CRUD through /api/v1/watchlists, adds backward-compatible watchlist_id/watchlist_ids fields to source/job contracts, and scopes source/job/run/item/count APIs by Watchlist while preserving legacy cluster route behavior. Focused Watchlists API/DB regression tests pass and Bandit reported no findings on touched backend files.
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
