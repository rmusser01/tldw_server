---
id: TASK-365
title: Complete Stage 1E Watchlists integration verification and API docs closeout
status: Done
assignee: []
created_date: '2026-05-15 03:36'
updated_date: '2026-05-15 04:15'
labels:
  - watchlists
  - stage1e
  - verification
  - docs
dependencies:
  - TASK-355
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
documentation:
  - Docs/API-related/Watchlists_API.md
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage1-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close out the Stage 1 first-class Watchlists implementation after the selector shell commit. Scope is limited to focused backend/frontend regression verification, fixing any scoped regressions found by those checks, updating Watchlists API documentation for the new first-class container contract, and recording final evidence. Preserve existing /watchlists route behavior, legacy cluster route compatibility, and the Stage 1 boundary that content-match alerts remain future work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused backend Watchlists regression suite passes or any failures are fixed/documented with concrete scope.
- [x] #2 Focused frontend Watchlists service/page/route suite passes or any failures are fixed/documented with concrete scope.
- [x] #3 Bandit is run against touched backend Watchlists Stage 1 files and results are recorded.
- [x] #4 Watchlists API documentation describes Watchlist CRUD, default migrated Watchlist behavior, watchlist_id scoping for sources/jobs/child lists, output provenance behavior, and the future-work boundary for content-match alerts.
- [x] #5 Stage 1 closeout records current browser/CDP evidence or reruns a scoped smoke if documentation/code changes affect rendered UI.
- [x] #6 Task notes and final summary capture verification commands, known baseline issues, and remaining post-Stage-1 product work.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 1E verification notes:
- Added and verified regression coverage for Watchlist-scoped outputs. New outputs now record metadata.watchlist_id, metadata.job_id, and metadata.run_id. GET /api/v1/watchlists/outputs?watchlist_id=<id> scopes through Watchlist job IDs so legacy job-linked outputs remain visible after migration. The job-ID lookup is paged rather than capped at a fixed job count.
- Backend focused suite: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Watchlists/test_first_class_watchlists_db.py tldw_Server_API/tests/Watchlists/test_first_class_watchlists_api.py tldw_Server_API/tests/Watchlists/test_watchlists_api.py tldw_Server_API/tests/Watchlists/test_runs_list_global.py tldw_Server_API/tests/Watchlists/test_watchlist_clusters_api.py tldw_Server_API/tests/Watchlists/test_preview_endpoint.py -q. Result: 42 passed, 5 warnings in 319.01s.
- Frontend focused suite from apps/packages/ui: ./node_modules/.bin/vitest run src/services/__tests__/watchlists-first-class.test.ts src/services/__tests__/watchlists-overview.test.ts src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx src/components/Option/Watchlists/__tests__/watchlists-selected-scope-contract.test.ts src/routes/__tests__/option-watchlists.route-state.test.tsx --maxWorkers=1 --no-file-parallelism. Result: 5 files passed, 21 tests passed.
- Bandit: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/DB_Management/Watchlists_DB.py tldw_Server_API/app/core/DB_Management/Collections_DB.py tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py -f json -o /tmp/bandit_watchlists_stage1e.json. Result: exit 0, JSON results count 0.
- Docs updated in Docs/API-related/Watchlists_API.md and mirrored to Docs/Published/API-related/Watchlists_API.md. cmp confirmed both docs match.
- Diff hygiene: git diff --check passed after final edits.
- Browser/CDP: no rendered UI changes in Stage 1E. Reused Stage 1D CDP smoke evidence: /watchlists at extension-sized 390x844 with mocked API routes rendered the selector, create flow, mobile tab select, no horizontal overflow, and scoped watchlist_id=42 source request.
- Remaining post-Stage-1 product work: setup wizard, content-match alerts, triage/change review, defensible report builder, lifecycle semantics, and full constrained-viewport management hardening.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Stage 1E closeout. Fixed the remaining backend contract gap for Watchlist-scoped outputs, updated API docs for first-class Watchlist behavior and Stage 1 alert/report boundaries, recorded fresh focused backend/frontend/Bandit verification, and preserved the existing Stage 1D browser/CDP evidence because this slice changed backend/docs only.
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
