---
id: TASK-378.2
title: Stage 3B Watchlist content alert API and docs
status: Done
assignee: []
created_date: '2026-05-15 14:53'
updated_date: '2026-05-15 15:10'
labels:
  - watchlists
  - stage3
  - api
  - docs
dependencies: []
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage3-content-alerts-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
documentation:
  - Docs/API-related/Watchlists_API.md
parent_task_id: TASK-378
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose nested content alert rule and alert inbox endpoints for selected Watchlists and document the content-alert versus health-rule boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Nested Watchlist content alert APIs support rule CRUD, alert list/detail, filters, and review-state updates.
- [x] #2 API validation returns clear errors for invalid regex, source constraints, and missing Watchlist scope.
- [x] #3 Docs distinguish content alerts from health rules and identify Topic Monitoring as an internal dependency boundary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Stage 3B after Stage 3A commit aebb3e4a6. Proceeding API test-first for nested content alert rule and alert inbox endpoints.

RED: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Watchlists/test_watchlist_content_alerts_api.py -q failed with 404 for missing nested content-alert routes. GREEN: API test file passed. Regression: 40 focused backend tests passed across content alert API/DB/matcher, first-class Watchlists, existing run-stat alert rules, and Topic Monitoring. Docs grep confirmed content-alert routes, health issue boundary, Topic Monitoring boundary, and watchlist_content terms. Verification: git diff --check passed. Bandit: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/DB_Management/Watchlists_DB.py tldw_Server_API/app/core/Watchlists/content_alerts.py tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py -f json -o /tmp/bandit_watchlists_stage3b_content_alerts_api.json passed with zero results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 3B content-alert API and docs: added Pydantic schemas for content alert rules and alerts, nested Watchlist endpoints for rule CRUD, alert list/detail, filtered inbox queries, and review-state updates, plus API documentation for content alerts and the health-rule boundary. Added API coverage for CRUD, invalid regex validation, filters, detail, read/dismiss states, and Watchlist scoping.
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
