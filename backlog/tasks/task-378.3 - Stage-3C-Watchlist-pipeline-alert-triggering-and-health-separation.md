---
id: TASK-378.3
title: Stage 3C Watchlist pipeline alert triggering and health separation
status: Done
assignee: []
created_date: '2026-05-15 14:53'
updated_date: '2026-05-15 15:16'
labels:
  - watchlists
  - stage3
  - pipeline
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
Trigger content alerts from the Watchlists ingestion pipeline and keep run-stat alert-rule notifications health-oriented.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Newly recorded matching items create deduped content alerts with item evidence.
- [x] #2 Pipeline content-alert evaluation is non-critical and cannot fail a scrape run.
- [x] #3 Run-stat alert rules remain backward compatible while using health-oriented notification type or metadata.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 3C RED: test_watchlist_alert_rules.py and test_watchlist_content_alerts_pipeline.py failed on missing health-oriented run-stat payload and missing pipeline content-alert invocation. GREEN: 11 focused tests passed after implementation. Regression: watchlist alert rules, content-alert pipeline tests, and Topic Monitoring tests passed: 26 passed. Bandit: /tmp/bandit_watchlists_stage3c_pipeline_alerts.json with zero errors/results. Documentation: API docs were already updated in Stage 3B; no additional public route contract changed in Stage 3C.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 3C pipeline integration. Ingested Watchlist items now invoke content-alert evaluation after record_scraped_item, scoped to the job Watchlist or source Watchlist fallback, with matcher failures logged as non-critical. Run-stat alert-rule notifications now identify as watchlist_health_issue while preserving legacy_kind=watchlist_alert and watchlist_run linkage for compatibility. Verification: red tests failed before implementation; focused tests passed (11 passed), regression with Topic Monitoring passed (26 passed), full Stage 3 backend suite passed (43 passed), git diff --check passed, and Bandit reported zero findings in /tmp/bandit_watchlists_stage3c_pipeline_alerts.json. Known skips or blockers: none for this stage.
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
