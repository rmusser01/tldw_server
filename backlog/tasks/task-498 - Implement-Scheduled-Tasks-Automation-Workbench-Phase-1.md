---
id: TASK-498
title: Implement Scheduled Tasks Automation Workbench Phase 1
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-01 19:33'
labels:
  - scheduled-tasks
  - webui
  - ux
  - phase-1
dependencies: []
references:
  - TASK-496
  - TASK-494
  - >-
    Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md
  - >-
    Docs/superpowers/plans/2026-06-01-scheduled-tasks-automation-workbench-phase1-implementation-plan.md
documentation:
  - >-
    Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md
  - >-
    Docs/superpowers/plans/2026-06-01-scheduled-tasks-automation-workbench-phase1-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Phase 1 of the Scheduled Tasks Automation Workbench UX from the approved PRD and implementation plan. Scope stays at the product/UI layer: unified /scheduled-tasks visibility, IA shell, status model, table/detail inspection, existing reminder plus Watchlists job run/result links, and safer reminder scheduling controls. Watchlists remains a separate full workspace and is not replaced or limited.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Scheduled task rows use user-facing statuses instead of raw API status strings.
- [ ] #2 /scheduled-tasks shows overview metrics, loading, empty, partial, unsupported, and error states with recovery actions.
- [ ] #3 Reminder tasks can be created and edited with safe one-time and recurring schedule controls instead of raw-first run_at/cron/timezone fields.
- [ ] #4 Watchlist jobs are visible as externally managed monitors with deep links to Watchlists settings, activity, latest run, reports, and latest output when identifiers exist.
- [ ] #5 Task detail inspection shows current state, schedule, last/next run, source metadata, and available actions without editing Watchlist jobs.
- [ ] #6 Existing Watchlists functionality and UX remain intact and separate.
- [ ] #7 Focused WebUI tests and relevant backend contract tests pass, or any skipped checks are documented with cause.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

Task 1 review clarification: failure-like status tokens (fail, error, missed) take precedence over result-like tokens when both appear in a single backend status string. This keeps states such as output_error in Needs attention rather than Found results. Helper tests now cover output ID aliases, token-boundary matching, and unsafe ID rejection.
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
