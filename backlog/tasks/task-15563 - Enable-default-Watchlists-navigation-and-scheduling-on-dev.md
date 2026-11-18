---
id: TASK-15563
title: Enable default Watchlists navigation and scheduling on dev
status: Done
assignee: []
created_date: '2026-11-18T02:34:00Z'
updated_date: '2026-11-18T02:41:00Z'
labels: []
dependencies: []
references:
  - https://github.com/rmusser01/tldw_server/pull/3505
documentation:
  - Docs/ADR/003-jobs-vs-scheduler-default.md
ordinal: 13969
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose Watchlists in hosted navigation, enable recurring workflow scheduling by default with an opt-out, and fix PostgreSQL schedule flag persistence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Watchlists appears in default hosted navigation.
- [x] #2 The recurring scheduler starts by default and honors an explicit opt-out.
- [x] #3 Workflow schedules can be created and paused on PostgreSQL and SQLite.
- [x] #4 Open a pull request targeting dev.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add regression tests for navigation, startup gates, and database writes. Make focused changes and verify them before opening the dev PR. ADR required: no. ADR-003 governs the existing Scheduler ownership, which does not change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Regressions reproduced the hidden hosted shortcuts, disabled default scheduler, and PostgreSQL type mismatch. With the fix, 35 frontend and 42 backend tests passed, including the project PostgreSQL 18 fixture and SQLite. Bandit found no issues in the changed scope. Ruff reported the same three existing findings before and after the change.

Verification was scoped to the changed navigation and scheduler paths. GitHub review and required CI are tracked on the PR.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Published a dev PR with default hosted Watchlists visibility, default recurring scheduler startup with an opt-out, and PostgreSQL-compatible boolean bindings.
<!-- SECTION:FINAL_SUMMARY:END -->
