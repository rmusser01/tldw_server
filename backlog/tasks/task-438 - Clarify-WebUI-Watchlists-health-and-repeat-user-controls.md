---
id: TASK-438
title: Clarify WebUI Watchlists health and repeat-user controls
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-19 15:59'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement WP10 Task 6 as a narrow frontend slice for /watchlists: keep the existing Watchlists runtime, make health/status and repeat-user jump controls visible before deep tabs, preserve first-run terminology and URL query handoff behavior, and verify with focused Vitest/Playwright coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /watchlists route is wrapped in the shared route error boundary without changing route path or runtime ownership.
- [x] #2 Health summary is the first operational status surface and empty watchlist data offers setup actions for feeds and monitors.
- [x] #3 Returning users can jump to Activity, Articles, Reports, and the command palette before deep tab content.
- [x] #4 Failed run recovery is surfaced next to Activity health state and navigates to Activity.
- [x] #5 Existing first-run terminology and query/deep-link handoff behavior remain covered by tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-17-webui-operations-integrations-implementation-plan.md#task-6-clarify-watchlists-health-and-repeat-user-controls
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a shared RouteErrorBoundary wrapper to the /watchlists route. Added a compact repeat-user action row below the health bar for Activity, Articles, Reports, and command palette access. Refined WatchlistsHealthBar empty and failed-run states with concrete setup and recovery actions. Extended route, page, health-bar, static guard, and focused E2E coverage. During E2E verification, the default port 8080 initially reused a stale Next dev server from .worktrees/character-chat-phase2-readiness; reran on isolated port 18080 to verify this checkout.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Clarified /watchlists as a status-first operations surface while preserving the existing Watchlists runtime and route path. Focused verification passed: Watchlists route/component Vitest coverage, Watchlists static guard, diff whitespace checks, and the focused Playwright Watchlists item/journey slice on an isolated local port. Bandit skipped because the touched implementation is frontend TypeScript/TSX and locale/test files only.
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
