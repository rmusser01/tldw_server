---
id: TASK-349.3.5
title: Stage 6E real-server constrained viewport QA and closeout
status: Done
dependencies:
- TASK-349.3.4
labels:
- watchlists
- stage6
- qa
- cdp
priority: medium
parent_task_id: TASK-349.3
documentation:
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md
- Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.scope-filter-summary.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.source-column.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunsTab.extension-management.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/SetupWizard/WatchlistSetupWizard.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsMobileNavigation.tsx
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run focused Stage 6 frontend verification, static checks, and real FastAPI plus real WebUI CDP smoke at extension-sized viewport, recording screenshots, console/network notes, known skips, and final Backlog closeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused Stage 6 frontend Vitest suite and `git diff --check` pass, or blockers are documented with exact failing commands and output.
- [x] #2 Bandit is run on touched Python scope if Python changed; otherwise the frontend-only skip is explicitly documented.
- [x] #3 Real FastAPI plus real Next WebUI CDP smoke runs without mocked server and covers constrained navigation plus Feeds, Monitors, Alerts, Updates, Activity, Reports, Templates, and Settings management reachability.
- [x] #4 CDP smoke records screenshots, console messages, request failures, horizontal-overflow checks, and any seed/setup caveats.
- [x] #5 Stage 6 plan and all `TASK-349.3*` Backlog records are updated with verification evidence, known skips/blockers, and final summaries.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Depends on `TASK-349.3.4`. Follow `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md` Task 5. Browser QA must use CDP/Playwright against real FastAPI and real WebUI. Do not mock the server.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 6E closed out constrained Watchlists management with real-server CDP evidence and focused regressions. The QA pass used the real FastAPI app plus real Next WebUI, seeded representative CTI and news Watchlists, and captured `/watchlists?view=all` at 420x760 and 1280x900. Extension-sized Overview, Feeds, Monitors, Updates, Activity, Reports, Templates, and Settings all measured zero horizontal overflow/offenders with screenshots under `/private/tmp/tldw-watchlists-stage6/screenshots/`. The final CDP report at `/private/tmp/tldw-watchlists-stage6/stage6e-cdp-watchlists-qa.json` recorded consoleCount=0, pageErrorCount=0, networkFailureCount=0, and badResponseCount=0. Stage 6E also fixed real API list calls to respect the backend `size <= 200` contract, compacted the constrained Activity CSV toolbar, and removed Watchlists-local AntD deprecation warnings. Verification passed: focused Stage 6 suite, API-contract Watchlists tests, `bun run test:watchlists:typecheck`, final real-server CDP, and `git diff --check`. Bandit was not applicable because no Python files changed.
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
