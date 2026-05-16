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
- apps/packages/ui/src/components/Option/Watchlists/AlertsTab/AlertsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/AlertsTab/__tests__/AlertsTab.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.advanced-filters.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.defensible-reports.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.extension-management.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/SetupWizard/WatchlistSetupWizard.tsx
- apps/packages/ui/src/components/Option/Watchlists/SetupWizard/__tests__/WatchlistSetupWizard.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceFormModal.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesBulkImport.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.forum-help.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourcesBulkImport.preflight-commit.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/TemplatesTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/__tests__/TemplatesTab.extension-management.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx
- backlog/tasks/task-349.3.5 - Stage-6E-real-server-constrained-viewport-QA-and-closeout.md
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
Stage 6E is closed on the clean branch after rebasing onto latest origin/dev; branch divergence was 0 behind / 42 ahead at the final verification point. A small AlertsTab test stabilization now waits for the async alert title render before asserting, matching the component's async data flow. The final real-server CDP pass ran after the final rebase using the real FastAPI server on 127.0.0.1:18011 plus the real Next WebUI on 127.0.0.1:18012, seeded CTI and news Watchlists through the real API, and covered overview, sources, jobs, runs, items, alerts, outputs, templates, and settings at 420x760 and 1280x900. Final report: /private/tmp/tldw-watchlists-first-class-clean-cdp-rerun/watchlists-cdp-report.json; screenshots: /private/tmp/tldw-watchlists-first-class-clean-cdp-rerun/screenshots. Results: 18 states, consoleMessages=0, pageErrors=0, badResponses=0, requestFailures=0; extension viewport overflow/offenders=0 on every tab. Desktop still shows the known 48px global app-shell overflow offset, not a Watchlists constrained-management regression. Verification passed after the final rebase: git diff --check; bun run test:watchlists:typecheck (1 file, 3 tests); focused Alert/template/source/output test group (7 files, 32 tests); additional touched mock group (3 files, 10 tests); branch-owned backend Watchlists tests (48 passed, 5 warnings); Bandit touched Python scope with zero high/medium/low findings in /tmp/bandit_watchlists_first_class_clean_final_post_rebase.json. Earlier broad Watchlists backend sweep still has unrelated baseline/environment failures already recorded separately.
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
