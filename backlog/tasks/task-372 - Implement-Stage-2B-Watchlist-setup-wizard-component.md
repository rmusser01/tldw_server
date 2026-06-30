---
id: TASK-372
title: Implement Stage 2B Watchlist setup wizard component
status: Done
assignee: []
created_date: '2026-05-15 04:55'
updated_date: '2026-05-15 05:34'
labels:
  - watchlists
  - stage2
  - frontend
dependencies:
  - TASK-371
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage2-setup-wizard-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build the React setup wizard component for Stage 2 using the Stage 2A model. Scope: Ant Design wizard modal/drawer, domain preset selection, start mode selection, objective/tracked scope fields, optional source/report/monitor fields, review step, validation, and component tests with injected service callbacks. No shell integration in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WatchlistSetupWizard renders domain presets, start modes, setup fields, and review step.
- [x] #2 Component tests cover CTI/news preset behavior, topic-only creation, source-backed creation, report-goal creation, and required-name validation.
- [x] #3 Component accepts service callbacks so tests do not mock the whole page shell.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 2B started after Stage 2A commit 35fde767c. Scope remains limited to the setup wizard component, component-local orchestration with injected callbacks, and component tests; no shell integration or service imports in this task.

Stage 2B TDD result: red run failed as expected because ../WatchlistSetupWizard did not exist. Implemented WatchlistSetupWizard.tsx and SetupWizard/index.ts with compact Ant Design modal layout, preset/start-mode controls, objective/scope/source/report/monitor fields, review step, required-name validation, and injected onCreateWatchlist/onCreateSources/onCreateJob/onComplete callbacks. Fixed choice-button accessible names so controls are targetable by their concise labels. Green verification passed: ./node_modules/.bin/vitest run src/components/Option/Watchlists/SetupWizard/__tests__/WatchlistSetupWizard.test.tsx src/components/Option/Watchlists/SetupWizard/__tests__/watchlist-setup-model.test.ts src/components/Option/Watchlists/__tests__/watchlists-stage2-copy-contract.test.ts --maxWorkers=1 --no-file-parallelism --reporter=verbose -> 3 files passed, 12 tests passed. git diff --check passed. Bandit not applicable because this task touched only frontend TypeScript/TSX and Backlog task files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Stage 2B WatchlistSetupWizard component. The wizard now renders first-class Watchlist presets and start modes, collects Watchlist objective/scope and optional collection/report settings, shows a review step, blocks advancement without a Watchlist name, and submits through injected callbacks rather than importing page-shell services. It can create topic-only Watchlists, source-backed Watchlists with initial sources and monitor creation, and report-goal Watchlists without sources that route to outputs. Verification: the new component test plus Stage 2A model/copy tests passed together (3 files, 12 tests), and git diff --check passed. Bandit was skipped as not applicable for frontend-only TS/TSX changes. Shell integration remains intentionally deferred to TASK-373.
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
