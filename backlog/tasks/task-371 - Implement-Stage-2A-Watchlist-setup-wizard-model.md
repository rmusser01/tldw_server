---
id: TASK-371
title: Implement Stage 2A Watchlist setup wizard model
status: Done
assignee: []
created_date: '2026-05-15 04:54'
updated_date: '2026-05-15 05:25'
labels:
  - watchlists
  - stage2
  - frontend
dependencies:
  - TASK-370
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
Build the Watchlist setup model for Stage 2. Scope: domain/start-mode types, CTI OSINT/news/general/blank presets, payload builders for Watchlist/source/monitor setup, source URL normalization, and copy contract tests. No React component or shell wiring in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Setup wizard model helpers exist with typed presets and start modes.
- [x] #2 Helper tests cover CTI/news/general/blank presets, topic-only no-source path, source-backed path, report-goal path, and URL normalization.
- [x] #3 Copy contract covers Stage 2 preset/start-mode labels and preserves Stage 3 alert boundary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 2 baseline before feature edits:
- Shell baseline: ./node_modules/.bin/vitest run src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx --maxWorkers=1 --no-file-parallelism --reporter=verbose --testTimeout=30000 --hookTimeout=30000 passed: 1 file, 2 tests.
- Quick setup helper baseline: ./node_modules/.bin/vitest run src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts --maxWorkers=1 --no-file-parallelism --reporter=verbose --testTimeout=30000 --hookTimeout=30000 passed: 1 file, 5 tests.
- Combined baseline and isolated OverviewTab.quick-setup.test.tsx both fail before Stage 2A changes. Isolated run result: 1 file failed, 11 failed / 6 passed in 560.48s. Combined run result: 2 files passed, OverviewTab.quick-setup.test.tsx failed with 13 failed / 4 passed in that file. Failures are timeouts around guided quick setup/pipeline tests plus a Run immediately label lookup in the pipeline modal. Treat as pre-existing baseline for Stage 2A; this task is limited to setup model/copy tests and should not edit OverviewTab behavior.

Stage 2A TDD result: red run failed as expected on the missing SetupWizard model import and missing setupWizard locale copy. Implemented typed setup presets, source URL normalization, Watchlist/source/job payload builders, and the Stage 2 setup copy contract. Green run passed: ./node_modules/.bin/vitest run src/components/Option/Watchlists/SetupWizard/__tests__/watchlist-setup-model.test.ts src/components/Option/Watchlists/__tests__/watchlists-stage2-copy-contract.test.ts --maxWorkers=1 --no-file-parallelism --reporter=verbose -> 2 files passed, 7 tests passed. git diff --check passed. Bandit not applicable because this task touched only frontend TypeScript and JSON locale files. Pre-existing OverviewTab.quick-setup.test.tsx failures remain documented above and are outside Stage 2A scope.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Stage 2 Watchlist setup model and copy contract for first-class Watchlist creation. The model now defines CTI/OSINT, news, general, and blank setup presets; typed start modes; source URL normalization; Watchlist/source payload construction; and monitor/job payload construction for later shell wiring. Locale copy now exposes Watchlist-first wizard labels and explicitly keeps alert-rule authoring as a later Stage 3 boundary. Verification: focused Vitest passed for the setup model and copy contract (2 files, 7 tests), and git diff --check passed. Bandit was skipped as not applicable for frontend-only TypeScript/JSON changes. Known pre-existing OverviewTab quick setup test failures remain out of scope for this Stage 2A slice.
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
