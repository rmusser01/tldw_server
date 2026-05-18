---
id: TASK-426
title: >-
  Implement Watchlists digest audio PR1 contract cadence and source settings
  slice
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-18 07:08'
labels:
  - watchlists
  - frontend
  - implementation
  - audio
  - cadence
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first PR slice from the Watchlists digest/audio implementation plan: frontend watchlists audio contract alignment, variable cadence controls, source settings preservation, and forum capability gating without changing backend audio artifact persistence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Watchlists frontend types and services represent backend-supported audio output fields and expose getWatchlistRunAudio(runId).
- [x] #2 Schedule picker supports variable every-N hours/minutes, weekdays, daily, weekly, and raw cron fallback without dropping existing cron parsing.
- [x] #3 Source form preserves unknown source settings, serializes website scrape_rules, passes draft settings to preflight, and gates forum source option from watchlists settings capability.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Commits:
- ab2f60201 feat: align watchlists audio contracts
- 5b249de37 feat: support variable watchlist cadence presets
- 4782a68d4 feat: preserve watchlist source settings
- 29a5d8d64 feat: show watchlist source diagnostics

Verification:
- bun run test -- src/services/__tests__/watchlists-audio.test.ts src/components/Option/Watchlists/__tests__/watchlists-static-guard.typecheck.test.ts src/components/Option/Watchlists/JobsTab/__tests__/schedule-utils.test.ts src/components/Option/Watchlists/JobsTab/__tests__/SchedulePicker.help.test.tsx src/components/Option/Watchlists/SourcesTab/__tests__/source-settings.test.ts src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.forum-help.test.tsx --maxWorkers=1 --no-file-parallelism
- bun run test -- src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.bulk-move.test.tsx src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.advanced-details.test.tsx src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.load-error-retry.test.tsx src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.delete-confirm.test.tsx --maxWorkers=1 --no-file-parallelism
- git diff --check

Bandit: skipped; touched runtime code is TypeScript frontend only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented PR1 frontend slice for watchlists digest/audio workflow: audio contract alignment, variable cadence controls, source settings preservation with scrape_rules, draft preflight settings, and forum capability gating. Verification recorded in final response.
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
