---
id: TASK-436
title: Implement Watchlists digest/audio PR5 power-user reuse and operator recovery
status: Done
labels:
- watchlists
- frontend
- backend
- prd-pr5
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/clone-utils.ts
- apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobsTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/clone-utils.ts
- apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx
- apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsCommandPalette.tsx
- apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx
- apps/packages/ui/src/services/watchlists.ts
- apps/packages/ui/src/types/watchlists.ts
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py
- tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR5 Tasks 9-10 from Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md: power-user reuse/batch operations plus operator recovery/diagnostics for the Watchlists digest and audio briefing workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clone monitor preserves scope, cadence, filters, output, delivery, and audio cast fields where supported while resetting runtime state.
- [ ] #2 Clone source preserves fetch/extraction/dedupe identity rules while resetting runtime status and seen state.
- [ ] #3 Command palette exposes create, clone, run, preview, retry, and export actions where backing operations exist or clearly reports unavailable actions.
- [ ] #4 Batch test/validation actions reuse safe existing APIs unless a backend endpoint is required.
- [ ] #5 Operator diagnostics/retry controls distinguish delivery/audio retries from full ingestion reruns.
- [ ] #6 Focused frontend and backend tests cover the implemented behavior, or skipped coverage is documented with the blocker.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implemented in PR5 local commit: clone helpers/actions for feeds and monitors; command-palette discovery for create/clone/run/preview/retry/export with disabled reasons when row context is required; stage-specific delivery/audio retry APIs and UI controls; diagnostics JSON export; quick-setup focus recovery hardening discovered during a11y verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Watchlists PR5 power-user reuse and recovery slice is implemented on codex/watchlists-pr5-reuse-recovery, rebased onto origin/dev. Verification passed: focused PR5 Vitest 20 tests, watchlists scale 53 tests, watchlists a11y 84 tests, watchlists static guard 3 tests, targeted backend pytest 11 tests, git diff --check, and Bandit on touched backend files with zero findings.
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
