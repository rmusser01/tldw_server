---
id: TASK-478
title: Implement Watchlists P0 demo blockers
status: In Progress
references:
- Docs/superpowers/plans/2026-05-22-watchlists-p0-demo-blockers-implementation-plan.md
- Docs/superpowers/specs/2026-05-22-watchlists-staged-demo-remediation-design.md
- Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md
modified_files:
- tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py
- tldw_Server_API/app/core/Watchlists/pipeline.py
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py
- tldw_Server_API/app/core/Scheduler/scheduler.py
- tldw_Server_API/app/core/Scheduler/__init__.py
- tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py
- tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py
- tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py
- apps/packages/ui/src/types/watchlists.ts
- apps/packages/ui/src/services/__tests__/watchlists-audio.test.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/Watchlists/watchlist-selection.ts
- apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlist-selection.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the focused P0 demo-blocker plan for /watchlists only: workflows queue worker availability, structured audio trigger results, run-audio fallback status, Reports live audio polling, active watchlist selection, and focused verification. Do not reopen the completed 2026-05-18 PRD checklist except for touched-file corrections required by these blockers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-22-watchlists-p0-demo-blockers-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 2 complete after review: run-audio fallback status reads Scheduler task state only from an already-started global scheduler; missing DB, no matching workflow run, matched run without id, and matched run without final audio artifact all expose Scheduler status or safe pending. Cancellation propagates at helper and endpoint boundary. Task 3 implemented Reports drawer live audio status polling for text digest outputs with requested audio, merges live status scalars over metadata artifacts/fallbacks, renders queue name, retries after transient status fetch failures, and stops polling on close. Task 4 implemented deterministic watchlist selection that preserves valid current selection and otherwise prefers active, non-archived/non-deleted, non-imported, news/CTI watchlists before falling back by recency and source order.
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
