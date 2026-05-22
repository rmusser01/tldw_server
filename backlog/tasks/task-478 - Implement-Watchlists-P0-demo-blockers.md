---
id: TASK-478
title: Implement Watchlists P0 demo blockers
status: Done
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
- [x] #1 Structured audio trigger results distinguish submitted, skipped, configuration-required, queue-unavailable, and enqueue-failed paths.
- [x] #2 Run-audio status exposes Scheduler-backed pending/queued/running/failed state when workflow artifacts are not yet available.
- [x] #3 Reports output preview polls live audio status for text digest outputs with requested audio and preserves metadata artifacts/fallbacks.
- [x] #4 Initial watchlist selection prefers the current valid selection or a first-class active watchlist instead of blindly selecting the first API item.
- [x] #5 Focused backend, frontend, Bandit, diff, and browser-smoke verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-22-watchlists-p0-demo-blockers-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 2 complete after review: run-audio fallback status reads Scheduler task state only from an already-started global scheduler; missing DB, no matching workflow run, matched run without id, and matched run without final audio artifact all expose Scheduler status or safe pending. Cancellation propagates at helper and endpoint boundary. Task 3 implemented Reports drawer live audio status polling for text digest outputs with requested audio, merges live status scalars over metadata artifacts/fallbacks, renders queue name, retries after transient status fetch failures, and stops polling on close. Task 4 implemented deterministic watchlist selection and review fix now passes the preferred selection into setWatchlists atomically to avoid the store's old items[0] transient fallback.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the focused Watchlists P0 demo-blocker slice: audio briefing trigger paths now return structured results and ensure a workflows queue worker before submit; run-audio status can report Scheduler-backed progress before durable workflow artifacts exist; Reports preview polls and merges live audio status for digest outputs with requested audio; and first-class active watchlists are selected atomically instead of transiently falling back to an imported placeholder. Verification on the rebased branch passed the focused backend Watchlists pytest subset, focused frontend Watchlists Vitest subset, touched-scope Bandit, and git diff whitespace check. Browser smoke with local FastAPI and Next.js verified `/watchlists` route load, Reports tab access, and Create report modal access; the first cold load produced a transient fetch error while services were warming, then the warm route/API pass reached the expected Watchlists and Reports states.
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
