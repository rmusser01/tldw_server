---
id: TASK-526
title: Reduce chat setup-blocked first-run overload
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 19:12'
labels:
  - chat
  - ux
  - first-run
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the /chat F4 first-time UX issue where a setup-blocked page shows provider setup, empty starter content, full cockpit rails, header controls, and composer controls all at once. Preserve restored rails while making the setup-blocked state visually dominant and progressive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When provider/model readiness is blocking because no usable provider is configured, desktop /chat does not present the full empty starter deck as a competing primary action.
- [x] #2 Setup-blocked cockpit state keeps context/runtime rails available but collapses or summarizes advanced rail detail enough to reduce first-run overload.
- [x] #3 The primary visible recovery action remains provider/model setup or refresh, with send still blocked for the same readiness reason.
- [x] #4 Regression coverage proves the setup-blocked state suppresses competing starter content without hiding restored rail affordances.
- [x] #5 Focused /chat tests pass and verification/known skips are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented setup recovery focus for /chat: PlaygroundChat hides PlaygroundEmpty when no-provider or no-model recovery is visible; Playground computes first-run setupRecoveryMode for no_models and provider_unconfigured blockers and passes it to context/runtime rails; rails keep primary context/runtime/model route visible while secondary prompt/search/session/assistant/tools/run sections start collapsed. Verification: RED tests failed as expected before implementation; focused Vitest suite passed 98 tests; git diff --check and git diff --cached --check passed; tsc remains blocked by known unrelated CharacterListContent GalleryCardDensity baseline; Bandit skipped because touched files are TS/TSX only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reduced setup-blocked first-run overload on /chat while preserving restored rails. Recovery banners now suppress the starter deck, and setup recovery mode collapses secondary rail sections by default while keeping context/runtime rails and model recovery actions available.
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
