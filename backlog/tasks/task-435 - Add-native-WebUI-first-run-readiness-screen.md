---
id: TASK-435
title: Add native WebUI first-run readiness screen
status: Done
labels:
- implementation
- setup
- frontend
- webui
documentation:
- Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md
- Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Option/Setup/ReadinessSetupScreen.tsx
- apps/packages/ui/src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx
- apps/packages/ui/src/routes/option-setup.tsx
- apps/packages/ui/src/routes/__tests__/option-setup-readiness.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 7 from Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md: a native setup readiness screen backed by useSetupReadiness, with profile/lane review, explicit secondary Provision Now action, Verify action, and backend /setup fallback for first-run guard failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Native `/setup` readiness screen renders readiness profiles, Chat, Embeddings/RAG, and Speech lanes.
- [x] TTS remains visible but secondary inside the Speech lane.
- [x] `Provision now` remains a separate secondary action and is not called by profile selection.
- [x] Backend `/setup` fallback link remains visible, including remote first-run guard states.
- [x] `/setup` keeps connection onboarding when server configuration is still missing or invalid.
- [x] `/setup` switches the same readiness screen to admin mode after first-run completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing setup route and component patterns.
2. Add failing screen tests for profile/lane rendering, fallback link, and explicit Provision Now behavior.
3. Implement ReadinessSetupScreen using useSetupReadiness without hidden provisioning.
4. Wire the native setup route to render readiness when available with fallback to legacy /setup.
5. Run focused frontend tests and update the plan/backlog.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `ReadinessSetupScreen` backed by `useSetupReadiness`, with profile selection, lane status cards, preview/provision/verify actions, skipped-lane consequences, and backend setup fallback.
- Added screen tests for lane rendering, secondary TTS copy, explicit provisioning, guard fallback, and admin mode pass-through.
- Updated `option-setup.tsx` to render onboarding only while connection setup still needs attention; configured backends now render the readiness screen.
- Added route tests for configured first-run mode, post-first-run admin mode, missing server URL fallback, and connection setup fallback.
- Verification: `bunx vitest run src/components/Option/Setup/__tests__/ReadinessSetupScreen.test.tsx src/routes/__tests__/option-setup-readiness.test.tsx` -> 8 passed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Native WebUI setup readiness screen is implemented and wired into /setup. It keeps onboarding for missing/invalid connection setup, uses first-run readiness endpoints before first-run completion, switches to admin mode after completion, keeps the backend /setup fallback visible, and keeps provisioning behind an explicit Provision now action.
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
