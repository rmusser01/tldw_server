---
id: TASK-446
title: Implement WebUI audio route ownership contract
status: Done
labels:
- webui
- ux-audit
- audio
- wp11a
priority: medium
documentation:
- Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md
modified_files:
- apps/packages/ui/src/routes/audio-route-jobs.ts
- apps/packages/ui/src/routes/__tests__/audio-route-jobs.test.ts
- backlog/tasks/task-446 - Implement-WebUI-audio-route-ownership-contract.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP11A Task 1 from the WebUI audio routes implementation plan. Add a pure audio route-job contract and focused route identity coverage for /audio, /speech, /stt, /tts, and /audiobook-studio. Keep the slice frontend metadata/test-only unless existing tests prove a route boundary or ownership mismatch that must be fixed in the same narrow scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audio route-job inventory covers /audio, /speech, /stt, /tts, and /audiobook-studio exactly once.
- [x] #2 Route-job contract records canonical route concept, owner, component, capability, route-state policy, and audit finding coverage for WP11A.
- [x] #3 Focused route identity tests preserve /tts as SpeechPlaygroundPage locked to listen mode and keep /speech as the combined route.
- [x] #4 The first route-job test is verified red before the contract is added, then green after implementation.
- [x] #5 Relevant focused Vitest checks and git diff checks are recorded in the task final summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Red/green TDD cycle completed for the WP11A audio route-job contract. The initial focused Vitest run first failed because fresh worktree dependencies were not installed; after running bun install in apps, the route-job test failed for the expected missing ../audio-route-jobs import. Added a pure typed route metadata module covering /audio, /speech, /stt, /tts, and /audiobook-studio, including canonical concepts, owners, components, capability inputs, route-state policy, primary action labels, and audit finding coverage.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the WP11A audio route ownership contract as a frontend metadata/test-only slice. Added AUDIO_ROUTE_JOBS and getAudioRouteJob in apps/packages/ui/src/routes/audio-route-jobs.ts, plus focused tests covering the five root audio routes, user-facing labels/jobs, audit finding coverage, and canonical ownership for overlapping /audio, /speech, /stt, /tts, and /audiobook-studio behavior. Verification passed: audio-route-jobs test 4/4, audio route identity plus route-job tests 7/7, and related hosted/metadata/visibility checks 9/9. Bandit is not applicable because this slice only touched frontend TypeScript tests/data and Backlog metadata; no Python code changed.
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
