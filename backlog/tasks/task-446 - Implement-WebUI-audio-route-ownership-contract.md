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
- apps/packages/ui/src/routes/route-registry.tsx
- apps/packages/ui/src/routes/__tests__/audio-route-jobs.test.ts
- apps/packages/ui/src/routes/__tests__/route-registry.visibility.test.ts
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
Implemented PR #1870 review fixes for the WP11A audio route ownership contract. /audio is now registered in the shared WebUI route registry as a RouteAliasNavigate alias to /speech, preventing the previous 404 fallback for legacy audio links. AUDIO_ROUTE_JOBS now derives finding values from AUDIO_ROUTE_FINDINGS, removes the unused extension_route owner, uses typed translation-key copy records with metadata-aligned fallback labels, and reflects the actual /audio alias mechanism. Verification passed: red-focused Vitest failures reproduced before implementation; focused audio-route/registry tests passed 10/10; adjacent audio route, metadata, visibility, and RouteAliasNavigate tests passed 25/25; git diff --check passed. Direct package tsc still fails on existing unrelated TypeScript debt across tests/components, including a pre-existing OptionPublicShare prop issue in route-registry.tsx; no errors referenced audio-route-jobs or the new /audio alias line. Bandit is not applicable because only frontend TypeScript/tests and task metadata changed.
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
