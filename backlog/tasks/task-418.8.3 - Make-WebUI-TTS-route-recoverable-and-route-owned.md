---
id: TASK-418.8.3
title: Make WebUI TTS route recoverable and route-owned
status: Done
labels:
- webui
- ux-audit
- audio
- wp11a
- tts
priority: medium
parent_task_id: TASK-418.8
documentation:
- Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Option/Audio/audio-readiness.ts
- apps/packages/ui/src/components/Option/Audio/__tests__/audio-readiness.test.ts
- apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/Speech/RenderStrip.tsx
- apps/packages/ui/src/components/Option/Speech/TtsProviderStrip.tsx
- apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx
- apps/packages/ui/src/components/Option/Speech/__tests__/RenderStrip.test.tsx
- apps/packages/ui/src/components/Option/Speech/__tests__/TtsProviderStrip.test.tsx
- apps/packages/ui/src/hooks/useTtsProviderData.ts
- apps/tldw-frontend/e2e/utils/page-objects/TTSPage.ts
- apps/tldw-frontend/e2e/workflows/tier-2-features/tts-synthesis.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP11A Task 4 from the WebUI audio routes implementation plan. Make /tts a single canonical synthesis route by preserving SpeechPlaygroundPage locked listen ownership, improving provider/voice/readiness/recovery states, keeping advanced synthesis controls discoverable, and preventing the legacy TtsPlaygroundPage from drifting into a second route owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /tts route-owner guard coverage verifies RouteErrorBoundary routeId=tts and routeLabel=TTS Playground around SpeechPlaygroundPage lockedMode=listen with hidden mode switcher.
- [x] #2 TTS readiness coverage includes missing provider, missing voice catalog, ElevenLabs loading/timeout/missing-key states, ffmpeg degraded output warning, Browser TTS local-output labeling, generated segment inspection, and advanced voice/model control discoverability.
- [x] #3 Implementation reuses existing Speech/TTS components and does not add a second TTS route surface or backend APIs.
- [x] #4 Focused TTS component and E2E verification is recorded; inherited repo-wide type debt is separated from touched-scope results.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Kept /tts routed through SpeechPlaygroundPage lockedMode=listen and tightened route-level synthesis recovery. Added readiness inputs for ElevenLabs catalog state and ffmpeg availability, labeled browser synthesis as local browser output, passed inferred server provider keys into TTS readiness, and gated Play with actionable recovery reasons for missing server audio, missing provider metadata, missing voice catalogs, ElevenLabs missing credentials/loading/timeout/error/empty catalogs. PR review follow-up made ffmpeg degraded output reporting consistent for ElevenLabs and simplified the ElevenLabs empty-catalog guard without adding backend APIs or a second route surface.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed WP11A Task 4 for /tts route readiness and addressed PR #1885 review comments. Focused tests passed: component suite 74/74, route contract 8/8, and TTS E2E 3/3. ESLint reported 0 errors with existing warnings in the large Speech page/test files. Broad frontend/UI TypeScript checks still fail on inherited unrelated debt outside this slice. Bandit was not applicable because no Python files were touched.
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
