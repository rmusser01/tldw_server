---
id: TASK-429
title: Implement TTS/STT route parity and TTS config truthfulness
status: Done
labels:
- implementation
- webui
- extension
- audio
- tts
- stt
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Slice 1 from Docs/superpowers/plans/2026-05-19-tts-stt-webui-extension-workflows-implementation-plan.md. Scope: extension #/stt parity with WebUI /stt, extension #/tts locked TTS route behavior, route copy/settings destination fixes, Browser TTS preview labeling, and provider/model/voice mismatch prevention. Keep work frontend-only and do not touch unrelated routes or backend APIs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Extension #/stt renders the dedicated SttPlaygroundPage core workflow.
- [x] #2 Extension #/tts mirrors WebUI /tts locked listen-mode behavior with the mode switcher hidden.
- [x] #3 TTS provider/model/voice render configs do not mix provider-specific model or voice defaults.
- [x] #4 Browser TTS is labeled as a local Browser preview/no-setup escape hatch.
- [x] #5 Speech settings copy points users to /settings/speech.
- [x] #6 Focused frontend and extension tests cover the route parity and TTS render-config behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented Slice 1 from `Docs/superpowers/plans/2026-05-19-tts-stt-webui-extension-workflows-implementation-plan.md`.

Touched files:

- `apps/tldw-frontend/extension/routes/option-stt.tsx`
- `apps/tldw-frontend/extension/routes/option-tts.tsx`
- `apps/tldw-frontend/extension/__tests__/audio-route-parity.guard.test.ts`
- `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/Speech/RenderStrip.tsx`
- `apps/packages/ui/src/components/Option/Speech/TtsProviderStrip.tsx`
- `apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx`
- `apps/packages/ui/src/components/Option/Speech/__tests__/RenderStrip.test.tsx`
- `apps/packages/ui/src/components/Option/Speech/__tests__/TtsProviderStrip.test.tsx`
- `apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx`
- `apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx`

Verification:

- `apps/tldw-frontend`: `../packages/ui/node_modules/.bin/vitest run extension/__tests__/audio-route-parity.guard.test.ts` passed, 2 tests.
- `apps/packages/ui`: `./node_modules/.bin/vitest run src/routes/__tests__/option-audio-route-identity.test.tsx src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx src/components/Option/Speech/__tests__/RenderStrip.test.tsx src/components/Option/Speech/__tests__/TtsProviderStrip.test.tsx src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx` passed, 37 tests.
- `git diff --check` passed.
- Bandit skipped because this slice touched frontend TypeScript/React and Backlog metadata only.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extension `#/stt` now uses the dedicated shared STT page, extension `#/tts` is locked to the dedicated listen-mode TTS workflow, TTS render defaults are provider-specific, Browser TTS is labeled as a local Browser preview, and speech settings copy points to `/settings/speech`.

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
