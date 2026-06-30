---
id: TASK-431
title: Implement TTS/STT audio error classification and recovery UI
status: Done
labels:
- implementation
- webui
- extension
- audio
- tts
- stt
- ux
modified_files:
- apps/packages/ui/src/components/Option/Audio/audio-error-classification.ts
- apps/packages/ui/src/components/Option/Audio/__tests__/audio-error-classification.test.ts
- apps/packages/ui/src/components/Option/STT/ComparisonPanel.tsx
- apps/packages/ui/src/components/Option/STT/RecordingStrip.tsx
- apps/packages/ui/src/components/Option/STT/__tests__/ComparisonPanel.test.tsx
- apps/packages/ui/src/components/Option/STT/__tests__/RecordingStrip.test.tsx
- apps/packages/ui/src/components/Option/Speech/RenderStrip.tsx
- apps/packages/ui/src/components/Option/Speech/__tests__/RenderStrip.test.tsx
- apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx
- apps/packages/ui/src/hooks/useComparisonTranscribe.ts
- apps/packages/ui/src/hooks/useMultiRenderState.ts
- apps/packages/ui/src/hooks/useTtsPlayground.tsx
- apps/packages/ui/src/hooks/__tests__/useComparisonTranscribe.test.ts
- apps/packages/ui/src/hooks/__tests__/useMultiRenderState.test.ts
- apps/packages/ui/src/hooks/__tests__/useTtsPlayground.test.tsx
- Docs/superpowers/plans/2026-05-19-tts-stt-webui-extension-workflows-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 3 from Docs/superpowers/plans/2026-05-19-tts-stt-webui-extension-workflows-implementation-plan.md. Scope: frontend-only shared error classification for visible TTS/STT workflows, mapping known backend/browser failures to safe user-facing recovery copy without exposing raw secrets or stack traces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Known TTS/STT errors map to stable categories such as missing credentials, missing model, engine unavailable, microphone blocked, network, timeout, and unknown.
- [x] #2 User-facing recovery copy points to concrete next actions such as Settings -> Speech, Audio Setup Guide, retry, or browser permission checks.
- [x] #3 Raw provider/debug details are not shown in user-facing cards when they may include secrets.
- [x] #4 TTS and STT visible error states consume the shared classifier where it improves current raw or vague messages.
- [x] #5 Focused tests cover classifier mappings and representative TTS/STT error rendering paths.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Starting Stage 3. First pass will inspect current visible TTS/STT error paths, then add a shared classifier and only wire it where it reduces raw/vague user-facing errors without broad backend changes.

Implemented a frontend-only shared audio error classifier and wired it into visible TTS/STT failure paths: STT comparison result cards, STT microphone start failures, SpeechPlayground dictation/streaming/long-form/voice-preview failures, TTS segment generation notifications, and TTS render strip generation failures. The classifier sanitizes debug details and maps known browser/backend errors to stable recovery categories.

Verification:
- `cd apps/packages/ui && ./node_modules/.bin/vitest run src/components/Option/Audio/__tests__/audio-error-classification.test.ts src/hooks/__tests__/useComparisonTranscribe.test.ts src/components/Option/STT/__tests__/ComparisonPanel.test.tsx src/components/Option/STT/__tests__/RecordingStrip.test.tsx src/hooks/__tests__/useTtsPlayground.test.tsx src/hooks/__tests__/useMultiRenderState.test.ts src/components/Option/Speech/__tests__/RenderStrip.test.tsx src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx` -> 8 files, 71 tests passed.
- `cd apps/packages/ui && ./node_modules/.bin/vitest run src/components/Option/Audio/__tests__/AudioReadinessStrip.test.tsx src/components/Option/Audio/__tests__/audio-readiness.test.ts src/components/Option/Audio/__tests__/audio-error-classification.test.ts src/hooks/__tests__/useTranscriptionModelsCatalog.test.tsx src/hooks/__tests__/useComparisonTranscribe.test.ts src/hooks/__tests__/useTtsPlayground.test.tsx src/hooks/__tests__/useMultiRenderState.test.ts src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx src/components/Option/STT/__tests__/ComparisonPanel.test.tsx src/components/Option/STT/__tests__/RecordingStrip.test.tsx src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx src/components/Option/Speech/__tests__/RenderStrip.test.tsx src/routes/__tests__/option-audio-route-identity.test.tsx` -> 13 files, 92 tests passed.
- `cd apps/tldw-frontend && ../packages/ui/node_modules/.bin/vitest run extension/__tests__/audio-route-parity.guard.test.ts` -> 1 file, 2 tests passed.
- `git diff --check` -> passed.
- `cd apps/packages/ui && ./node_modules/.bin/tsc --noEmit --pretty false` -> failed on existing unrelated frontend baseline errors outside touched audio files.
- Bandit skipped: frontend-only TypeScript/TSX slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 3 audio error classification is implemented for the shared WebUI/extension TTS and STT surfaces. Known credentials, missing-model, engine-unavailable, unsupported, microphone, network, timeout, and unknown failures now produce stable user-facing recovery copy and settings recovery links where available, while raw provider details remain redacted from visible cards/notifications. Focused Vitest coverage passes; extension audio route parity still passes. Full package TypeScript remains blocked by existing unrelated frontend baseline errors outside touched audio files. Bandit is skipped because this slice only changes frontend TypeScript/TSX and Backlog metadata.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run for touched code when applicable or document frontend-only skip
- [x] #4 Final summary added
- [x] #5 Known skips or blockers documented
<!-- DOD:END -->
