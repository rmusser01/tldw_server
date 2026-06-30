---
id: TASK-430
title: Implement TTS/STT readiness and capability disclosure from existing APIs
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
- Docs/superpowers/plans/2026-05-19-tts-stt-webui-extension-workflows-implementation-plan.md
- apps/packages/ui/src/components/Option/Audio/audio-readiness.ts
- apps/packages/ui/src/components/Option/Audio/AudioReadinessStrip.tsx
- apps/packages/ui/src/components/Option/Audio/__tests__/audio-readiness.test.ts
- apps/packages/ui/src/components/Option/Audio/__tests__/AudioReadinessStrip.test.tsx
- apps/packages/ui/src/hooks/useTranscriptionModelsCatalog.ts
- apps/packages/ui/src/hooks/__tests__/useTranscriptionModelsCatalog.test.tsx
- apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/STT/ComparisonPanel.tsx
- apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx
- apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 2A from Docs/superpowers/plans/2026-05-19-tts-stt-webui-extension-workflows-implementation-plan.md. Scope: frontend-only readiness summaries and capability labels for /tts and /stt using existing provider, voice catalog, transcription model catalog, and transcription health APIs. Preserve unknown states and avoid backend endpoint changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TTS readiness uses existing provider/voice/settings state and labels Browser preview as available without setup.
- [x] #2 STT readiness uses existing transcription model catalog and health data without unbounded health checks.
- [x] #3 Capability labels distinguish supported, unsupported, and unknown with accessible text.
- [x] #4 STT model options preserve serverModels while exposing modelOptions metadata.
- [x] #5 Readiness strips render on TTS and STT pages without horizontal-overflow-prone layout.
- [x] #6 Focused tests cover readiness formatting, STT catalog metadata, and TTS/STT page rendering.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented frontend-only readiness disclosure using existing TTS provider state, STT model catalog, and bounded STT health checks. Added a reusable AudioReadinessStrip plus helper mapping that preserves unknown capability states. The STT catalog hook now exposes modelOptions metadata while preserving serverModels and avoids disabled-state rerender loops. STT readiness health-checks the configured default model instead of an arbitrary alphabetic model when available. TTS readiness labels Browser preview as the no-setup fallback and does not incorrectly require server audio for ElevenLabs when a key is saved.
Updated the implementation plan status after Slice 2A completion so Stage 1 and Stage 2A reflect completed/deferred items accurately.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 2A readiness disclosure is implemented for the TTS and STT WebUI/extension-shared surfaces. Focused Vitest suites pass; extension route parity guard still passes. Full package TypeScript check was attempted and fails on pre-existing repo-wide frontend type debt outside this touched slice; no touched-file TypeScript errors were observed in the reported output. Bandit is not applicable because this slice only changes frontend TypeScript/TSX and Backlog task metadata.
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
