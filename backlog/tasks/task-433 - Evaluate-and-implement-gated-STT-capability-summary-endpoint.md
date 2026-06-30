---
id: TASK-433
title: Evaluate and implement gated STT capability summary endpoint
status: Done
labels:
- audio
- stt
- webui
- backend
modified_files:
- Docs/superpowers/plans/2026-05-19-tts-stt-webui-extension-workflows-implementation-plan.md
- tldw_Server_API/app/api/v1/endpoints/audio/audio_health.py
- tldw_Server_API/tests/Audio/test_stt_capabilities_endpoint.py
- apps/packages/ui/src/components/Option/Audio/audio-readiness.ts
- apps/packages/ui/src/components/Option/Audio/AudioReadinessStrip.tsx
- apps/packages/ui/src/components/Option/Audio/__tests__/audio-readiness.test.ts
- apps/packages/ui/src/hooks/useTranscriptionModelsCatalog.ts
- apps/packages/ui/src/hooks/__tests__/useTranscriptionModelsCatalog.test.tsx
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/tldw/domains/models-audio.ts
- apps/packages/ui/src/services/tldw/client-ownership.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Stage 5 from the TTS/STT WebUI and extension workflow plan. Document the Phase 2A capability metadata gap first, then implement the optional `/api/v1/audio/transcriptions/capabilities` backend/frontend enhancement only if existing APIs cannot support clear STT capability UX. Preserve no-warm/no-download behavior and keep frontend fallback behavior when the endpoint is unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Phase 2A gap note documents which visible STT UX states cannot be derived from current APIs
- [x] #2 Backend owner, auth/rate-limit pattern, and no-warm/no-download behavior are confirmed before endpoint implementation
- [x] #3 If implemented, capability endpoint distinguishes supported, unsupported, and unknown with source/confidence fields
- [x] #4 Frontend treats the capability endpoint as an enhancement and preserves existing catalog/health fallback
- [x] #5 Focused backend/frontend tests and hygiene checks are run and recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-19-tts-stt-webui-extension-workflows-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 5 gate satisfied. Existing APIs left model capabilities unknown because the static catalog has no provider/capability metadata and health is one-model availability only. Implemented read-only `/api/v1/audio/transcriptions/capabilities` under the audio health endpoint owner with no warm/download behavior, provider/source metadata, and frontend enhancement fallback.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 5 Phase 2B STT capability summary. The backend endpoint combines static model catalog metadata, lightweight health availability, provider adapter batch/streaming/diarization support, and response-schema timestamp/segment support while preserving unknown states when provider metadata is unavailable. The frontend now attempts the capability summary first and falls back to the existing catalog plus selected-model health behavior if the endpoint is unavailable. Verification: backend capability+health pytest passed 6/6; focused frontend Vitest passed 27/27; ownership guard passed; Bandit touched backend endpoint produced 0 results/errors; git diff --check passed. Full frontend tsc still fails on the inherited package-wide baseline outside the Stage 5 capability endpoint path.
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
