---
id: TASK-12923
title: Default STT model to Parakeet MLX on macOS and ONNX elsewhere
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-08 17:15
labels:
- backend
- audio
- stt
dependencies: []
modified_files:
- tldw_Server_API/app/core/config.py
- tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py
- tldw_Server_API/Config_Files/config.txt
- tldw_Server_API/Config_Files/README.md
- apps/packages/ui/src/config/ui-constants.ts
- apps/packages/ui/src/hooks/__tests__/useSttSettings.test.tsx
- apps/packages/ui/src/components/Option/Settings/SSTSettings.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx
- apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundVoiceChat.ts
- apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx
- apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.audio-source.test.tsx
- tldw_Server_API/tests/Logging/test_config_loading_sections.py
- tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py
- Docs/API-related/Audio_Transcription_API.md
- Docs/Published/API-related/Audio_Transcription_API.md
- Docs/Getting_Started/First_Time_Audio_Setup_CPU.md
- Docs/Published/Getting_Started/First_Time_Audio_Setup_CPU.md
- Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md
- Docs/Published/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md
- tldw_Server_API/app/core/TTS/TTS-DEPLOYMENT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Change the default transcription model selection so fresh/default STT settings use Parakeet MLX on macOS and Parakeet ONNX on Linux/Windows, while preserving explicit user overrides.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the frontend initialized sttModel to whisper-1, so chat dictation and voice-chat sent an explicit model and bypassed backend defaults. Backend config also hard-coded ONNX, so it could not choose MLX on macOS.

Implementation: frontend STT defaults now leave model empty/server-default; backend STT config default uses auto and resolves to parakeet-mlx on macOS and parakeet-tdt-0.6b-v3-onnx on Linux/Windows. Batch and streaming default paths share the resolver. Docs updated.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Frontend no longer defaults STT to whisper-1 and Speech Playground no longer promotes the first catalog model into request options when settings use the server default. Backend STT config supports auto, resolving to parakeet-mlx on macOS and parakeet-tdt-0.6b-v3-onnx on Linux/Windows. Verification included frontend STT/dictation/Speech tests, backend STT tests, typecheck, git diff --check, Bandit on touched backend files, live backend audio health showing parakeet-mlx, and review follow-up regression confirming Speech Playground omits model for server default.
<!-- SECTION:FINAL_SUMMARY:END -->

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
