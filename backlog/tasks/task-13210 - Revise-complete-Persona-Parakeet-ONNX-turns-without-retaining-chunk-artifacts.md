---
id: TASK-13210
title: Revise complete Persona Parakeet ONNX turns without retaining chunk artifacts
status: Done
created_date: 2026-09-06 16:07
assignee:
- '@codex'
priority: high
references:
- TASK-13202
- TASK-13208
- TASK-13209
modified_files:
- tldw_Server_API/app/core/Persona/turn_transcriber.py
- tldw_Server_API/app/core/Persona/parakeet_transcriber.py
- tldw_Server_API/app/core/Persona/whisper_transcriber.py
- tldw_Server_API/app/core/Persona/live_stt.py
- tldw_Server_API/tests/Persona/test_persona_parakeet_turn.py
- tldw_Server_API/tests/Persona/test_persona_whisper_transcriber.py
- tldw_Server_API/tests/Persona/test_persona_live_voice_runtime.py
- Docs/ADR/046-persona-live-conversation-and-voice-runtime.md
- Docs/Reviews/MIGU_VOICE_FOLLOWUP_2026_09_06.md
updated_date: 2026-09-06 16:16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Physical Migu UAT sent words the user did not say. A paced local Parakeet ONNX probe reproduced a mistaken early final retained alongside a later corrected decode. Preserve coherent, revisable speech within one bounded Persona turn.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Parakeet ONNX revises the whole current Persona turn across streaming chunk boundaries without concatenating stale fragments; intentional repetition and empty corrections remain valid.
- [x] #2 Parakeet ONNX ingestion, Stop, reset, timeout and VAD finalization retain bounded work ownership and reject oversized turns without dropping earlier audio.
- [x] #3 Existing Whisper voice behavior and generic streaming backends remain covered; real local ONNX probes include leading/trailing silence and cadence matching browser capture.
- [x] #4 Document exact human and synthetic evidence separately, including remaining speech-recognition limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR; extend Docs/ADR/046-persona-live-conversation-and-voice-runtime.md because this applies the existing bounded whole-turn Persona contract to ONNX without changing provider or auth boundaries.
1. Reproduce chunk artifact retention with a failing production-factory test and record physical evidence.
2. Extract the existing owned whole-turn scheduler for reuse by Whisper and Persona Parakeet ONNX; retain existing backend decoding and exact VAD boundary handling.
3. Run focused lifecycle regressions, real local paced speech/silence probes, lint/Bandit, independent review and update evidence. Physical acceptance stays open until user confirms.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the existing bounded whole-turn Persona contract for explicit Parakeet ONNX selection. Extracted the reviewed scheduler from Whisper; each backend keeps its loader/decoder and model naming. Production-factory regression failed with retained 'Right.' before the fix and passed after. Shared model-parametrized lifecycle and real TestClient VAD/Stop/disconnect tests passed: final focused 72, plus 128 other passing regression cases. Real ONNX phrase and repetition probes passed with 3 s leading/trailing silence, no silence words, complete corrected final text and preserved repeated words. Ruff/Black pass seven Python files; Bandit zero issues; independent review no actionable findings. ADR046 and published mirror extended; user guides and source-bound review evidence updated. Intermediate ASR hypotheses can still be incorrect. Physical acceptance remains TASK-13202; no claim of perfect human transcription.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Parakeet ONNX now revises one bounded Persona turn instead of concatenating stale chunk fragments. It shares the existing owned background decoding, exact VAD boundary and cleanup behavior with Whisper. Normal ONNX selection and generic streaming backends are preserved. Automated and real local-model validation passed; physical voice acceptance remains open.
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
