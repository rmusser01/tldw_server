---
id: TASK-13209
title: Keep Parakeet ONNX silence markers out of live speech transcripts
status: Done
created_date: 2026-09-06 15:39
references:
- TASK-13202
updated_date: 2026-09-06 15:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Normal Parakeet ONNX browser UAT rendered forty backend '[No speech detected]' markers as recognized words during a user-confirmed silent recording. Prevent this backend no-speech status from entering streaming transcript/history or being sent as a spoken command, preserving real speech. No chat/provider submission occurred in the failed UAT.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Parakeet ONNX silence produces no transcript words or finalized history in partial, final and flush paths.
- [x] #2 Real speech, including ordinary words 'no speech detected', is preserved; normalization is confined to the exact ONNX backend status sentinel.
- [x] #3 Focused regression tests and a real local silence/speech probe pass without another microphone recording.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR; existing Docs/ADR/046-persona-live-conversation-and-voice-runtime.md applies.
Reason: Routine repair of the existing no-synthetic-transcript boundary; no change to model selection, transport or file API.
1. Reproduce the observed sentinel at the canonical streaming ONNX adapter with targeted partial/final/flush tests.
2. Translate the ONNX decoder's exact no-speech status to empty recognition at its streaming adapter; retain legacy file API behavior and real words.
3. Run focused Parakeet/Persona tests, scoped lint/format/Bandit and local-model silence/speech validation; document the stopped human attempt and user-controlled retry.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Converted the exact ONNX backend no-speech sentinel to empty text only in the streaming variant adapter, before partial/final/flush handling. Three failing regressions became passing; all 57 focused Parakeet and Persona voice tests passed. Changed files pass Ruff/Black, scoped Bandit has zero findings, independent review found no actionable issues. Real cached Parakeet ONNX CPU probe produced no frames/history for silence and preserved the complete synthetic notebook phrase. Evidence and stopped human attempt in Docs/Reviews/MIGU_VOICE_FOLLOWUP_2026_09_06.md and assets/migu-parakeet-silence-2026-09-06. Existing ADR046 applies; no new ADR needed for this boundary bugfix. No additional microphone test or full test suite run. TASK-13202 retains physical voice acceptance.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prevented Parakeet ONNX silence statuses from becoming spoken transcript content. 57 focused tests, lint/format/Bandit, independent review and actual local silence/speech probe passed. Human recording retry remains pending.
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
