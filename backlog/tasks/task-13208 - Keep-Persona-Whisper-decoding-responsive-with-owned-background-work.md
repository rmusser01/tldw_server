---
id: TASK-13208
title: Keep Persona Whisper decoding responsive with owned background work
status: Done
created_date: 2026-09-06 14:54
references:
- TASK-13202
- https://github.com/rmusser01/tldw_server/pull/2908#discussion_r3942849695
updated_date: 2026-09-06 15:28
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the whole-turn Whisper blocking/cadence defect deferred from PR #2908. Preserve bounded audio and transcript accuracy while socket control remains responsive, and keep runtime cleanup owned until inference really finishes. TASK-13202 retains human UAT using the user's normal Parakeet ONNX CPU configuration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Slow recognition does not block the asyncio event loop or incoming audio/control processing.
- [x] #2 Only one decoder runs per transcriber; audio remains bounded and update cadence starts after completed inference rather than replaying every queued chunk.
- [x] #3 Stop/reset/cleanup suppress stale results and do not release a model while an owned worker still uses it; capacity exhaustion fails explicitly.
- [x] #4 Focused regressions and a real local-model probe record responsiveness and retained transcript behavior without opening the microphone.
- [x] #9 Automatic VAD commitment waits for recognition through its exact audio boundary, retaining subsequent audio for the next turn; manual commitment retains displayed-transcript semantics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, amendment to existing ADR
ADR path: Docs/ADR/046-persona-live-conversation-and-voice-runtime.md
Reason: Background recognition must preserve the existing per-session voice ownership and process-wide task capacity boundary.
1. Reproduce event-loop blocking and pre-decode cadence with deterministic delayed-model regressions on current dev.
2. Reuse bounded streaming work capacity for one background decode per transcriber; coalesce buffered audio, publish only current-generation results, and retain cleanup until the worker exits.
3. Verify reset, Stop, disconnect, failures and capacity saturation without synthetic transcript substitution or overlapping workers.
4. Run focused tests, Ruff/Black/Bandit and a local model probe; record evidence and link back to TASK-13202. Configure subsequent human UAT for the normal Parakeet ONNX CPU backend, not the temporary tiny.en selection.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Independent review identified a VAD race introduced by background snapshots: automatic commitment must retain a boundary until its final recognition, with later audio carried into the next turn. Adding a focused regression and correction before completion. Normal Parakeet ONNX CPU preparation passed on isolated server (0.37 s), with no microphone or provider message.
Implemented bounded background decoding, completion-based cadence, late-worker cleanup, same-connection retry admission, and exact VAD boundary/carry semantics. 165 focused tests passed. Transcriber/tests Ruff+Black clean; endpoint only preexisting SIM114. Bandit zero findings. Independent review found a VAD race, fixed and re-reviewed with no actionable findings. Local Whisper/Kokoro probe retained full phrase: max ingestion 0.154 ms; heartbeat gap 18.294 ms; decode interval >=350 ms. Normal Parakeet ONNX CPU synthetic recognition passed (first cold callback 1832 ms). ADR046 and published guide updated; plan Docs/superpowers/plans/2026-09-06-persona-whisper-responsiveness.md and receipts Docs/Reviews/MIGU_VOICE_FOLLOWUP_2026_09_06.md. No microphone opened; physical acceptance remains TASK-13202. No full suite run.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed Whisper decode blocking with bounded worker ownership, exact automatic turn boundaries and next-turn audio carry. 165 targeted tests, independent review and local-model probes passed. Physical Parakeet/Kokoro floating-Buddy acceptance remains TASK-13202.
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
