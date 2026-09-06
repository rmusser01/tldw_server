---
id: TASK-13200
title: Filter non-speech audio in Persona Whisper recognition
status: Done
assignee: []
created_date: '2026-09-06 00:45'
updated_date: '2026-09-06 00:49'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Human Migu browser UAT confirmed audible DeepSeek/Kokoro output but denied saying the Thank you prefix recorded before the prompted speech. Persona currently disables Whisper speech filtering, allowing non-speech audio to enter recognition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona Whisper filters non-speech audio independently of manual versus automatic turn commitment.
- [x] #2 The existing recognizer receives the filter option and genuine spoken phrases are preserved without text blacklists.
- [x] #3 Focused regressions and local real-model silence/speech probes document results and remaining human UAT limits.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR; clarify existing Docs/ADR/046-persona-live-conversation-and-voice-runtime.md. Reason: preserve local speech/provider boundaries and repair recognition setup within the current contract. Compare the real local Whisper model on synthetic silence/noise and known speech with the existing filter off/on; add failing config/adapter regression, enable the existing filter only for Whisper, then run focused checks and record human playback acceptance separately.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Enabled the existing local faster-whisper speech filter for Persona Whisper independently of manual/automatic turn commitment. No text blacklist or provider change. Two parameterized adapter regressions failed on vad_filter=False, then the 129-test Persona/Whisper scope passed. Real tiny.en comparison: five seconds of silence produced You with filtering off and empty output with it on; known local Kokoro speech remained correct. Production Persona config with the real model confirmed empty silence and correct speech, including leading silence. Bandit zero findings; Ruff one unchanged endpoint SIM114 baseline finding, no new findings. ADR046, user guide and published mirrors updated. Human playback passed on the previous source, while human transcript accuracy remains pending a new microphone run under TASK-13195. Sanitized receipts are in Docs/Reviews/assets/migu-buddy-browser-voice-2026-09-05.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
