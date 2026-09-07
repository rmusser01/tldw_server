---
id: TASK-13214
title: Honor configured Persona Live speech provider without Kokoro lock-in
status: Done
created_date: 2026-09-07 16:06
labels:
- buddy
- voice
- bug
priority: high
updated_date: 2026-09-07 16:44
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the UAT-specific Kokoro restriction so Persona Live prepares and speaks through the configured TTS provider. Audit related UAT assumptions and retain explicit errors, provider choice, and Stop/cleanup behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Configured registered TTS providers prepare and synthesize without silently substituting Kokoro or its model/voice defaults.
- [x] #2 Unsupported or unavailable configurations fail clearly before recording where detectable; preparation does not synthesize user content.
- [x] #3 Regression coverage verifies a non-Kokoro provider through preparation, audio delivery and stop/cleanup, including provider failure and no fallback.
- [x] #4 Audit related voice assumptions, correct relevant guides and record remaining limitations and actual UAT evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/004-persona-live-tts-provider-selection.md. Reason: preserve selected TTS provider/model/voice through authenticated preparation and synthesis, including optional persisted model and lazy selected Kitten model readiness.
1. Reproduce provider/default failures and audit voice paths.
2. Restore shared TTS resolution, effective credentials and owned cleanup; retain browser speech and explicit model transport.
3. Verify selected provider, failure, stop and actual native browser speech/server route.
4. Review, update guides and evidence, run focused checks and commit.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented provider/model/voice selection, shared authenticated TTS credential scope, browser playback recovery and temporary adapter cleanup. Kitten preparation warms the requested model through its cache; Chat preparation no longer requires a static server key. Guides corrected in both repositories. Final verification/evidence recording in progress.
Verification complete: 265 focused Python and 111 frontend tests passed; Bandit found zero issues in all seven changed production Python files. Targeted new Python Ruff/Black checks pass. Frontend ESLint has zero errors and 58 confirmed baseline warnings; large legacy Python formatting/lint debt remains unchanged. Real restarted server prepared Parakeet + browser TTS, dispatched the authorized DeepSeek Persona reply, acknowledged voice Stop, and stopped the session with HTTP 200. Real native SpeechSynthesis through the production controller emitted start/end and stopped a long reply with no queued speech. The browser input/WS boundary was controlled; no microphone/human-audibility or paid remote-TTS claim. Independent review resolved credential-scope, selected Kitten model readiness, and browser explicit-voice substitution findings. Canonical/Published guide mirrors match and MkDocs build passed. ADR: backlog/decisions/004-persona-live-tts-provider-selection.md. Evidence and limitations: Docs/Reviews/PERSONA_TTS_PROVIDER_CHOICE_2026_09_07.md. Added incident-based lesson; temporary browser harness removed from product public files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed Kokoro-only Persona Live preparation, preserved configured provider/model/voice through effective credentials and cleanup, and corrected both Buddy guides. Verified 265 Python and 111 frontend tests, zero Bandit findings, actual browser synthesis/Stop, and real Parakeet/DeepSeek server route. Remaining physical voice, provider deployment and Buddy animation limits are explicitly recorded.
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
