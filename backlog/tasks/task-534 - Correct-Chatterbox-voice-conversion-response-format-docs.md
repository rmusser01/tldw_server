---
id: TASK-534
title: Correct Chatterbox voice conversion response format docs
status: Done
labels:
- docs
- tts
- chatterbox
modified_files:
- Docs/STT-TTS/CHATTERBOX_SETUP.md
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Correct the Chatterbox setup runbook so documented voice-conversion output response_format values match the endpoint's supported content-type map.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs-only correction: compare the runbook's Chatterbox voice-conversion response_format list to the endpoint's _AUDIO_CONTENT_TYPE_MAP, remove unsupported output formats, and record the upload cap near the voice-conversion examples.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Corrected the Chatterbox setup runbook so voice-conversion output response_format values match the current endpoint support: wav, mp3, flac, opus, aac, and pcm. Added the 50 MiB per-upload cap note. Verified with git diff --check.
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
