---
id: TASK-538
title: Defer Chatterbox streaming voice-reference cleanup
status: Done
labels:
- tts
- chatterbox
- streaming
- bugfix
references:
- https://github.com/devnen/Chatterbox-TTS-Server
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py
- tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix ChatterboxAdapter so temporary voice-reference files created for streaming TTS are deleted after the audio stream is consumed or closed, not immediately when generate() returns the streaming response.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add a failing adapter regression test showing streaming Chatterbox voice-reference generation keeps the temporary reference file available until the stream is consumed, wrap the Chatterbox streaming generator with deferred cleanup, then verify with the focused test, full adapter mock suite, Bandit, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Chatterbox streaming TTS voice-reference cleanup by wrapping returned audio streams with a cleanup generator. Temporary voice-reference files now remain available while the stream is consumed and are removed when the stream finishes or closes; non-streaming requests and pre-return error paths still clean up immediately. Added a red/green regression test for the streaming lifetime boundary. Verification: focused regression failed before the fix and passed after it; full Chatterbox adapter mock suite passed; Bandit on chatterbox_adapter.py reported no findings; git diff --check passed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
