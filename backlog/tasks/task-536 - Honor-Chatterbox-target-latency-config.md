---
id: TASK-536
title: Honor Chatterbox target latency config
status: Done
labels:
- tts
- chatterbox
- config
modified_files:
- tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py
- tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make ChatterboxAdapter honor target_latency_ms / chatterbox_target_latency_ms instead of always using the hardcoded 200 ms progressive streaming hint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add a failing adapter unit test for target_latency_ms / chatterbox_target_latency_ms, wire the config into ChatterboxAdapter initialization with positive-int fallback behavior, update the Chatterbox parity plan, then verify with the Chatterbox adapter mock suite, Bandit, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
ChatterboxAdapter now honors target_latency_ms and chatterbox_target_latency_ms for progressive streaming chunk hints, with prefixed config taking precedence and invalid/non-positive values falling back to 200 ms. Verified with the Chatterbox adapter mock suite, Bandit on chatterbox_adapter.py, and git diff --check.
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
