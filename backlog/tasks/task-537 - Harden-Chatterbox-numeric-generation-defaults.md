---
id: TASK-537
title: Harden Chatterbox numeric generation defaults
status: Done
labels:
- tts
- chatterbox
- config
- hardening
modified_files:
- tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py
- tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make ChatterboxAdapter parse generation default numeric config defensively, honoring unprefixed aliases and falling back to safe defaults instead of raising on malformed values.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add a failing adapter unit test for Chatterbox generation default numeric config, replace direct float(...) conversions with defensive coercion that honors prefixed and unprefixed config keys, update the Chatterbox parity plan, then verify with the Chatterbox adapter mock suite, Bandit, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
ChatterboxAdapter now parses generation default numeric config defensively. Prefixed keys take precedence, unprefixed aliases are honored, and malformed values for default exaggeration, cfg weight, temperature, repetition penalty, min-p, and top-p fall back to safe defaults instead of raising during initialization. Verified with the Chatterbox adapter mock suite, Bandit on chatterbox_adapter.py, and git diff --check.
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
