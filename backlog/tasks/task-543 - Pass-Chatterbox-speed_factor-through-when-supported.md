---
id: TASK-543
title: Pass Chatterbox speed_factor through when supported
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 00:00'
labels:
  - tts
  - chatterbox
  - compatibility
  - generation
dependencies: []
references:
  - 'https://github.com/devnen/Chatterbox-TTS-Server'
  - Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add safe Chatterbox speed_factor support by mapping explicit extra_params.speed_factor or non-default TTSRequest.speed into generation kwargs, relying on the existing runtime signature filter to drop the kwarg for upstream versions that do not support it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default speed does not add speed_factor to Chatterbox generation kwargs.
- [x] #2 Non-default TTSRequest.speed is offered as speed_factor.
- [x] #3 Explicit extra_params.speed_factor takes precedence over request.speed.
- [x] #4 Turbo and standard/multilingual generation paths use the same safe pass-through behavior.
- [x] #5 Setup docs and parity plan document the speed_factor behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add failing adapter tests proving explicit speed_factor and non-default request.speed become candidate generation kwargs, while default speed does not add a kwarg. Implement the pass-through for standard/multilingual and Turbo kwargs, update docs/plan, then verify with the full Chatterbox adapter mock suite, Bandit on the adapter, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added _resolve_speed_factor() to prefer explicit extra_params.speed_factor and otherwise translate non-default TTSRequest.speed into a speed_factor candidate kwarg. Standard, multilingual, and Turbo generation kwargs now include speed_factor only when requested; _filter_generation_kwargs still drops it for runtimes whose generate signature does not accept it.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed safe Chatterbox speed_factor pass-through. Verification: python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v -> 41 passed; python -m bandit -r tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py -f json -o /tmp/bandit_chatterbox_speed_factor_task543.json -> clean; git diff --check -> clean.
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
