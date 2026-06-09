---
id: TASK-544
title: Expose Chatterbox generation and chunking controls in capabilities
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 00:06'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add discoverable Chatterbox capability metadata for the upstream-aligned controls now supported by the adapter/service, including generation parameters, Turbo-specific controls, split_text/chunk_size aliases, speed_factor handling, and BF16 modes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chatterbox adapter capabilities metadata lists standard/multilingual generation controls, Turbo controls, speed factor support, chunking aliases, and BF16 modes.
- [x] #2 Capability metadata remains static/discoverable and does not claim unsupported speech-rate support through the generic capabilities flag.
- [x] #3 Adapter tests cover the new metadata keys and fail before implementation.
- [x] #4 Touched backend Python path passes focused pytest and Bandit; diff whitespace check is clean.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented capability metadata in ChatterboxAdapter.get_capabilities() for generation_parameters, speed_factor request fields, chunking aliases, and BF16 config/env modes. Added adapter test coverage that first failed on missing generation_parameters, then passed after implementation. Updated CHATTERBOX_SETUP.md and the upstream parity plan with the discoverability note.

Verification:
- RED: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k capabilities_expose_family_metadata -v (failed with KeyError: generation_parameters)
- GREEN: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k capabilities_expose_family_metadata -v (1 passed)
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v (41 passed)
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py -f json -o /tmp/bandit_chatterbox_capability_metadata_task544.json (results: [])
- git diff --check (clean)
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added discoverable Chatterbox capability metadata for standard/multilingual generation controls, Turbo controls, speed_factor request fields, upstream-style split_text/chunk_size chunking aliases, and BF16 modes. Kept generic supports_speech_rate false and documented the expanded metadata surface in the Chatterbox setup runbook and implementation plan.
<!-- SECTION:FINAL_SUMMARY:END -->

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
