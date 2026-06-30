---
id: TASK-535
title: Normalize Chatterbox provider YAML config aliases
status: Done
labels:
- tts
- chatterbox
- config
modified_files:
- tldw_Server_API/app/core/TTS/adapter_registry.py
- tldw_Server_API/tests/TTS/test_tts_adapters.py
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expand Chatterbox provider config alias normalization so unprefixed YAML settings for family selection, model paths, auto-download, conditionals cache, and generation defaults are exposed under the adapter's chatterbox_* keys.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add a failing registry test for Chatterbox provider YAML config alias normalization, expand _apply_provider_aliases for Chatterbox generic keys, update the Chatterbox parity plan, then verify with the TTS adapter registry test file, Bandit, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Expanded Chatterbox provider config alias normalization so YAML-style generic keys are duplicated to adapter-prefixed chatterbox_* keys for variant, model paths, VC model path, device/runtime toggles, auto-download, conditionals cache size, and generation defaults. Verified with python -m pytest tldw_Server_API/tests/TTS/test_tts_adapters.py -v, Bandit on adapter_registry.py, and git diff --check.
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
