---
id: TASK-533
title: Bound Chatterbox voice conditionals cache
status: Done
labels:
- tts
- chatterbox
- hardening
modified_files:
- tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py
- tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py
- tldw_Server_API/Config_Files/tts_providers_config.yaml
- Docs/STT-TTS/CHATTERBOX_SETUP.md
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a bounded cache for prepared Chatterbox voice conditionals so repeated reference audio remains fast without letting long-running adapters retain unbounded conditionals entries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add a failing adapter test for LRU eviction, replace the plain dict conditionals cache with an OrderedDict bounded by chatterbox_conditionals_cache_size / conditionals_cache_size, document the setting, then run focused pytest, Bandit, and diff hygiene checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented a bounded Chatterbox voice-conditionals cache. Cache hits now refresh recency, inserts evict the least recently used entry beyond the configured max, and operators can tune the max with chatterbox_conditionals_cache_size / conditionals_cache_size (default 16, 0 disables retention). Verified with adapter tests, provider default-policy test, Bandit on the adapter, and git diff --check.
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
