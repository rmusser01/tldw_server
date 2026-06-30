---
id: TASK-548
title: Expose OpenAI-style TTS voice catalog format
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 00:26'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an opt-in OpenAI-compatible flattened response shape for the provider voice catalog so clients can discover Chatterbox and other TTS voices without conflicting with the existing custom voice /api/v1/audio/voices endpoint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /api/v1/audio/voices/catalog?format=openai returns an object with data entries and object=list for provider voice catalog data.
- [x] #2 Each flattened voice entry includes id, object=voice, provider, name, language when available, and provider-specific metadata without changing the default catalog response.
- [x] #3 The format option supports provider filtering and keeps existing /api/v1/audio/voices custom voice list behavior untouched.
- [x] #4 Focused endpoint tests fail before implementation and pass after; touched backend Python path passes Bandit and git diff --check is clean.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented opt-in OpenAI-style voice catalog formatting on GET /api/v1/audio/voices/catalog?format=openai. The formatter flattens provider voice mappings to object=list with data entries, preserves provider filtering, includes provider/name/language and extra voice fields under metadata, and leaves the default catalog response unchanged. Added endpoint tests covering flattened catalog, provider-filtered flattened catalog, and the existing /api/v1/audio/voices custom voice route with a format query. Updated CHATTERBOX_SETUP.md and the Chatterbox upstream parity plan.

Verification: RED focused endpoint test failed on missing object wrapper; GREEN focused endpoint test passed 3 tests. Broader endpoint slice passed 8 tests with ProviderManagementEndpoints or VoiceManagementEndpoints. Bandit on audio_tts.py wrote /tmp/bandit_chatterbox_voice_catalog_task548.json with results empty. git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added opt-in OpenAI-style provider voice discovery via /api/v1/audio/voices/catalog?format=openai while preserving the existing provider catalog default and /api/v1/audio/voices custom voice route. Documented the Chatterbox discovery mapping and verified with focused red/green endpoint tests, broader provider/voice endpoint tests, Bandit, and git diff --check.
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
