---
id: TASK-549
title: Expose TTS provider model-info endpoint
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 00:32'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a focused provider model-info endpoint for TTS providers so Chatterbox clients can discover loaded state, supported model IDs, family metadata, and unload route without parsing the full providers or health payload.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GET /api/v1/audio/tts/providers/{provider}/model-info returns provider status, loaded/initialized state, capabilities, supported model IDs, and Chatterbox family metadata when present.
- [x] #2 Unknown providers return HTTP 404 instead of an empty model-info payload.
- [x] #3 The endpoint does not expose provider config secrets or raw filesystem paths.
- [x] #4 Focused endpoint tests fail before implementation and pass after; touched backend Python path passes Bandit and git diff --check is clean.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented GET /api/v1/audio/tts/providers/{provider}/model-info. The endpoint combines provider status and sanitized capabilities into a focused payload with provider, status, initialized/loaded flags, model IDs, model family metadata, optional voice-conversion metadata, capabilities, and the matching unload endpoint. Unknown providers return HTTP 404. Added aggregate audio module exports and setup/plan docs.

Verification: RED focused provider_model_info tests failed before implementation with missing route. GREEN focused provider_model_info tests passed 2 tests. Broader provider/voice endpoint slice passed 10 tests. Bandit on audio_tts.py, audio.py, and audio/__init__.py wrote /tmp/bandit_chatterbox_model_info_task549.json with results empty. git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a focused TTS provider model-info endpoint for Chatterbox-style discovery of loaded state, supported model IDs, family metadata, voice-conversion metadata, and unload route. The endpoint returns 404 for unknown providers, sanitizes capability/status values, is documented, and passed focused red/green tests, broader endpoint tests, Bandit, and git diff --check.
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
