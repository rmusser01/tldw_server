---
id: TASK-545
title: Support safe Chatterbox predefined_voice_id custom voice alias
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 00:11'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Map upstream-style Chatterbox extra_params.voice_mode=predefined plus extra_params.predefined_voice_id to tldw stored custom voice resolution, without enabling arbitrary reference_audio_filename filesystem reads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 OpenAI-compatible Chatterbox speech requests with extra_params.voice_mode='predefined' and extra_params.predefined_voice_id resolve stored custom voice audio through VoiceManager when no direct voice_reference is provided.
- [x] #2 The alias is scoped to Chatterbox and does not change existing custom:<voice_id> behavior for other providers.
- [x] #3 Arbitrary upstream reference_audio_filename values are not read from disk by this compatibility path.
- [x] #4 Integration tests cover the alias and fail before implementation; focused pytest, Bandit on touched backend code, and git diff --check are clean.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a safe Chatterbox-only stored voice alias in TTSServiceV2. extra_params.voice_mode=predefined plus extra_params.predefined_voice_id now resolves through VoiceManager.load_voice_reference_audio() when no direct voice_reference is present. Existing voice=custom:<voice_id> handling remains intact for other providers, and reference_audio_filename is preserved as inert request metadata rather than read from disk. Updated OpenAI speech schema docs, CHATTERBOX_SETUP.md, and the Chatterbox parity plan.

Verification:
- RED: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_custom_voice_resolution.py -k chatterbox -v (1 failed, 1 passed; predefined alias reached adapter without voice_reference)
- GREEN: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_custom_voice_resolution.py -k chatterbox -v (2 passed)
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_custom_voice_resolution.py -v (5 passed)
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/TTS/test_tts_service_v2.py -v (32 passed)
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/TTS/tts_service_v2.py tldw_Server_API/app/api/v1/schemas/audio_schemas.py -f json -o /tmp/bandit_chatterbox_predefined_voice_task545.json (results: [])
- git diff --check (clean)
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Chatterbox-only compatibility for upstream-style predefined voice selection by resolving extra_params.voice_mode=predefined plus extra_params.predefined_voice_id through the authenticated user custom voice store. Kept arbitrary reference_audio_filename values inert and documented the safe mapping in API schema docs and the Chatterbox setup runbook.
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
