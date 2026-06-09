---
id: TASK-546
title: Support Chatterbox output_format speech request alias
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 00:14'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Accept upstream-style Chatterbox output_format on OpenAI-compatible speech requests as a safe alias for response_format when response_format was not provided.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 OpenAI speech requests for Chatterbox may set output_format without response_format and receive the matching internal AudioFormat.
- [x] #2 If both response_format and output_format are provided, response_format keeps precedence.
- [x] #3 The alias is scoped to Chatterbox-family model IDs and does not alter non-Chatterbox request conversion.
- [x] #4 Tests fail before implementation and pass after implementation; focused pytest, Bandit on touched backend code, and git diff --check are clean.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added optional OpenAISpeechRequest.output_format and mapped it in TTSServiceV2._convert_request for Chatterbox-family models only when response_format was not explicitly provided. Explicit response_format keeps precedence, and non-Chatterbox requests ignore output_format. Updated CHATTERBOX_SETUP.md and the parity plan with the alias behavior.

Verification:
- RED: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/TTS/test_tts_service_v2.py -k output_format_alias -v (1 failed, 2 passed; Chatterbox output_format stayed MP3)
- GREEN: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/TTS/test_tts_service_v2.py -k output_format_alias -v (3 passed)
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/TTS/test_tts_service_v2.py -v (35 passed)
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_custom_voice_resolution.py -v (5 passed)
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/TTS/tts_service_v2.py tldw_Server_API/app/api/v1/schemas/audio_schemas.py -f json -o /tmp/bandit_chatterbox_output_format_task546.json (results: [])
- git diff --check (clean)
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added upstream-style output_format compatibility for Chatterbox speech requests while preserving OpenAI response_format precedence. The alias is scoped to Chatterbox-family model IDs, leaves non-Chatterbox conversion unchanged, and is documented in schema/runbook/plan notes.
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
