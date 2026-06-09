---
id: TASK-2321
title: Address PR 2320 Chatterbox review feedback and rebase
status: Done
labels:
- chatterbox
- tts
- review
references:
- https://github.com/rmusser01/tldw_server/pull/2320
modified_files:
- apps/packages/ui/src/services/background-proxy.ts
- apps/packages/ui/src/services/__tests__/background-proxy.test.ts
- pyproject.toml
- tldw_Server_API/Config_Files/privilege_catalog.yaml
- tldw_Server_API/app/api/v1/endpoints/audio/audio_voice_conversion.py
- tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py
- tldw_Server_API/app/core/TTS/adapter_registry.py
- tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py
- tldw_Server_API/app/core/TTS/tts_exceptions.py
- tldw_Server_API/app/core/TTS/tts_service_v2.py
- tldw_Server_API/tests/TTS/test_tts_adapters.py
- tldw_Server_API/tests/TTS/test_tts_service_v2.py
- tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py
- tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2320 on latest dev, evaluate all PR comments, remediate accepted review findings, rerun focused verification, and update the PR branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2320 onto latest origin/dev and addressed the current Gemini, Qodo, and CodeRabbit review comments. The second-pass fixes pin the Perth Git dependency to a commit SHA, make direct single-file upload fallback preserve the new `files` field while appending the legacy `file` alias, require admin access plus a cataloged `audio.tts_provider_unload` scope for provider unloads, return 409 when unload is blocked by active TTS work, clean up unload cache state even when adapter close fails, prevent failed voice-conversion upload writes from leaking temp files, fail Chatterbox local-only runtime loading without process-global offline env mutation, report Chatterbox watermark metadata correctly, strip upstream chunk aliases from child chunk requests, normalize Chatterbox `output_format` back onto the request object, and run voice conversion through the same concurrency/accounting guards as speech generation. Backlog DoD checkboxes and duplicate final-summary markers called out by CodeRabbit were also cleaned up.

Verification recorded:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m py_compile tldw_Server_API/app/api/v1/endpoints/audio/audio_voice_conversion.py tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py tldw_Server_API/app/core/TTS/adapter_registry.py tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py tldw_Server_API/app/core/TTS/tts_service_v2.py tldw_Server_API/app/core/TTS/tts_exceptions.py` -> passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/TTS/test_tts_service_v2.py tldw_Server_API/tests/TTS/test_tts_adapters.py tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v` -> 123 passed, 3 skipped.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py -v` -> 48 passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/PrivilegeCatalog/test_endpoint_scope_catalog_sync.py tldw_Server_API/tests/PrivilegeCatalog/test_privilege_catalog_loader.py -v` -> 10 passed.
- `bunx vitest run src/services/__tests__/background-proxy.test.ts src/services/__tests__/server-capabilities.test.ts src/services/tldw/__tests__/audio-models.test.ts src/services/tldw/__tests__/voice-cloning.test.ts` -> 4 files passed, 80 tests passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/audio/audio_voice_conversion.py tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py tldw_Server_API/app/core/TTS/adapter_registry.py tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py tldw_Server_API/app/core/TTS/tts_service_v2.py tldw_Server_API/app/core/TTS/tts_exceptions.py -f json -o /tmp/bandit_pr2320_review_scope_2.json` -> 0 findings.
- `git diff --check` -> passed.
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
