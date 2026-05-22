---
id: TASK-455
title: Implement OmniVoice managed sidecar real synthesis
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-22 20:21
labels:
- tts
- omnivoice
- implementation
dependencies: []
references:
- TASK-453
- TASK-454
- Docs/superpowers/specs/2026-05-22-omnivoice-real-sidecar-synthesis-design.md
documentation:
- Docs/superpowers/plans/2026-05-22-omnivoice-real-sidecar-synthesis-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_protocol.py
- tldw_Server_API/app/core/TTS/adapters/omnivoice_adapter.py
- tldw_Server_API/app/core/TTS/tts_validation.py
- tldw_Server_API/app/core/TTS/adapters/omnivoice_runtime.py
- tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_server.py
- tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_supervisor.py
- Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py
- tldw_Server_API/app/core/Setup/install_manager.py
- tldw_Server_API/Config_Files/tts_providers_config.yaml
- tldw_Server_API/tests/TTS/adapters/test_omnivoice_adapter_mock.py
- tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_runtime.py
- tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_server.py
- tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_supervisor.py
- tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_installer.py
- tldw_Server_API/tests/TTS_NEW/unit/test_tts_validation_omnivoice.py
- tldw_Server_API/tests/TTS_NEW/integration/test_omnivoice_real_runtime.py
- Docs/STT-TTS/TTS-SETUP-GUIDE.md
- tldw_Server_API/app/core/TTS/TTS-README.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved implementation plan to finish the existing managed OmniVoice TTS sidecar so it uses the real OmniVoice Python API instead of returning stub silent WAV audio.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Task 2 implemented: normalized OmniVoice adapter sidecar payloads to canonical keys with design/clone conflict validation, generation object allowlist/coercion, scratch-dir direct reference materialization, native sample-rate header handling, structured sidecar error mapping, OmniVoice validation passthrough/parameter checks, and service no-fallback policy for explicit OmniVoice semantics.

Verification recorded for Task 2: red run failed 9 expected tests; focused suite later passed 29 tests; nearby OmniVoice protocol/registry/service sanitization checks passed 19 selected tests; Bandit code/tests returned 0 findings; scoped diff check passed. Full git diff --check is blocked by unrelated pre-existing trailing whitespace in Docs/Design/Agents.md.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the OmniVoice managed sidecar path so provider requests now reach a real lazy-loaded OmniVoice runtime instead of the old silent-WAV stub. The implementation adds strict sidecar protocol validation, adapter request normalization and structured sidecar error mapping, a sidecar-local runtime wrapper with local model path validation and managed clone-reference handling, runtime-backed FastAPI sidecar endpoints, supervisor config propagation/readiness handling, installer/setup-manager local model path requirements, disabled-by-default config hints, opt-in real runtime smoke tests, and setup documentation. Verification on 2026-05-22: focused OmniVoice suite `169 passed`; broad TTS unit slice `447 passed, 1 xfailed, 1 xpassed`; docs link/check smoke `4 passed`; opt-in real runtime test default run `3 skipped`; Bandit on touched code reported zero findings in `/tmp/bandit_omnivoice_real_sidecar.json`. Full `git diff --check` remains blocked by unrelated pre-existing trailing whitespace at `Docs/Design/Agents.md:127`.
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
