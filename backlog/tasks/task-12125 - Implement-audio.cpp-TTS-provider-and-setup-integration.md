---
id: TASK-12125
title: Implement audio.cpp TTS provider and setup integration
status: Done
labels:
- audio
- tts
- implementation
- setup
references:
- Docs/superpowers/specs/2026-07-03-audio-cpp-tts-integration-design.md
- https://github.com/0xShug0/audio.cpp
- https://raw.githubusercontent.com/0xShug0/audio.cpp/release-0.1/app/server/README.md
documentation:
- docs/superpowers/plans/2026-07-03-audio-cpp-tts-provider-implementation-plan.md
modified_files:
- Docs/STT-TTS/TTS-SETUP-GUIDE.md
- Docs/superpowers/plans/2026-07-03-audio-cpp-tts-provider-implementation-plan.md
- backlog/tasks/task-12125 - Implement-audio.cpp-TTS-provider-and-setup-integration.md
- Helper_Scripts/install_tts_audio_cpp.py
- tldw_Server_API/Config_Files/tts_providers_config.yaml
- tldw_Server_API/app/core/TTS/adapter_registry.py
- tldw_Server_API/app/core/TTS/adapters/audio_cpp_adapter.py
- tldw_Server_API/app/core/TTS/adapters/audio_cpp_client.py
- tldw_Server_API/app/core/TTS/adapters/audio_cpp_config.py
- tldw_Server_API/app/core/TTS/adapters/audio_cpp_sidecar_supervisor.py
- tldw_Server_API/tests/TTS_NEW/fixtures/empty_config.txt
- tldw_Server_API/tests/TTS_NEW/integration/test_audio_cpp_tts_service.py
- tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_adapter.py
- tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_client.py
- tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_config.py
- tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_sidecar_supervisor.py
- tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_installer.py
- tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_registry.py
- tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_tts_config.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the accepted Approach A design for `0xShug0/audio.cpp`: a disabled-by-default `audio_cpp` TTS provider that routes through the existing tldw_server TTS service, can call an external `audiocpp_server`, can optionally manage a loopback sidecar, and includes setup/admin documentation plus tests for registry, configuration, request safety, adapter behavior, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `audio_cpp` is registered as a first-class TTS provider with explicit aliases, namespaced model aliases, disabled-by-default config, and no regression to existing `pocket_tts` routing.
- [x] #2 The adapter/client can synthesize through `audiocpp_server` using the existing `/api/v1/audio/speech` flow, with tested request translation, WAV response handling, one-shot streaming compatibility, format conversion handoff, and sanitized error mapping.
- [x] #3 Reference-audio and option passthrough behavior is safe by default: loopback-only base URLs unless explicitly allowed, external reference audio disabled unless configured, server-local scratch paths constrained, and only allowlisted scalar options sent upstream.
- [x] #4 Managed sidecar support can render upstream server config, choose a loopback port, wait for health, avoid tight restart loops, and shut down cleanly without exposing arbitrary command args or process output.
- [x] #5 Installer/setup helpers and documentation cover explicit clone/build/config/model steps without silent network downloads during normal server startup or inference.
- [x] #6 Focused pytest, Ruff, and Bandit verification are recorded for the touched implementation scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow `docs/superpowers/plans/2026-07-03-audio-cpp-tts-provider-implementation-plan.md`. Use test-driven slices and update this task with touched files, verification output, and any scoped deviations before finalization.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

- Created the implementation task and linked plan after the accepted design task `TASK-12124`.
- Current workspace note: active branch is `codex/parakeet-onnx-wav-fallback`, with unrelated untracked runtime/template files present. Source edits should proceed in an isolated worktree or after an explicit in-place decision.
- Implementation decision: using isolated git worktree `C:\Users\GDesktop-1\Working\tldw\.worktrees\audio-cpp-tts-provider` on branch `codex/audio-cpp-tts-provider`. The parent checkout remains on `dev` with unrelated untracked files left untouched.
- Baseline attempt with `..\..\.venv\Scripts\python.exe -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/test_fish_s2_registry.py tldw_Server_API/tests/TTS_NEW/unit/test_pocket_tts_cpp_registry.py` stopped during collection because the shared venv was missing declared dependency `pytest-asyncio>=1.3.0`.
- Installed declared dependency `pytest-asyncio==1.4.0` into the shared project venv from cache, then reran the same baseline: 5 passed, 6 warnings in 83.26s.
- Stage 1 red test: `..\..\.venv\Scripts\python.exe -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_registry.py tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_tts_config.py` failed as expected with 5 failures for missing `TTSProvider.AUDIO_CPP`, missing `providers.audio_cpp`, and missing `format_preferences.audio_cpp`.
- Stage 1 implementation added `TTSProvider.AUDIO_CPP`, provider/model aliases, lazy default adapter mapping, disabled `audio_cpp` YAML config, and format preferences limited to `wav`, `mp3`, `opus`, `flac`, `aac`, and `pcm`.
- Stage 1 green test: the same audio.cpp test command passed with 5 passed, 5 warnings in 35.09s.
- Stage 1 adjacent regression: `..\..\.venv\Scripts\python.exe -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/test_fish_s2_registry.py tldw_Server_API/tests/TTS_NEW/unit/test_pocket_tts_cpp_registry.py` passed with 5 passed, 6 warnings in 40.86s.
- Stage 2 red test: `..\..\.venv\Scripts\python.exe -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_client.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_config.py` failed during collection with missing `audio_cpp_client` and `audio_cpp_config` modules.
- Stage 2 implementation added `AudioCppClient`, `AudioCppSpeechResult`, `AudioCppConfig`, base URL and managed-host validation, option allowlist filtering, server-config rendering, path containment checks, and generated scratch reference names.
- Stage 2 green test: the same client/config command passed with 11 passed, 10 warnings in 52.02s, then after Ruff fixes passed again with 11 passed, 10 warnings in 68.63s.
- Stage 2 combined focused test: `..\..\.venv\Scripts\python.exe -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_registry.py tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_tts_config.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_client.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_config.py` passed with 16 passed, 10 warnings in 99.73s.
- Stage 2 adjacent regression: `..\..\.venv\Scripts\python.exe -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/test_fish_s2_registry.py tldw_Server_API/tests/TTS_NEW/unit/test_pocket_tts_cpp_registry.py` passed with 5 passed, 6 warnings in 64.47s.
- Stage 2 Ruff check passed for `audio_cpp_client.py`, `audio_cpp_config.py`, `test_audio_cpp_client.py`, and `test_audio_cpp_config.py`; `git diff --check` also passed for the Stage 2 touched files.
- Stage 3 red test: `..\..\.venv\Scripts\python.exe -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_adapter.py tldw_Server_API/tests/TTS_NEW/integration/test_audio_cpp_tts_service.py` failed during collection with missing `tldw_Server_API.app.core.TTS.adapters.audio_cpp_adapter`, as expected before implementation.
- Stage 3 implementation added `AudioCppTTSAdapter`, tested capabilities, allowlisted request translation, full-byte one-shot streaming compatibility, managed/shared reference-audio staging, catalog-only voice behavior, and service routing through `TTSServiceV2`.
- Stage 3 green test: the same adapter/service command passed with 9 passed, 7 warnings in 80.59s, then after Ruff fixes passed again with 9 passed, 7 warnings in 79.52s.
- Stage 3 combined focused test: `..\..\.venv\Scripts\python.exe -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_registry.py tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_tts_config.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_config.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_client.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_adapter.py tldw_Server_API/tests/TTS_NEW/integration/test_audio_cpp_tts_service.py` passed with 25 passed, 12 warnings in 117.02s.
- Stage 3 adjacent regression: `..\..\.venv\Scripts\python.exe -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/test_fish_s2_registry.py tldw_Server_API/tests/TTS_NEW/unit/test_pocket_tts_cpp_registry.py` passed with 5 passed, 6 warnings in 64.60s.
- Stage 3 Ruff check passed for `audio_cpp_adapter.py`, `test_audio_cpp_adapter.py`, and `test_audio_cpp_tts_service.py`.
- Stage 3 post-review refactor moved reference-audio file writes off the event loop; final focused rerun passed with 9 passed, 7 warnings in 76.92s.
- Stage 4 red test: `..\..\.venv\Scripts\python.exe -m pytest -q --tb=short tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_sidecar_supervisor.py tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_installer.py` failed as expected with 9 missing-module/import failures for `audio_cpp_sidecar_supervisor` and `Helper_Scripts.install_tts_audio_cpp`.
- Stage 4 implementation added `AudioCppSidecarSupervisor`, managed-mode adapter startup wiring, `Helper_Scripts/install_tts_audio_cpp.py`, sidecar/installer tests, and `Docs/STT-TTS/TTS-SETUP-GUIDE.md` setup guidance.
- Stage 4 green test: the same sidecar/installer command passed with 9 passed, 6 warnings in 71.75s.
- Stage 4 self-review found that omitting an explicit subprocess environment would inherit parent secrets. Added a failing sidecar test that set `HF_TOKEN` and `OPENAI_API_KEY` and expected them not to be passed to the child process; it failed with missing `env`, then passed after adding an allowlisted sidecar env.
- Stage 4 Ruff check passed for `audio_cpp_sidecar_supervisor.py`, `audio_cpp_adapter.py`, `install_tts_audio_cpp.py`, `test_audio_cpp_sidecar_supervisor.py`, and `test_audio_cpp_installer.py` after env hardening.
- Stage 4 adapter regression: `..\..\.venv\Scripts\python.exe -m pytest -q --tb=short tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_adapter.py tldw_Server_API/tests/TTS_NEW/integration/test_audio_cpp_tts_service.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_sidecar_supervisor.py tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_installer.py` passed with 18 passed, 9 warnings in 85.71s.
- Stage 5 focused audio.cpp suite: `..\..\.venv\Scripts\python.exe -m pytest -q --tb=short tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_registry.py tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_tts_config.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_config.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_client.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_adapter.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_audio_cpp_sidecar_supervisor.py tldw_Server_API/tests/TTS_NEW/unit/test_audio_cpp_installer.py tldw_Server_API/tests/TTS_NEW/integration/test_audio_cpp_tts_service.py` passed with 34 passed, 14 warnings in 124.60s.
- Stage 5 adjacent regression note: the plan-named `tldw_Server_API/tests/TTS_NEW/unit/adapters/test_pocket_tts_cpp_adapter.py` is absent in this checkout. Substituted `tldw_Server_API/tests/TTS_NEW/unit/test_pocket_tts_cpp_registry.py` plus `test_pocket_tts_cpp_installer.py`.
- Stage 5 adjacent regression: `..\..\.venv\Scripts\python.exe -m pytest -q --tb=short --basetemp .pytest_tmp_adjacent tldw_Server_API/tests/TTS_NEW/unit/test_fish_s2_registry.py tldw_Server_API/tests/TTS_NEW/unit/test_pocket_tts_cpp_registry.py tldw_Server_API/tests/TTS_NEW/unit/test_pocket_tts_cpp_installer.py` passed with 14 passed, 6 warnings in 44.55s. The first substitute run without `--basetemp` failed only because pytest tried to create temp paths under `C:\Users\GDesktop-1\AppData\Local\Temp\pytest-of-GDesktop-1`, outside the writable sandbox.
- Stage 5 Ruff check passed for all touched Python files in the implementation plan.
- Stage 5 Bandit check passed: `..\..\.venv\Scripts\python.exe -m bandit -r tldw_Server_API/app/core/TTS/adapters/audio_cpp_client.py tldw_Server_API/app/core/TTS/adapters/audio_cpp_config.py tldw_Server_API/app/core/TTS/adapters/audio_cpp_sidecar_supervisor.py tldw_Server_API/app/core/TTS/adapters/audio_cpp_adapter.py Helper_Scripts/install_tts_audio_cpp.py -f json -o models/audio_cpp/test_artifacts/bandit_audio_cpp_tts.json`; report summary had 0 results, 0 high, 0 medium, 0 low findings.
- Known limitations: no live `audiocpp_server` smoke test was run in this environment; managed mode remains documented as CUDA-first; generic `/v1/tasks/run`, STT, VAD, diarization, and other Approach C task surfaces are not included in this TTS slice.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

Approach A is implemented for TTS: `audio_cpp` is registered as a disabled-by-default provider, routes through the existing `TTSServiceV2` flow, includes client/config/adapter/managed-sidecar support, has explicit setup helpers and docs, and keeps request options, paths, reference audio, process args, and subprocess environment constrained by default. Verification passed for focused audio.cpp tests, adjacent provider regressions, Ruff, and Bandit. Follow-up Approach C work should reuse `AudioCppClient` and `AudioCppSidecarSupervisor` for broader `/v1/tasks/run`-style audio processing surfaces after a real runtime smoke test is available.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or documented as non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
