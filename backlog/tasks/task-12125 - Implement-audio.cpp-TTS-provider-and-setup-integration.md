---
id: TASK-12125
title: Implement audio.cpp TTS provider and setup integration
status: In Progress
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
- Docs/superpowers/plans/2026-07-03-audio-cpp-tts-provider-implementation-plan.md
- backlog/tasks/task-12125 - Implement-audio.cpp-TTS-provider-and-setup-integration.md
- tldw_Server_API/Config_Files/tts_providers_config.yaml
- tldw_Server_API/app/core/TTS/adapter_registry.py
- tldw_Server_API/tests/TTS_NEW/fixtures/empty_config.txt
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
- [ ] #2 The adapter/client can synthesize through `audiocpp_server` using the existing `/api/v1/audio/speech` flow, with tested request translation, WAV response handling, one-shot streaming compatibility, format conversion handoff, and sanitized error mapping.
- [ ] #3 Reference-audio and option passthrough behavior is safe by default: loopback-only base URLs unless explicitly allowed, external reference audio disabled unless configured, server-local scratch paths constrained, and only allowlisted scalar options sent upstream.
- [ ] #4 Managed sidecar support can render upstream server config, choose a loopback port, wait for health, avoid tight restart loops, and shut down cleanly without exposing arbitrary command args or process output.
- [ ] #5 Installer/setup helpers and documentation cover explicit clone/build/config/model steps without silent network downloads during normal server startup or inference.
- [ ] #6 Focused pytest, Ruff, and Bandit verification are recorded for the touched implementation scope.
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

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

Pending implementation.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or documented as non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
