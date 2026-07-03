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
- backlog/tasks/task-12125 - Implement-audio.cpp-TTS-provider-and-setup-integration.md
- docs/superpowers/plans/2026-07-03-audio-cpp-tts-provider-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the accepted Approach A design for `0xShug0/audio.cpp`: a disabled-by-default `audio_cpp` TTS provider that routes through the existing tldw_server TTS service, can call an external `audiocpp_server`, can optionally manage a loopback sidecar, and includes setup/admin documentation plus tests for registry, configuration, request safety, adapter behavior, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `audio_cpp` is registered as a first-class TTS provider with explicit aliases, namespaced model aliases, disabled-by-default config, and no regression to existing `pocket_tts` routing.
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
