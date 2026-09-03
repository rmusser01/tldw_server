---
id: TASK-12124
title: Design audio.cpp TTS provider and setup integration
status: Done
labels:
- audio
- tts
- design
- setup
references:
- https://github.com/0xShug0/audio.cpp
- https://raw.githubusercontent.com/0xShug0/audio.cpp/release-0.1/app/server/README.md
- https://raw.githubusercontent.com/0xShug0/audio.cpp/release-0.1/README.md
- https://raw.githubusercontent.com/0xShug0/audio.cpp/release-0.1/LICENSE
documentation:
- Docs/superpowers/specs/2026-07-03-audio-cpp-tts-integration-design.md
modified_files:
- Docs/superpowers/specs/2026-07-03-audio-cpp-tts-integration-design.md
- backlog/tasks/task-12124 - Design-audio.cpp-TTS-provider-and-setup-integration.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the accepted design for integrating `0xShug0/audio.cpp` as a tldw_server audio backend using Approach A first: a first-class `audio_cpp` TTS provider with external-server and optional managed-sidecar modes, setup/admin installer guidance, reference-audio constraints, and a staged path toward broader Audio Studio support after TTS proves out.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design documents `audio_cpp` as a TTS provider that uses existing TTS service, adapter registry, fallback, quota, history, and storage behavior.
- [x] #2 Design covers external-server and managed-sidecar runtime modes, setup/admin installer flow, and explicit model installation boundaries.
- [x] #3 Design records reviewed constraints around CUDA-first server support, server-local paths, single-chunk streaming compatibility, configured voice mappings, and licensing/bundling boundaries.
- [x] #4 Design defines testing, security, error mapping, and the follow-up path from Approach A to broader Audio Studio/Approach C support.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Write `Docs/superpowers/specs/2026-07-03-audio-cpp-tts-integration-design.md` from the approved brainstorming sections and review corrections. Keep implementation out of scope until the user reviews the written spec and approves transition to planning.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

- Created `Docs/superpowers/specs/2026-07-03-audio-cpp-tts-integration-design.md` from the approved brainstorming sections.
- Kept the first slice scoped to Approach A: a first-class `audio_cpp` TTS provider and setup story, with Approach C deferred until TTS proves the runtime and path-safety model.
- Recorded review corrections for upstream server support, server-local reference-audio paths, single-chunk streaming compatibility, configured voice mappings, model-manager wrapping, and optional external-component licensing boundaries.
- Amended the spec after review to tighten current-codebase integration risks: config-schema preservation, provider enum/routing, namespaced model aliases, loopback-only default `base_url`, external reference-audio opt-in, sidecar lifecycle controls, verified voice request fields, and format-advertising tests.
- Verification is documentation-only: no backend code paths were changed, so Bandit is not applicable for this task.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

Design spec written and amended for integrating `0xShug0/audio.cpp` as an optional tldw_server TTS backend. The spec covers architecture, provider registration, configuration, setup/installer scope, TTS request and response behavior, security, testing, documentation, and the staged follow-up path from Approach A to broader Audio Studio support.

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
