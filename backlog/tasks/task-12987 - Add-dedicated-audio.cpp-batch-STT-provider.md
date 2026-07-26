---
id: TASK-12987
title: Add dedicated audio.cpp batch STT provider
status: In Progress
assignee: []
created_date: '2026-07-26 03:27'
updated_date: '2026-07-26 03:30'
labels:
  - stt
  - benchmark
  - audio-cpp
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-25-audio-cpp-batch-stt-provider-design.md
  - 'https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a first-class external-server-only audio.cpp STT adapter and provider registration. The adapter must connect to a user-managed audiocpp_server, validate the pinned HTTP contract, support ordinary batch transcription and the native STT benchmark, and never download, build, launch, restart, terminate, or silently fall back from audio.cpp.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Registers canonical audio-cpp provider with audiocpp and audio_cpp aliases.
- [ ] #2 Uses network-free planning followed by consent-gated health/model discovery and WAV multipart transcription.
- [ ] #3 Records descriptive audio.cpp backend/model metadata while leaving weight identity unresolved and policy gates ineligible.
- [ ] #4 Supports strict and normalized benchmark scoring with separate cold-first and warm timing.
- [ ] #5 Normal CI uses fake transports and upstream-shaped fixtures; live audio.cpp coverage is opt-in.
- [ ] #6 Configuration and user documentation describe setup, limitations, network consent, and true cold-start procedure.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-25: Human-approved design recorded in Docs/superpowers/specs/2026-07-25-audio-cpp-batch-stt-provider-design.md. Scope is external-server-only, WAV batch transcription, network-free planning, consent-gated discovery, unresolved weight identity, no fallback/retry/download/process supervision. Design-only changed-file pre-commit and git diff --check passed using the repository-level .venv.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
