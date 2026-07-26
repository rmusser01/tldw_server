---
id: TASK-12987
title: Add dedicated audio.cpp batch STT provider
status: In Progress
assignee: []
created_date: '2026-07-26 03:27'
updated_date: '2026-07-26 04:05'
labels:
  - stt
  - benchmark
  - audio-cpp
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-25-audio-cpp-batch-stt-provider-design.md
  - >-
    Docs/superpowers/plans/2026-07-25-audio-cpp-batch-stt-provider-implementation-plan.md
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-07-25-audio-cpp-batch-stt-provider-implementation-plan.md task-by-task using TDD, focused commits, independent review, Bandit, and PR gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-25: Human-approved design recorded in Docs/superpowers/specs/2026-07-25-audio-cpp-batch-stt-provider-design.md. Scope is external-server-only, WAV batch transcription, network-free planning, consent-gated discovery, unresolved weight identity, no fallback/retry/download/process supervision. Design-only changed-file pre-commit and git diff --check passed using the repository-level .venv.

Independent design review found and the spec now addresses four blocking gaps: explicit ordinary API selectors/default-model routing; frozen origin/model/timeout/transport with pre-I/O plan verification; a bounded allowlisted metadata finalizer extension; and real RIFF/WAVE PCM validation with byte-zero upload. It also clarifies empty-output scoring, strict config parsing, and cache lock/reset behavior.

Independent spec re-review approved the revised design with no remaining issues. The final revision additionally makes adapter-side selector normalization mandatory so original REST/Jobs model strings cannot leak upstream, and pins the six retained artifact metadata keys.

Implementation plan drafted with TDD stages for bounded artifact metadata, strict config/selector parsing, upstream contract and WAV validation, secure no-retry HTTP/cache execution, immutable adapter registration, ordinary API and benchmark integration, documentation, and final verification/review/PR gates.

Independent plan review found three blocking issues and the plan was revised: cache discovery now uses concurrent-future single-flight with a short-held threading lock and generation-safe reset; origin validation rejects raw dot/non-root paths and reconstructs a no-trailing-slash canonical origin; WAV validation requires pre/post-open regular-file identity and streams the complete declared PCM payload to detect late truncation.

Independent implementation-plan re-review approved the revised plan with no blocking issues. Its advisory request for an explicit canonical STT config projection test was added to Task 2.
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
