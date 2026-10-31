---
id: TASK-12124
title: Plan Stage 2 Persona transparent-video packs
status: Done
created_date: 2026-08-24 05:11
dependencies:
- TASK-12123
labels:
- persona
- persona-visuals
- video
- planning
priority: High
documentation:
- Docs/superpowers/specs/2026-08-23-persona-ambient-companion-transparent-video-design.md
- Docs/superpowers/plans/2026-08-23-persona-ambient-companion-stage-1-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-08-23-persona-transparent-video-packs-stage-2-implementation-plan.md
updated_date: 2026-08-24 05:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the execution-ready implementation plan for Stage 2 of the approved Persona Ambient Companion design: native video_clips packs, authenticated video rendering, local conversion Jobs, review/publication, dsh-pet import, and Chatbook-compatible raster fallback export. Planning only; no feature implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan consumes the exact Stage 1 interfaces and maps current renderer, Jobs, storage, archive, API, and frontend creator files.
- [x] #2 Plan decomposes Stage 2 into TDD tasks with failing tests, commands, minimal implementation steps, verification, and commits.
- [x] #3 Plan covers native video contracts, required static/sprite fallback, authenticated bounded Blob loading, capability/alpha probing, conversion validation and cancellation, immutable review/activation, and source cleanup.
- [x] #4 Plan covers streaming-safe dsh-pet ZIP/TGZ mapping and current Chatbook fallback-only export with golden compatibility checks.
- [x] #5 Plan includes backend/frontend/E2E/media fixture/Bandit verification, is self-reviewed with no placeholders, and is committed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created after final human approval of TASK-12122. Stage 2 planning depends on the Stage 1 plan so video reuses the shared engine, preference, generation, and authenticated-asset interfaces.
Plan saved at Docs/superpowers/plans/2026-08-23-persona-transparent-video-packs-stage-2-implementation-plan.md and explicitly depends on the Stage 1 plan. Self-review covered approved-spec traceability, placeholder scan, exact path/interface checks, v53/v54 migration sequencing, browser-versus-creator capability separation, archive streaming, cleanup boundary, and Chatbook compatibility. Planning-only task: Bandit is intentionally deferred to the implementation gates specified in the plan.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created a seven-task, TDD-first Stage 2 implementation plan covering video_clips v1 and schema v53, fallback-first authenticated browser playback, alpha probing, bounded local conversion, schema v54 conversion Jobs and lifecycle, guided creator UI, streaming-safe dsh-pet ZIP/TGZ review import, Chatbook fallback-only export, golden fixtures, E2E media smoke, documentation, and security gates. No feature implementation was performed.
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
