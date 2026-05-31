---
id: TASK-454
title: Plan OmniVoice managed sidecar real synthesis implementation
status: Done
labels:
- tts
- omnivoice
- planning
references:
- TASK-453
- Docs/superpowers/specs/2026-05-22-omnivoice-real-sidecar-synthesis-design.md
- https://github.com/k2-fsa/OmniVoice
documentation:
- Docs/superpowers/plans/2026-05-22-omnivoice-real-sidecar-synthesis-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-22-omnivoice-real-sidecar-synthesis-implementation-plan.md
- backlog/tasks/task-454 - Plan-OmniVoice-managed-sidecar-real-synthesis-implementation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for finishing the existing managed OmniVoice TTS sidecar so it runs the real OmniVoice Python API instead of returning stub silent WAV audio.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and reviewed the implementation plan for finishing the OmniVoice managed sidecar real synthesis integration. The plan covers protocol schemas, adapter canonical payload normalization, real sidecar runtime, sidecar API wiring, supervisor config propagation, installer local model-path enforcement, opt-in real-runtime tests, docs, verification, and Bandit.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
