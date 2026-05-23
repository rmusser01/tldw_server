---
id: TASK-416
title: Plan media viewer read-along TTS implementation
status: Done
labels:
- plan
- webui
- extension
- tts
- media
modified_files:
- Docs/superpowers/plans/2026-05-17-media-viewer-read-along-tts-implementation-plan.md
references:
- Docs/superpowers/specs/2026-05-17-media-viewer-read-along-tts-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a detailed implementation plan for the approved media viewer read-along TTS design, including file ownership, TDD steps, staged tasks, verification commands, and execution handoff.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan starts with the required superpowers writing-plans header and references the approved design spec.
- [x] #2 Plan maps concrete shared UI files, new read-along modules, Dexie cache files, provider changes, tests, and verification commands.
- [x] #3 Plan decomposes implementation into TDD-first tasks that can be executed independently with frequent commits.
- [x] #4 Plan calls out critical risks: annotation selection mediation, large-content lazy segmentation, abort/stale suppression, cache privacy/quota, embedded media overlap, route parity, and accessibility.
- [x] #5 Plan includes execution handoff options.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan saved at Docs/superpowers/plans/2026-05-17-media-viewer-read-along-tts-implementation-plan.md.

The plan decomposes the approved design into eight staged tasks:
- Baseline and guardrail tests.
- Segment and scope primitives.
- Cache keys, Dexie store, and provider abort surface.
- Mediated content selection actions.
- Read-along playback session hook.
- ContentViewer UI integration.
- Accessibility, route parity, and regression hardening.
- Browser verification and final cleanup.

Critical risks covered:
- Existing annotation selection behavior must move to a mediated selection action path before read-along UI lands.
- Browser TTS provider must use a no-cache SpeechSynthesis path.
- Large-content read-from-here/full-item behavior must segment canonical content lazily without forcing full rendering.
- Stop/media/content changes must abort in-flight TTS work and suppress stale async completions.
- Cache metadata must avoid raw selected text and handle browser quota failures.

Verification:
- Marker scan found no unfinished-work markers in the plan or task.
- git diff --check passed for the plan and task files.
- Bandit is not applicable because this task changes documentation/task tracking only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-05-23 plan-hardening review:
- Re-reviewed the implementation plan against current `origin/dev` after the read-along implementation and PR review fixes had already landed.
- Confirmed the planned shared UI ownership, read-along module layout, `ContentViewer` wiring, annotation selection mediation, TTS provider abort surface, Dexie cache table, route parity guard, and focused test surfaces all exist in current code.
- Updated the plan with an execution-status note so future workers do not mistake the preserved unchecked task-step checklist for remaining backlog work.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the media viewer read-along TTS implementation plan at Docs/superpowers/plans/2026-05-17-media-viewer-read-along-tts-implementation-plan.md. The plan is TDD-first, scoped to shared apps/packages/ui surfaces, and includes file ownership, focused test commands, staged commits, final verification, and execution handoff options. Plan-review subagent dispatch was not performed in this turn because active tool policy requires explicit user authorization for subagents.
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
