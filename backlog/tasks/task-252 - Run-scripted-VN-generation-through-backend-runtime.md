---
id: TASK-252
title: Run scripted VN generation through backend runtime
status: Done
assignee: []
created_date: '2026-05-10 23:51'
updated_date: '2026-05-11 00:33'
labels:
  - vn
  - scripted-generation
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1535'
documentation:
  - Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md
  - Docs/superpowers/specs/2026-05-10-vn-scripted-model-generation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 5 from Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md: integrate model-backed generate opcodes into scripted VN Play runtime while preserving literal generation behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Automatic generation creates request/revision/generation rows and advances scene
- [x] #2 Confirmation-gated generation pauses without model call
- [x] #3 Batch cap of one pauses before a second automatic generation
- [x] #4 scene_update persists applied and rejected visual resolver outcomes on the active revision
- [x] #5 Model failure persists failed revision/request and does not advance cursor
- [x] #6 Existing literal generation tests still pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing scripted-runtime tests for automatic generation, confirmation gating, batch cap, model failure, and literal preservation. 2. Refactor script execution so model-backed generate opcodes return generation descriptors instead of requiring literal text. 3. Resolve pinned profile snapshots from published script-version maps and call the Task 4 generation orchestration helper. 4. Persist generation events and scene updates for successful outputs while leaving failed attempts at the generation point for retry. 5. Verify focused VN Play/VN Scripts tests, compile, Bandit, and diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented scripted runtime model-backed generate execution. Automatic generate opcodes now create generation/request/action/revision rows through execute_script_generation_call, confirmation-gated opcodes pause without provider calls, and one-generation batch cap pauses before a following automatic generate.

scene_update generation revisions now persist applied_visuals and rejected_visuals from visual directive resolution without failing text generation on missing or failed visual resolution.

Verification: focused VN Play pytest suite passed with 119 passed and 8 warnings. compileall passed. Bandit on touched backend files reported 0 findings. git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Integrated model-backed scripted VN generate opcodes into the backend runtime. Script advance/choice endpoints now await async scripted execution; automatic generation calls reuse the generation orchestration helper; confirmation-gated generation records pending confirmation without calling the model; batch cap pauses before chained automatic generation; failed model calls persist failed request/revision state without advancing; scene_update outputs persist applied and rejected visual resolver outcomes on revisions. Added focused runtime tests and updated the stale API expectation for no-literal generate opcodes.
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
