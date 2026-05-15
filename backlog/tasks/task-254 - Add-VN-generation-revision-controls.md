---
id: TASK-254
title: Add VN generation revision controls
status: Done
assignee: []
created_date: '2026-05-11 04:30'
updated_date: '2026-05-11 04:30'
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
Implement Task 7 from Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md: revision activation, regeneration, cancellation, and checkpoint restore for scripted VN generation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cancel with on_cancel jumps and advances through normal script flow
- [x] #2 Cancel without on_cancel leaves stable canceled state
- [x] #3 Regenerate creates revision history while preserving active revision semantics
- [x] #4 Activation changes current public output without rewriting original events
- [x] #5 Activation is blocked after downstream material events
- [x] #6 Checkpoint restore restores exact active revision map
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing generation action, checkpoint, save-slot, and restore helpers.
2. Add red tests for Task 7 command semantics in the focused VN Play suites.
3. Implement service/repository revision controls in small increments.
4. Extend checkpoint/save-slot restore snapshots to carry active generation revision maps.
5. Run focused/full VN Play verification, compileall, Bandit, and diff checks before commit.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Implemented scripted VN generation cancel, regenerate, and revision activation service flows.
- Added active generation revision snapshots to checkpoints/save slots and restore support for both active_revision_id and latest_request_id.
- Addressed review findings by blocking regeneration after downstream material events, making cancel-without-on_cancel non-advanceable, prevalidating invalid on_cancel branches before mutation, and exposing active generation public output in script state.
- Verification: `python -m pytest tldw_Server_API/tests/VN_Play/test_vn_play_scripted_generation_runtime.py -q --tb=short` -> 19 passed; `python -m pytest tldw_Server_API/tests/VN_Play -q` -> 198 passed; `python -m compileall ...` on touched backend files passed; `python -m bandit -r ... -f json -o /tmp/bandit_vn_task254.json` -> 0 results and 0 errors; `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added scripted VN generation revision controls with cancel/regenerate/activate service paths, downstream material guards, public active-generation output, and active revision checkpoint/save-slot restore semantics. Regression coverage now exercises cancel stability, invalid cancel branch recovery, revision history, activation safety, downstream blocking, and exact active revision map restore.
<!-- SECTION:FINAL_SUMMARY:END -->
