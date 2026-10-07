---
id: TASK-13515
title: Resolve VN durability task-ID collisions and finalize merged tracking
status: In Progress
labels:
- vn-assets
- docs
- tracking
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Requester-approved narrow manual exception: renumber only the VN-owned TASK-13356, TASK-13358 and TASK-13369 files and update current VN references while preserving historical sections and every unrelated colliding task. Finalize the already merged PR3016 record through official Backlog operations after its identity is unique. Publish a separate documentation-only PR; the Change summary waiver applies only to PR3060.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Three VN records have unique unused IDs and matching filenames; unrelated colliding records are byte-identical to dev.
- [x] #2 Current VN task dependencies and design associations use the new IDs; historical narrative references remain intact with explicit old-to-new mappings.
- [x] #3 Merged PR3016 review tracking is finalized through official Backlog operations with accurate merge evidence and all earlier notes preserved.
- [ ] #4 Owned documentation checks and independent review pass; publish the scoped follow-up without runtime changes or applying PR3060 summary waiver.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-06: requester approved the previously requested scoped manual task-ID migration exception. Existing official MCP/CLI task-edit interfaces do not support changing an existing ID. Inventory of 136 registered worktrees and 347175 task paths found maximum root ID 13514; reserved tracking TASK-13515 and migration IDs TASK-13516/TASK-13517/TASK-13518. PR3060 remains on its separate published branch at 0f8bd6c922bbf1f6066390969c596c48c8c3b780 with current-head hosted reviews/CI pending. Implementation plan: Docs/Plans/IMPLEMENTATION_PLAN_vn_task_id_finalization.md.
2026-10-06 local migration verified: TASK-13356/TASK-13358/TASK-13369 VN records are now TASK-13516/TASK-13517/TASK-13518. Both VN dependencies and four current design/plan associations use the new IDs. Structured verification proves every original historical section/note is preserved and all four unrelated colliding records remain byte-identical to dev; the new IDs are unused in the other 135 registered worktrees. TASK-13518 is finalized Done through official Backlog editing after verified PR3016 merge, seven required gates, and 102 resolved threads. Scoped format/diff checks pass. Independent review, publication, requester-owned Change summary for the new PR, and new-head CI remain pending; TASK-13515 stays In Progress.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
