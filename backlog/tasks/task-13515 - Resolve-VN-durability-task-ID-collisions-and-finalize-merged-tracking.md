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
- [x] #4 Owned documentation checks and independent review pass; publish the scoped follow-up without runtime changes or applying PR3060 summary waiver.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-06: requester approved the previously requested scoped manual task-ID migration exception. Existing official MCP/CLI task-edit interfaces do not support changing an existing ID. Inventory of 136 registered worktrees and 347175 task paths found maximum root ID 13514; reserved tracking TASK-13515 and migration IDs TASK-13516/TASK-13517/TASK-13518. PR3060 remains on its separate published branch at 0f8bd6c922bbf1f6066390969c596c48c8c3b780 with current-head hosted reviews/CI pending. Implementation plan: Docs/Plans/IMPLEMENTATION_PLAN_vn_task_id_finalization.md.
2026-10-06 local migration verified: TASK-13356/TASK-13358/TASK-13369 VN records are now TASK-13516/TASK-13517/TASK-13518. Both VN dependencies and four current design/plan associations use the new IDs. Structured verification proves every original historical section/note is preserved and all four unrelated colliding records remain byte-identical to dev; the new IDs are unused in the other 135 registered worktrees. TASK-13518 is finalized Done through official Backlog editing after verified PR3016 merge, seven required gates, and 102 resolved threads. Scoped format/diff checks pass. Independent review, publication, requester-owned Change summary for the new PR, and new-head CI remain pending; TASK-13515 stays In Progress.
2026-10-06 publication: the verified migration was normally pushed first as f0ccbc66d2fcc8a3e2795704c130811180a144b3 and published as https://github.com/rmusser01/tldw_server/pull/3207 against dev 1047ce10ecc191f78b0bee7e1b8ad19780c68015. Independent read-only reviewer Nash found no concrete issue in the full immutable migration diff and separately verified historical preservation and the unchanged unrelated records. Scoped structured checks, canonical task formatting, applicable pre-commit hooks and diff checks pass; code-only hooks were skipped, not passes. The new PR has no runtime changes, no automatic merge, and no summary waiver. Requester-written Change summary and final-head hosted reviews/CI remain required. Task and plan Stage3 remain In Progress until verified normal merge. PR3060 stays separate: exact 0f8 CodeRabbit full review finished with both source/covered commit IDs and no actionable finding; six required gates pass, container gate is pending, Qodo manual request is unacknowledged and its current-head qualification is incomplete. One bounded properly formatted retry has been requested from the requester; do not repeat without that decision.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The scoped VN identity migration is implemented and published in PR3207. TASK-13356/TASK-13358/TASK-13369 VN records now have unique identities TASK-13516/TASK-13517/TASK-13518, and current dependencies/associations are corrected. Every original historical section is preserved; all four unrelated colliding records remain byte-identical to dev. The merged PR3016 review record is finalized Done through official Backlog editing after its verified normal merge, seven required gates, and 102 resolved threads.

Local structured checks, applicable pre-commit checks, and independent migration review passed. No product code, configuration, dependencies, environment, or CI changes are included, and no fresh product-test/Bandit claim is made. This cleanup task remains In Progress until PR3207 has its own requester-written Change summary, completed final-head hosted reviews, successful live required gates, and a verified normal merge. PR3060 is separate and its summary waiver is not reused.
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
