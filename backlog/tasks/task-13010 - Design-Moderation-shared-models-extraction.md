---
id: TASK-13010
title: Design Moderation shared models extraction
status: In Progress
created_date: 2026-08-12 00:44
labels:
- moderation
- refactor
- design
priority: medium
references:
- https://github.com/rmusser01/tldw_server/pull/2770
- codex/moderation-shared-models-design@5d33b21ca4
documentation:
- Docs/superpowers/specs/2026-08-01-moderation-shared-models-extraction-design.md
modified_files:
- Docs/superpowers/specs/2026-08-01-moderation-shared-models-extraction-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Transplant and preserve the approved design for moving ModerationPolicy, PatternRule, and ModerationEvaluationResult into a neutral canonical models module while retaining exact moderation_service.py imports and behavior. This record replaces stale stacked-branch TASK-12988, which now collides on current dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The design defines canonical model ownership and exact moderation_service.py re-export compatibility.
- [ ] #2 The design freezes behavior, metadata, import, serialization, dispatch, and non-goal boundaries for a structural-only PR.
- [ ] #3 The design is reconciled to current dev and linked to the implementation task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved stacked-branch design will be transplanted onto current dev under this collision-free task identity before implementation PR preparation.
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
