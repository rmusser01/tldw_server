---
id: TASK-13010
title: Design Moderation shared models extraction
status: Done
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
updated_date: 2026-08-12 00:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Transplant and preserve the approved design for moving ModerationPolicy, PatternRule, and ModerationEvaluationResult into a neutral canonical models module while retaining exact moderation_service.py imports and behavior. This record replaces stale stacked-branch TASK-12988, which now collides on current dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design defines canonical model ownership and exact moderation_service.py re-export compatibility.
- [x] #2 The design freezes behavior, metadata, import, serialization, dispatch, and non-goal boundaries for a structural-only PR.
- [x] #3 The design is reconciled to current dev and linked to the implementation task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Transplanted the previously approved and independently corrected design onto current dev. Reconciled the predecessor to merged TASK-12992 / PR #2770, documented the fresh-branch transplant strategy, and linked implementation TASK-13011. Placeholder scan and git diff --check pass. Bandit is not applicable to this design/tracking record because production-code security verification belongs to TASK-13011.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Defined models.py as the standard-library-only canonical owner of ModerationPolicy, PatternRule, and ModerationEvaluationResult; preserved exact moderation_service.py re-exports and runtime behavior; froze metadata, serialization, policy_types dispatch, namespace, testing, scope, and rollback boundaries; and reconciled rollout to current dev under TASK-13011.
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
