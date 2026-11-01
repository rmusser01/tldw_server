---
id: TASK-15205
title: Design Moderation compiler-side delegate cleanup
status: In Progress
assignee: []
created_date: '2026-11-01 16:46'
updated_date: '2026-11-01 16:56'
labels:
  - moderation
  - refactor
  - design
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-11-01-moderation-compiler-delegate-cleanup-design.md
  - >-
    Docs/superpowers/plans/2026-11-01-moderation-compiler-delegate-cleanup-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the next narrow structural cleanup after merged PR #3176: remove only repository-unused private ModerationService compiler delegates while preserving compiler behavior, active dispatch, logging, and supported APIs. User selected this follow-up on 2026-11-01. Implementation is gated on short design approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audit exact candidate names and production/test references on current dev.
- [x] #2 Document five-method removal scope, retained active helpers, test migration, compatibility risk, and verification gates.
- [x] #3 Obtain human design approval before implementation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only audit against origin/dev 54b321dcec8cdbb6358581f42c1548696260570b. No production callers found for _split_action_directive, _resolve_categories_override, _parse_categories_override, _compile_user_rule, or _coalesce_bool. Three tests call _compile_user_rule; one asserts sanitized service logging and must move to a production service path, not lose that coverage.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Human approved the exact five-method design on 2026-11-01. Compilation and 67 focused tests passed before implementation (146 existing warnings on Python 3.14.3). Implementation planning now underway; preserving the earlier subagent-driven execution choice. Current changes are documentation only, so Bandit is not applicable at this checkpoint. MCP task_edit could not find the newly created task; official CLI fallback successfully resolves it.

Implementation plan written and self-reviewed at Docs/superpowers/plans/2026-11-01-moderation-compiler-delegate-cleanup-implementation-plan.md. Three stages: migrate test coverage against unchanged production, remove the exact five delegates with a five-case red/green structural test, then run the full downstream matrix and independent review. No production or test code changed. Design approved; awaiting implementation-plan review before starting the preserved subagent-driven workflow. git diff --check passed; Bandit is not applicable to this documentation-only change.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
