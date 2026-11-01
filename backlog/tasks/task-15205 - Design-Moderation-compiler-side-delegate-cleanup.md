---
id: TASK-15205
title: Design Moderation compiler-side delegate cleanup
status: In Progress
created_date: 2026-11-01 16:46
labels:
- moderation
- refactor
- design
documentation:
- Docs/superpowers/specs/2026-11-01-moderation-compiler-delegate-cleanup-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the next narrow structural cleanup after merged PR #3176: remove only repository-unused private ModerationService compiler delegates while preserving compiler behavior, active dispatch, logging, and supported APIs. User selected this follow-up on 2026-11-01. Implementation is gated on short design approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Audit exact candidate names and production/test references on current dev.
- [ ] #2 Document five-method removal scope, retained active helpers, test migration, compatibility risk, and verification gates.
- [ ] #3 Obtain human design approval before implementation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only audit against origin/dev 54b321dcec8cdbb6358581f42c1548696260570b. No production callers found for _split_action_directive, _resolve_categories_override, _parse_categories_override, _compile_user_rule, or _coalesce_bool. Three tests call _compile_user_rule; one asserts sanitized service logging and must move to a production service path, not lose that coverage.
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
