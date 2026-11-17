---
id: TASK-15406
title: Design Moderation compiler-side delegate cleanup
status: Done
assignee: []
created_date: 2026-11-01 16:46
updated_date: 2026-11-01 17:33
labels:
- moderation
- refactor
- design
dependencies: []
references:
- 93d132a766:Docs/superpowers/plans/2026-11-01-moderation-compiler-delegate-cleanup-implementation-plan.md
documentation:
- Docs/superpowers/specs/2026-11-01-moderation-compiler-delegate-cleanup-design.md
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only audit against origin/dev 54b321dcec8cdbb6358581f42c1548696260570b. No production callers found for _split_action_directive, _resolve_categories_override, _parse_categories_override, _compile_user_rule, or _coalesce_bool. Three tests call _compile_user_rule; one asserts sanitized service logging and must move to a production service path, not lose that coverage.

Human approved the exact five-method design on 2026-11-01. Compilation and 67 focused tests passed before implementation (146 existing warnings on Python 3.14.3). Implementation planning now underway; preserving the earlier subagent-driven execution choice. Current changes are documentation only, so Bandit is not applicable at this checkpoint. MCP task_edit could not find the newly created task; official CLI fallback successfully resolves it.

Implementation plan written and self-reviewed at Docs/superpowers/plans/2026-11-01-moderation-compiler-delegate-cleanup-implementation-plan.md. Three stages: migrate test coverage against unchanged production, remove the exact five delegates with a five-case red/green structural test, then run the full downstream matrix and independent review. No production or test code changed. Design approved; awaiting implementation-plan review before starting the preserved subagent-driven workflow. git diff --check passed; Bandit is not applicable to this documentation-only change.

User authorized correcting both review findings and continuing implementation. Plan corrected to use private-rule-id-92741 instead of a sentinel contained in dangerous_regex, explicitly mark the new module unit, and select its five cases with -m unit. Task headings aligned to SDD brief extraction; no scope change. Documentation-only checks: git diff --check; source/test files unchanged at this checkpoint, so Bandit not applicable.

Implementation completed in TASK-15403. All three execution stages are complete. Per repository instructions, only this task-owned implementation plan is removed from the working tree; its completed version is retained in git at 93d132a766:Docs/superpowers/plans/2026-11-01-moderation-compiler-delegate-cleanup-implementation-plan.md. Design spec remains tracked.
PR preparation: renumbered this branch-owned design task because TASK-15205 already exists on dev for unrelated completed work. Highest observed id across dev/open PRs was 15405; TASK-15406 allocated to this design. Canonicalized with backlog-py per ADR-059; historical implementation-plan references remain immutable Git history.
Current implementation task is TASK-15407 (formerly TASK-15403); design was formerly TASK-15205. ADR check: ADR required: no, because this cleanup removes only unused private wrappers and does not establish a new durable architecture rule. Task metadata follows ADR-059.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved exact five-method cleanup design and implementation plan completed; both pre-implementation review findings corrected. Execution handed to a linked implementation task with subagent-driven per-task review and final whole-branch review.
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
