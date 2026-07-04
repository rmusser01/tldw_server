---
id: TASK-12126
title: Implement Chat Macros v1 and wrapup command
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-04 00:03'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-07-03-chat-macros-design.md
  - Docs/superpowers/plans/2026-07-03-chat-macros-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Chat Macros v1 system and built-in /wrapup command according to the design spec and implementation plan. Scope includes backend macro definitions/storage/run records, Jobs execution, chat slash routing, minimal frontend settings/status UI, tests, docs, Bandit, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Backend Chat_Macros module supports built-in /wrapup, user macro storage, settings/output profiles, run records, branch records, and validation.
- [ ] #2 Macro invocation works from chat/workspace surfaces with chat-native branch execution, background Jobs mode, cancellation, final result persistence, and idempotent post-back.
- [ ] #3 WebUI exposes minimal macro settings/manager controls and renders macro status/final output/run detail states.
- [ ] #4 Focused backend/frontend tests pass and Bandit is run on touched backend scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation should follow Docs/superpowers/plans/2026-07-03-chat-macros-implementation-plan.md. The plan was produced from the approved design spec and reviewed in three subagent passes; blocking review comments were folded into the plan. Start implementation with the plan Task 1 and keep commits task-sized.

Implementation started in worktree .worktrees/chat-macros-v1 on branch codex/chat-macros-v1.

Task 1 complete: added Chat_Macros parser/model foundation, built-in /wrapup MACRO.yaml, README, and parser tests. Commits: 331e6276fb (initial parser), 8dc047de8b (skill permission test), c1f4e6eb95 (parser hardening). Verification: clean baseline command-router suite passed before implementation (26 passed); focused parser tests passed at Task 1 HEAD (11 passed, 3 warnings); Bandit on tldw_Server_API/app/core/Chat_Macros reported no findings. Reviews: spec compliance passed after adding skills permission test; code-quality review passed after hardening defaults, alias collisions, duplicate non-repeated args, and prompt step support. Minor non-blocking note: parse_macro_args still trusts directly supplied hand-built arg_specs for alias uniqueness; intended call path uses validated MacroDefinition args.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Implementation plan followed or deviations documented
- [ ] #3 Focused backend tests passing
- [ ] #4 Focused frontend tests passing
- [ ] #5 Bandit run for touched backend scope and new findings fixed
- [ ] #6 Documentation updated
- [ ] #7 Final summary added
- [ ] #8 Known skips or blockers documented
<!-- DOD:END -->
