---
id: TASK-12126
title: Implement Chat Macros v1 and wrapup command
status: To Do
assignee: []
created_date: ''
updated_date: '2026-07-03 23:38'
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
