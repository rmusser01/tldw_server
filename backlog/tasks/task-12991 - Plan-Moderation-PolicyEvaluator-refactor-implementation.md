---
id: TASK-12991
title: Plan Moderation PolicyEvaluator refactor implementation
status: Done
created_date: 2026-08-08 17:23
dependencies:
- TASK-12990
labels:
- moderation
- planning
- refactor
priority: medium
documentation:
- Docs/superpowers/specs/2026-07-20-moderation-policy-evaluator-refactor-design.md
- Docs/superpowers/plans/2026-07-23-moderation-policy-evaluator-refactor-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-07-20-moderation-policy-evaluator-refactor-design.md
- Docs/superpowers/plans/2026-07-23-moderation-policy-evaluator-refactor-implementation-plan.md
- backlog/tasks/task-12991 - Plan-Moderation-PolicyEvaluator-refactor-implementation.md
- backlog/tasks/task-12992 - Implement-Moderation-PolicyEvaluator-refactor.md
updated_date: 2026-08-08 17:33
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write an actionable implementation plan for the approved Moderation PolicyEvaluator structural extraction. Preserve public ModerationService dispatch and exact evaluation/redaction quirks, use TDD, name concrete files and commands, include compatibility and real-service caller regressions, and remain planning-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The implementation plan references the approved TASK-12990 design.
- [x] #2 The strict structural extraction is decomposed into small TDD tasks with concrete files, commands, expected outcomes, and commit checkpoints.
- [x] #3 The plan covers characterization, immutable limits, evaluator extraction, exact scan and redaction semantics, service compatibility, caller regressions, compilation, Bandit, and mergeability.
- [x] #4 The plan is self-reviewed for coverage, placeholders, type consistency, executable ordering, and no-behavior-change scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The plan maps the approved design to seven TDD tasks and explicit final verification. Independent review refinements preserved replacement-lookup exceptions, expanded literal behavior matrices, staged caller fixtures correctly, strengthened descriptor checks, and added review-fix checkpoints. Historical detailed planning notes remain available in PR #2770 history under the superseded colliding planning record.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the executable TDD plan for extracting PolicyEvaluator behind ModerationService while preserving dispatch, signatures, exception boundaries, and literal scan/redaction behavior. Production implementation remained deferred to TASK-12992.
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
