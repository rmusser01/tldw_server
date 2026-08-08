---
id: TASK-12990
title: Design Moderation PolicyEvaluator refactor
status: Done
created_date: 2026-08-08 17:22
labels:
- moderation
- design
- refactor
priority: medium
references:
- https://github.com/rmusser01/tldw_server/pull/2528
- tldw_Server_API/app/core/Moderation/moderation_service.py
- tldw_Server_API/app/core/Moderation/policy_compiler.py
documentation:
- Docs/superpowers/specs/2026-07-20-moderation-policy-evaluator-refactor-design.md
modified_files:
- Docs/superpowers/specs/2026-07-20-moderation-policy-evaluator-refactor-design.md
- backlog/tasks/task-12990 - Design-Moderation-PolicyEvaluator-refactor.md
updated_date: 2026-08-08 17:33
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the next compatibility-preserving Moderation refactor slice after PolicyCompiler: extract mostly pure rule evaluation and redaction behavior into a PolicyEvaluator behind ModerationService while preserving public service methods, dynamic public redaction dispatch, exact scan/redaction semantics, and current exception behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current ModerationService responsibilities and extraction boundary are documented.
- [x] #2 The design preserves public dispatch, exact evaluation and redaction behavior, exception boundaries, and configuration ownership.
- [x] #3 PolicyEvaluator statelessness and immutable per-call EvaluationLimits are specified.
- [x] #4 Characterization, direct evaluator, delegation, caller, compilation, lint, Bandit, and mergeability verification are defined.
- [x] #5 The approved design is recorded in Docs/superpowers/specs.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The approved design defines a strict structural extraction, with ModerationService retaining configuration, locking, I/O, persistence, and compatibility dispatch while PolicyEvaluator receives explicit policies and frozen limit snapshots. Independent review refinements added wrapper dispatch coverage, unsupported-limit cases, real lock/writer behavior, and non-string action semantics. Historical detailed execution notes remain available in PR #2770 history under the superseded colliding design record.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved a behavior-preserving PolicyEvaluator design that separates stateless decision, scan, snippet, count, and redaction mechanics from the stateful ModerationService facade. The design keeps behavior hardening and shared-model relocation out of scope.
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
