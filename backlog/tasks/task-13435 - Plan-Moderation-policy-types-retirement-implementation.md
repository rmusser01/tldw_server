---
id: TASK-13435
title: Plan Moderation policy_types retirement implementation
status: In Progress
created_date: 2026-10-03 21:43
dependencies:
- TASK-13421
labels:
- Moderation
- planning
- refactor
- TDD
priority: medium
references:
- TASK-13421
- Docs/superpowers/specs/2026-10-02-moderation-policy-types-retirement-design.md
documentation:
- Docs/superpowers/specs/2026-10-02-moderation-policy-types-retirement-design.md
- Docs/superpowers/plans/2026-10-03-moderation-policy-types-retirement-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-10-02-moderation-policy-types-retirement-design.md
- Docs/superpowers/plans/2026-10-03-moderation-policy-types-retirement-implementation-plan.md
updated_date: 2026-10-03 21:50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the executable implementation plan for the approved Moderation policy_types() retirement design. The plan must use TDD, preserve import-isolation and runtime namespace contracts, keep production changes to PolicyCompiler and PolicyEvaluator, and run compilation before focused and downstream verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The plan maps every approved design requirement to exact production and test files.
- [x] #2 The plan defines a verified red/green cycle for legacy subclass hooks no longer controlling model selection.
- [x] #3 The plan replaces hook-based clean-process import checks with representative compiler/evaluator operations.
- [x] #4 The plan specifies compilation-first focused, downstream, quality, security, source-audit, and current-dev gates with exact commands.
- [ ] #5 The implementation plan is self-reviewed, committed, and ready for user review.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan self-review found that removing PolicyEvaluator.policy_types() leaves the evaluator's private _ModerationPolicy import unused. Ruff confirms F401 for that import pattern. The approved design was clarified to retain the evaluator aliases used by runtime operations (_PatternRule and _ModerationEvaluationResult) while removing the now-unused evaluator-only _ModerationPolicy import; TYPE_CHECKING annotations remain unchanged.
Self-review completed against the approved design: all production/test requirements map to exact files and steps; the red fixtures now fail through canonical-type assertions rather than incidental exceptions; placeholder scan returned no hits; type and test names are consistent across red/green commands; compilation, focused/downstream, Ruff, Black, Bandit, source-audit, and current-dev gates are explicit. Bandit is not applicable to this planning-only task because it changes documentation and Backlog records, not runtime code.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the executable TDD implementation plan for retiring both Moderation policy_types() hooks and created follow-on implementation task TASK-13436. The plan fixes production and test scope, preserves import and namespace contracts, specifies compilation-first verification through all affected callers, and records the intentional direct-call/subclass compatibility break. A lint-driven clarification removes the evaluator's now-unused _ModerationPolicy runtime import while preserving TYPE_CHECKING annotations and the private aliases used by runtime operations.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
