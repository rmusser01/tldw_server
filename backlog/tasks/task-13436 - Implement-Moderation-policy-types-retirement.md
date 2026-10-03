---
id: TASK-13436
title: Implement Moderation policy_types retirement
status: To Do
created_date: 2026-10-03 21:45
dependencies:
- TASK-13435
labels:
- moderation
- refactor
- behavior-preserving
priority: medium
documentation:
- Docs/superpowers/specs/2026-10-02-moderation-policy-types-retirement-design.md
- Docs/superpowers/plans/2026-10-03-moderation-policy-types-retirement-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/Moderation/policy_compiler.py
- tldw_Server_API/app/core/Moderation/policy_evaluator.py
- tldw_Server_API/tests/unit/test_moderation_models_characterization.py
- tldw_Server_API/tests/unit/test_moderation_models_imports.py
- tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Retire PolicyCompiler.policy_types() and PolicyEvaluator.policy_types() per the approved structural design. Replace internal hook lookups with the existing private canonical model aliases while preserving supported ModerationService behavior, import boundaries, runtime type-hint behavior, and canonical result types. The undocumented subclass model-substitution seam and direct hook calls are intentionally removed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PolicyCompiler and PolicyEvaluator no longer define or call policy_types().
- [ ] #2 Compiler policy/rule construction and evaluator checks/results use the existing private canonical model aliases.
- [ ] #3 Legacy-named subclass policy_types() methods are ignored by representative compile/evaluate operations.
- [ ] #4 Clean-process import tests exercise real compile/evaluate operations without loading moderation_service.
- [ ] #5 Focused and downstream Moderation tests pass after compilation-first verification.
- [ ] #6 Ruff, scoped Black check, Bandit, git diff --check, and source audit pass on the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow the approved implementation plan: establish the two inverse subclass regression tests red, replace hook-dependent import tests with real operations, make the minimal private-alias production edits, then run compilation-first focused, downstream, formatting, lint, security, and source-audit gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
