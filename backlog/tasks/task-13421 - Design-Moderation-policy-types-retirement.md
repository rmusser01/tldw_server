---
id: TASK-13421
title: Design Moderation policy_types retirement
status: In Progress
created_date: 2026-10-03 01:11
labels:
- Moderation
- design
- refactor
- compatibility
priority: medium
references:
- TASK-13112
- Docs/superpowers/specs/2026-08-23-moderation-compatibility-seams-cleanup-design.md
- tldw_Server_API/app/core/Moderation/policy_compiler.py
- tldw_Server_API/app/core/Moderation/policy_evaluator.py
documentation:
- Docs/superpowers/specs/2026-10-02-moderation-policy-types-retirement-design.md
modified_files:
- Docs/superpowers/specs/2026-10-02-moderation-policy-types-retirement-design.md
- backlog/tasks/task-13421 - Design-Moderation-policy-types-retirement.md
updated_date: 2026-10-03 01:16
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the next structural Moderation refactor slice: retire PolicyCompiler.policy_types() and PolicyEvaluator.policy_types(), bind compiler/evaluator behavior directly to canonical models, preserve service imports and moderation semantics, explicitly accept the undocumented subclass model-substitution break, and retain import-isolation coverage without private-layout tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current policy_types() production, test, documentation, and history usage is documented.
- [x] #2 The design defines the exact compatibility break and preserves all public ModerationService and model import contracts.
- [x] #3 Compiler/evaluator canonical-type wiring, runtime namespace behavior, and import-isolation replacements are specified.
- [x] #4 Compilation-first, focused, downstream, lint, security, and mergeability verification gates are specified.
- [ ] #5 The approved design is committed in Docs/superpowers/specs.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Audit completed on current origin/dev 86e287fee7. policy_types() exists only in PolicyCompiler/PolicyEvaluator internal production calls; no endpoint, service caller, public documentation, or integration caller uses it. Hook-specific tests preserve descriptors, tuple/cache identity, clean-process service isolation, and subclass model substitution. The approved design retires direct calls and subclass substitution, preserves canonical model/service imports, replaces hook-based import tests with representative operations, and avoids private-layout assertions. Focused baseline: 119 passed, 250 warnings across compiler, evaluator, model characterization, canonical-model, and import suites.
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
