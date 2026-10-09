---
id: TASK-13421
title: Design Moderation policy_types retirement
status: Done
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
updated_date: 2026-10-03 21:40
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
- [x] #5 The approved design is committed in Docs/superpowers/specs.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Audit completed on current origin/dev 86e287fee7. policy_types() exists only in PolicyCompiler/PolicyEvaluator internal production calls; no endpoint, service caller, public documentation, or integration caller uses it. Hook-specific tests preserve descriptors, tuple/cache identity, clean-process service isolation, and subclass model substitution. The approved design retires direct calls and subclass substitution, preserves canonical model/service imports, replaces hook-based import tests with representative operations, and avoids private-layout assertions. Focused baseline: 119 passed, 250 warnings across compiler, evaluator, model characterization, canonical-model, and import suites.
Design specification self-review completed: no TBD/TODO placeholders, unresolved ambiguity, or cross-scope implementation dependency remains. The review added an explicit TDD red/green proof for ignoring legacy subclass hooks, exact implementation file scope, representative clean-process import-isolation operations, and a ban on private-layout tests. git diff --cached --check passed before commit. Design commit: 18659bd296. Bandit is not applicable because this task changes documentation and its Backlog record only.
Written-spec review identified five corrections before implementation planning: reconcile the source-audit gate with the intentional legacy-hook regression fixtures; narrow behavior-preservation language to supported ModerationService/runtime callers; declare private runtime-alias rebinding outside compatibility because removing the evaluator cache changes that monkeypatch behavior; describe rollback as one-PR rather than one-commit; and keep the task open until the amended written spec is approved.
Applied all five written-spec review corrections. The verification gate now allows only the intentional legacy-hook regression fixtures; behavior-preservation claims are limited to supported service/runtime paths; private runtime-alias rebinding is explicitly unsupported and covered as a risk; rollback is one PR; TASK-13421 remains In Progress pending approval of the amended written specification.
After the review-correction commit, the branch was rebased cleanly onto current origin/dev 4c4f197f68; no intervening commit touched the Moderation production/test scope. Rebased commits are 1d5c9aeb58 (design), 67a1120a5f (initial task completion record), and 3d2233a575 (review corrections). The focused current-dev baseline was rerun after rebase: 119 passed, 250 warnings in 1.86s. The task remains In Progress pending approval of the amended written spec.
The user approved the amended written specification on 2026-10-03. TASK-13421 is complete; implementation planning proceeds as a separate tracked work item.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved and committed a narrow, current-dev design to retire PolicyCompiler.policy_types() and PolicyEvaluator.policy_types(). The implementation will use existing private canonical model aliases, preserve supported ModerationService/runtime behavior and model import contracts, replace hook-based import tests with representative operations, and explicitly remove direct calls and subclass model substitution. Private runtime-alias rebinding is outside compatibility. No runtime code changed in this design task.
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
