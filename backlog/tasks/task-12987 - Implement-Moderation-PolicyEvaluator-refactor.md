---
id: TASK-12987
title: Implement Moderation PolicyEvaluator refactor
status: In Progress
created_date: 2026-07-26 03:49
dependencies:
- TASK-12985
labels:
- moderation
- implementation
- refactor
priority: medium
references:
- tldw_Server_API/app/core/Moderation/moderation_service.py
- tldw_Server_API/app/core/Moderation/policy_compiler.py
documentation:
- Docs/superpowers/specs/2026-07-20-moderation-policy-evaluator-refactor-design.md
- Docs/superpowers/plans/2026-07-23-moderation-policy-evaluator-refactor-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-07-20-moderation-policy-evaluator-refactor-design.md
- Docs/superpowers/plans/2026-07-23-moderation-policy-evaluator-refactor-implementation-plan.md
- tldw_Server_API/app/core/Moderation/policy_evaluator.py
- tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
- tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py
- backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md
updated_date: 2026-07-26 04:58
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved PolicyEvaluator implementation plan as a strict structural extraction: add a stateless explicit-input evaluator and immutable limit snapshots behind ModerationService while preserving public dispatch, private callable compatibility, exact evaluation/redaction behavior, caller contracts, and all verification gates. TASK-12987 replaces the stale-worktree implementation tracker that collided with canonical TASK-12986 after current dev was fetched.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Literal pre-extraction characterization tests lock current service behavior and quirks.
- [ ] #2 EvaluationLimits and a stateless PolicyEvaluator own evaluation, scanning, snippets, and redaction logic without service I/O, logging, or mutable runtime state.
- [ ] #3 ModerationService delegates logic while preserving all public signatures, tuple ordering, public dynamic dispatch, and callable private compatibility methods.
- [ ] #4 Direct evaluator, delegation-invariant, moderation, Guardian, Chat, Workflow, STT, endpoint, compilation, Bandit, diff, and current-dev mergeability gates pass.
- [ ] #5 No endpoint/schema changes, behavior hardening, shared-model relocation, or unrelated refactor is included.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation begins from a fresh branch based on current origin/dev because the stale isolated branch became 18 commits behind and its proposed TASK-12986 ID collided with the merged trusted-license CI task. The approved design and plan are being carried forward without the obsolete duplicate task record.
Execution resumed on 2026-07-25 in `.worktrees/moderation-policy-evaluator-design` on fresh branch `codex/moderation-policy-evaluator-refactor`, based on current `origin/dev`. The stale branch is preserved separately. TASK-12987 replaces the collided local TASK-12986 ID, which current dev uses for unrelated trusted-license CI work. Pre-Task-1 baseline: 44/44 focused existing moderation tests passed. Task 1 decision/dispatch characterization is in progress under the subagent-driven workflow.
Task 1 implementation commit `6ed86653fc` added the literal decision/dispatch characterization oracle without production changes. The first and final focused runs both passed 30/30; the pre-task baseline remained 44/44. Black, Ruff, and diff checks passed. Default Bandit reported only expected low-severity B101 findings on pytest assertions; the test-only scan excluding B101 exited 0 with no findings. Task-scoped independent review is pending.
Task 1 independent task review is APPROVED with no Critical, Important, or Minor findings. The initial reviewer concern about design/plan entries in `modified_files` was withdrawn after verification that Backlog tracks cumulative full-work-item provenance and those files were added in the preceding TASK-12987 setup commit. A `-W error` diagnostic confirmed the warning noise originates from existing repository/environment configuration and cleanup (`PytestConfigWarning` for the pre-existing `plugins` option, Loguru closed-stream noise, and sqlite ResourceWarning), not the new characterization test.
Task 2 implementation commit `7c5e78cab5` completed the literal scan/redaction/limit oracle. The approved Task 1 baseline passed 30/30; the first and final expanded runs passed 67/67. The 37 added cases cover sequential redaction/counts, short/long replacement-limit asymmetry, chunk/search bounds, all raw limit fields, bounded full-text redaction, zero-length behavior, malformed raw-rule propagation, exception boundaries, and borrowed policy/rule/category identity. Black, Ruff, test-only Bandit with B101 excluded, and diff checks passed. Existing repository/environment warning noise remains unchanged. Acceptance criterion #1 is complete; independent Task 2 review is pending.
Task 2 review identified one Important oracle weakness: the original identity-only immutability probe could miss in-place mutations to enabled categories, list contents/order, mutable rule fields, and policy scalars. Review-fix commit `a9af5bb8de` adds copied value snapshots plus collection/rule/category identity checks across two rules. The strengthened suite passed 67/67 before and after Black; Ruff, test-only Bandit, and diff checks passed. Fresh Task 2 re-review is pending.
Fresh Task 2 re-review of the complete amended range is APPROVED with no Critical, Important, or Minor findings. The reviewer verified copied enabled-category values and identity, strict ordered rule identity, every mutable rule field/category snapshot, policy scalar snapshots, and intact literal scan/redaction/limit coverage. Task 2 is complete; Task 3 direct decision evaluator implementation is starting.
Task 3 implementation commit `ff7346f570` added frozen `EvaluationLimits`, stateless direct decision evaluation, deferred service-type loading, phase/category eligibility, chunk/original-string scanning, action ranking, match selection, snippets, and result construction. RED was a genuine `ModuleNotFoundError` for the absent evaluator module. GREEN: 34/34 direct tests and 101/101 combined direct plus characterization tests passed. Compileall, Black, Ruff, production-scope Bandit (zero findings), and diff checks passed. Redaction remains the planned Task 4 placeholder; ModerationService is unchanged. Independent Task 3 review is pending.
Task 3 review judged production behavior compliant but found three Important direct-test boundary gaps. Test-only fix commit `ff546f5ad6` now locks the static descriptor/exact deferred type tuple/empty instance state; strengthens two-rule policy, collection, rule-field, category, scalar, and all-limit immutability snapshots; proves constructor-boundary identity preservation for four raw `EvaluationLimits` sentinels; and temporarily anchors the planned missing-redaction boundary until Task 4 supersedes it. Focused direct tests passed 37/37 and combined direct plus characterization tests passed 104/104. Black, Ruff, test-only Bandit, and diff checks passed; the prior production Bandit remains zero findings. Fresh Task 3 re-review is pending.
Fresh Task 3 re-review is APPROVED with no Critical, Important, or Minor findings. The reviewer verified the static deferred loader contract, exact service type tuple, empty evaluator state, temporary Task 3 missing-redaction boundary, identity-preserving raw limits, strengthened borrowed-input snapshots, unchanged production diff, and intact decision semantics. Task 3 is complete; Task 4 will replace the temporary boundary test with real evaluator redaction and nested-limit coverage.
Task 4 implementation commit `f98963074b` added stateless evaluator redaction, count-returning redaction, full-text long-path match collection, sequential/action-agnostic literal substitution, current replacement-limit asymmetry, and nested reuse of the identical supplied limits object. RED: 21 expected missing-method failures with 38 passing. GREEN: 59/59 direct tests and 126/126 combined direct plus 67-case oracle passed. Compileall, Black, Ruff, production Bandit (zero findings), and diff checks passed. The temporary Task 3 missing-redaction assertion was replaced by real nested-redaction coverage. Independent Task 4 review is pending.
Task 4 review found one Important path-coverage asymmetry and one Minor no-call gap. Test-only fix commit `b53bfb042b` now independently anchors exact text through both separately implemented redaction APIs for sequential/disabled, full-text long, short/long limit, and zero-length cases; it also proves requested redacted output never invokes redaction for warn/block decisions. Production is unchanged. Focused direct tests passed 61/61 and combined direct plus characterization tests passed 128/128. Black, Ruff, test-only Bandit, and diff checks passed. Fresh Task 4 re-review is pending.
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
