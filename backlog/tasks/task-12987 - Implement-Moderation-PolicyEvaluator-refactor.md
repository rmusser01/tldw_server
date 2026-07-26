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
- tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py
- backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md
updated_date: 2026-07-26 03:59
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved PolicyEvaluator implementation plan as a strict structural extraction: add a stateless explicit-input evaluator and immutable limit snapshots behind ModerationService while preserving public dispatch, private callable compatibility, exact evaluation/redaction behavior, caller contracts, and all verification gates. TASK-12987 replaces the stale-worktree implementation tracker that collided with canonical TASK-12986 after current dev was fetched.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Literal pre-extraction characterization tests lock current service behavior and quirks.
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
