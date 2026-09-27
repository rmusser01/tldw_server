---
id: TASK-12992
title: Implement Moderation PolicyEvaluator refactor
status: Done
created_date: 2026-08-08 17:23
dependencies:
- TASK-12991
labels:
- moderation
- implementation
- refactor
priority: medium
references:
- tldw_Server_API/app/core/Moderation/moderation_service.py
- tldw_Server_API/app/core/Moderation/policy_compiler.py
- https://github.com/rmusser01/tldw_server/pull/2770
documentation:
- Docs/superpowers/specs/2026-07-20-moderation-policy-evaluator-refactor-design.md
- Docs/superpowers/plans/2026-07-23-moderation-policy-evaluator-refactor-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-07-20-moderation-policy-evaluator-refactor-design.md
- Docs/superpowers/plans/2026-07-23-moderation-policy-evaluator-refactor-implementation-plan.md
- tldw_Server_API/app/core/Moderation/policy_evaluator.py
- tldw_Server_API/app/core/Moderation/moderation_service.py
- tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
- tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py
- tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py
- tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py
- tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py
- backlog/tasks/task-12992 - Implement-Moderation-PolicyEvaluator-refactor.md
updated_date: 2026-08-08 17:34
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved PolicyEvaluator implementation plan as a strict structural extraction: add a stateless explicit-input evaluator and immutable limit snapshots behind ModerationService while preserving public dispatch, private callable compatibility, exact evaluation/redaction behavior, caller contracts, and all verification gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Literal pre-extraction characterization tests lock current service behavior and quirks.
- [x] #2 EvaluationLimits and a stateless PolicyEvaluator own evaluation, scanning, snippets, and redaction without service I/O, logging, or mutable runtime state.
- [x] #3 ModerationService delegates logic while preserving all public signatures, tuple ordering, public dynamic dispatch, and callable private compatibility methods.
- [x] #4 Direct evaluator, delegation, moderation, Guardian, Chat, Workflow, STT, endpoint, compilation, Bandit, diff, and current-dev mergeability gates pass.
- [x] #5 No endpoint or schema changes, behavior hardening, shared-model relocation, or unrelated refactor is included.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the seven-task extraction with literal service characterization, direct evaluator coverage, mutation-sensitive delegation checks, and real Chat and Workflow callers. All planned compile, format, lint, focused regression, Bandit, scope, and independent-review gates passed before PR preparation. Historical per-task commit and review evidence remains available in PR #2770 history under the superseded colliding implementation record. PR review remediation and current-dev revalidation are tracked separately in TASK-12989.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented a stateless explicit-input PolicyEvaluator and frozen lossless EvaluationLimits for decision, scan, snippet, count, and redaction mechanics. ModerationService remains the stateful compatibility facade; production changes are limited to the service and evaluator, with no intentional behavior, endpoint, or schema changes.
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
