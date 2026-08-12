---
id: TASK-13011
title: Implement Moderation shared models extraction
status: In Progress
created_date: 2026-08-12 00:45
dependencies:
- TASK-13010
- TASK-12992
labels:
- moderation
- refactor
- implementation
priority: medium
references:
- https://github.com/rmusser01/tldw_server/pull/2770
- codex/moderation-shared-models-design@5d33b21ca4
- origin/dev@414e81a12aa71df97c4fad17df084aa7a78c474b
documentation:
- Docs/superpowers/specs/2026-08-01-moderation-shared-models-extraction-design.md
- Docs/superpowers/plans/2026-08-01-moderation-shared-models-extraction-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-08-01-moderation-shared-models-extraction-design.md
- Docs/superpowers/plans/2026-08-01-moderation-shared-models-extraction-implementation-plan.md
- backlog/tasks/task-13010 - Design-Moderation-shared-models-extraction.md
- backlog/tasks/task-13011 - Implement-Moderation-shared-models-extraction.md
- tldw_Server_API/app/core/Moderation/models.py
- tldw_Server_API/app/core/Moderation/moderation_service.py
- tldw_Server_API/app/core/Moderation/policy_compiler.py
- tldw_Server_API/app/core/Moderation/policy_evaluator.py
- tldw_Server_API/tests/unit/test_moderation_models_characterization.py
- tldw_Server_API/tests/unit/test_moderation_models_canonical.py
- tldw_Server_API/tests/unit/test_moderation_models_imports.py
updated_date: 2026-08-12 00:49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Transplant the reviewed shared-model extraction onto current dev: make models.py the canonical owner of ModerationPolicy, PatternRule, and ModerationEvaluationResult; preserve exact service facade imports and behavior; remove compiler/evaluator service type edges; rerun current-dev tests, security checks, and review. This record replaces stale stacked-branch TASK-12989, which collides on current dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 models.py canonically owns exactly the three approved dataclasses and remains standard-library-only.
- [ ] #2 moderation_service.py re-exports the exact canonical class objects with unchanged supported constructors, defaults, to_dict mapping, and runtime behavior.
- [ ] #3 PolicyCompiler and PolicyEvaluator no longer load moderation_service.py for canonical runtime types while preserving policy_types descriptors and subclass dispatch.
- [ ] #4 Focused and caller regression tests, compilation, Black/Ruff, Bandit, diff/scope checks, and independent review pass on current dev.
- [ ] #5 The PR contains only the shared-model extraction and collision-free tracking records, with a requester-authored Change summary required before merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-08-01-moderation-shared-models-extraction-implementation-plan.md Task 5: transplant reviewed post-predecessor commits, reconcile IDs, rerun all verification on current dev, obtain independent review, and prepare a PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Transplant: created codex/moderation-shared-models-dev directly from origin/dev@414e81a12aa71df97c4fad17df084aa7a78c474b and applied the nine reviewed post-predecessor commits without conflicts as one pending consolidated change. The original codex/moderation-shared-models-design@5d33b21ca4 remains untouched as a recovery reference. Reconciled predecessor references to merged TASK-12992 / PR #2770 and replaced colliding stale task IDs with TASK-13010 and TASK-13011. Current production scope is exactly models.py, moderation_service.py, policy_compiler.py, and policy_evaluator.py. Fresh current-dev verification and independent review remain pending; PR creation remains gated on the requester's own Change summary.
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
