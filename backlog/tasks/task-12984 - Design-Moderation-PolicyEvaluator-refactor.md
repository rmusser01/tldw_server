---
id: TASK-12984
title: Design Moderation PolicyEvaluator refactor
status: Done
created_date: 2026-07-24 00:12
dependencies:
- TASK-2432
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
- Docs/superpowers/specs/2026-06-24-moderation-policy-compiler-refactor-design.md
- Docs/superpowers/specs/2026-07-20-moderation-policy-evaluator-refactor-design.md
modified_files:
- Docs/superpowers/specs/2026-07-20-moderation-policy-evaluator-refactor-design.md
- backlog/tasks/task-12984 - Design-Moderation-PolicyEvaluator-refactor.md
- backlog/tasks/task-12974 - Design-Moderation-PolicyEvaluator-refactor.md
updated_date: 2026-07-24 00:32
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the next compatibility-preserving Moderation refactor slice after PolicyCompiler: extract mostly pure rule evaluation and redaction behavior into a PolicyEvaluator behind ModerationService. Preserve public service methods, dynamic public redaction dispatch, exact scan/redaction semantics, and current exception behavior while making evaluation limits and decision behavior explicit and testable. This task is design-only; implementation planning and code changes follow after review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current evaluation, redaction, compatibility wrapper, caller, and test boundaries are documented.
- [x] #2 The design compares viable extraction approaches and selects a pragmatic compatibility-preserving boundary.
- [x] #3 PolicyEvaluator inputs, outputs, configuration limits, ownership, error handling, exact scan mechanics, and service delegation are specified.
- [x] #4 The design defines non-tautological behavior-preservation tests and concrete real-service verification gates without implementing production code.
- [x] #5 The written spec is self-reviewed, committed, and presented for user approval before implementation planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Migrated to a fresh task after rebasing onto current dev exposed that TASK-12974 had independently been allocated to the frontend licensing design. The superseded evaluator task record on this branch must be removed after migration. Final review corrections preserve public redact_text dispatch, use locked lossless limit snapshots, pin exact scanning and redaction quirks, and require direct evaluator plus real-service caller coverage.
The branch was rebased onto origin/dev before revision. The design now distinguishes direct evaluator composition from service-facade composition: direct evaluator evaluation can reuse one supplied limits value, while service evaluation obtains a decision-only result and retains dynamic dispatch through the public redact_text method. The old evaluator TASK-12974 file is removed because current dev owns that ID for a different completed task.
Independent final review found four additional precision gaps: public evaluation-wrapper dispatch tests, unsupported values for all four limits, actual lock/writer behavior, and full non-string action semantics. All four were incorporated; focused re-review confirmed them resolved. Verification is documentation-only: git diff --check passed, stale-marker search returned no matches, TASK-12974 and TASK-12984 each resolve to exactly one current task record after migration, and the unrelated frontend licensing task has no diff. Bandit and runtime tests are not applicable because no production or test code changed.
Revised design checkpoint committed as c8e0772b4a (`docs: harden PolicyEvaluator design after review`) and prepared for final user approval before implementation planning.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed and approved the Moderation PolicyEvaluator structural-extraction design. The final design preserves all public ModerationService dispatch points, pins exact evaluation and redaction quirks, uses stateless explicit-input evaluation with locked lossless limit snapshots, retains borrowed policy identity and aliasing, and defines literal characterization, direct evaluator, delegation-invariant, real-service caller, compilation, security, and mergeability gates. Final user approval was received on 2026-07-23; implementation planning is the next separate work item.
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
