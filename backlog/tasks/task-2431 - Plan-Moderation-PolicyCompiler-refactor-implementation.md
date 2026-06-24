---
id: TASK-2431
title: Plan Moderation PolicyCompiler refactor implementation
status: Done
created_date: 2026-06-24 20:22
labels:
- moderation
- planning
- refactor
priority: medium
documentation:
- Docs/superpowers/specs/2026-06-24-moderation-policy-compiler-refactor-design.md
- Docs/superpowers/plans/2026-06-24-moderation-policy-compiler-refactor-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-24-moderation-policy-compiler-refactor-implementation-plan.md
- backlog/tasks/task-2431 - Plan-Moderation-PolicyCompiler-refactor-implementation.md
- backlog/tasks/task-2432 - Implement-Moderation-PolicyCompiler-refactor.md
updated_date: 2026-06-24 20:42
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write an implementation plan for the approved compiler-first Moderation refactor design. The plan must be actionable task-by-task, preserve ModerationService compatibility, use TDD, cover verification gates, and remain planning-only with no code implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is written under Docs/superpowers/plans and references the approved design spec.
- [x] #2 Plan decomposes the compiler-first refactor into small TDD tasks with exact files, commands, expected outcomes, and commit checkpoints.
- [x] #3 Plan covers ResolvedModerationConfig, PolicyCompiler/report types, parser/lint compatibility, PII rule boundary, service integration, compatibility wrappers, and regression tests.
- [x] #4 Plan is self-reviewed for spec coverage, placeholders, type consistency, and scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wrote the implementation plan for the approved Moderation PolicyCompiler refactor. Self-review confirmed the plan covers the design spec, uses exact file paths and commands, keeps PolicyEvaluator out of scope, keeps file I/O out of PolicyCompiler, preserves compatibility boundaries, and includes TDD slices plus commit checkpoints. Verification for this plan-only branch: targeted draft-marker scan found no matches; git diff --check passed for the plan/task files. Bandit is not applicable because no code is touched in this planning step.
Post-plan review tightened the implementation plan for subagent execution: corrected implementation tracking to TASK-2432, added missing PolicyCompilationInput service import, made service config resolution explicit, preserved legacy bool-like quick-rule parsing, and marked the plan self-review checklist complete after the review pass.
Follow-up plan review also made Task 6 regression tests deterministic by using temporary moderation config paths and persist=False for update_settings coverage.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation plan created for the compiler-first Moderation refactor. The plan decomposes the work into TDD tasks for compiler dataclasses, blocklist parsing, service wrapper/lint compatibility, global service integration, per-user policy assembly, behavior regression coverage, and final verification. Created TASK-2432 as the execution task for the later implementation phase.
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
