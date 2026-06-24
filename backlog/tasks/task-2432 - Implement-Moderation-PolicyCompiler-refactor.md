---
id: TASK-2432
title: Implement Moderation PolicyCompiler refactor
status: To Do
created_date: 2026-06-24 20:27
dependencies:
- TASK-2430
- TASK-2431
labels:
- moderation
- implementation
- refactor
priority: medium
documentation:
- Docs/superpowers/specs/2026-06-24-moderation-policy-compiler-refactor-design.md
- Docs/superpowers/plans/2026-06-24-moderation-policy-compiler-refactor-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-24-moderation-policy-compiler-refactor-implementation-plan.md
- backlog/tasks/task-2432 - Implement-Moderation-PolicyCompiler-refactor.md
updated_date: 2026-06-24 20:42
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved compiler-first Moderation refactor using the plan in Docs/superpowers/plans/2026-06-24-moderation-policy-compiler-refactor-implementation-plan.md. Preserve ModerationService and ModerationPolicy compatibility, keep I/O/persistence/logging in the service, and move deterministic policy assembly into PolicyCompiler.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PolicyCompiler and related dataclasses are implemented with deterministic policy assembly and sanitized reports.
- [ ] #2 ModerationService integrates the compiler while preserving public methods, file I/O boundaries, lint output behavior, helper wrapper compatibility, and existing policy types.
- [ ] #3 Compiler, service compatibility, blocklist/lint, PII, per-user override, and supervised overlay regression tests are added or updated.
- [ ] #4 Focused pytest, py_compile, git diff --check, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation plan reviewed for subagent execution readiness. The plan now directs implementation progress and verification updates to this task, includes explicit service config-resolution code, imports PolicyCompilationInput where service integration uses it, and preserves existing bool-like is_regex behavior for loaded user quick rules.
Plan review follow-up made Task 6 regression tests deterministic by monkeypatching service config to temporary blocklist/user/runtime override paths and keeping update_settings persistence disabled.
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
