---
id: TASK-15403
title: Implement Moderation compiler-side delegate cleanup
status: In Progress
assignee: []
created_date: '2026-11-01 17:13'
updated_date: '2026-11-01 17:19'
labels:
  - moderation
  - refactor
dependencies:
  - TASK-15205
documentation:
  - >-
    Docs/superpowers/specs/2026-11-01-moderation-compiler-delegate-cleanup-design.md
  - >-
    Docs/superpowers/plans/2026-11-01-moderation-compiler-delegate-cleanup-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved five-method cleanup design and reviewed implementation plan on codex/moderation-compiler-delegate-cleanup. Preserve supported behavior, compiler implementations, active dispatch, logging sanitization, and public imports. Remove only the five repository-unused private ModerationService delegates; no regex hardening or PR publication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Direct compiler tests and the public effective-policy logging regression preserve all migrated assertions.
- [ ] #2 Exactly the five approved private delegates are removed; compiler and active service paths stay unchanged.
- [ ] #3 Five structural cases demonstrate red then green and are selected by pytest -m unit.
- [ ] #4 Compilation, complete Moderation/downstream regression matrix, lint, formatting and scoped Bandit checks pass with no new findings.
- [ ] #5 Independent per-task and whole-branch reviews are resolved and evidence recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Pre-edit baseline compiled service/compiler and completed 400 combined Moderation/Guardian/Chat/Audio cases plus 12 Workflow adapter cases (47 deselected); overall command exit 0. Existing baseline deprecations and background Audit Loguru closed-stream errors appeared during teardown before source changes. Production and test Bandit baselines each have zero findings/errors, with B101 excluded only for test assertions. Starting Task 1 coverage migration at 1d719a5023; controller owns tracked plan/task records.

Task 1 complete at d381f01af1: only three test files changed. Three replacements passed against unchanged production, then all 67 focused cases passed with 146 existing warnings; compilation, changed-function Black checks, Ruff and Bandit passed. Fresh task reviewer approved spec compliance and code quality with no actionable issues. Controller confirmed its prerequisites (approved spec, caller audit, task tracking, 412-test baseline, zero-finding Bandit baseline) that were outside the task-only diff.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
