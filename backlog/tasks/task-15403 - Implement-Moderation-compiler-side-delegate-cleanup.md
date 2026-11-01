---
id: TASK-15403
title: Implement Moderation compiler-side delegate cleanup
status: In Progress
assignee: []
created_date: '2026-11-01 17:13'
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
- [ ] #1 Direct compiler tests and the public effective-policy logging regression preserve all migrated assertions.
- [ ] #2 Exactly the five approved private delegates are removed; compiler and active service paths stay unchanged.
- [ ] #3 Five structural cases demonstrate red then green and are selected by pytest -m unit.
- [ ] #4 Compilation, complete Moderation/downstream regression matrix, lint, formatting and scoped Bandit checks pass with no new findings.
- [ ] #5 Independent per-task and whole-branch reviews are resolved and evidence recorded.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
