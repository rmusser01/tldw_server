---
id: TASK-530.9
title: Implement Skills seed overwrite confirmation
status: Done
assignee: []
created_date: 2026-06-28 15:38
updated_date: 2026-06-28 15:39
labels:
- skills
- webui
- safe-operations
dependencies: []
documentation:
- Docs/superpowers/specs/2026-06-28-skills-seed-overwrite-confirmation-design.md
- Docs/superpowers/plans/2026-06-28-skills-seed-overwrite-confirmation.md
parent_task_id: TASK-530
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2543
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-530 Safe Operations after TASK-530.8 by adding an explicit frontend confirmation before Seed and Overwrite Existing calls the Skills seed endpoint with overwrite=true. Keep Seed Missing Only one-click, keep backend seed behavior unchanged, and keep version-aware delete, bulk delete, export feedback, and permission metadata panels out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Seed and Overwrite Existing opens a confirmation before calling seedSkills with overwrite=true.
- [x] #2 Cancelling the confirmation does not call the seed mutation.
- [x] #3 Confirming the modal calls seedSkills({ overwrite: true }) exactly once and uses destructive button affordance.
- [x] #4 Seed Missing Only remains one-click and continues to call seedSkills({ overwrite: false }).
- [x] #5 Focused Manager Vitest coverage records the safe overwrite workflow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: Docs/superpowers/specs/2026-06-28-skills-seed-overwrite-confirmation-design.md
Plan: Docs/superpowers/plans/2026-06-28-skills-seed-overwrite-confirmation.md
PR: https://github.com/rmusser01/tldw_server/pull/2543

Implementation completed:
- Added a `Modal.confirm` guard for the Skills manager `Seed and Overwrite Existing` dropdown action.
- Kept `Seed Missing Only` and the empty-state `Seed built-ins` actions as immediate missing-only seed operations.
- Added focused Manager tests for confirmation configuration, no mutation before accept, and accepted overwrite mutation.

Verification:
- `cd apps/packages/ui && bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot` -> PASS, 27 tests passed.
- `git diff --check` -> PASS.
- Bandit skipped; touched scope is TypeScript/React only.

Notes:
- Local `apps/packages/ui/node_modules/antd` symlink target in the tracked worktree points to a missing Bun package hash, so Vitest requires a temporary local symlink to an installed `antd@6.2.1` Bun package hash for execution. The symlink was restored to the tracked target after verification and is not part of this change.
- PR body includes a placeholder noting that a human-written change summary is required before merge per repo policy.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
