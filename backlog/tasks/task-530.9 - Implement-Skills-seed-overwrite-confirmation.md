---
id: TASK-530.9
title: Implement Skills seed overwrite confirmation
status: In Progress
assignee: []
created_date: '2026-06-28 15:38'
updated_date: '2026-06-28 15:39'
labels:
  - skills
  - webui
  - safe-operations
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-06-28-skills-seed-overwrite-confirmation-design.md
parent_task_id: TASK-530
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-530 Safe Operations after TASK-530.8 by adding an explicit frontend confirmation before Seed and Overwrite Existing calls the Skills seed endpoint with overwrite=true. Keep Seed Missing Only one-click, keep backend seed behavior unchanged, and keep version-aware delete, bulk delete, export feedback, and permission metadata panels out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Seed and Overwrite Existing opens a confirmation before calling seedSkills with overwrite=true.
- [ ] #2 Cancelling the confirmation does not call the seed mutation.
- [ ] #3 Confirming the modal calls seedSkills({ overwrite: true }) exactly once and uses destructive button affordance.
- [ ] #4 Seed Missing Only remains one-click and continues to call seedSkills({ overwrite: false }).
- [ ] #5 Focused Manager Vitest coverage records the safe overwrite workflow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: Docs/superpowers/specs/2026-06-28-skills-seed-overwrite-confirmation-design.md
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
