---
id: TASK-45.44.1
title: 'Migrate design-system product state: Chat and Playground'
status: In Progress
assignee: []
created_date: '2026-05-14 03:18'
updated_date: '2026-05-14 04:28'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1658'
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub product-area migration issue. Closure requires zero current product-state baseline exceptions for the owned path map area and the verification gates recorded in the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The linked GitHub issue owns current count and public status.
- [ ] #2 Implementation PR tasks are created under this child when the area is too broad for one PR.
- [ ] #3 Backlog notes record PR links and before/after count evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the two current Chat and Playground product-state baseline entries and their owning files. 2. Add a focused failing regression test for canonical design-system state labels in the affected Chat/Playground surface. 3. Migrate the affected UI text to the canonical design-system state registry or shared primitive without changing unrelated behavior. 4. Remove the resolved Chat and Playground baseline entries only after the verifier reports zero findings for this product area. 5. Run focused Vitest, bun run verify:design-system-state from apps/packages/ui, git diff --check, and document Bandit as skipped for UI-only work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Continuation after tracker PR #1679 merged. Starting first product-area slice for GitHub issue #1658 / TASK-45.44.1 in isolated worktree .worktrees/design-system-chat-playground-migration.

Migrated Chat and Playground target labels. Target guard findings are now zero and target baseline entries are now zero. Focused guard suite passes. Full design-system verifier still fails on inherited WorkspaceACPHistoryModal Alert outside this task. Bandit skipped for UI-only changes.
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
