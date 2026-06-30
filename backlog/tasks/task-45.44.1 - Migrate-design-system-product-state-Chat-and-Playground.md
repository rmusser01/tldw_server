---
id: TASK-45.44.1
title: 'Migrate design-system product state: Chat and Playground'
status: Done
assignee: []
created_date: '2026-05-14 03:18'
updated_date: '2026-05-23'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1658'
  - 'https://github.com/rmusser01/tldw_server/pull/1683'
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
- [x] #1 The linked GitHub issue owns current count and public status.
- [x] #2 Implementation PR tasks are created under this child when the area is too broad for one PR.
- [x] #3 Backlog notes record PR links and before/after count evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect the two current Chat and Playground product-state baseline entries and their owning files. 2. Add a focused failing regression test for canonical design-system state labels in the affected Chat/Playground surface. 3. Migrate the affected UI text to the canonical design-system state registry or shared primitive without changing unrelated behavior. 4. Remove the resolved Chat and Playground baseline entries only after the verifier reports zero findings for this product area. 5. Run focused Vitest, bun run verify:design-system-state from apps/packages/ui, git diff --check, and document Bandit as skipped for UI-only work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Continuation after tracker PR #1679 merged. Started first product-area slice for GitHub issue #1658 / TASK-45.44.1 in isolated worktree .worktrees/design-system-chat-playground-migration.

Migrated Chat and Playground target labels. Target guard findings are now zero and target baseline entries are now zero. Focused guard suite passes. Bandit skipped for UI-only changes.

PR #1683 opened against dev and issue #1658 body updated with current count 0 and PR link. Before this slice: 2 Chat/Playground baseline exceptions. After this slice: 0 Chat/Playground baseline exceptions. No narrower implementation PR task was needed because the area contained only the two target canonical-state-label findings.

PR #1683 review fixes: added defensive optional state-label access for the reviewed getDesignSystemState call sites. After rebasing onto current dev, WorkspaceACPHistoryModal uses the shared design-system Alert primitive instead of AntD Alert. Full bun run verify:design-system-state now exits 0.

Closeout 2026-05-23:
- Verified GitHub issue #1658 records current Chat/Playground baseline debt as Total 0, `antd-product-state-import` 0, and `canonical-state-label` 0, refreshed by PR #1683.
- PR #1683 is merged into `dev` at `e4663b9b6cb06730ef901ccb44fac930ad1a8fec`.
- Current `apps/packages/ui/scripts/design-system-product-state-baseline.json` has zero rows for the Chat/Playground owned path map (`src/components/Option/Playground`, `src/components/Common/Playground`, `src/components/Sidepanel/Chat`, and `src/routes/sidepanel-chat.tsx`).
- Follow-up child TASK-45.44.1.1 closed the later unbaselined Playground Ready labels.
- Current full `bun run verify:design-system-state` was rerun after repairing the local UI dependency symlink and exits 1 on unrelated repo-wide product-state drift outside the Chat/Playground owned paths; this closeout does not claim global verifier cleanliness.
- Bandit skipped for this closeout because only Backlog markdown is changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the Chat and Playground design-system product-state tracker. The product area was migrated through PR #1683, the public issue #1658 records zero current baseline debt for the owned paths, the local baseline file has zero Chat/Playground rows, and the later Playground Ready-label follow-up is already closed under TASK-45.44.1.1. No application code changed in this closeout.
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
