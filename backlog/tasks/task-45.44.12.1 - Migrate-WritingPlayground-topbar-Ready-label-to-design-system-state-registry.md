---
id: TASK-45.44.12.1
title: Migrate WritingPlayground topbar Ready label to design-system state registry
status: Done
assignee: []
created_date: '2026-05-15 03:14'
updated_date: '2026-05-15 03:42'
labels:
  - design-system
  - webui
  - extension
  - product-state
  - writing-playground
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1669'
  - apps/packages/ui/src/components/Option/WritingPlayground/index.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - 'https://github.com/rmusser01/tldw_server/pull/1716'
parent_task_id: TASK-45.44.12
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the remaining hardcoded WritingPlayground topbar diagnostics Ready product-state label with the canonical design-system state registry value. Scope is limited to the WritingPlayground topbar ready status label and the matching product-state baseline entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WritingPlayground topbar ready diagnostics label resolves through the design-system ready state label instead of a hardcoded canonical literal.
- [x] #2 The matching WritingPlayground canonical-state-label baseline exception is removed and the design-system product-state verifier passes.
- [x] #3 Focused WritingPlayground and design-system guard tests cover the registry-backed ready label behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing WritingPlayground regression test that mocks the design-system ready label and expects the topbar ready status to render that registry label.
2. Replace the topbar ready fallback with the exported design-system ready label without changing warning or busy behavior.
3. Remove the resolved WritingPlayground baseline exception and run the focused WritingPlayground tests plus the product-state guard verifier.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the WritingPlayground topbar Ready-label migration in isolated worktree .worktrees/design-system-next-slice-5. Red/green proof: the new focused topbar design-system test fails against the hardcoded Ready fallback and passes when the topbar uses READY_STATE_LABEL. Removed the matching canonical-state-label baseline entry and refreshed only the shifted WritingPlayground AntD Alert baseline IDs caused by the added import.

Verification: bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingPlayground.topbar-design-system-state.test.tsx src/components/Option/WritingPlayground/__tests__/WritingPlaygroundDiagnosticsPanel.design-system-state.test.tsx src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed with 3 files and 54 tests. bun run verify:design-system-state passed with 483 baseline exceptions, 479 AntD product-state imports and 4 canonical-state-label exceptions remaining. JSON parse and git diff --check passed. bunx tsc --noEmit --pretty false still exits 2 on existing repo-wide type debt; touched-path filter found no diagnostics for WritingPlayground.topbar-design-system-state, WritingPlayground/index.tsx, design-system-product-state-baseline, or the task file. Bandit skipped because touched runtime scope is UI TypeScript, JSON baseline, and Backlog metadata only.

Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1716
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the remaining WritingPlayground topbar Ready diagnostics label to the design-system ready state label, added focused regression coverage for registry-backed rendering, removed the resolved canonical-state-label baseline exception, and refreshed only pre-existing AntD Alert baseline IDs shifted by the new import.
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
