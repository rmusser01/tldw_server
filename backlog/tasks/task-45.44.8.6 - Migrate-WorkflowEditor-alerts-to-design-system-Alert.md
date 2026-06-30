---
id: TASK-45.44.8.6
title: Migrate WorkflowEditor alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
priority: medium
parent_task_id: TASK-45.44.8
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove WorkflowEditor AntD Alert product-state exceptions from the design-system baseline by migrating ExecutionPanel, NodeConfigPanel, and NodePalette alert surfaces to the shared design-system Alert primitive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ExecutionPanel, NodeConfigPanel, and NodePalette render the design-system Alert primitive instead of AntD Alert for product-state alert surfaces.
- [x] #2 The product-state baseline contains no WorkflowEditor Alert exceptions touched by this slice.
- [x] #3 Focused regression coverage verifies the migrated WorkflowEditor alert surfaces expose the design-system Alert marker and preserve user-facing copy.
- [x] #4 Relevant verification commands are recorded, with Bandit skipped only if the touched scope remains TypeScript/UI-only.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started from latest origin/dev after PR #2188 merged. Scope selected from live design-system baseline: WorkflowEditor ExecutionPanel, NodeConfigPanel, and NodePalette AntD Alert product-state exceptions.

Migrated WorkflowEditor ExecutionPanel, NodeConfigPanel, and NodePalette from AntD Alert to the shared design-system Alert primitive. Preserved alert copy and actions, including diagnostics retry and node-library retry controls.

Removed 4 WorkflowEditor Alert entries from the product-state baseline, reducing live baseline exceptions from 86 to 82 and leaving no WorkflowEditor baseline entries.

Verification recorded:
- `bunx vitest run src/components/WorkflowEditor/__tests__/ExecutionPanel.design-system.test.tsx src/components/WorkflowEditor/__tests__/NodeConfigPanel.test.tsx src/components/WorkflowEditor/__tests__/NodePalette.test.tsx --reporter=dot`
- `bun run verify:design-system-state`
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`

Bandit skipped because this is TypeScript/UI-only work.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
WorkflowEditor AntD Alert product-state exceptions were migrated to the design-system Alert primitive in ExecutionPanel, NodeConfigPanel, and NodePalette. Focused regression coverage now asserts DS Alert markers and preserved copy/actions. Verification passed for the focused WorkflowEditor Vitest suite, design-system product-state guard with 82 remaining exceptions and zero WorkflowEditor entries, and TypeScript with the larger Node heap. Bandit skipped because this is TypeScript/UI-only work.
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
