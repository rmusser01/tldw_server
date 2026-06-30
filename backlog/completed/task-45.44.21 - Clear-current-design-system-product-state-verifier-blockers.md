---
id: TASK-45.44.21
title: Clear current design-system product-state verifier blockers
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-30 16:47'
labels:
  - design-system
  - webui
  - product-state
dependencies: []
references:
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - >-
    apps/packages/ui/src/components/Option/WritingPlayground/WritingActionBar.tsx
  - apps/packages/ui/src/components/Notes/NotesEditorPane.tsx
  - apps/packages/ui/src/components/Notes/NotesManagerPage.tsx
  - apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx
  - >-
    apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/index.tsx
  - >-
    apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceCapabilityRemediation.tsx
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the current unbaselined product-state verifier blockers in WritingPlayground, Notes, and ResearchWorkspace so `bun run verify:design-system-state` can pass on this branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WritingPlayground no longer imports AntD Alert for product-state UI and its Ready label comes from the design-system state registry.
- [x] #2 Notes linked-note unavailable labels come from the design-system state registry without changing existing note-link behavior.
- [x] #3 ResearchWorkspace Ready, Degraded, and Blocked labels come from the design-system state registry without changing status semantics.
- [x] #4 `bun run verify:design-system-state` passes or any remaining failures are documented as outside this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Inspect the shared state registry and existing DS Alert usage patterns.', 'Add focused regression coverage for the affected visible labels/alerts where practical.', 'Replace direct AntD Alert usage and hard-coded canonical labels with design-system primitives/registry labels.', 'Run focused tests, full product-state verifier, TypeScript, and diff hygiene; record evidence.']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation notes:
- Replaced the remaining WritingActionBar AntD product-state Alert with the shared design-system Alert primitive.
- Routed WritingActionBar ready status, Notes linked-note unavailable status, ResearchWorkspace provider/source readiness labels, and workspace remediation blocked/degraded labels through the design-system state registry.
- Added focused registry-backed regression coverage for WritingActionBar, Notes graph chips, ResearchWorkspace ChatPane provider status, and WorkspaceCapabilityRemediation.

Verification:
- RED: bun run verify:design-system-state failed on WritingPlayground Alert/Ready, Notes Unavailable, and ResearchWorkspace Ready/Degraded/Blocked findings.
- RED focused tests failed for registry-backed WritingActionBar ready status/DS confirmation Alert, Notes unavailable graph chip labels, WorkspaceCapabilityRemediation blocked/degraded labels, and ChatPane provider status labels.
- GREEN: WritingActionBar test passed 9/9; Notes graph panels passed 7/7; WorkspaceCapabilityRemediation passed 6/6; ChatPane stage1 passed 24/24; SourcesPane design-system passed 2/2.
- Full guard: bun run verify:design-system-state exited 0 and reported no blocked product-state findings.
- TypeScript: env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false passed.
- Whitespace: git diff --check passed.
- Bandit skipped: touched files are frontend TS/TSX tests/components and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Cleared the current unbaselined design-system product-state verifier blockers in WritingPlayground, Notes, and ResearchWorkspace by moving remaining canonical labels through the registry and migrating the WritingActionBar confirmation warning to DS Alert.
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
