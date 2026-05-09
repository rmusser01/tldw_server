---
id: TASK-45.25
title: Adapt Agent DiffViewer file status badges to shared Badge
status: Done
assignee: []
created_date: '2026-05-09 05:21'
updated_date: '2026-05-09 05:25'
labels:
  - design-system
  - webui
  - guard
dependencies: []
references:
  - apps/packages/ui/src/components/Agent/DiffViewer.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate Agent DiffViewer FileStatusBadge from local span styling to the shared design-system Badge primitive with explicit canonical state mapping, preserving visible file status labels and removing its local-status-badge baseline exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DiffViewer file status labels for new, deleted, and renamed files render through the shared Badge primitive
- [x] #2 File status badge variants are selected from design-system state registry mappings
- [x] #3 The local-status-badge baseline exception for src/components/Agent/DiffViewer.tsx is removed without new unbaselined findings
- [x] #4 Focused DiffViewer tests, product-state guard tests, design-system verifier, diff checks, and touched-file TypeScript filter are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after verifying PR #1403 merged into dev at 739b1a2b52b71132e4339bf4ac69621488977c36. New isolated worktree: .worktrees/design-system-agent-file-status-badge on branch codex/design-system-agent-file-status-badge. Remaining local-status-badge baseline entries include Agent/DiffViewer, Agent/SessionHistoryPanel, Layouts/ConnectionStatus, PresentationStudioStatusBadge, SyncStatusBadge, and Sidepanel/Chat StatusDot; this slice targets Agent/DiffViewer only.

Red evidence: bunx vitest run src/components/Agent/__tests__/DiffViewer.file-status-badge.test.tsx --reporter=dot failed after dependency bootstrap because NEW was not inside data-ds-component="Badge".

Implementation: FileStatusBadge now maps new/deleted/renamed file statuses to canonical design-system states, selects shared Badge variants from state severity, preserves visible labels NEW/DEL/RENAME, and removes the Agent/DiffViewer local-status-badge baseline exception.

Verification: bunx vitest run src/components/Agent/__tests__/DiffViewer.file-status-badge.test.tsx --reporter=dot passed 1/1; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 46/46; bun run verify:design-system-state passed with baseline exceptions 511 and local-status-badge 5; git diff --check passed.

TypeScript caveat: bunx tsc --noEmit --pretty false still fails on existing repo-wide frontend baseline errors, but filtering the output for DiffViewer, DiffViewer.file-status-badge, and design-system-product-state-baseline returned no touched-file errors.

Bandit: skipped because this slice only changes TypeScript/TSX, JSON, and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted Agent DiffViewer FileStatusBadge to the shared Badge primitive with design-system state mapping and removed its local-status-badge baseline exception. Focused tests, product-state guard tests, design-system verifier, and diff checks passed; full tsc remains blocked by unrelated baseline errors.
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
