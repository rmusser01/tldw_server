---
id: TASK-2261
title: Add local Research Workspace reconciliation panel
status: Done
labels:
- workspaces
- webui
- research-workspace
- migration
priority: high
documentation:
- Docs/superpowers/specs/2026-06-04-canonical-workspaces-manager-project-creation-design.md
modified_files:
- apps/packages/ui/src/components/Option/Workspaces/workspace-local-reconciliation.ts
- apps/packages/ui/src/components/Option/Workspaces/WorkspaceReconciliationPanel.tsx
- apps/packages/ui/src/components/Option/Workspaces/WorkspacesManagerPage.tsx
- apps/packages/ui/src/store/research-workspace-legacy-storage-inventory.ts
- apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-local-reconciliation.test.ts
- apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspaceReconciliationPanel.test.tsx
- apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspacesManagerPage.test.tsx
- apps/packages/ui/src/store/__tests__/research-workspace-legacy-storage-inventory.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next canonical Workspaces manager roadmap slice: detect local-only Research Workspace entries, show them separately from server-backed Workspaces, dry-run reconciliation states, and allow explicit server metadata creation or linking by writing a minimal reconciliation marker without rewriting local source, note, artifact, chat, or IndexedDB payloads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local Research Workspace entries from the existing local store can be detected without rewriting local payloads.
- [x] #2 The manager separates local-only entries from server-backed Workspaces.
- [x] #3 Dry-run states cover local-only, server row exists, name conflict, possible duplicate, unsupported local payload, and ready to create metadata.
- [x] #4 Users can create server metadata for eligible local entries or link a local entry to an existing Workspace.
- [x] #5 A minimal reconciliation marker is written only after confirmed metadata promotion or link.
- [x] #6 Local tombstones and undo behavior remain preserved.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-04-canonical-workspaces-manager-project-creation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed TASK-2261. Verification: red tests first failed on missing reconciliation helper/panel; focused reconciliation suite passed with 19 tests across helper, panel, manager, and storage inventory; broader Workspaces route/model/client suite passed with 61 tests across 10 files; git diff --check passed. Design-system product-state guard still fails on unrelated baseline labels in Onboarding FirstChatStep and ACP readiness only; no Workspaces files were reported. Package TypeScript check still fails on unrelated baseline DynamicUI missing modules and ResearchWorkspace fixture typing; no new Workspaces errors were reported. Bandit skipped because this slice touched frontend TypeScript/tests and Backlog only.
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
