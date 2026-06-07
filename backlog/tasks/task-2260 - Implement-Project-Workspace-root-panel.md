---
id: TASK-2260
title: Implement Project Workspace root panel
status: Done
labels:
- workspaces
- webui
- project-workspace
- sandbox
priority: high
documentation:
- Docs/superpowers/specs/2026-06-04-canonical-workspaces-manager-project-creation-design.md
modified_files:
- apps/packages/ui/src/components/Option/Workspaces/WorkspaceProjectRootPanel.tsx
- apps/packages/ui/src/components/Option/Workspaces/__tests__/WorkspaceProjectRootPanel.test.tsx
- apps/packages/ui/src/components/Option/Workspaces/WorkspacesManagerPage.tsx
- apps/packages/ui/src/components/Option/Workspaces/WorkspaceList.tsx
- apps/packages/ui/src/services/tldw/domains/workspace-api.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next canonical Workspaces manager roadmap slice: add a Project Workspace root panel for upgrading Research Workspaces, attaching host-local roots, provisioning sandbox-managed roots, showing root/inventory status, and gating inventory actions until the Workspace API reports file inventory availability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Research Workspaces expose an Upgrade to Project Workspace action that patches workspace_profile to project.
- [x] #2 Project Workspaces without a primary root expose host-local and sandbox-managed root setup choices.
- [x] #3 Host-local root attachment uses the existing primary-root attach endpoint with expected_workspace_version.
- [x] #4 Sandbox-managed root provisioning uses the Workspace-owned sandbox-volume endpoint with an Idempotency-Key and visible provisioning state.
- [x] #5 The panel can recover visible provisioning state from active operations in workspace context.
- [x] #6 Inventory scan actions are disabled until file_inventory.available is true, with specific remediation copy for sandbox roots before mount availability.
- [x] #7 Passive root displays do not expose raw host-local paths.
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
Completed TASK-2260. Verification: focused red/green suite passed with 25 tests across WorkspaceProjectRootPanel, WorkspacesManagerPage, and workspace API client; broader Workspaces route/model/client suite passed with 46 tests across 7 files; git diff --check passed. Design-system product-state guard still fails on unrelated baseline labels in Onboarding FirstChatStep and ACP readiness only; no Workspaces files were reported. Package TypeScript check still fails on unrelated baseline DynamicUI missing modules and ResearchWorkspace fixture typing; no new Workspaces errors were reported. Bandit skipped because this slice touched frontend TypeScript/tests and Backlog only.
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
