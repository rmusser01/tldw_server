---
id: TASK-2256
title: Implement Workspaces manager WebUI API client models
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-04 05:46'
labels:
  - workspaces
  - frontend
  - project-workspace
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-06-04-canonical-workspaces-manager-project-creation-design.md
  - >-
    Docs/superpowers/plans/2026-06-04-canonical-workspaces-manager-project-creation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 from the canonical Workspaces manager plan: add WebUI Workspace API client parity and canonical manager normalization/copy helpers. This task must not build the /workspaces route or UI manager screen; those are later slices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace API client exposes typed methods for patch, raw delete, roots, primary-root attach, file-inventory scan/status/items, and current Workspace sub-resources; operation polling and sandbox-root provision callables are deferred until the backend endpoints land.
- [x] #2 Workspace API client response types include workspace_profile, project root, file inventory availability, context attention state, and operation envelopes from context active_operations.
- [x] #3 Canonical manager model helpers normalize workspace/context/root state without importing ACP, MCP, or prototype workspace response types.
- [x] #4 Copy helpers pin canonical labels and tests reject Workspace Playground and Shared Workspace as canonical manager labels.
- [x] #5 Focused frontend tests are written red-first and pass after implementation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 2 from Docs/superpowers/plans/2026-06-04-canonical-workspaces-manager-project-creation.md: WebUI Workspace API Client Parity And Normalized Models.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-04: Red client route check:
  `bunx vitest run apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts`
  failed as expected with 9 missing Workspace API methods (`patchWorkspace`,
  `deleteWorkspace`, roots, file inventory, operation, and sandbox provision
  methods).
- 2026-06-04: Red manager model/copy check:
  `bunx vitest run apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-manager-models.test.ts`
  failed as expected because `workspace-manager-copy` did not exist yet.
- 2026-06-04: Green focused frontend check:
  `bunx vitest run apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-manager-models.test.ts`
  passed with 2 test files and 17 tests.
- 2026-06-04: `git diff --check` passed with no output.
- 2026-06-04: Bandit N/A; this task only changes frontend TypeScript and
  Backlog task metadata, with no Python touched.
- 2026-06-04: Code-quality review found callable operation polling and sandbox
  provision methods pointed at Task 4 backend routes that do not exist yet.
  Removed those future-only callables from Task 2 and deferred them until the
  backend operation/provision endpoints land; retained operation envelope types
  for context active_operations.
- 2026-06-04: Final verification:
  `bunx vitest run apps/packages/ui/src/services/__tests__/tldw-api-client.workspace-api.test.ts apps/packages/ui/src/components/Option/Workspaces/__tests__/workspace-manager-models.test.ts`
  passed with 2 files and 18 tests; `git diff --check` passed with no output.
  Spec review and code-quality review both approved after endpoint sequencing
  fixes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Task 2 WebUI client/model parity slice for canonical Workspaces. Added current backend-backed Workspace client methods and response/request types for metadata patch/delete, roots, primary-root attach, file inventory scan/status/items, source/artifact/note sub-resources, context attention state, project-root file inventory availability, and active operation envelopes. Added canonical manager normalization/copy helpers and focused tests for attention defaults, project-root inventory availability, active operations, canonical labels, ACP/MCP/prototype label guardrails, segment ID encoding, slash-ID rejection, source-status URLs, and wider artifact request fields. Operation polling and sandbox-root provision callables are intentionally deferred until the Task 4 backend endpoints exist.
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
