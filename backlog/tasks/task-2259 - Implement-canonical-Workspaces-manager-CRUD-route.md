---
id: TASK-2259
title: Implement canonical Workspaces manager CRUD route
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-04 06:42'
labels:
  - workspaces
  - webui
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
Implement Task 5 from the canonical Workspaces manager plan: add the server-backed /workspaces product directory with create/edit/archive/unarchive/open flows for Research and Project Workspace shells. Keep project root setup, MCP policy editing, ACP launch, hard delete, and local Research Workspace migration out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /workspaces route constant, metadata, registry entry, and route shell exist with no aliases or redirects.
- [x] #2 Manager renders server-backed Workspace list with loading, unavailable, empty, Research, Project, archived, and needs-attention states.
- [x] #3 Manager supports search and filters for profile, archived visibility, and attention state.
- [x] #4 Manager supports create Research Workspace and create Project Workspace shell without root setup.
- [x] #5 Manager supports edit metadata, archive, unarchive, and open in Research Workspace actions.
- [x] #6 Manager does not expose hard delete, soft-delete restore, MCP policy editing, ACP launch, or project root setup in this slice.
- [x] #7 Focused route and manager tests are written red-first and pass after implementation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Red evidence: package-local Vitest failed because /workspaces metadata, option-workspaces route shell, and WorkspacesManagerPage were missing. Green evidence: ./node_modules/.bin/vitest run src/routes/__tests__/option-workspaces.route.test.tsx src/routes/__tests__/route-metadata.coverage.test.ts src/routes/__tests__/route-registry.visibility.test.ts src/components/Option/Workspaces/__tests__/workspace-manager-models.test.ts src/components/Option/Workspaces/__tests__/WorkspacesManagerPage.test.tsx -> 5 files passed, 25 tests passed. TypeScript check with 8 GB heap reached existing unrelated errors in DynamicUI/OpenUI, ResearchWorkspace test fixture typing, and route AST helper dependency resolution; no TASK-2259 files were reported. Design-system guard runs after repairing local dependency symlinks but fails on existing unrelated blocked labels in Onboarding FirstChatStep and ACP readiness; new Workspaces files were not reported. git diff --check exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the canonical /workspaces manager CRUD route. Added route constant, metadata, audited route coverage, route registry entry, and option route shell with no aliases or redirects. Added WorkspacesManagerPage, WorkspaceList, create dialog, and metadata dialog for server-backed list/search/filter, create Research Workspace, create Project Workspace shell, edit metadata, archive, unarchive, and open in Research Workspace. Extended workspace manager normalization with source counts and added focused route/manager tests.
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
