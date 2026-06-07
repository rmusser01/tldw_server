---
id: TASK-2315
title: Design Workspace cross-resource membership foundation
status: In Progress
labels:
- workspaces
- project-workspace
- design
- membership
priority: high
references:
- https://github.com/rmusser01/tldw_server/issues/1990
- https://github.com/rmusser01/tldw_server/issues/1984
- Docs/superpowers/specs/2026-06-03-canonical-workspace-core-project-model-design.md
- Docs/superpowers/specs/2026-06-04-canonical-workspaces-manager-project-creation-design.md
documentation:
- Docs/superpowers/specs/2026-06-07-workspace-cross-resource-membership-design.md
modified_files:
- Docs/superpowers/specs/2026-06-07-workspace-cross-resource-membership-design.md
- backlog/tasks/task-2315 - Design-Workspace-cross-resource-membership-foundation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a focused design/spec for GitHub issue #1990 cross-resource Workspace membership. Scope is to define how canonical Workspace identity associates notes, media/sources, artifacts, chats, prompts, files, and future agent/sandbox outputs without making /workspaces a global filter that hides existing records. Produce an implementation-ready design and task roadmap; do not implement runtime code in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design defines a canonical membership model covering resource types, owner/user scoping, membership roles, provenance, soft-delete/archive behavior, and global visibility guardrails.
- [ ] #2 Design distinguishes server-backed membership from Research Workspace source selection, MCP trusted root bindings, ACP execution workspaces, and Sandbox project roots.
- [ ] #3 Design proposes API/read-model slices and migration/backfill behavior without destructive reassignment of existing resources.
- [ ] #4 Design includes a sequential implementation roadmap with parallelizable tasks, tests, validation, and failure states.
- [ ] #5 Design verification is recorded; Bandit applicability is documented for docs-only scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Drafted the Workspace cross-resource membership design for GitHub issue #1990. The design recommends a generic server-backed membership table plus fail-closed resource adapters, keeps Research Workspace source selection and Project Workspace root/runtime bindings separate, defines API/read-model/backfill behavior, distinguishes scoped workspace_note from global note membership, and records a sequential implementation roadmap with parallelizable slices. Verification: git diff --check passed. Bandit is not applicable because this task only changes Markdown design/tracking files.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
