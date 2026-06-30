---
id: TASK-468
title: Draft Workspace Persona Defaults PRD
status: Done
labels:
- persona
- workspace
- prd
- docs
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/issues/1911
- https://github.com/rmusser01/tldw_server/issues/1902
modified_files:
- Docs/Product/Workspace_Persona_Defaults_PRD.md
- backlog/tasks/task-468 - Draft-Workspace-Persona-Defaults-PRD.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Draft a repo-grounded PRD for Workspace-scoped Persona defaults, replacing old project_id terminology with current Workspace-level persona/style/voice/tool default semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PRD is grounded in current Workspace, chat, and Persona contracts.
- [x] #2 Scope, non-goals, precedence rules, risks, staged implementation, and validation plan are documented.
- [x] #3 Issue #1911 and tracker #1902 are referenced.
- [x] #4 Docs-only verification is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create the dedicated GitHub tracker issue for Workspace Persona Defaults. 2. Inspect current Workspace, chat, and Persona contracts to ground the PRD. 3. Draft the PRD with scope, non-goals, precedence, degraded-state behavior, staged implementation, risks, and validation. 4. Run docs-only verification and update the task status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Workspace Persona Defaults PRD and grounded it in existing Workspace CRUD schemas, Chat Workspace scope/runtime state, Workspace-scoped chat session metadata, and Persona profile references. Documented reference-backed defaults, precedence, non-goals, staged delivery, risks, and validation. Verification: git diff --check passed. Bandit skipped because this is docs/backlog only.
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
