---
id: TASK-573
title: MCP Unified Stage 4L editable profile CRUD design
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-31 15:36'
labels:
  - mcp-unified
  - design
  - stage-4l
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and review the Stage 4L design spec for manager-first editable profile CRUD, covering create, limited patch, guarded delete, FastAPI/CLI surfaces, validation, audit behavior, and test scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Stage 4L design spec captures manager-first editable profile CRUD scope.
- [ ] #2 Spec covers FastAPI and CLI contracts for create, limited patch, and guarded delete.
- [ ] #3 Spec documents safety rules for duplicate create, unsupported patch fields, default disable, default delete, and assigned-profile delete.
- [ ] #4 Spec documents non-goals for assignment CRUD, approval policy editing, path scopes, grants, and UI changes.
- [ ] #5 Spec review loop is completed and results are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design-only task. After human and reviewer approval, invoke the writing-plans skill to create the Stage 4L implementation plan in a separate plan artifact.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spec drafted at Docs/superpowers/specs/2026-05-31-mcp-unified-stage4l-editable-profile-crud-design.md. Spec reviewer approved with no blocking issues; advisory clarifications for disabled default create, scoped delete guards, malformed JSON tests, and the disabled-default-create manager test were incorporated.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
