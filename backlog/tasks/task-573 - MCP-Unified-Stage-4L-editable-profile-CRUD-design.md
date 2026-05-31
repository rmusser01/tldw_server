---
id: TASK-573
title: MCP Unified Stage 4L editable profile CRUD design
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-31 15:36
labels:
- mcp-unified
- design
- stage-4l
dependencies: []
modified_files:
- Docs/superpowers/specs/2026-05-31-mcp-unified-stage4l-editable-profile-crud-design.md
- backlog/tasks/task-573 - MCP-Unified-Stage-4L-editable-profile-CRUD-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and review the Stage 4L design spec for manager-first editable profile CRUD, covering create, limited patch, guarded delete, FastAPI/CLI surfaces, validation, audit behavior, and test scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stage 4L design spec captures manager-first editable profile CRUD scope.
- [x] #2 Spec covers FastAPI and CLI contracts for create, limited patch, and guarded delete.
- [x] #3 Spec documents safety rules for duplicate create, unsupported patch fields, default disable, default delete, and assigned-profile delete.
- [x] #4 Spec documents non-goals for assignment CRUD, approval policy editing, path scopes, grants, and UI changes.
- [x] #5 Spec review loop is completed and results are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design-only task. After human and reviewer approval, invoke the writing-plans skill to create the Stage 4L implementation plan in a separate plan artifact.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spec drafted at Docs/superpowers/specs/2026-05-31-mcp-unified-stage4l-editable-profile-crud-design.md. Spec reviewer approved with no blocking issues; advisory clarifications for disabled default create, scoped delete guards, malformed JSON tests, and the disabled-default-create manager test were incorporated. A follow-up self-review tightened effective-default wording, no-op patch handling, nested policy patch rejection, reason-code specificity, and persistent-store guarded delete requirements.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4L design spec was written and reviewed. It captures manager-first editable profile CRUD scope, FastAPI and CLI contracts, safety rules for duplicate create, unsupported/no-op patch, default protections, assigned-profile delete protection, compact audit posture, persistent-store guarded delete requirements, and explicit non-goals. Follow-up implementation planning and execution are tracked in TASK-574 and TASK-575. Known skips/blockers: Bandit not applicable for design-only task; no blockers.
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
