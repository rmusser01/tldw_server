---
id: TASK-13155
title: Establish safe Collections output-template ownership and deletion
status: To Do
assignee: []
created_date: '2026-09-03 02:30'
updated_date: '2026-09-03 02:41'
labels:
  - collections
  - output-templates
  - api
  - deletion
dependencies: []
references:
  - 'tldw_chatbook:TASK-18919'
  - >-
    tldw_chatbook:Docs/superpowers/specs/2026-09-01-collections-followup-backlog-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the existing `/outputs/templates` API but define which template types are governed by the
Collections umbrella and expose the current live reference owner: Reading digest schedules.
Historical rendered outputs do not block deletion because they are self-contained. Update and
delete remain user-scoped; deletion refuses a template referenced by a Reading digest schedule
unless that reference is first removed or reassigned. The task records a source-level reference
inventory at its branch base; finding another live owner requires updating the task acceptance
criteria before implementation rather than silently expanding scope. The API returns bounded
conflict reasons without leaking another user's objects. Docs-info advertises exact
`hasCollectionsOutputTemplateManagementV1=true` only when bounded paging, CRUD, reference-safe
deletion, and the documented ownership set are active. Existing template rendering remains intact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The API and documentation identify the exact output-template types governed by Collections and record a branch-base inventory of live template reference owners.
- [ ] #2 Template listing remains bounded and user-scoped, while create/get/update/delete preserve the existing rendering contract and expose bounded errors.
- [ ] #3 Deleting a template referenced by a Reading digest schedule is refused atomically without exposing another user's objects; historical self-contained outputs do not block deletion.
- [ ] #4 Docs-info advertises `hasCollectionsOutputTemplateManagementV1=true` only when ownership, CRUD, paging, and reference-safe deletion guarantees are active.
- [ ] #5 Focused database/API/security tests and the required Server ADR check pass.
<!-- AC:END -->
