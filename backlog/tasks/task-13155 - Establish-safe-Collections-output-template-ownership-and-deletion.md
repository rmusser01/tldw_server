---
id: TASK-13155
title: Establish safe Collections output-template ownership and deletion
status: To Do
assignee: []
created_date: '2026-09-03 02:30'
updated_date: '2026-09-03 02:31'
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
Keep the existing `/outputs/templates` API, define the exact template types governed by Collections, and record the current live reference owner: Reading digest schedules. Historical rendered outputs are self-contained and do not block deletion. Update/delete remain user-scoped; deletion refuses a template referenced by a Reading digest schedule unless removed or reassigned. Record a source-level branch-base reference inventory; another live owner requires acceptance-criteria revision before implementation, not silent scope expansion. Advertise exact `hasCollectionsOutputTemplateManagementV1=true` only when bounded paging, CRUD, reference-safe deletion, and the ownership set are active.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The API and documentation identify the exact output-template types governed by Collections and record a branch-base inventory of live template reference owners.
- [ ] #2 Template listing remains bounded and user-scoped, while create/get/update/delete preserve the existing rendering contract and expose bounded errors.
- [ ] #3 Deleting a template referenced by a Reading digest schedule is refused atomically without exposing another user's objects; historical self-contained outputs do not block deletion.
- [ ] #4 Docs-info advertises `hasCollectionsOutputTemplateManagementV1=true` only when ownership, CRUD, paging, and reference-safe deletion guarantees are active.
- [ ] #5 Focused database/API/security tests and the required Server ADR check pass.
<!-- AC:END -->
