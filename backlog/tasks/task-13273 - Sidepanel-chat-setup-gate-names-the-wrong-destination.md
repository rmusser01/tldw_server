---
id: TASK-13273
title: Sidepanel chat setup gate names the wrong destination
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - extension
  - chat
  - copy
  - ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Navigating to chat in the extension sidepanel renders "Finish setup to open Companion Home". The user asked for chat and is told what a different surface needs. Reproduced against a freshly built extension from dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The setup gate names the surface the user actually requested.
- [ ] #2 The gate copy is derived from the requested route rather than hard-coded.
<!-- AC:END -->
