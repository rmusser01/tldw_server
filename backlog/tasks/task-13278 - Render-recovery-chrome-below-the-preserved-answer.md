---
id: TASK-13278
title: Render recovery chrome below the preserved answer
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - error-handling
  - ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After stopping a stream the partial answer survives intact, which is a genuine strength. But the interruption notice and four recovery buttons render above it: measured at character index 32 for the notice and 345 for the prose. The user content the feature exists to preserve is pushed below the apology. The notice also offers a provider fallback action when no provider failed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Preserved assistant content renders above the recovery chrome.
- [ ] #2 The recovery action set matches the cause; a user-initiated stop does not offer provider fallback.
- [ ] #3 The recovery actions are listed once, not repeated as prose above the buttons.
<!-- AC:END -->
