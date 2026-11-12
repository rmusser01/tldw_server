---
id: TASK-13283
title: Fix Start chatting contrast on the chat empty state
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - a11y
  - ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The primary empty-state call to action measures 3.13:1 between its white label and its blue fill, where the standard requires 4.5:1 for text at this size. Confirmed by pixel sampling as well as computed styles. It is the only contrast failure on the page, which otherwise passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Start chatting reaches at least 4.5:1 against its background.
- [ ] #2 The button still reads as the primary action.
- [ ] #3 No other chat control regresses below its required ratio.
<!-- AC:END -->
