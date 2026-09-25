---
id: TASK-13271
title: Consolidate the three chat shortcuts surfaces and correct the documented binding
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - consistency
  - ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three controls whose names are variants of "shortcuts" open three different panels: a left-rail control opens the page navigator, a top-bar control opens the keyboard reference, and a chat toolbar control opens a third chat-specific panel. The chat panel documents Shift+/ as "Open keyboard shortcuts"; pressing Shift+/ actually opens the page navigator. The keyboard reference lists its own binding as a third value. Verified live by probing each control by accessible name.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One concept has one name; navigation is not labelled as shortcuts.
- [ ] #2 One keyboard-shortcuts surface exists, on one binding.
- [ ] #3 Every advertised binding performs the action it documents.
- [ ] #4 The chat-specific bindings appear as a section of the single surface.
<!-- AC:END -->
