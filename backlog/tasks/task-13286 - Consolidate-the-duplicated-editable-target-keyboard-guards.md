---
id: TASK-13286
title: Consolidate the duplicated editable-target keyboard guards
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - webui
  - keyboard
  - refactor
  - ux-audit
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The predicate answering "is the user typing right now" is reimplemented at least fourteen times across the shared UI package under at least five different names, including isEditableTarget, isInputFocused, shouldIgnoreShortcut, isInputField and isTypingTarget. Every other screen applies it correctly; the chat page was the one that computed it and applied it too late, which produced the question-mark defect. The duplication is why one site could drift without anyone noticing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One shared helper covers input, textarea, select and contenteditable targets.
- [ ] #2 Every global keyboard handler in the shared UI package uses it.
- [ ] #3 The duplicate local definitions are removed.
- [ ] #4 A test covers each element kind the helper must treat as editable.
<!-- AC:END -->
