---
id: TASK-13279
title: Keep copy and regenerate visible on the last assistant message
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - recognition
  - ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Copy, Edit, Regenerate and Generation Info measure 0x0 and are absent from the accessibility tree until the message is hovered, then become 32x32. The two most frequent actions in any chat client are hidden while a rarely-used overflow menu is permanent. There are two overflow menus on every message; the always-visible one measures 37x22, below the accessibility target floor. Touch users have no hover.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Copy and Regenerate are persistently visible on the last assistant message.
- [ ] #2 Only one overflow menu renders per message.
- [ ] #3 Every persistent message action meets the minimum target size.
- [ ] #4 Hover-reveal remains acceptable for older turns.
<!-- AC:END -->
