---
id: TASK-13277
title: Announce or remove automatic focus-mode entry on narrow viewports
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - responsive
  - ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Below 768px the chat page enters focus mode without being asked. An Exit focus control appears that the user never requested, and the rails vanish. There is also a dead band between 768 and 1023px where the mobile rail tab strip renders alongside the full desktop toolbar, because the mobile breakpoint and the rail breakpoint disagree. A narrow extension window lands in that band.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Automatic focus-mode entry is either announced or removed.
- [ ] #2 The mobile breakpoint and the rail breakpoint agree.
- [ ] #3 No viewport width renders both the mobile rail strip and the full desktop toolbar.
<!-- AC:END -->
