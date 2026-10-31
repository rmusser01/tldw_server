---
id: TASK-13275
title: Progressive disclosure for the chat composer toolbar
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - density
  - ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured 63 visible interactive controls on the default chat screen before any conversation exists, of which the composer toolbar alone contributes a row of 13. The first-run task is one control: type a sentence and press Enter. Three nested disclosure layers already exist inside the composer, so a user hunting one control may open several independent disclosures. The prior audit measured 93 controls, so density is improving and the remaining reduction is layout-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The default first-run composer presents the message field, model, attach and the slash hint.
- [ ] #2 Remaining tools are reachable behind one labelled disclosure.
- [ ] #3 A returning user keeps whatever disclosure state they last chose.
- [ ] #4 Toolbar groups carry a visible separator, not only a screen-reader group label.
<!-- AC:END -->
