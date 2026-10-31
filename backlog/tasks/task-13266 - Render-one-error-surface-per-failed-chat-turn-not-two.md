---
id: TASK-13266
title: Render one error surface per failed chat turn, not two
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
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A single connection refusal renders the same message twice: an inline bubble on the failed turn and a composer banner. Measured live at nine buttons across seven distinct concepts for one failure, consuming roughly 35 percent of the viewport. The bubble also lists its own actions as prose above the buttons, so the same sentence appears three times. "Switch model" against "Switch provider" and "Retry same model" against "Retry chat" force the user to work out whether the difference is meaningful.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One failure renders one alert region.
- [ ] #2 The surviving surface is anchored to the turn that failed.
- [ ] #3 The recovery actions are deduplicated to a single named set.
- [ ] #4 Only one alert is announced to assistive technology per failure.
<!-- AC:END -->
