---
id: TASK-13282
title: Raise sub-24px chat tap targets to the accessibility floor
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
Measured at 390x844 with touch. Two genuine failures below the 24x24 minimum: Take a quick tour at 118x16 and Select character or persona at 16x16. The always-visible message overflow control is 37x22. The skip link and the hidden file inputs also measure 1x1 but are standard visually-hidden patterns and are not failures. Separately, 55 of 63 desktop controls are under 44x44, which is within the rules but heavy for touch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No interactive chat control has a hit area below 24x24.
- [ ] #2 Hit area is increased via padding without changing the visual size.
- [ ] #3 Visually-hidden skip links and file inputs are excluded from the check by design.
- [ ] #4 A test pins the minimum for chat controls.
<!-- AC:END -->
