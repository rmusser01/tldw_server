---
id: TASK-13276
title: Fix the composer occluding chat content at sidepanel width
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - extension
  - responsive
  - ux-audit
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 400px, the width of the extension sidepanel, the composer panel paints over the empty state. Measured: a heading clipped to "CHA (chevron) DES", a sentence truncated mid-word, and three controls whose centre point resolves to a different element and are therefore unreachable. The toolbar wraps into five ungrouped rows, two unlabelled slider icons land in different rows, and the model is named twice about 200px apart. Stable across repeated sampling, not a transient re-layout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No chat content is occluded by the composer at 390 to 400px.
- [ ] #2 Every visible control is hit-testable at its own centre point.
- [ ] #3 The composer toolbar wraps to at most two rows.
- [ ] #4 The model is named once in the composer area.
<!-- AC:END -->
