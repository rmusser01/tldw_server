---
id: TASK-13281
title: Label the composer token counter operands
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - copy
  - ux-audit
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The composer shows an unlabelled three-term formula, for example "635 + ~0 = 635 tokens". The operands are the conversation total and the unsent draft estimate, but nothing says so, and the tilde is unexplained. Confirmed the arithmetic is correct, so this is a labelling problem rather than a counting bug.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each operand is identifiable without prior knowledge, via label or tooltip.
- [ ] #2 The meaning of the approximation marker is explained.
- [ ] #3 The control keeps its compact footprint.
<!-- AC:END -->
