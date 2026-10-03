---
id: TASK-13280
title: Plain-language pass on chat internal vocabulary
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - copy
  - i18n
  - ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Visible on the default chat surface without opening anything: "Restore context sidechannel", "Context rail", "Runtime rail", "Cockpit rails hidden", "Legacy sheet view", "General chat starter selected", the assistant byline prefixed "tldw:", and "Temp" for temporary chat in a product that also has a temperature setting. Screen-reader users additionally hear a checkpoint count every few seconds while generating. Region-label chips name the interface back at the user. All are translation defaults, so this is one copy pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No shipped chat string uses sidechannel, rail, cockpit or checkpoint as user-facing vocabulary.
- [ ] #2 Temp is spelled Temporary.
- [ ] #3 The assistant byline shows the model name without an internal prefix.
- [ ] #4 Region-label chips that only restate the interface are removed.
- [ ] #5 Domain terms that are real standards keep their name and gain a first-use gloss.
<!-- AC:END -->
