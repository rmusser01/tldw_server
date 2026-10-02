---
id: TASK-13267
title: Gate Continue from partial on an actual partial response
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - error-prevention
  - ux-audit
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue from partial is offered after a connection refusal where zero tokens were received. Its only current gate covers image generation. On an error turn the stored assistant message is the serialized error payload, so continuing seeds the next request with that payload. The action is genuinely useful after a user-initiated stop and should survive there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Continue from partial is hidden when no assistant text was received.
- [ ] #2 Continue from partial is hidden when the turn holds an error payload rather than prose.
- [ ] #3 Continue from partial remains available after a user-initiated stop with preserved text.
- [ ] #4 The gate is applied at every site that renders the action.
<!-- AC:END -->
