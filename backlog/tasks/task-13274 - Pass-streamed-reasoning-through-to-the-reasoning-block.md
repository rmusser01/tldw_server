---
id: TASK-13274
title: Pass streamed reasoning through to the reasoning block
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - streaming
  - ux-audit
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reasoning models appear frozen. Verified against live llama.cpp: the model emitted 57 reasoning chunks in a 60-token stream, while the UI showed a generating placeholder for 15.5 seconds with zero disclosure elements before any text appeared. The streaming parser handles the reasoning field and a reasoning block component exists to render it; the transport layer yields only the plain token and discards reasoning, so the two never meet. A thinking-budget control already exists in model settings, so reasoning is a first-class concept in the product.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Streamed reasoning content reaches the reasoning block on the default chat path.
- [ ] #2 Reasoning is visible to the user while the response is still generating.
- [ ] #3 Reasoning is collapsed by default and does not displace the answer.
- [ ] #4 A non-reasoning model shows no empty reasoning affordance.
<!-- AC:END -->
