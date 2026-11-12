---
id: TASK-13269
title: Show per-row provider health in the chat model picker
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
The model dropdown lists configured providers identically regardless of reachability. Reproduced live: llama.cpp (running) and Ollama (refusing connections) appeared with the same presentation and no health indicator, so the user discovers the truth only after composing and sending. The catalog decides usability from static server metadata; the word reachable appears only in a code comment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each model row carries a health indicator derived from a reachability signal.
- [ ] #2 Unreachable models are visually de-emphasised and sorted below reachable ones.
- [ ] #3 Section headings do not describe unprobed models as usable.
- [ ] #4 The list exposes a scroll affordance when it exceeds its container.
<!-- AC:END -->
