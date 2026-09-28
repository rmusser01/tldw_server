---
id: TASK-13265
title: Name the failing provider in chat error copy instead of the tldw server
status: To Do
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-19 14:47'
labels:
  - chat
  - webui
  - error-handling
  - truthfulness
  - ux-audit
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A provider connection refusal renders "Something went wrong while talking to your tldw server" and directs the user to inspect server health. Reproduced live against a stopped Ollama: the tldw server was healthy and returned a correct 502; the unreachable hop was the provider. The user is dispatched to debug the one component that was working. Provider 401, 403, 429 and 5xx responses all fall through to this generic branch, and one branch matches an Anthropic credential error string while pointing the user at tldw server settings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A provider failure names the provider and, where known, its endpoint.
- [ ] #2 The suggested next action targets the provider, not the local server.
- [ ] #3 Provider auth, rate-limit and upstream-failure responses each map to their own message rather than the generic fallback.
- [ ] #4 The Anthropic credential error string no longer routes to tldw server settings.
<!-- AC:END -->
