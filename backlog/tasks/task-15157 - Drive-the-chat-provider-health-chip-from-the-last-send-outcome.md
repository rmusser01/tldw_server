---
id: TASK-15157
title: Drive the chat provider health chip from the last send outcome
status: To Do
assignee: []
created_date: 2026-09-19 14:47
updated_date: 2026-09-19 14:47
labels:
- chat
- webui
- truthfulness
- ux-audit
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The provider chip beside the model name reports only a probe of the local tldw server liveness endpoint, polled every 30 seconds and held green across consecutive failures. No chat send outcome ever writes to it. Reproduced live: with Ollama stopped, the chip read "Ollama / gemma3:1b Healthy" while two error panels about that same provider were on screen. A status indicator that contradicts the screen it sits on is worse than no indicator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failed provider request drives the chip out of the healthy state immediately.
- [ ] #2 The chip stays non-healthy until a later request or probe for that provider succeeds.
- [ ] #3 When health is cached rather than current, the chip communicates staleness instead of asserting current health.
- [ ] #4 A test covers the case of a provider failure while the local server probe is healthy.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Renumbered from TASK-13264 to TASK-15157 on 2026-10-31 (TASK-13960): TASK-13264 now names only "Prepare 0.1.44 release with all changes since v0.1.43 and land prior closure records" (backlog/tasks/task-13264 - Prepare-0.1.44-release-with-all-changes-since-v0.1.43-and-land-prior-closure-records.md). Before 2026-10-31 both tasks used TASK-13264; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
