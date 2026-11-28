---
id: TASK-15692
title: Preserve transcript order when reopening browser-only Document Workspace chats
status: To Do
labels:
- document-workspace
- chat
- bug
dependencies:
- TASK-15586
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During TASK-15586 private-beta verification, a browser-only EPUB chat displayed the assistant reply above the question after switching conversations in the in-app browser. A browser-only PDF chat showed the same issue after reloading Chrome. To reproduce, clear Save to server on a new chat, send a question and wait for the reply, then switch away and back or reload and reopen the chat. Both messages are present, but their display order can reverse.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Existing browser-only PDF and EPUB conversations reopen in their original message order after switching tabs or reloading.
- [ ] #2 Regression coverage verifies the original question and reply remain intact and in order when an existing browser-only chat is reopened.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Observed during same-account PDF and EPUB verification for Hosted PR #336 (https://github.com/rmusser01/tldw-hosted/pull/336). The server-saved copies reopened in question-then-reply order. A starting point for investigation is getChatHistory in apps/packages/ui/src/db/dexie/chat.ts, which retrieves local messages by history_id without specifying chronological order.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
