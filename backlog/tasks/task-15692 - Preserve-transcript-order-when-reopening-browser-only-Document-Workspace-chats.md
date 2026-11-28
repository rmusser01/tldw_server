---
id: TASK-15692
title: Preserve transcript order when reopening browser-only Document Workspace chats
status: In Progress
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
- [x] #1 Existing browser-only PDF and EPUB conversations reopen in their original message order after switching tabs or reloading.
- [x] #2 Regression coverage verifies the original question and reply remain intact and in order when an existing browser-only chat is reopened.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Observed during same-account PDF and EPUB verification for Hosted PR #336 (https://github.com/rmusser01/tldw-hosted/pull/336). The server-saved copies reopened in question-then-reply order. A starting point for investigation is getChatHistory in apps/packages/ui/src/db/dexie/chat.ts, which retrieves local messages by history_id without specifying chronological order.
ADR required: no. The fix restores browser-only message order and preserves the native document-chat association described in Docs/ADR/100-document-workspace-chat-association.md.
IndexedDB returned local rows in message-ID order. Document Workspace now sorts deduplicated rows by saved timestamps before projecting both the display and model context. The Vitest regression and both real-document Playwright cases reproduced the inversion before the fix. With the fix, 37 focused Vitest checks and all six Document Workspace chat Playwright checks passed, including PDF and EPUB tab switches and reloads, cross-device sync, and Save to server. Bandit does not apply to TypeScript and Markdown changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored the original order when reopening existing browser-only Document Workspace chats. Verified PDF and EPUB conversations after switching chat tabs and reloading. Pending Dev review.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
