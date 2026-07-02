---
id: TASK-12111
title: Fix character chat delete/edit targeting wrong Dexie row (greeting index offset)
status: Done
labels:
- bug
- high
- chat
- character
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: High (data corruption).** Round-2 audit findings R1 + R4.

In a character chat *with a greeting*, the greeting seed message is inserted at UI-array index 0 (`useMessage.tsx:1354`, `useChatActions.ts:1382`) but is **never written to Dexie** — `saveMessageOnSuccess` (`hooks/chat-helper/index.ts:438-491`) only persists the `user` and `assistant` rows. So the UI list `[greeting, user, assistant]` and the Dexie list `[user, assistant]` are off by one.

`deleteMessage(index)` (`useMessage.tsx:2929`) and `editMessage(index, …)` (`:2781-2873`) pass the **UI-array index** into `removeMessageByIndex`/`updateMessageByIndex`/`deleteChatForEdit`, which index into the Dexie history sorted by `createdAt` (`db/dexie/helpers.ts:406`). Result: deleting the user bubble removes the *assistant* Dexie row; editing the user bubble overwrites the *assistant* row; deleting the greeting deletes the *user* row. The server delete is correct (uses `serverMessageId`), so Dexie permanently diverges from server/UI.

Verified: `removeMessageByIndex` uses `sortedHistory[index]` (helpers.ts:406) against the Dexie list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Delete/edit target the correct stored message regardless of a greeting at UI index 0 — address messages by stable id (serverMessageId / message id), not by array position, OR reconcile the index offset when a non-persisted greeting is present.
- [ ] #2 The greeting seed is either persisted to Dexie (so UI and Dexie lists align) or explicitly excluded from index-based operations — pick one and make delete/edit consistent with it.
- [ ] #3 A character conversation rehydrates with the same message count from Dexie and from the server (greeting no longer vanishes/reappears by load source).
- [ ] #4 Tests: delete the user bubble in a greeting-led character chat and assert the Dexie assistant row is preserved and the user row removed; edit the user bubble and assert the user row (not assistant) is updated.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
