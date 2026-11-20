---
id: TASK-15584
title: Add conversation tabs to Document Workspace chat
status: In Progress
labels:
- document-workspace
- chat
- webui
dependencies:
- TASK-15579
references:
- Docs/superpowers/specs/2026-11-20-document-workspace-conversation-tabs-design.md
- Docs/ADR/049-chat-history-selection-ownership.md
- https://github.com/rmusser01/tldw_server/pull/3534
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user start and switch among multiple independent conversations for the open document in the Chat sidebar.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Start a new chat and switch between named conversations for the same document from the sidebar.
- [ ] #2 Restore each conversation’s own messages after reloading or reopening the document, including conversations saved before this change.
- [ ] #3 Bind the conversation list, active history, and pending replies to the verified account, document, and conversation. Verify keyboard switching and a narrow layout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Document Workspace currently loads the newest owner-matching history for each document. The existing history store can hold multiple rows for a document. Reuse the existing document association and history identities. ADR-049 governs owner-bound history. A new ADR is unnecessary if the implementation retains the existing storage and ownership contracts. Include document and account switches plus the live duplicate-reply reproduction in the restoration checks.
The requester selected browser-scoped tabs. The design applies ADR-049 to the existing document index and ownership contract. Cross-device conversation discovery is tracked in TASK-15586.
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
