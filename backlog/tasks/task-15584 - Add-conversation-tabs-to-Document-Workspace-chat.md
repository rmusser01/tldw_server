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
- Docs/superpowers/plans/2026-11-20-document-workspace-conversation-tabs.md
- Docs/ADR/049-chat-history-selection-ownership.md
- https://github.com/rmusser01/tldw_server/pull/3534
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user start and switch among multiple independent conversations for the open document in the Chat sidebar.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Start a new chat and switch between named conversations for the same document from the sidebar.
- [x] #2 Restore each conversation’s own messages after reloading or reopening the document, including conversations saved before this change.
- [ ] #3 Bind the conversation list, active history, and pending replies to the verified account, document, and conversation. Verify keyboard switching and a narrow layout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Document Workspace currently loads the newest owner-matching history for each document. The existing history store can hold multiple rows for a document. Reuse the existing document association and history identities. ADR-049 governs owner-bound history. A new ADR is unnecessary if the implementation retains the existing storage and ownership contracts. Include document and account switches plus the live duplicate-reply reproduction in the restoration checks.
The requester selected browser-scoped tabs. The design applies ADR-049 to the existing document index and ownership contract. Cross-device conversation discovery is tracked in TASK-15586.
174 targeted Vitest cases passed, including the production Document Chat dispatcher with two server-backed chats, owner and document checks, and switching during reads and stopped replies. Frontend typecheck, scoped ESLint and git diff --check passed. An independent review caught server identity reload, stopped-turn activity ownership, temporary transcript retention and saved message identity cases. Fixed those paths and added regression coverage. Bandit does not apply to the TypeScript-only changes.
Local Next preview rendered the workspace. The synthetic API fixture has not opened the PDF yet, so the browser overflow check remains pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added browser-scoped conversation tabs, per-conversation drafts, exact history restore and scoped reply cancellation in Document Workspace. TASK-15586 tracks cross-device discovery.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
