---
id: TASK-15586
title: Sync Document Workspace conversations across devices
status: In Progress
labels:
- document-workspace
- chat
- sync
dependencies:
- TASK-15584
references:
- Docs/superpowers/specs/2026-11-20-document-workspace-conversation-tabs-design.md
- Docs/ADR/049-chat-history-selection-ownership.md
- Docs/superpowers/specs/2026-11-23-document-workspace-conversation-sync-design.md
- Docs/ADR/098-document-workspace-chat-association.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a signed-in user discover and resume the same document conversations from another browser or device.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Browsers signed in to the same account can open the same document and find its synced conversations.
- [ ] #2 Conversation discovery uses verified document access and chat ownership, including after switching accounts.
- [ ] #3 Existing browser-scoped document conversations remain accessible as the association is reconciled across devices.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Requested as a follow-up to the browser-scoped conversation tabs. Assess the server-owned document-to-chat association and migration behavior during design. ADR-049 governs owner-bound chat history. A durable server-side contract needs a separate ADR assessment.
The approved first-release design covers same-server discovery and a per-chat Save to server action for existing browser-only conversations. The [design spec](../../Docs/superpowers/specs/2026-11-23-document-workspace-conversation-sync-design.md) and [proposed ADR-098](../../Docs/ADR/098-document-workspace-chat-association.md) are ready for review. ADR required: yes. ADR path: Docs/ADR/098-document-workspace-chat-association.md. The association is a durable Chat persistence and Media access contract.
2026-11-24 design review checked the conversation-tabs branch and current Dev. Native history currently refuses document-grounded continuation. A synthetic schema check confirmed native import rejects browser-issued chat IDs and raw source rows. The revised design requires a qualified document-grounded send, stable import recovery with citation preservation, explicit confirmation for numeric-only legacy document links, and access checks at open and send. PR #3534 remains the Public Dev delivery dependency.
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
