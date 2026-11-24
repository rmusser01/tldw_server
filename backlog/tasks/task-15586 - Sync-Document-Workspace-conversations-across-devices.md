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
- Docs/superpowers/plans/2026-11-24-document-workspace-conversation-sync.md
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
The [implementation plan](../../Docs/superpowers/plans/2026-11-24-document-workspace-conversation-sync.md) is ready for review. The merge preview of Public Dev PR #3534 head 3dc7c90f3a and Dev ee06a040d7 has one conflict in translate.py. Preserve both the merged provider lookup and the tabs PR account check, then verify native document-grounded continuation before releasing sync. ADR-098 remains proposed.
Resolved the Translate import conflict in Dev PR #3534 at commit 76cc496428. Translation and account-scope tests passed (16 backend), along with the tabs and text selection regression suite (35 frontend). The sync branch uses that verified merge as its baseline.
Stage 1 passed 39 integration and schema tests with PostgreSQL required. ChaCha binds chat links to the original native record under owner lock; authorized Media listing hides empty, deleted, and other-account chats.
Stage 2 passed four native turn tests on SQLite and PostgreSQL and 341 focused frontend tests. A linked send rechecks Media access, refuses stale history, and preserves bounded source citations through native settlement and readback.
Browser sidebar integration and per-chat browser save are implemented. 419 focused frontend tests and WebUI typechecking passed. The SQL-backed import replay tests passed on SQLite and PostgreSQL, including citation round-trip and the active Sync v2 refusal. Release qualification is running against real Chromium contexts and real PDF and EPUB files.
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
