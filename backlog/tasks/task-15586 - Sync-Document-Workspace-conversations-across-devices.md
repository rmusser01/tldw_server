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
- Docs/ADR/100-document-workspace-chat-association.md
- Docs/superpowers/plans/2026-11-24-document-workspace-conversation-sync.md
- https://github.com/rmusser01/tldw_server/pull/3647
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a signed-in user discover and resume the same document conversations from another browser or device.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Browsers signed in to the same account can open the same document and find its synced conversations.
- [x] #2 Conversation discovery uses verified document access and chat ownership, including after switching accounts.
- [x] #3 Existing browser-scoped document conversations remain accessible as the association is reconciled across devices.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Requested as a follow-up to the browser-scoped conversation tabs. Assess the server-owned document-to-chat association and migration behavior during design. ADR-049 governs owner-bound chat history. A durable server-side contract needs a separate ADR assessment.
The approved first-release design covers same-server discovery and a per-chat Save to server action for existing browser-only conversations. The [design spec](../../Docs/superpowers/specs/2026-11-23-document-workspace-conversation-sync-design.md) and [proposed ADR-100](../../Docs/ADR/100-document-workspace-chat-association.md) are ready for review. ADR required: yes. ADR path: Docs/ADR/100-document-workspace-chat-association.md. The association is a durable Chat persistence and Media access contract.
2026-11-24 design review checked the conversation-tabs branch and current Dev. Native history currently refuses document-grounded continuation. A synthetic schema check confirmed native import rejects browser-issued chat IDs and raw source rows. The revised design requires a qualified document-grounded send, stable import recovery with citation preservation, explicit confirmation for numeric-only legacy document links, and access checks at open and send. PR #3534 remains the Public Dev delivery dependency.
The [implementation plan](../../Docs/superpowers/plans/2026-11-24-document-workspace-conversation-sync.md) is ready for review. The merge preview of Public Dev PR #3534 head 3dc7c90f3a and Dev ee06a040d7 has one conflict in translate.py. Preserve both the merged provider lookup and the tabs PR account check, then verify native document-grounded continuation before releasing sync. ADR-100 remains proposed.
Resolved the Translate import conflict in Dev PR #3534 at commit 76cc496428. Translation and account-scope tests passed (16 backend), along with the tabs and text selection regression suite (35 frontend). The sync branch uses that verified merge as its baseline.
Stage 1 passed 39 integration and schema tests with PostgreSQL required. ChaCha binds chat links to the original native record under owner lock; authorized Media listing hides empty, deleted, and other-account chats.
Stage 2 passed four native turn tests on SQLite and PostgreSQL and 341 focused frontend tests. A linked send rechecks Media access, refuses stale history, and preserves bounded source citations through native settlement and readback.
Browser sidebar integration and per-chat browser save are implemented. 419 focused frontend tests and WebUI typechecking passed. The SQL-backed import replay tests passed on SQLite and PostgreSQL, including citation round-trip and the active Sync v2 refusal. Release qualification is running against real Chromium contexts and real PDF and EPUB files.
Local release qualification passed in Chromium against real PDF and EPUB documents. Two independent sign-ins on one account discovered and continued the same chat, retained their local drafts, and reopened saved citations. A different account saw no private chat titles. The browser suite also verified explicit legacy confirmation and a first-chat create replay with the original server ID. Run with TLDW_E2E_DOCUMENT_CHAT_SYNC=1 against a disposable multi-user backend. The regular tier-2 CI job uses single-user auth.
Final focused verification passed 74 SQLite and PostgreSQL API and import tests, 451 frontend tests, WebUI type checking, and the OpenAPI drift check. The backend cases include active Sync v2 admission and create refusal alongside an authorized read. Bandit found zero issues on changed production lines. Public Dev PR #3534 is the queued prerequisite, and release CI and deployment are pending.
The final opt-in Chromium run passed all four cases without skips. After a lost create response, it replayed the original ID, proved that a refused first question left no discoverable chat, then verified that retry saved the question and published exactly one tab.
The [full shared UI typecheck](https://github.com/rmusser01/tldw_server/pull/3607) exposed two inherited Document Workspace message fixtures without source lists. Added empty lists to those fixtures. The shared UI typecheck and 436 focused shared UI tests passed on the preview built from current Dev and PR #3534.
Dev merged [PR #3534](https://github.com/rmusser01/tldw_server/pull/3534) at 42e8a3fcf9 after all eight required checks passed. Rebased TASK-15586 onto that merge. All 492 focused shared UI tests and 74 SQLite and PostgreSQL tests passed on the new base. Shared UI and WebUI type checking and the OpenAPI drift check passed. Bandit reported no findings on changed production lines.
Dev PR #3647 exposed a duplicate ADR number and invalid published links. Assigned the proposed decision ADR-100, updated its links in the spec and plan, and regenerated the public docs from their canonical sources.
Fixed Dev CI size and endpoint-patching checks by reusing the shared compare helpers, grouping the chat actor and optional document dependency, extracting greeting and message access work, and supplying the test Sync profile through the service factory. The generated API diff changes only descriptions on the create-chat and send-message operations. The focused SQLite and PostgreSQL suite passed all 74 cases, and the shared UI lint and import-cycle gates passed locally.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Document Workspace discovers and continues native chats for the verified document and signed-in account. Existing browser histories can be saved individually and recover without creating duplicate server chats. Local verification used real documents, both database backends, and an isolated multi-user browser run. The feature is based on the merged Dev prerequisite and awaits its own Dev PR checks.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
