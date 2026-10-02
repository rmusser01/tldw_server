---
id: TASK-13389
title: >-
  Temporary mode cannot open saved server chats: restore read access per the H1
  ruling
status: Done
assignee: []
created_date: '2026-09-27 21:18'
updated_date: '2026-09-30 04:27'
labels:
  - bug
  - chat
  - webui
  - history-selection
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With temporaryChat on (the default in Firefox private windows), opening a saved server chat fails. useHistorySelection.loadConversation (apps/packages/ui/src/hooks/chat/useHistorySelection.ts:684) returns an unavailable owner with code temporary_history_unavailable before any read, and useServerChatLoader (useServerChatLoader.ts:822) turns that into a failed load (history_owner_unavailable), so the user sees an error instead of the conversation.

The H1 decisions log (Docs/Reviews/CHATBOOK_H1_HISTORY_SELECTION_DECISIONS_2026_09_17.md:103) allows returning unsupported for temporary owners before any IndexedDB/profile/bookmark/native write, but also says to preserve read access and never coerce a temporary request into local durability. The zero-write half is implemented; the read-access half is not. Replying in temporary mode (memory-only pending state) is Task4.2 and temporary fork/Save semantics are H3; this task is only the read path.

Found while fixing Playground coordinator tests on PR #3011; not reproduced in a browser yet (code reading).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 In temporary mode, opening a saved server chat loads and displays its transcript instead of failing with history_owner_unavailable
- [x] #2 A test proves the temporary load performs zero writes: no IndexedDB mirror, profile, bookmark, or server-side selection write
- [x] #3 Replies in a temporary session over a loaded saved chat stay memory-only until Task4.2/H3 define persistence (no silent coercion into local durability)
- [x] #4 Reproduced in a browser (e.g. Firefox private window) before and verified after
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: useServerChatLoader (not useHistorySelection) mishandled the H1 temporary owner. With temporaryChat on, it called loadConversation({temporary:true}), which correctly publishes the zero-write unavailable owner temporary_history_unavailable, but the loader then (a) returned early because settingsMode stayed 'pending' (first open: transcript never rendered, no error), and (b) on every later open hit the 'owner unavailable -> failed history_owner_unavailable' gate before reading.

Fix (apps/packages/ui/src/hooks/chat/useServerChatLoader.ts, commit 3511438237): in temporary mode the loader publishes the temporary owner once and continues with the display-only canonical read (getChat/listChatMessages -> setMessages). useHistorySelection is unchanged, so the controller keeps exposing the unsupported capability (HistorySelectionReview already says 'You can still read this conversation'). syncChatSettingsForServerChat is skipped for temporary reads because it writes local and server settings. The mirror path was already gated on !temporaryChat. A 404/403 on a temporary read now sets serverChatLoadState=failed instead of hanging (rejectCurrentOwner ignored non-native owners). AC3: sends/edits still throw temporary_history_unavailable (normalChatMode, chat-action-utils, messageHandlers, native-history-character-send), so nothing is coerced into local durability. Memory-only replies remain Task4.2/H3.

Tests (apps/packages/ui/src/hooks/__tests__/useServerChatLoader.scope.test.tsx, real useHistorySelection controller): temporary load renders transcript and ends 'loaded' both fresh and after a prior temporary owner; zero calls to ensureLocalProfileId, saveHistoryBookmark, captureHistorySnapshot, ensureServerChatHistoryId, reconcileServerChatMirror, linkServerChatMirror, syncChatSettingsForServerChat; missing chat reports failed/server_chat_not_found. Loader+selection+Playground sibling suites: 13 files, 3 new tests pass; useChatActions.saved-normal has 2 failures that also fail on origin/dev without this change ('deletes a qualified mirror row ... canonical request ID'). tsc --noEmit -p apps/packages/ui: no errors in touched files (pre-existing errors elsewhere). Bandit: N/A (TypeScript only). Docs: none needed; ruling doc already describes the behavior.

Pending: AC #4 browser repro (Firefox private window) left for the orchestrator. Not changed: sidepanel openServerChat opens server chats as a durable tab (temporaryChat:false) by design; toggling temporary off while a temporary owner is published still hits the history_owner_unavailable gate until the next navigation resets the controller (pre-existing, not reproduced).

Browser verification 2026-09-29 (headless Chromium via Playwright; the Chrome extension was not connected). Setup: isolated backend on :8766 (single-user, fresh DBs) plus WebUI next dev on :8080, with a seeded 4-message saved server chat. BEFORE, on dev with 3511438237: in temporary mode, clicking the chat in Recent conversations did nothing. Root cause: ChatSidebar.tsx wrapped the Server/Folders lists in pointer-events-none opacity-50 when temporaryChat was set (a pre-H1 leftover), so the click hit the list container and selectServerChat never ran. The loader fix was unreachable from the UI, and the unit tests drove the loader directly. Control (normal mode) loaded fine. AFTER (38da7aa302): the gate is removed. The same click loads all 4 messages, temporaryChat stays true, historyId stays null, and the H1 notice shows 'You can still read this conversation'. The first after-run also showed one POST /api/v1/rag/feedback/implicit (dwell_time, carrying the message text). Implicit feedback is now disabled in temporary mode at both call sites (useMessageState.ts, Message.tsx), matching the existing canSaveKnowledge gate. Final run: 0 IndexedDB store count changes, 0 non-GET backend requests, and server chat/messages/history-selection byte-identical before and after. Tests: ChatSidebar.lazy-history (temporary mode keeps list clickable) and visual-identity-message-state (implicit feedback off in temporary, on otherwise) were red before the fix and green after. Sidebar suites 34 passed; Playground/__tests__ + useImplicitFeedback + useServerChatLoader.scope 197/198 passed. The 1 failure, Message.dynamic-ui-surface.guard, is pre-existing on dev (it reads the untouched PlaygroundChat.tsx) and is tracked separately.

Qodo on #3064: removing the sidebar gate also re-enabled mutation controls. Temporary mode now keeps saved chats read-only. Rows open the chat, but pin and the actions menu are hidden (ServerChatRow readOnly), bulk selection is off, and the Folders tab stays disabled (folder management). New tests: a real ServerChatList read-only click selects without management buttons; ChatSidebar passes readOnly/selectionMode=false and gates Folders; PlaygroundMessage disables implicit feedback in temporary mode; useImplicitFeedback sends nothing when disabled. Affected suites: 202/203 passed (the 1 failure is pre-existing TASK-13397).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Temporary mode now opens saved server chats read-only end to end. useServerChatLoader publishes the temporary owner and does a display-only canonical read (3511438237). ChatSidebar no longer disables the history lists in temporary mode, and implicit feedback is off there (38da7aa302). Verified in a real browser: the transcript renders with zero IndexedDB writes, zero backend writes, and unchanged server state. Unit tests cover the loader, the sidebar gate and the feedback gate. Pre-existing guard failure tracked as TASK-13397.
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
