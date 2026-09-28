---
id: TASK-13389
title: >-
  Temporary mode cannot open saved server chats: restore read access per the H1
  ruling
status: In Progress
assignee: []
created_date: '2026-09-27 21:18'
updated_date: '2026-09-28 00:19'
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
- [ ] #4 Reproduced in a browser (e.g. Firefox private window) before and verified after
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: useServerChatLoader (not useHistorySelection) mishandled the H1 temporary owner. With temporaryChat on, it called loadConversation({temporary:true}), which correctly publishes the zero-write unavailable owner temporary_history_unavailable, but the loader then (a) returned early because settingsMode stayed 'pending' (first open: transcript never rendered, no error), and (b) on every later open hit the 'owner unavailable -> failed history_owner_unavailable' gate before reading.

Fix (apps/packages/ui/src/hooks/chat/useServerChatLoader.ts, commit 3511438237): in temporary mode the loader publishes the temporary owner once and continues with the display-only canonical read (getChat/listChatMessages -> setMessages). useHistorySelection is unchanged, so the controller keeps exposing the unsupported capability (HistorySelectionReview already says 'You can still read this conversation'). syncChatSettingsForServerChat is skipped for temporary reads because it writes local and server settings. The mirror path was already gated on !temporaryChat. A 404/403 on a temporary read now sets serverChatLoadState=failed instead of hanging (rejectCurrentOwner ignored non-native owners). AC3: sends/edits still throw temporary_history_unavailable (normalChatMode, chat-action-utils, messageHandlers, native-history-character-send), so nothing is coerced into local durability. Memory-only replies remain Task4.2/H3.

Tests (apps/packages/ui/src/hooks/__tests__/useServerChatLoader.scope.test.tsx, real useHistorySelection controller): temporary load renders transcript and ends 'loaded' both fresh and after a prior temporary owner; zero calls to ensureLocalProfileId, saveHistoryBookmark, captureHistorySnapshot, ensureServerChatHistoryId, reconcileServerChatMirror, linkServerChatMirror, syncChatSettingsForServerChat; missing chat reports failed/server_chat_not_found. Loader+selection+Playground sibling suites: 13 files, 3 new tests pass; useChatActions.saved-normal has 2 failures that also fail on origin/dev without this change ('deletes a qualified mirror row ... canonical request ID'). tsc --noEmit -p apps/packages/ui: no errors in touched files (pre-existing errors elsewhere). Bandit: N/A (TypeScript only). Docs: none needed; ruling doc already describes the behavior.

Pending: AC #4 browser repro (Firefox private window) left for the orchestrator. Not changed: sidepanel openServerChat opens server chats as a durable tab (temporaryChat:false) by design; toggling temporary off while a temporary owner is published still hits the history_owner_unavailable gate until the next navigation resets the controller (pre-existing, not reproduced).
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
