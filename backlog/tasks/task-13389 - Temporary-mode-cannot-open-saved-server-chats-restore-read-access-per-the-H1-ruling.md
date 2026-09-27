---
id: TASK-13389
title: >-
  Temporary mode cannot open saved server chats: restore read access per the H1
  ruling
status: To Do
assignee: []
created_date: '2026-09-27 21:18'
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
- [ ] #1 In temporary mode, opening a saved server chat loads and displays its transcript instead of failing with history_owner_unavailable
- [ ] #2 A test proves the temporary load performs zero writes: no IndexedDB mirror, profile, bookmark, or server-side selection write
- [ ] #3 Replies in a temporary session over a loaded saved chat stay memory-only until Task4.2/H3 define persistence (no silent coercion into local durability)
- [ ] #4 Reproduced in a browser (e.g. Firefox private window) before and verified after
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
