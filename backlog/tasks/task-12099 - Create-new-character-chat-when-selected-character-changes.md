---
id: TASK-12099
title: Create new character chat when selected character changes
status: Done
labels:
- codex
- bug
- webui
- chat
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2573
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix WebUI/extension chat behavior so selecting a different tracked character while a character chat is loaded clears the prior server chat and starts a new conversation for the selected character instead of keeping the old conversation active.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a Playground reset effect that checks loaded server-chat metadata (serverChatId + serverChatMetaLoaded + serverChatCharacterId) against the current selected tracked character. When they differ, the page clears the local history id, visible history/messages, server chat metadata, serverChatId, and persisted session state, so the next message starts a fresh conversation for the selected character. Added a cockpit shell regression test for switching from character A to character B while chat A is loaded. Verification: bun run test src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx passed (36 tests). git diff --check passed. Bandit not applicable because only frontend TypeScript and Backlog task files were touched.
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
