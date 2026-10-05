---
id: TASK-13454
title: 'Chat UX stage 1: reliable saving and character conversation transitions'
status: To Do
labels:
- chat
- ux
- webui
- extension
- nng
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve findings 1, 2, and 8 through truthful persistence and safe new-chat/character transitions. Reuse or integrate existing fixes after they reach dev; preserve prior chats, drafts, selected ancestry, and ownership checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Empty, temporary, local-acknowledged, server-acknowledged, pending, and failed states describe actual writes; a connected server or bare conversation ID never proves the latest turn was saved.
- [ ] #2 A standard local chat can be left, found, and reopened with its correct messages and configuration through the main conversation UI.
- [ ] #3 Standard local chat to explicit new character conversation to first turn succeeds with resolved provider and a new owner-bound native session; stale callbacks cannot bind the wrong view or account.
- [ ] #4 Unusable model/provider, unsupported context, invalid ownership, and server failures show a reason plus a valid recovery action before or after Send as appropriate, retaining the draft.
- [ ] #5 All new-chat entry points identify Saved, Temporary, or Character behavior consistently; Use no prior messages has an accurate effect and does not promote local history or disable owner guards.
- [ ] #6 An unconfigured first visit names connection/provider/model prerequisites and links to the existing setup UI, preserves the draft during configuration, and reaches a usable first send without introducing a new setup wizard.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Program: TASK-13453. Design: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Base: origin/dev 75ab224081bf140ef52017c1a9b0a04f6878d488, verified 2026-10-04. User approved equal audience priority and all ten findings plus enhancements in five stages. Implementation waits for review of the written design. ADR required: no for the client remediation; ADR-049 and ADR-050 govern owner-bound history and native fork boundaries, ADR-059 governs canonical task editing. Reassess if implementation changes public APIs or persistence authority. Reuse existing shared UI, loaders, presets, and test harnesses; preserve temporary-chat privacy and account/workspace fences. Findings 1, 2, 8. Integrate/reverify #3147 and assess #3165/#3174/#3175 only when their contracts are accepted and merged. Do not implement a second save-status store or implicit local-to-server conversion.
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
