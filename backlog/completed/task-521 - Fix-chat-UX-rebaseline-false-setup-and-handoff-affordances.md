---
id: TASK-521
title: Fix chat UX rebaseline false setup and handoff affordances
status: Done
assignee:
- codex
labels:
- chat
- ux
- webui
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the next small /chat UX rebaseline slice after rail restoration: suppress the false no-provider banner when usable models are present, make chat-title editing self-explanatory/accessibility-safe, and retarget the sidepanel chat dashboard handoff to /chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
['Observed on current branch with backend http://127.0.0.1:18016 and frontend http://127.0.0.1:18015: rails present, CharacterControlRail absent, but /chat showed No LLM provider configured while runtime/status showed tldw:gemma3:1b ready and provider status API reported local providers configured.', 'Observed mobile title click turned Untitled into a blank unlabeled textbox in the chat header.', 'Source evidence: SidepanelHeaderSimple openFullScreen uses /options.html#/chat, but openDashboard still uses /options.html#/flashcards.']
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the false no-provider /chat setup banner when usable chat models are present, added an accessible label and Untitled placeholder to the chat title editor, and retargeted the sidepanel chat dashboard button to /chat. Verification: focused Vitest suite passed (19 tests), git diff --check passed, live browser verified the no-provider banner removal and title editor label. TypeScript was run and failed only on the pre-existing CharacterListContent GalleryCardDensity baseline issue. Bandit skipped because this slice touched only TS/TSX and Backlog files.
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
