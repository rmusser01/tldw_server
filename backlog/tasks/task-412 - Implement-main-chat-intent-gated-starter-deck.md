---
id: TASK-412
title: Implement main chat intent-gated starter deck
status: Done
labels:
- webui
- chat
- cockpit
- ux
- frontend
priority: medium
documentation:
- Docs/superpowers/plans/2026-05-16-chat-cockpit-intent-gated-starter-deck.md
related:
- https://github.com/rmusser01/tldw_server/pull/1795
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved main WebUI /chat first-paint density slice. Keep the full starter deck only for a true blank state, collapse or omit it when the user has typed a draft or a conversation exists, and avoid bottom controls, extension sidepanel/sidebar scope, or unrelated pages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Full /chat starter deck remains visible on true blank state with no draft and no conversation.
- [x] #2 Starter deck hides or collapses when the composer has draft text before send.
- [x] #3 Starter deck stays hidden for existing/loaded conversations.
- [x] #4 Clearing the draft before any send can restore the starter deck.
- [x] #5 No bottom bar or composer-adjacent replacement summary is introduced.
- [x] #6 Focused tests cover blank, draft, existing conversation, and restored blank states.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created the implementation plan at Docs/superpowers/plans/2026-05-16-chat-cockpit-intent-gated-starter-deck.md. Scope is main WebUI /chat only, with no sidepanel/sidebar or bottom replacement summary.

Implemented parent-owned starter deck visibility in `Playground.tsx`, a narrow composer draft callback in `PlaygroundForm.tsx`, and a `showStarterDeck` gate in `PlaygroundChat.tsx`. Provider/model availability warnings remain independent of the starter deck gate so chat feedback can still appear when relevant.

Verification:
- Red test: focused cockpit-shell suite failed before implementation on draft, active conversation, and restored-draft starter deck assertions.
- Green tests: `bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/PlaygroundChat.server-load-state.test.tsx src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.signals.guard.test.ts --config vitest.config.ts` passed with 4 files and 22 tests.
- TypeScript: `bunx tsc --noEmit --project tsconfig.json --pretty false` still fails on existing repo-wide baseline; filtered log found no touched-file errors for the Playground files/tests in this task.
- Whitespace: `git diff --check` passed.
- Browser proof: not captured because no tldw_server2 frontend/backend listener was running on the expected local ports. `lsof` showed unrelated tldw_chatbook on `8837` and llama-server on `9099`; no mocked data or alternate fake server was used.
- Bandit skipped because touched runtime scope is frontend TypeScript plus Markdown task/plan files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Main `/chat` starter deck is now intent-gated for the first slice: visible only for true blank state, hidden after unsent draft text or active conversation identity, and restored when draft text is cleared before send. No bottom bar, composer replacement summary, extension sidepanel/sidebar, or unrelated page changes were introduced.
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
