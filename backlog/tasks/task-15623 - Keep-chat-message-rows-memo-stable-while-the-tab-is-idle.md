---
id: TASK-15623
title: Keep /chat message rows memo-stable while the tab is idle
status: In Progress
labels:
- webui
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On /chat every PlaygroundChat re-render re-renders every message row: renderBlock passes new inline closures and unstable hook callbacks, so PlaygroundMessage's strict memo comparator never holds. The linked-research poll re-renders PlaygroundChat on every refetch because it reads dataUpdatedAt/errorUpdatedAt, and every row runs useTldwAudioStatus with its own polling observers. Port the TASK-13521 sidepanel stable-handler pattern (not the TASK-13511 handlers TASK-13570 removed), move linked-research failure counting into refetchInterval, and hoist audio status to one PlaygroundChat call passed to rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Re-rendering PlaygroundChat with unchanged messages re-renders zero message rows
- [x] #2 Stable edit handler forwards all four arguments and returns the underlying promise; delete source exposes .current and passes its result through; handlers call the latest closure
- [x] #3 An identical linked-research refetch does not re-render PlaygroundChat and three consecutive errors still switch the poll to 60 s
- [x] #4 Rows given an audioStatus prop call useTldwAudioStatus disabled; PlaygroundChat passes one referentially stable audioStatus
- [x] #5 Existing PlaygroundChat, Message, VirtualChatTimeline edit-retention, TASK-13511 and playground a11y tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no ADR required. Searched Docs/ADR/README.md; this is a render-performance bug fix inside existing components that keeps every action's arguments, return values and target row unchanged. It adds no authority, protocol, storage or dependency. ADR-049 (chat history selection ownership) governs the action sources and is preserved: actions still dispatch through the hook's latest per-render closures, and Delete resolves its source when the confirmation opens.
Relation to TASK-13570: this ports the TASK-13521 side-panel pattern (Sidepanel/Chat/body.tsx), not the TASK-13511 handlers 742cb26271 removed. Edit forwards both editor predicates and returns the hook's promise; Delete is a callable with a .current getter that PlaygroundMessage resolves when its confirmation opens, so the confirmed delete keeps the action live then and its result reaches the row; Stop forwards to the focus-handoff stop (useExplicitStopFocus), never the raw abort. Every wrapper calls the hook's latest per-render action at call time, which is what the always-re-rendered inline closures did before. Handlers are interned by row index with what they capture (isBot, message id) as deps, so each acts on the row as rendered; the cache is never cleared and is bounded by the longest chat shown.
Root cause verified: renderBlock passed fresh closures and per-render hook actions (regenerateLastMessage, stopStreamingFromUi, runContinue/onSubmit, runSteeredContinue, bridged dynamic-UI action, handleVariantSwipe) so the row memo never held; linkedResearchRunsQuery's dataUpdatedAt read re-rendered PlaygroundChat twice per poll even with identical runs. Not changed: the compare-cluster path (PlaygroundCompareCluster still receives raw actions) and totalMessages, which changes for every row when a message is added.
Verification: new PlaygroundChat.row-memo.test.tsx (8 tests) red 6/8 before the fix (rows re-rendered, handlers changed identity, first-render actions called, identical research refetch re-rendered PlaygroundChat 6 vs 4 renders, no audioStatus prop) and green after; Message.memoization.test.tsx audio tests red 2/2 before the Message change, green after. PlaygroundChat.research-status backoff test now feeds refetchInterval the query state as TanStack does. Neighbour run: 78 files / 1892 tests (all PlaygroundChat*, Message*/message*, VirtualChatTimeline.edit-retention, the three dfa4dcc014 test files, every test importing PlaygroundChat, Message or useTldwAudioStatus) green; 42 files / 458 tests (Playground.*, VirtualChatTimeline.*, source guards reading PlaygroundChat.tsx/Message.tsx) green; bun run test:playground:a11y 10 files / 63 tests green. eslint on changed files: 0 errors. Narrow tsc over the changed files: no errors in them.
DoD: no documentation describes this internal render behavior, so none changed; Bandit N/A (TypeScript only); no known skips or blockers.
PR: https://github.com/rmusser01/tldw_server/pull/3576 (base dev).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PlaygroundChat rows now receive stable, ref-backed handlers (TASK-13521 pattern) and one memoized audio status, so a PlaygroundChat re-render with unchanged messages re-renders no rows; the linked-research poll counts failures in refetchInterval from query.state instead of reading dataUpdatedAt/errorUpdatedAt in render, so an unchanged poll no longer re-renders PlaygroundChat; PlaygroundMessage takes an optional audioStatus and keeps its own useTldwAudioStatus disabled when given one.
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
