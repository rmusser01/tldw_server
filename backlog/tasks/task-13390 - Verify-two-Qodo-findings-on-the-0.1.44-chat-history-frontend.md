---
id: TASK-13390
title: Verify two Qodo findings on the 0.1.44 chat-history frontend
status: Done
assignee: []
created_date: '2026-09-27 21:39'
updated_date: '2026-09-28 19:59'
labels:
  - frontend
  - chat
  - qodo
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qodo raised both on #3035, the main-to-dev sync for v0.1.45. The code shipped in 0.1.44 through the Chatbook history/fork work (#2968, #3002). TASK-13264.9 triaged eleven Qodo findings on #3027, but these two are not visibly among them. Neither has been reproduced yet; verify each before fixing.

1. Chat selection spills across instances. requestServerChatSelection writes a transient selection intent to the module-level usePlaygroundSessionStore (apps/packages/ui/src/store/playground-session.tsx ~145-147). Every mounted useServerChatLoader subscribes to it (apps/packages/ui/src/hooks/chat/useServerChatLoader.ts ~712-713), so with several loaders mounted for the same chat, one intent could drive loadConversation in each instance instead of only in the initiating flow.

2. History links fail after account changes. initializePlayground returns early when canAutomaticallyLoad() is false, before it parses an explicit history-selection URL handoff (apps/packages/ui/src/components/Option/Playground/Playground.tsx ~1766-1786; useHistorySelection.ts ~648-653). After an account configuration change invalidates the native-history lease, the handoff is skipped, and the one-shot initialization does not retry it while the page stays mounted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each finding reproduced with a failing test, or disproven with evidence recorded here
- [x] #2 Confirmed defects fixed with regression tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed 2026-09-28. FINDING 2 (history links fail after account changes): REPRODUCED AND FIXED. With canAutomaticallyLoad() false, the state after an account/config change invalidates the native-history lease, Playground's one-shot initializePlayground returned before parsing an explicit historySelection handoff, so the link was dropped while the page stayed mounted. Now the gate applies only when there is no handoff, which matches useServerChatLoader treating deliberate selections as exempt. Regression test: Playground.search.integration.test.tsx 'TASK-13390: an explicit history link still opens...' (red before the fix; the file passes 24/24). FINDING 1 (selection spills across instances): REPRODUCED IN ISOLATION, NOT REACHABLE TODAY. Two useServerChatLoader instances mounted together both call loadConversation for one requestServerChatSelection (repro: 2 calls instead of 1). But only five callers enable the loader (hydrateServerChat: Playground, DocumentChat, WorkspaceChatPanel, ModelPlayground, ResearchWorkspace ChatPane), each on its own route, and ChatWorkspaceConsole renders a single WorkspaceChatPanel, so no screen mounts two. Latent: if a future view co-mounts two enabled loaders, make the intent claim-once. Unrelated: 15 Playground tests in 4 files (research-context, jsx-extension guard, modal footers, locale mirror) fail identically without this change.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
