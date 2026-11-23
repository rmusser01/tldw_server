---
id: TASK-15621
title: Stop health-poll and notification-heartbeat re-renders of the chat page
status: In Progress
labels:
- webui
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Idle /chat re-mounts message DOM on every connection health poll (30 s connected, 5 s disconnected) because useConnectionState() returns the whole connection state object, which checkOnce replaces at least twice per poll, so Playground and every useConnectionState consumer re-render even when the fields they read did not change. Separately the notification SSE heartbeat (every 10 s) is appended to the NotificationLifecycleProvider events buffer and changes its context value, re-rendering the app shell. Fix: give useConnectionState an optional selector compared with shallow equality, convert the chat-path consumers to select only what they read (derived values where the raw fields change per poll), make the object-returning helper hooks return stable references, and drop heartbeat frames in the provider's onEvent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 useConnectionState(selector) re-renders a consumer only when the selected values change; useConnectionState() still returns the whole state
- [x] #2 Playground, PlaygroundForm, PlaygroundEmpty, usePlaygroundSessionPersistence, useServerChatHistory, ServerChatList, FolderChatList and IndependentBuddyHost select only what they read
- [x] #3 useConnectionActions, useKnowledgeStatus and useConnectionUxState return a stable reference while their values are unchanged
- [x] #4 NotificationLifecycleProvider ignores heartbeat frames (event heartbeat or payload.heartbeat true): no events append and no context change
- [x] #5 New tests fail before the fix and pass after; existing tests of every touched consumer pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: not required. Searched Docs/ADR/README.md; no ADR governs frontend store subscriptions, and this is a render-performance bug fix inside existing hooks with no new durable architecture rule.
Root cause verified: useConnectionState() selected s.state, which checkOnce replaces with an isChecking set and a final set on every poll; while the server is unreachable phase also flips SEARCHING -> ERROR each 5 s poll, so Playground needed the derived readiness boolean, not raw isConnected/phase. Heartbeats: services/notifications.ts resets reconnectAttempt before calling options.onEvent, so filtering in the provider keeps the reconnect reset.
Brief deviation: selections of derived values (Playground catalogConnectionReady, PlaygroundForm connectionUxState/connectionPending, PlaygroundEmpty connectionPending) are undefined under a mock that ignores the selector. Four test files broke that way (Playground.cockpit-shell, Playground.cockpit-controls, Playground.coordinator.integration, PlaygroundEmpty.disconnected); their mocks now apply the selector like the real hook. Raw-field consumers still work with selector-ignoring mocks.
Verification: new tests red then green (useConnectionState.selector 2 failed -> 3 passed; provider heartbeat 2 failed -> 53 passed). Neighbouring tests: 34 shared-UI files 1285 passed; 7 web files 185 passed; 7 helper-hook consumer files 33 passed. eslint: no new warnings or errors on changed files.
PR: https://github.com/rmusser01/tldw_server/pull/3575 (base dev).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
useConnectionState takes an optional shallow-compared selector; the /chat-path consumers select only what they read (Playground, PlaygroundForm and PlaygroundEmpty select derived values); the object-returning helper hooks are shallow-compared; NotificationLifecycleProvider drops SSE heartbeat frames. A health poll or heartbeat no longer re-renders Playground or the notification consumers. Four Playground test mocks now apply the selector.
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
