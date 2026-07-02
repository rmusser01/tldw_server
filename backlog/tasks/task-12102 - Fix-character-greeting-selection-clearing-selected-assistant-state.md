---
id: TASK-12102
title: Fix character greeting selection clearing selected assistant state
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-02 16:37'
labels:
  - webui
  - chat
  - characters
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Using the chat greeting selector after opening a character chat can leave the greeting rendered while the cockpit/runtime assistant state reverts to no selected character. Trace root cause, add regression coverage, and keep Miku/character state selected after greeting selection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening a character route must not restore a stale persisted chat before applying the requested character.
- [x] #2 Opening a character route over an active server chat must clear the old conversation and select the requested character.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: route character hydration could be blocked by persisted-session restore or by an existing server chat/history state, leaving the route/greeting UI visible while selectedCharacter/server assistant metadata stayed null or stale.

Fix: let explicit character routes take precedence over persisted restore, clear active conversation/server-chat metadata when a bare character route is opened, and then fetch/apply the route character. Draft-only composer state is no longer treated as an active conversation for route hydration, so a typed draft cannot keep invalidating the character fetch.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed explicit character-route handling so selecting/opening a new character starts a fresh character chat instead of keeping a stale chat loaded. Added coordinator coverage for persisted-session precedence and active-server-chat replacement.

Verification:
- bunx vitest run src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx (38 tests passed)
- npx playwright test --config=/private/tmp/tldw-miku-storage-probe.config.ts --project=chromium --reporter=line
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r apps/packages/ui/src/components/Option/Playground/Playground.tsx apps/packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx -f json -o /tmp/bandit_task_12102.json (0 findings; TSX parser errors recorded because Bandit is Python-only)
<!-- SECTION:FINAL_SUMMARY:END -->

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
