---
id: TASK-462
title: Add Persona Buddy interaction browser coverage
status: Done
labels:
- persona
- buddy
- frontend
- e2e
references:
- TASK-457
- TASK-461
- Docs/superpowers/plans/2026-05-20-persona-buddy-interaction-text-slice.md
- Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md
- 'issue #1510'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 5 from the Persona Buddy interaction text-slice plan: Playwright coverage for opening the Buddy shell, sending text through the Persona stream, and routing to Visuals.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 E2E test mocks live session list/create/focus/stop and visual-pack responses.
- [x] #2 E2E test captures Persona stream WebSocket user_message payload with session_id, client_message_id, and text.
- [x] #3 E2E test verifies Choose/Change Buddy routes to the Visuals workflow.
- [x] #4 Focused Playwright command passes or any environment blocker is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `apps/tldw-frontend/e2e/workflows/persona-buddy-interaction.spec.ts` with mocked Persona catalog/profile/visual-pack/live-control endpoints and a mocked Persona stream WebSocket.
- The test opens `/persona`, opens the Buddy dock popover, sends text through the compact composer, asserts the captured `user_message` contains `session_id`, `client_message_id`, and `text`, then follows the Buddy Visuals route.
- Default-port Playwright initially hit local sandbox binding/stale-server issues, so verification used an isolated worktree server on `127.0.0.1:18080`.
- Bandit is not applicable for this frontend-only Playwright coverage slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 5 browser coverage is complete for the first Persona Buddy text interaction slice. The focused Playwright spec passes against an isolated Next dev server and covers live-control mocks, Persona stream payload capture, and Choose/Change Buddy routing to the Visuals tab.

Draft PR: https://github.com/rmusser01/tldw_server/pull/1901.
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
