---
id: TASK-458
title: Implement Persona Buddy live-control backend API
status: In Progress
labels:
- persona
- buddy
- backend
- implementation
references:
- TASK-457
- Docs/superpowers/plans/2026-05-20-persona-buddy-interaction-text-slice.md
- Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md
- 'issue #1510'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 1 from the Persona Buddy interaction text-slice plan: backend live-control schemas, shared session materialization, live-control service, FastAPI routes, WebSocket stream-presence/client_message_id metadata, and focused backend tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Backend live-control schemas are added without changing existing /session, /sessions, or /stream response shapes.
- [ ] #2 Existing Persona session materialization is extracted and reused by /persona/session and live-control create/resume.
- [ ] #3 Live-control list/create/focus/stop routes are authenticated, user-scoped, and tested.
- [ ] #4 Lifecycle, terminal allowed-actions, focus single-winner behavior, stream presence, idempotency, and redaction tests are covered.
- [ ] #5 Focused backend tests and git diff checks are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-20-persona-buddy-interaction-text-slice.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
