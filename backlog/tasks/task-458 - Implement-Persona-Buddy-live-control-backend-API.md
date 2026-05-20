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
Implemented backend live-control API slice in progress.

Verification (using parent checkout venv because this worktree has no `.venv`):
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_live_control_api.py -q` -> 15 passed, 5 warnings
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_sessions.py -q` -> 8 passed, 5 warnings
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/app/core/Persona/session_materialization.py tldw_Server_API/app/core/Persona/live_control.py -f json -o /tmp/bandit_persona_buddy_live_control.json` -> passed, JSON written
- `git diff --check` -> passed

Note: exact command prefix `source .venv/bin/activate` fails in this worktree because `.venv/bin/activate` is absent; the reusable project venv exists at `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate`.

Implemented:
- live-control schemas and validators
- shared session materialization helper reused by `/persona/session` and live-control create/resume
- live-control service for list/create/focus/stop, stream presence, idempotency, focus metadata, lifecycle/actions/capabilities, and redacted summaries
- FastAPI live-control routes
- WebSocket stream presence registry updates and `client_message_id` turn metadata

Follow-up review fixes:
- preserved bounded `client_message_id` metadata through manual and auto `voice_commit` paths
- marked WebSocket stream presence when `voice_config` and `audio_chunk` resolve a known session
- normalized omitted live-control create/resume surface to the default surface before reuse/idempotency lookup
- added regressions for default-surface reuse and voice commit `client_message_id` propagation

Second follow-up coverage fixes:
- added endpoint-level `/api/v1/persona/stream` regression coverage for `voice_commit` preserving bounded `client_message_id` in turn metadata
- added endpoint-level stream-presence regressions for `user_message` and `voice_config` paths, including connected lifecycle while open and idle lifecycle after WebSocket cleanup
- focused live-control test file now covers 20 tests

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
