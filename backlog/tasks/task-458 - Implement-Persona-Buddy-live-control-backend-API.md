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

Third follow-up code-quality fixes:
- ref-counted live stream presence so overlapping WebSockets on the same session keep lifecycle `connected` until the last socket closes
- changed live-control focus/stop not-owned sessions to owned-only lookup with 404 semantics
- normalized materialized session policy rules through canonical `normalize_policy_rules`
- stopped persisting `companion_activity_surface` into `preferences_json`; the surface remains in the `activity_surface` column and runtime manager preferences
- added regressions for multi-WebSocket presence, not-owned focus/stop, focus/stop preference preservation, materialized policy normalization, and `/persona/sessions` persisted preference surface leakage

Fourth follow-up review fixes:
- redacted `/persona/sessions` and `/persona/sessions/{session_id}` preferences through the public persisted-preference normalizer so live-control focus/idempotency metadata and runtime activity-surface state are not exposed
- resolved live-control requested persona IDs before idempotency/resume lookup so unknown persona fallback reuses the default-backed session consistently
- removed unused live-control `session_manager` parameters from list/focus/stop helpers and route call sites
- added regressions for public session preference redaction, unknown persona idempotency/resume reuse, and WebSocket `user_message` bounded `client_message_id` persistence

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_live_control_api.py tldw_Server_API/tests/Persona/test_persona_sessions.py -q` -> 35 passed, 5 warnings
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/core/Persona/live_control.py -f json -o /tmp/bandit_persona_buddy_live_control_app.json` -> passed

Fifth follow-up review fixes:
- preserved private `persona_live_control` metadata during WebSocket preference persistence while keeping it redacted from public session responses
- changed live-control create/resume to reject explicit unknown persona IDs with 404 while still ensuring the built-in default profile exists when requested
- kept `/persona/sessions` response compatibility for unrelated preference keys while redacting known internal/sensitive fields
- blocked WebSocket user/voice/audio/config work on terminal persona sessions
- made focus clearing update only the target plus previously-focused rows, and paginated idempotency/reuse/focus scans beyond the first page
- made stop idempotent for already-terminal sessions without rewriting archived status

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_live_control_api.py tldw_Server_API/tests/Persona/test_persona_sessions.py -q` -> 40 passed, 5 warnings
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/core/Persona/live_control.py -f json -o /tmp/bandit_persona_buddy_live_control_app.json` -> passed
- `git diff --check` -> passed

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
