---
id: TASK-12108
title: Add persistent HttpOnly sessions for same-origin single-user WebUI auth
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-07-10 22:44
labels: []
dependencies: []
references:
- TASK-12106
- TASK-12030
- TASK-12127
- https://github.com/rmusser01/tldw_server/issues/2590
documentation:
- Docs/superpowers/specs/2026-07-10-single-user-http-only-session-design.md
- docs/superpowers/plans/2026-07-10-single-user-http-only-session-implementation-plan.md
- .superpowers/sdd/http-task-2-report.md
- .superpowers/sdd/http-task-3-report.md
- .superpowers/sdd/http-task-4-report.md
priority: high
modified_files:
- tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py
- tldw_Server_API/tests/AuthNZ/unit/test_auth_principal_service_and_single_user_tokens.py
- tldw_Server_API/app/core/AuthNZ/websocket_session_auth.py
- tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py
- tldw_Server_API/app/api/v1/endpoints/acp_multiplex.py
- tldw_Server_API/app/api/v1/endpoints/persona.py
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/api/v1/endpoints/workflows.py
- tldw_Server_API/app/api/v1/endpoints/meetings.py
- tldw_Server_API/app/api/v1/API_Deps/Meetings_DB_Deps.py
- tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_websocket.py
- tldw_Server_API/app/api/v1/endpoints/sandbox.py
- tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py
- tldw_Server_API/app/core/MCP_unified/server.py
- tldw_Server_API/app/api/v1/endpoints/voice_assistant.py
- tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py
- tldw_Server_API/app/core/Audio/streaming_service.py
- tldw_Server_API/tests/AuthNZ/unit/test_websocket_session_auth.py
- tldw_Server_API/tests/AuthNZ/test_websocket_cookie_route_contract.py
- tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py
- tldw_Server_API/tests/sandbox/test_ws_signed_validation.py
- apps/tldw-frontend/pages/api/_tldw-webui/runtime-auth-policy.ts
- apps/tldw-frontend/pages/api/_tldw-webui/session.ts
- apps/tldw-frontend/pages/api/_tldw-webui/runtime-config.ts
- apps/tldw-frontend/scripts/validate-networking-config.mjs
- apps/tldw-frontend/__tests__/pages/api/runtime-config.test.ts
- apps/tldw-frontend/__tests__/pages/api/runtime-session.test.ts
- apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts
- Dockerfiles/docker-compose.yml
- Dockerfiles/docker-compose.single-user.yml
- Dockerfiles/docker-compose.host-storage.yml
- Dockerfiles/docker-compose.webui.yml
- tldw_Server_API/app/core/AuthNZ/settings.py
- tldw_Server_API/tests/AuthNZ/unit/test_settings_single_user_session_cookie.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace browser-visible runtime API-key provisioning in the runtime-enabled loopback quickstart WebUI with a backend-issued persistent HttpOnly session cookie, including CSRF protection, API-key-rotation invalidation, logout revocation, WebSocket compatibility, and browser relaunch coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Runtime-enabled loopback quickstart WebUI bootstrap authenticates without returning the API key to browser JavaScript.
- [ ] #2 The persistent session cookie is HttpOnly, host-only, SameSite=Lax, Secure outside explicit loopback HTTP, and has a bounded lifetime.
- [ ] #3 Cookie-authenticated state-changing requests require the existing double-submit CSRF token while header-authenticated API clients remain unaffected.
- [ ] #4 A single-user API-key rotation invalidates previously issued cookie sessions after settings reload/process restart.
- [ ] #5 Logout revokes the current server-side session and clears the session cookie.
- [ ] #6 Regression coverage includes hard reload, close/reopen of the same browser profile, and representative same-origin WebSockets without API-key re-entry.
- [ ] #7 Runtime bootstrap fails closed and does not fall back to exposing or persisting the API key in browser JavaScript.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: docs/superpowers/plans/2026-07-10-single-user-http-only-session-implementation-plan.md. Stages: opaque session primitive; HTTP principal/CSRF/endpoints; shared WebSocket auth; non-secret Next runtime bootstrap; cookie-mode WebUI clients; lifecycle/deployment/security verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Hybrid design: reuse the existing AuthNZ SessionManager/sessions table for a high-entropy opaque single-user session, authenticate loopback quickstart WebUI requests through an HttpOnly host-only cookie, and reuse the existing double-submit CSRF middleware. Independent review added shared HTTP/WebSocket cookie-principal resolution, exact trusted-Origin WebSocket checks, upgraded-profile key scrubbing, and CSRF_BIND_TO_USER pre-resolution. Final pre-implementation corrections: do not path-exclude session mint; fail closed if effective CSRF is disabled; use a constant single-user-cookie:v1 type tag and existing API-key-derived HMAC token hashes for rotation; retain the current quickstart/loopback/no-forwarding guard; activate atomically across HTTP, CSRF, and WebSockets; preserve multiple Set-Cookie headers with getSetCookie semantics. Design: Docs/superpowers/specs/2026-07-10-single-user-http-only-session-design.md
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 4 remaining medium finding fixed from reviewed base 35461f5743. WebUI and backend Settings now apply the same cookie-name contract: default only when absent; reject explicit empty, invalid token syntax, exact csrf_token, and case-insensitive __Host-/__Http-/__Secure- prefixes; accept valid custom names including case-distinct CSRF_TOKEN. Compose `${VAR:-default}` behavior remains intentionally unchanged. TDD: frontend RED 8/114 pass plus 2-case __Http- RED, then GREEN 155/155; backend RED 9/3 pass plus 1-case __Http- RED, then GREEN 21/21. ESLint, strict targeted tsc, Python compile, Bandit (0 findings), and diff check all exit 0. Full evidence appended to `.superpowers/sdd/http-task-4-report.md`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
