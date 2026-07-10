---
id: TASK-12108
title: Add persistent HttpOnly sessions for same-origin single-user WebUI auth
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-10 22:44'
labels: []
dependencies: []
references:
  - TASK-12106
  - TASK-12030
  - TASK-12127
  - 'https://github.com/rmusser01/tldw_server/issues/2590'
documentation:
  - Docs/superpowers/specs/2026-07-10-single-user-http-only-session-design.md
  - >-
    docs/superpowers/plans/2026-07-10-single-user-http-only-session-implementation-plan.md
priority: high
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
