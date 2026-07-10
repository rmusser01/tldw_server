---
id: TASK-12108
title: Add persistent HttpOnly sessions for same-origin single-user WebUI auth
status: In Progress
priority: High
references:
- TASK-12106
- TASK-12030
- TASK-12127
- https://github.com/rmusser01/tldw_server/issues/2590
documentation:
- Docs/superpowers/specs/2026-07-10-single-user-http-only-session-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace browser-visible runtime API-key provisioning in quickstart same-origin deployments with a backend-issued persistent HttpOnly session cookie, including CSRF protection, API-key-rotation invalidation, logout revocation, and browser relaunch coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Same-origin single-user WebUI runtime bootstrap establishes authentication without returning the API key to browser JavaScript.
- [ ] #2 The persistent session cookie is HttpOnly, host-only, SameSite=Lax, Secure outside explicit loopback HTTP development, and has a bounded lifetime.
- [ ] #3 Cookie-authenticated state-changing requests require the existing double-submit CSRF token while header-authenticated API clients remain unaffected.
- [ ] #4 A single-user API-key rotation invalidates previously issued cookie sessions.
- [ ] #5 Logout revokes the current server-side session and clears the session cookie.
- [ ] #6 Regression coverage includes hard reload and close/reopen of the same browser profile without API-key re-entry.
- [ ] #7 Runtime bootstrap fails closed and does not fall back to exposing or persisting the API key in browser JavaScript.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Hybrid design selected: reuse the existing AuthNZ SessionManager/sessions table for a high-entropy opaque single-user session, bind each record to a fingerprint of the current configured API key, authenticate same-origin WebUI requests through an HttpOnly host-only cookie, and reuse the existing double-submit CSRF middleware. The Next runtime endpoint becomes non-secret and a separate same-origin POST route performs the server-to-server API-key exchange. Design: Docs/superpowers/specs/2026-07-10-single-user-http-only-session-design.md
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
