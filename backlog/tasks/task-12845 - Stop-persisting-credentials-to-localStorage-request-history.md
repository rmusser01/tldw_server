---
id: TASK-12845
title: Stop persisting credentials to localStorage request-history
status: Done
labels:
- bug
- critical
- security
- frontend
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: Critical (credential exposure at rest).** From the 2026-07-02 frontend audit (finding C2).

`apps/tldw-frontend/lib/api.ts:468` runs `applyBrowserHeaders` (adds `Authorization: Bearer <JWT>`, `X-API-KEY`, `X-CSRF-Token`), then `:470` passes those headers into `buildRequestHistoryConfig`. `recordSuccess` (`:384-404`) writes `requestHeaders` **and** `responseBody` into `localStorage['tldw-request-history']` (200-entry ring, `lib/history.ts:16-28`) with no redaction. For a `/auth/login` response, `responseBody` includes the `access_token`.

`clearRequestHistory` (`lib/history.ts:41`) exists but is **never called** anywhere, and logout (`lib/auth.ts:203-213`, `TldwAuth.logout`) does not touch the key. Result: bearer tokens and API keys sit in plaintext localStorage indefinitely and survive logout — exfiltratable by any XSS (see TASK-12093) or anyone with access to the machine/profile.

Verified by direct read of `lib/history.ts` and `lib/api.ts:366-404,455-554`; confirmed `recordSuccess` is called on the success path (`:554`) and `clearRequestHistory` has zero call sites.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Auth-bearing headers (`authorization`, `x-api-key`, `x-csrf-token`, org headers) are redacted before being written to request-history.
- [ ] #2 Response bodies for auth endpoints (at minimum `/auth/login`, `/auth/refresh`, `/auth/magic-link`) do not persist tokens to request-history (redact `access_token`/`refresh_token` or skip body capture for these routes).
- [ ] #3 `clearRequestHistory()` is invoked on logout, and the history key is cleared.
- [ ] #4 A test asserts that after a login + an authenticated request, `localStorage['tldw-request-history']` contains no bearer token, API key, or CSRF token.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
