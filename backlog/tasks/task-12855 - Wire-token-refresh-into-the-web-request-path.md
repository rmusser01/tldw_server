---
id: TASK-12855
title: Wire token refresh into the web request path
status: Done
labels:
- bug
- high
- auth
- frontend
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: High (multi-user sessions break on token expiry).** From the 2026-07-02 frontend audit (finding H4).

Refresh-and-retry exists (`apps/packages/ui/src/services/tldw/request-core.ts:470-514`) but requires `runtime.refreshAuth`, which is wired **only** in the extension background (`entries/background.ts:1248-1263`). The web (non-extension) direct fallback passes only `{ getConfig }` (`services/background-proxy.ts:835-847`, also `:582,635,1504`), so no refresh happens in the browser. `TldwAuth`'s pre-expiry timer is armed only inside `login`/`verifyMagicLink`/`refreshToken` (`TldwAuth.ts:384-401`) and is discarded on page reload.

Result: a multi-user user who logs in and reloads will, on access-token expiry, have every shared-stack request 401 and the UI misreport "backend unavailable" (via `notifyBackendUnavailable`) while a valid refresh token sits unused in `tldwConfig`. Recovery requires manual re-login.

Related (fold in): `TldwAuth.refreshToken()` (`:231-261`) has no single-flight guard outside the extension worker, so the UI auto-refresh timer can race a 401 refresh and persist a rotated/dead token (relevant once web refresh is wired). The post-refresh retry also re-serializes the body with `JSON.stringify` (`request-core.ts:497-501`), turning a `FormData`/`Blob` upload into `"{}"` — fix while here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The web direct fallback in `background-proxy.ts` supplies a working `refreshAuth` so `request-core.ts` refresh-and-retry runs in the browser.
- [ ] #2 `refreshAuth` is single-flighted (concurrent 401s trigger one refresh) in both web and extension contexts.
- [ ] #3 The refresh timer is (re-)armed on page load when a valid refresh token is present.
- [ ] #4 The post-refresh retry preserves binary bodies (`FormData`/`Blob`) instead of `JSON.stringify`-ing them.
- [ ] #5 A test simulates access-token expiry mid-session and asserts a transparent refresh + retry (no "backend unavailable", no forced re-login).
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
