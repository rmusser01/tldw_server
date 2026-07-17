# Same-Origin Single-User HttpOnly Session Design

**Backlog:** TASK-12108
**Related:** TASK-12106 (remote WebUI and extension device persistence)
**Status:** Proposed for specification review
**Date:** 2026-07-10

## Problem

The quickstart WebUI currently receives the configured single-user API key from a Next.js runtime-config endpoint and installs it into browser JavaScript so requests can send `X-API-KEY`. This removes repeated manual entry, but it gives every script running in the page access to a reusable server credential and relies on client storage behavior for continuity.

For the existing runtime-enabled loopback quickstart, where the WebUI and API are presented through one browser origin, the server can provide the same low-friction experience without exposing the API key: exchange the server-side runtime key for a bounded, revocable, opaque session stored in an HttpOnly cookie.

## Goals

- A runtime-enabled loopback quickstart WebUI authenticates without API-key entry.
- The configured API key is never returned to browser JavaScript, browser storage, logs, or diagnostics.
- Authentication survives hard reloads and closing/reopening the same browser profile.
- Sessions are bounded, revocable, and invalidated by single-user API-key rotation.
- Cookie-authenticated mutations use the existing double-submit CSRF defense.
- Existing `X-API-KEY` clients continue working without cookie or CSRF requirements.
- Failure is closed: session-bootstrap failure never falls back to exposing the key.

## Non-Goals

- Using cookies for a manually configured cross-origin WebUI or browser extension. Host-only cookies cannot provide those surfaces with a reliable shared auth context.
- Replacing multi-user JWT/refresh-token authentication.
- Creating a new general-purpose identity provider.
- Making a network-exposed single-user WebUI private by itself. Enabling automatic runtime auth means anyone allowed to reach that WebUI receives the single-user session; deployment network controls remain part of the security boundary.
- Infinite sessions. “Remember” means a bounded persistent session, not a credential that never expires.
- Expanding automatic session provisioning to LAN hosts or reverse-proxied deployments. Those environments require an explicit trusted-proxy, public-origin, TLS, and network-access policy before they can safely opt in.

## Chosen Architecture

Use a server-side opaque session backed by the existing AuthNZ `sessions` table and `SessionManager`:

1. Browser JavaScript reads non-secret runtime capability metadata.
2. When cookie-session auth is available, it sends a same-origin `POST /api/_tldw-webui/session` with credentials included.
3. The Next.js route validates the deployment/runtime-auth guard and the browser request origin.
4. The route calls the internal backend `POST /api/v1/auth/single-user/session` with the configured API key in a server-to-server `X-API-KEY` header. It forwards the browser's current session cookie when present.
5. The backend reuses a valid matching session or creates random opaque access and refresh values, persists their hashes/encrypted values through `SessionManager`, and returns an HttpOnly session cookie.
6. The Next.js route forwards `Set-Cookie` to the browser without exposing the token in a JSON body.
7. Same-origin API requests rely on the browser cookie. Remote/manual and extension requests continue using the explicit API-key persistence design in TASK-12106.

The cookie value is a high-entropy opaque random token, not the single-user API key and not a JWT. The API key remains a provisioning credential used only on the server-to-server bootstrap hop and by explicit API clients.

The same-origin deployment must proxy both HTTP requests and WebSocket upgrades under the WebUI origin. The implementation verifies the existing quickstart rewrite in the standalone runtime with an upgrade integration test. If that runtime does not preserve WebSocket upgrades, the deployment layer must add an explicit same-origin WebSocket proxy before cookie-session auth is enabled; connecting directly to a different backend origin is not an acceptable fallback.

## Session Record and Key-Rotation Binding

Create two independent random values for each session so existing access/refresh hash uniqueness constraints remain valid. Set both expiries to the single-user session lifetime; the refresh value remains server-side and reserved for safe future rotation.

Tag the session record's existing `device_id` field with the constant versioned type marker `single-user-cookie:v1`. Do not store an API-key fingerprint.

`SessionManager` already indexes session tokens with HMAC keys derived from the current single-user API key. After an API-key change and settings reload/process restart, old token hashes no longer resolve. This provides key-rotation invalidation without adding a second verifier or a database value that could serve as an offline API-key-guessing oracle.

Cookie validation requires all of the following:

- `AUTH_MODE == "single_user"`.
- `SessionManager.validate_session(cookie_token)` returns an active, non-expired record.
- The record user matches the canonical single-user account.
- The record `device_id` equals the `single-user-cookie:v1` type marker.

An API-key change therefore invalidates old sessions when the new settings become active, normally after the required service restart. A wrong-type session is treated as unauthenticated and may be revoked best-effort.

The session-validation repository query must return `device_id`; no new table or migration is required.

## Cookie Contract

Add dedicated single-user session settings rather than overloading the CSRF cookie:

- Name: `tldw_single_user_session` by default.
- `HttpOnly=true` unconditionally.
- Host-only: omit `Domain`.
- `Path=/api`.
- `SameSite=Lax` by default.
- `Secure` follows `SESSION_COOKIE_SECURE`. The loopback HTTP quickstart explicitly sets it false; HTTPS deployments keep it true. Never infer this from untrusted forwarding headers.
- Persistent `Max-Age`/`Expires`, default 30 days, matching the server-side access expiry.

The backend returns only non-secret metadata such as `{ authenticated: true, expires_at }`. Responses use `Cache-Control: no-store`.

Deleting the cookie repeats the same Path/SameSite/Secure attributes needed for reliable removal.

## Backend Endpoints

### `POST /api/v1/auth/single-user/session`

- Available only in single-user mode.
- Requires a valid `X-API-KEY`; it is excluded from cookie fallback so an existing cookie cannot mint replacement sessions by itself.
- If a valid matching session cookie is supplied, returns it without creating another database row.
- Reusing a session does not reset the browser cookie beyond the record's existing server-side expiry. It may omit the auth `Set-Cookie` header entirely.
- Otherwise creates an opaque server-side session and sets the cookie.
- Uses the normal single-user IP allowlist and request audit context.
- Never includes the API key or opaque cookie value in the response body.

### `DELETE /api/v1/auth/single-user/session`

- Authenticates from the session cookie.
- Revokes that exact session by ID and deletes the cookie.
- Is CSRF protected.
- Is idempotent from the browser's perspective: an absent/invalid session still returns a cleared cookie and a non-secret logged-out result.
- The idempotent result is evaluated after the CSRF gate when a session cookie is present. A caller with a stale cookie but no CSRF token first performs a safe GET/bootstrap to obtain a token; the server does not weaken logout CSRF protection merely to clear an already unusable cookie.

The general `/api/v1/auth/logout` endpoint may delegate to the same current-session revocation helper when the authenticated credential came from this cookie, but cookie cleanup remains explicit in the response.

## Principal Resolution

Extend the unified principal resolver with a narrowly scoped cookie path:

1. Preserve current precedence for explicit `Authorization` and `X-API-KEY` headers.
2. Only when both headers are absent and auth mode is single-user, read the configured single-user session cookie.
3. Validate the opaque session and its constant single-user-cookie type marker.
4. Build the existing canonical single-user `AuthPrincipal` with a distinct token/source marker such as `single_user_session`.
5. Attach the validated session ID to request state for exact logout and auditing.

Explicit header credentials always win. Cookie extraction is never enabled for multi-user mode in this change.

Put opaque cookie validation and principal construction in one reusable AuthNZ helper. HTTP principal resolution, WebSocket principal resolution, CSRF pre-resolution, bootstrap reuse, and exact-session logout call this helper rather than implementing separate cookie checks.

## WebSocket Authentication

Browser WebSocket APIs cannot set `Authorization` or `X-API-KEY` headers. Existing first-party clients therefore put the API key or token in query parameters. Cookie-session mode must remove that browser-visible secret without regressing live features.

Add a shared WebSocket principal resolver with this precedence:

1. Preserve explicit header, subprotocol, and legacy query credentials for API clients, remote WebUI mode, extension mode, and multi-user compatibility.
2. Only when no explicit credential is supplied, auth mode is single-user, and the dedicated session cookie is present, validate the opaque session through the shared helper.
3. Before accepting cookie auth, require a non-null HTTP(S) `Origin` that exactly matches a configured trusted WebUI origin. Reuse the normalized runtime allowed-origin/public-WebUI configuration; wildcard origins never authorize cookie WebSockets.
4. Reject malformed, `null`, cross-origin, unconfigured reverse-proxy, expired, revoked, or key-mismatched cookie connections with code `4401`/`4403` before `accept()`.
5. Attach the canonical principal and session ID to `websocket.state` for ownership, quota, and audit code.

All first-party WebSocket authentication entry points reachable from the WebUI must delegate to this resolver, including representative persona, ACP, audio/voice, meetings, workflows/watchlists, prompt-studio, sandbox, and MCP transports. Endpoint-specific scope and ownership checks still run after authentication. Add a route-audit contract test so a newly added first-party WebSocket cannot silently omit the shared cookie-auth path.

In cookie-session mode, first-party WebSocket URL builders:

- Resolve to the same WebUI `ws:`/`wss:` origin and the existing `/api/...` proxy path.
- Omit `api_key` and `token` query parameters.
- Rely on the browser to attach the host-only HttpOnly cookie.

Remote/manual WebUI and extension builders retain their explicit credential behavior because their WebSocket origin does not share the WebUI cookie.

## Runtime Bootstrap

The existing runtime-config endpoint becomes non-secret. Its successful payload reports capability only for the existing runtime-enabled loopback quickstart guard:

```json
{
  "runtimeAuth": {
    "available": true,
    "authMode": "single-user",
    "transport": "cookie-session"
  },
  "networking": {
    "deploymentMode": "quickstart",
    "serverUrl": ""
  }
}
```

It never contains `apiKey`.

Add a separate POST bootstrap route rather than creating sessions from a GET. The Next route reuses the existing runtime-auth deployment guard and additionally:

- Requires a same-origin `Origin` matching the request Host for browser POSTs.
- Rejects cross-site Fetch Metadata when present.
- Uses only a validated `TLDW_INTERNAL_API_ORIGIN` as its backend target.
- Sends the API key only to that fixed internal origin.
- Forwards only the known session and CSRF cookies, not arbitrary browser authorization headers.
- Copies only the expected authentication-session and `csrf_token` `Set-Cookie` headers, content type, cache, and status data back to the browser. Use the runtime's `Headers.getSetCookie()` support (with a tested equivalent only if unavailable) so multiple cookie headers are never comma-folded.
- Retains the existing quickstart, loopback host/peer, and no-forwarding-header restrictions. A same-origin URL alone does not make a non-loopback deployment trusted.

Runtime bootstrap failure reports a generic unavailable/authentication error and never falls back to the old API-key response.

## Client Request Flow

On startup:

1. Fetch non-secret runtime config.
2. If `transport == "cookie-session"`, POST the same-origin bootstrap route with credentials included.
3. Mark the in-memory connection source as cookie session without populating `apiKey`.
4. Probe the normal authenticated profile endpoint using the ambient cookie.
5. Clear any legacy runtime-owned key copies only after the cookie-authenticated probe succeeds.

After that successful probe, perform a fail-closed secret scrub for the same-origin runtime context: remove all legacy runtime/session bridge slots and every `tldwConfig.apiKey` or session record that lacks complete explicit manual source, persistence scope, normalized remote origin, and matching connection metadata. A complete manual/device credential may remain atomically in `tldwConfig`; a complete manual/session credential may remain in its session store. Missing, malformed, partially written, or contradictory ownership metadata never preserves a browser-readable key. Preserve non-secret connection metadata. This scrub is idempotent and runs before publishing cookie-session readiness.

For same-origin requests, the shared request client:

- Does not require or attach `X-API-KEY` when the active source is cookie session.
- Uses same-origin credentials (explicitly stated in fetch options for clarity and tests).
- Reads the existing non-HttpOnly `csrf_token` cookie and echoes it as `X-CSRF-Token` on state-changing requests.
- Never attaches the CSRF header to cross-origin/manual requests unless that transport explicitly requires it.

If session bootstrap or the authenticated probe fails, preserve any pre-existing manual remote configuration and show a connection error. Do not expose, persist, or reconstruct the runtime API key.

## CSRF

The existing `CSRFProtectionMiddleware` already implements the double-submit cookie pattern and skips requests carrying explicit API-key or Bearer headers. Extend its activation rules so single-user cookie sessions are protected:

- Enable the middleware in single-user mode.
- In single-user mode, require CSRF only when the dedicated session cookie is present; public/headerless endpoints without that cookie retain existing behavior.
- Do not add the session path to the middleware's path exclusions. The mint POST is already exempt because its server-to-server request carries `X-API-KEY`; keeping the path protected ensures DELETE logout still requires CSRF.
- Protect session deletion and all other state-changing cookie-authenticated API calls.
- When `CSRF_BIND_TO_USER=true`, the middleware's pre-dependency user resolver validates the session cookie through the shared helper and recovers the canonical user ID. Bound-cookie validation must not depend on an endpoint dependency that runs later.
- Centralize effective-CSRF-state resolution so middleware setup and session issuance cannot disagree. If CSRF is explicitly disabled, the backend refuses to mint cookie sessions and bootstrap fails closed. Tests that exercise cookie sessions explicitly enable CSRF.

The readable CSRF cookie is expected; the authentication session cookie remains HttpOnly.

## Expiry, Reload, and Relaunch

- Hard reload reuses the cookie and the bootstrap route reuses the existing valid session rather than inserting a duplicate.
- Closing and reopening the same browser profile preserves the persistent cookie until expiry.
- A fresh browser profile has no session and receives a new one only through runtime-enabled same-origin bootstrap.
- Expired, revoked, corrupted, or key-mismatched cookies produce 401 and are cleared when practical.
- The initial release uses a fixed 30-day lifetime. Rolling extension and background refresh are out of scope; a later bootstrap after expiry transparently creates a new session while runtime auth remains enabled.

Because runtime-enabled same-origin bootstrap is automatic, normal expiry does not ask the user to enter a key.

## Error Handling

- Backend unavailable: return a generic 502/503 from the Next route and leave the browser unconfigured.
- Invalid or missing server-side key: return unavailable/401 without leaking whether the key value was malformed.
- Session database unavailable: fail closed; never issue an untracked cookie.
- Cookie write rejected: the authenticated follow-up probe fails and the UI explains that browser cookies must be enabled.
- CSRF token missing/mismatched: return the existing 403 response; the client may perform one safe GET to obtain a fresh CSRF cookie before retrying only if current request semantics permit a retry.
- API-key rotation: after settings reload/process restart, the old cookie receives 401; the next runtime bootstrap uses the new server-side key and creates a new session.

No error or log includes the API key, opaque token, cookie header, or CSRF token.

## Test Strategy

### Backend Unit and Integration

- Session mint requires single-user mode and a valid API key.
- Mint sets the exact HttpOnly/host-only/Path/SameSite/Secure/Max-Age contract and returns no token.
- Repeated mint with a valid cookie reuses the session.
- Cookie principal resolution succeeds without auth headers.
- Explicit headers take precedence over cookies.
- Expired, revoked, malformed, wrong-user, and wrong-type sessions fail.
- API-key rotation invalidates an existing cookie session.
- Logout revokes the exact session and clears the cookie.
- Cookie-authenticated mutation without matching CSRF fails; matching CSRF succeeds.
- Cookie-authenticated mutation with `CSRF_BIND_TO_USER=true` resolves the cookie user before dependency execution and succeeds only with the correctly bound token.
- `X-API-KEY` mutation remains exempt from CSRF.
- Explicitly disabled CSRF prevents cookie-session minting and runtime bootstrap.
- Shared WebSocket resolution accepts a valid cookie only for an exact trusted Origin and rejects missing, null, malformed, cross-origin, expired, revoked, and key-mismatched cases before accept.
- A route-audit test covers every first-party WebSocket endpoint intended for WebUI use.

### Next.js Route and Client

- Runtime config advertises `cookie-session` and never serializes `apiKey`.
- Bootstrap rejects non-POST, cross-origin, forwarded/untrusted, disabled, wrong-mode, and invalid-internal-origin cases according to the deployment guard.
- Bootstrap sends the key only in the fixed internal request and forwards only expected response headers.
- Client cookie-session mode omits `X-API-KEY`, includes same-origin credentials, and adds CSRF on mutations.
- Cookie-session WebSocket builders use the same-origin proxy and omit query credentials; remote/extension builders retain explicit credentials.
- Failed bootstrap never writes a key to local/session storage.
- Successful cookie probe removes legacy bridge/runtime artifacts, scrubs ambiguous `tldwConfig.apiKey` and session records, and preserves only complete new-format origin-bound manual credentials.
- Runtime config does not advertise cookie-session outside the loopback quickstart guard, and partial deployments do not retain a browser-visible runtime-key fallback.

### Browser Lifecycle

- A fresh runtime-enabled loopback setup reaches an authenticated route with no key in localStorage, sessionStorage, IndexedDB, extension storage, page state, or runtime-config response. An upgraded profile may retain only a separately entered, complete origin-bound manual remote credential.
- Save/bootstrap then hard reload remains authenticated.
- Close Chromium completely, relaunch with the same Playwright `userDataDir`, and remain authenticated without API-key entry.
- Logout clears the cookie; relaunch does not reuse the revoked token.
- Rotate the configured API key and confirm the old browser session is rejected and safely reprovisioned.
- Open representative persona, ACP, and audio WebSockets through the same-origin proxy; confirm the HttpOnly session authenticates them and neither URL nor frames contain the API key.
- Attempt a cookie-bearing WebSocket from a different Origin and confirm rejection before accept.

The close/reopen case uses two browser processes against one persistent profile; another page in the same context is insufficient.

## Security Review Checklist

- API key never crosses the server/browser boundary.
- Opaque session token has at least 256 bits of entropy.
- Session record is server-side, bounded, exact-session revocable, and key-rotation-bound.
- Cookie is HttpOnly, host-only, appropriately Secure, and SameSite=Lax.
- Cookie mutations require CSRF.
- Cookie WebSockets require exact trusted-Origin validation before accept.
- Bootstrap target cannot be influenced by request input.
- Bootstrap validates same-origin browser intent and fails closed.
- No secrets appear in bodies, URLs, logs, diagnostics, storage migrations, screenshots, or test artifacts.
- Header-based API clients preserve existing behavior.

## Rollout

1. Add backend session primitives and tests without enabling client bootstrap.
2. Add the non-secret runtime capability and Next bootstrap route.
3. Switch same-origin request hydration from runtime key to cookie session.
4. Enable single-user cookie-aware CSRF and client CSRF headers in the same release.
5. Add lifecycle E2E, migration cleanup, deployment docs, and security verification.
6. Enable the existing runtime-auth flag's `cookie-session` capability only after HTTP, CSRF, standalone WebSocket proxy, and first-party WebSocket route coverage pass together.
7. Keep the old runtime key response permanently removed once the cookie path ships; do not retain a browser-visible compatibility fallback.
