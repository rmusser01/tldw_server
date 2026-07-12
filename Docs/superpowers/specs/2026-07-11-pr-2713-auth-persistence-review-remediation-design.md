# PR 2713 Authentication Persistence Review Remediation Design

## Context

PR 2713 introduced two intentional secret-storage models:

- same-origin quickstart uses a verified HttpOnly cookie session recorded by the non-secret `tldwCookieSessionConfig` marker;
- manually configured remote servers store device keys in local storage and session keys in the browser or extension session store.

The review found that request code still has several independent configuration readers. `TldwApiClient` hydrates the two models correctly, while WebUI direct requests, extension background requests, uploads, and streams read only local `tldwConfig`. That divergence can make session-only credentials unusable and can prevent cookie requests from stripping explicit credentials and attaching CSRF.

## Considered approaches

### Patch each request path independently

This is the smallest change per file but preserves the root cause: every new transport must remember two storage records and their validation rules. It is rejected because another caller can drift again.

### Route every operation through `TldwApiClient`

This removes duplicate readers but would require a large rewrite of extension service-worker uploads, streaming ports, and direct fallback behavior. It adds risk outside the reviewed defects.

### Shared effective-auth resolver

Add one storage-agnostic resolver beside the existing credential-policy helpers. It reads local connection metadata, optionally accepts a verified WebUI cookie-session candidate, otherwise hydrates only an origin-matched manual device/session key. WebUI direct runtimes and `TldwApiClient` may opt into cookie-session state only for a same-origin quickstart page whose in-memory marker is not invalidated. Extension workers never opt into cookie sessions, but use the same resolver for session-key hydration.

This is the selected approach because it fixes the shared boundary without changing persistence formats or transport APIs.

## Effective configuration contract

The resolver returns one in-memory `TldwConfig` and never persists a hydrated session key.

1. Read `tldwConfig` from persistent/local storage.
2. If the caller supplies the current quickstart WebUI origin, the cookie marker is not invalidated, and `tldwCookieSessionConfig` exactly matches that origin, return a sanitized cookie-session configuration with no explicit credentials.
3. Otherwise, validate manual credential metadata against the normalized configured origin.
4. For device scope, retain the complete origin-bound local key.
5. For session scope, merge the matching session-store record into the returned in-memory object only.
6. On missing, malformed, mismatched, or unreadable secret state, return metadata without an API key and fail closed at the request layer.

The resolver is reused by the WebUI direct runtime, extension background request/upload/stream paths, and client initialization. Cookie eligibility remains an explicit caller input so extension contexts cannot accidentally activate it.

## Logout and clearing

Cookie-session logout sends the existing CSRF-protected DELETE through the effective WebUI runtime. The backend route has no mandatory principal dependency: it always emits the clearing cookie and a `Cache-Control: no-store` response, and revokes only when the cookie resolves to an active canonical single-user session. A missing, expired, malformed, or already revoked cookie therefore produces an idempotent successful logout without mutating another session.

After a successful response, the client invalidates and removes the non-secret cookie-session marker, then rehydrates any preserved manual connection. Network failure remains visible and does not falsely claim a completed logout.

Manual credential clearing attempts both stores even when one operation fails. It sanitizes persistent metadata when readable and reports an error if persistent read/write or session removal fails, so callers cannot report successful logout/reset while a browser-readable secret may remain.

## Bootstrap scrubbing

Quickstart bootstrap preserves a session secret only when the complete active `tldwConfig` declares `single-user`, `manual`, `session`, and the same normalized remote origin as the session record. Device, cookie, ambiguous, or unrelated connection metadata causes the session record to be removed.

## Verification and UAT

Regression tests first reproduce each defect:

- cookie-session POST/PATCH requests use the cookie marker, omit preserved manual headers, and attach CSRF;
- extension/direct requests, uploads, and streams hydrate a session key without persisting it locally;
- active and stale cookie logout clear correctly and return no-store;
- clear failures propagate and bootstrap removes mismatched session records;
- WebUI and extension lifecycle suites call a protected fixture endpoint after reload/relaunch, then prove session scope stops authenticating after a full restart.

The required frontend gate runs the three auth-persistence browser suites when security-relevant frontend code changes. Final UAT uses a real Chromium WebUI profile and a real unpacked extension installation for device, session, cookie, logout, reload, and relaunch flows.
