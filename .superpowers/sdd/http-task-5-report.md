# HttpOnly Session Task 5 Report

## Implementation summary

- Replaced the browser-visible runtime-key bridge with capability-only cookie bootstrap: same-origin session POST, authenticated profile probe, and `authSource: "cookie-session"` publication only after both succeed.
- Prevented public/runtime API-key persistence in quickstart mode. Successful probes scrub ambiguous legacy local/session/runtime key slots while preserving complete origin-bound manual device/session credentials. Bootstrap, probe, capability, or storage failures leave manual configuration intact.
- Added cookie auth to same-origin HTTP requests only. Cookie requests use `credentials: "same-origin"`, remove stale API-key/Bearer/CSRF headers, and echo the readable `csrf_token` only on unsafe methods. Hosted and explicit remote transports retain their existing auth behavior.
- Added page-origin, query-secret-free cookie WebSockets for persona, watchlists, prompt studio, ACP client/hook, voice conversation, and streaming TTS. ACP REST receives the same cookie/CSRF contract. The extension-only background transport remains explicit remote auth and has a regression proving the cookie source marker cannot suppress that branch.
- Extended canonical/client config handling so cookie quickstart works without an API key and never rehydrates `NEXT_PUBLIC_X_API_KEY` into storage or request headers.

Implementation base: `0bd06eaec4` (`fix(auth): validate single-user session cookie name`).

## TDD RED

The first runnable focused browser/UI run produced 11 expected behavior failures with 46 passing tests. Failures were grouped as:

- shared HTTP: missing same-origin credentials/CSRF and stale header suppression;
- persona/watchlists/prompt studio: remote-origin URLs with `api_key`/`token` query secrets;
- ACP REST/WebSocket/hook: missing cookie request options and secret-free page-origin URLs;
- API client: public quickstart key persistence and cookie configs rejected without a key;
- voice/audio: token-bearing remote WebSocket URL in cookie mode.

The runtime-bootstrap RED cases were written before implementation for POST+probe success, bootstrap failure, probe failure, unavailable capability, manual-config preservation, post-probe-only legacy scrub, complete manual credential preservation, secret-free storage, and extension-protocol exclusion. Initial collection was blocked by a dangling worktree `antd` dependency link; verification used a temporary untracked link to the matching installed root-workspace package.

## GREEN and verification

Fresh final focused Vitest aggregate:

```text
Test Files  11 passed (11)
Tests       118 passed (118)
```

The aggregate includes runtime bootstrap, hosted/quickstart HTTP, persona/watchlists/prompt studio, ACP REST/WebSocket/hook, API client quickstart config, voice/audio WebSockets, and the extension background regression.

Additional verification:

- Focused ESLint over all touched TypeScript/TSX files: exit 0, no errors; existing baseline warnings remain.
- Targeted strict TypeScript run completed with 47 unrelated workspace baseline errors. A path filter found zero diagnostics in Task 5 touched files. The repository-wide run was also attempted and exhausted Node's 4 GB heap.
- Prettier check reports the existing formatting baseline across the touched legacy files; no bulk whole-file rewrite was applied. `git diff --check` exits 0.
- The temporary worktree dependency link was removed after the final runtime-bootstrap/aggregate run, and both `test ! -e` and `test ! -L` passed.
- Bandit is not applicable because Task 5 changes only TypeScript/TSX, tests, Backlog metadata, and this report.

## Security coverage

- No runtime API key or cookie value is written to local/session storage or returned through runtime config.
- Cookie auth activates only for `authSource="cookie-session"` plus same-origin HTTP transport.
- Safe cookie requests omit CSRF; unsafe cookie requests replace stale CSRF with the readable cookie value.
- Cookie HTTP requests never attach stale `X-API-KEY` or `Authorization` headers.
- Cookie WebSockets use the browser page `ws:`/`wss:` origin and contain no `api_key` or `token` parameters.
- Failed bootstrap/probe does not scrub or rewrite valid manual remote configuration.
- Ambiguous upgraded-profile keys are scrubbed only after the authenticated cookie probe succeeds.

## Known limits

- Lifecycle/relaunch and live proxy-upgrade coverage belongs to Task 6.
- `background.ts` is extension-service-worker-only and cannot use the WebUI host-only cookie; its explicit remote WebSocket behavior was intentionally left unchanged.
- The implementation commit message is `feat(web): use cookie auth for quickstart requests`; its hash is reported by the committed task handoff.

## Review follow-up

The five Task 5 review findings were reproduced from base `56ed4dbb74` and fixed with focused RED/GREEN coverage.

- Top-level authentication now recognizes a successfully probed keyless cookie-session config. Quickstart auth state and legacy auth storage no longer fall back to `NEXT_PUBLIC_X_API_KEY`; advanced/manual environment auth remains available.
- The canonical complete manual connection remains atomically stored in `tldwConfig`. Verified cookie activation uses the separate non-secret `tldwCookieSessionConfig` record. Bootstrap failure removes only that active marker, and repeated bootstrap plus `TldwApiClient.initialize()/getConfig()` leave the manual key, server URL, origin, source, persistence, and metadata unchanged.
- Sidepanel Chat and Playground use the same cookie-aware voice auth-ready resolver, including `authSource` dependencies. Caller coverage proves the keyless gate becomes available and produces the page-origin audio chat WebSocket URL.
- `isCookieSessionBrowserTransport()` is the single browser transport predicate. It requires single-user cookie source, quickstart mode, same-origin transport, and an HTTP(S) page origin. Hosted, advanced, extension, and non-HTTP stale markers retain their existing auth headers, organization behavior, and remote origins.
- ACP REST now exposes one cookie-aware request path used by the ACP client, ACP Playground health, and Agent Registry health/diagnostics. ACP workspace SSH uses a secret-free page-origin WebSocket only when the central cookie predicate succeeds.

Review TDD evidence:

```text
Initial RED: 11 failed, 65 passed across the first runnable review aggregate.
Voice/ACP caller RED: 2 failed, 14 passed.
ACP direct-path RED: 3 failed, 18 passed.
Final focused aggregate: 17 files passed; 138 tests passed.
Background transport regression: 1 file passed; 45 tests passed.
Total focused review evidence: 183 tests passed.
```

Static verification:

- Frontend ESLint and the temporary in-base UI-package ESLint pass with zero errors; warnings are existing baseline warnings in legacy files.
- Frontend TypeScript reports 15 unrelated baseline diagnostics and zero diagnostics in Task 5 touched paths. The standalone UI TypeScript run reached the existing 4 GB heap limit.
- `git diff --check` passes.
- Bandit remains not applicable to this TypeScript/TSX-only follow-up.
- Temporary dependency and lint symlinks were removed and verified absent.

Review-fix commit message: `fix(web): complete cookie-session client activation`.
