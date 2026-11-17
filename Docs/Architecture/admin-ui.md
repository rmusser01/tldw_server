# The admin web UI (admin-ui)

This document describes the sysadmin console: its deliberately separate stack, the same-origin proxy that keeps tokens out of the browser, edge-middleware auth with local JWT verification, the triple-gated API-key login, deployment, and testing. It is a client of the API only — no database, no direct filesystem.

## What it is (and is not)

A Next.js 16 (App Router) + React 19 + Radix + Tailwind 4 console on port **3001**, with its own `package.json` and `bun.lock` — deliberately **not** part of the `apps/` bun workspace and deliberately a different stack from the end-user WebUI (`apps/tldw-frontend`: Pages Router, React 18, antd, port 8080). Never cross-import between the two. The browser never talks to the backend directly: all privileged traffic goes through a same-origin server-side proxy (`/api/proxy/*`), enforced by CSP `connect-src 'self'`.

## Authoritative files

All paths under `admin-ui/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Auth edge | `middleware.ts` | token validation with an in-process LRU cache (hashed keys); local HS256/384/512 JWT verify with secondary-secret rotation support; backend `/users/me` fallback probe; matcher excludes `/login`, `/api`, `_next`, files |
| Server proxy | `app/api/proxy/[...path]/route.ts` | rewrites to `${NEXT_PUBLIC_API_URL}/api/${version}`, 30 s timeout, one GET-only retry, 502/504 mapping |
| Cookie handling | `lib/server-auth.ts` | httpOnly cookie constants; cookie → `Authorization: Bearer` / `X-API-KEY`; safe-header passthrough allowlist; `set-cookie` stripped from responses |
| Login routes | `app/api/auth/{login,mfa/login,apikey,logout}/route.ts` | forwards to backend auth; **strips tokens from the JSON the browser sees**; logout invalidates the middleware cache |
| Health | `app/api/health{,/ready}/route.ts` | liveness; readiness probes backend `/health` (503 when down) |
| Browser API | `lib/http.ts`, `lib/api-client.ts` (typed client), `lib/api-config.ts` | everything is `/api/proxy<endpoint>`; real-backend E2E port map |
| Client auth | `lib/auth.ts` | non-httpOnly session-marker polling + auth-change events |
| Rate limiting | `lib/rate-limiter.ts` | 10 req / 60 s per IP (requires `TRUST_PROXY_HEADERS` behind a proxy) |
| Env/flags | `lib/env.ts`, `lib/admin-ui-flags.ts` | zod-validated `NEXT_PUBLIC_*` at build; the unsafe-local-tools gate for DSR/backup/monitoring mutations |
| Build | `next.config.mjs` | standalone output (repo-root tracing); full CSP/HSTS headers |
| Packaging | `Dockerfiles/Dockerfile.admin-ui`, `docker-compose.admin-ui.yml` | bun builder → node:20-slim runtime, non-root UID, healthcheck against `/api/health/ready`; starts only after the backend is healthy |
| Backend counterpart | `tldw_Server_API/app/api/v1/endpoints/admin.py`, `router_groups/admin.py`, `endpoints/test_support/admin_e2e.py` | the `/api/v1/admin/*` surface (16 router specs) and the E2E test-support router (flag-gated) |

## Login (dataflow)

1. Browser POSTs credentials to same-origin `/api/auth/login` (rate-limited).
2. The route forwards to backend `/api/v1/auth/login`, then sets **httpOnly** cookies (`access_token`, optional refresh, plus non-httpOnly `admin_session`/`admin_auth_mode` markers) and returns the payload **with the tokens deleted** — browser JS never sees them. MFA follows the same shape.
3. The client polls for the session marker, fetches `/users/me` through the proxy, and emits an auth-change event.

API-key login (single-user convenience) is **triple-gated**: refused if enterprise mode is on, if not explicitly enabled, or if the backend is not in single-user mode; the enterprise flag hard-disables it regardless.

## Page request and privileged call (dataflow)

1. Edge middleware checks cookies (then headers): JWT verified **locally** when `JWT_SECRET_KEY` is present, else via a backend probe; results cached (positive ≤30 s bounded by token expiry, negative 5 s, LRU cap).
2. Unauthenticated → redirect to `/login?redirectTo=…`; otherwise the response carries a request id.
3. Page code calls the typed client → same-origin `/api/proxy/admin/...` → the proxy attaches the server-side credential from cookies, passes safe headers, and maps timeout → 504 / network error → 502 (GET retried once).

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Backend down via proxy | 504 (30 s timeout) / 502 (network); GET-only single retry |
| Backend down at readiness | 503 (Docker healthcheck fails) |
| Login rate limit | 429 + `Retry-After` |
| Revoked token | Accepted up to ~30 s per instance (best-effort cache; logout proactively invalidates) |
| Backend `set-cookie` | Stripped — backend sessions never reach the browser |
| Malformed/oversized tokens | Rejected pre-verification (length/charset caps) |

## Config keys (exact)

Build-time (baked into the client bundle; changes require a rebuild): `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`), `NEXT_PUBLIC_API_VERSION` (v1), `NEXT_PUBLIC_DEFAULT_AUTH_MODE`, `NEXT_PUBLIC_ALLOW_ADMIN_API_KEY_LOGIN` (false), `NEXT_PUBLIC_BILLING_ENABLED` (false), `NEXT_PUBLIC_ADMIN_UI_ENABLE_UNSAFE_LOCAL_TOOLS` (false), `NEXT_PUBLIC_SENTRY_DSN`. Runtime: `JWT_SECRET_KEY` / `JWT_SECONDARY_SECRET` / `JWT_ALGORITHM`, `ADMIN_UI_ALLOW_API_KEY_LOGIN` (false), `ADMIN_UI_ENTERPRISE_MODE`, `AUTH_MODE`, `TRUST_PROXY_HEADERS` (false), `SERVER_X_API_KEY`, `SENTRY_AUTH_TOKEN`, `ADMIN_UI_PORT` (3001). E2E: `TLDW_ADMIN_E2E_*` family plus the backend's `ENABLE_ADMIN_E2E_TEST_MODE`.

## Verified gotchas

1. Two stacks, two lockfiles — admin-ui vs the `apps/` monorepo; also two Next versions and two React versions. Cross-imports are the classic breakage.
2. The auth cache is per-instance and non-persistent — multi-replica deployments have per-pod caches and revoked tokens linger ≤30 s.
3. Without `JWT_SECRET_KEY`, every protected page request becomes a backend probe — local verify is the fast path.
4. The middleware matcher excludes `/api/*`; API routes enforce their own auth (the proxy reads cookies server-side).
5. `ai-overview` and `invitations` pages exist but are missing from the README's feature list (minor drift).
6. ADR-029 explicitly keeps admin-ui outside the PyPI static-bundle decision (its middleware/app-routes design needs a separate one).

## Governing decisions and docs

ADR-029 (`Docs/ADR/029-tldw-frontend-static-pypi-bundle.md` — admin-ui excluded), ADR-052 tangentially (MCP admin claims the console manages). Current references: `admin-ui/README.md`, `admin-ui/docs/{architecture,api-reference,deployment-guide,troubleshooting}.md`, feature guides, the living `Release_Checklist.md`. Tests: colocated Vitest (incl. axe a11y lane), Playwright smoke (stubbed), and real-backend E2E (two sequential projects: JWT and single-user, port-mapped against real backends, `--workers=1`).

## Related docs

- [authnz.md](./authnz.md) — the backend auth the console drives
- [sdks.md](./sdks.md) — the other client surfaces
