# AuthNZ: authentication, users, and the admin surface

This document describes the server's dual-mode auth core (single-user API key vs multi-user JWT+sessions with optional TOTP), claim-first authorization through `AuthPrincipal`, API keys, the ~40-module admin-services pattern, billing/BYOK, and the governance/audit cores. `Docs/Code_Documentation/AuthNZ-Developer-Guide.md` predates the claim-first API and is partially stale; the module README at `tldw_Server_API/app/core/AuthNZ/README.md` is current.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Settings | `app/core/AuthNZ/settings.py` | `Settings(BaseSettings)` — every key below; `is_single_user_mode` / `is_multi_user_mode` / `get_profile` |
| Principal resolution | `app/core/AuthNZ/auth_principal_resolver.py`, `principal_model.py` | `get_auth_principal(request)` — JWT → API-key → single-user-cookie precedence; `AuthPrincipal` kinds (user / api_key / service / anonymous), `AuthContext` |
| Endpoint dependencies | `app/api/v1/API_Deps/auth_deps.py` | `get_auth_principal` (adds the maintenance-mode gate), `require_permissions` (AND), `require_roles` (OR; `RequireRole`), `require_api_key_scope`, `require_token_scope`, `require_expected_user` (412 on stale user), rate-limit deps |
| JWT | `app/core/AuthNZ/jwt_service.py` | `JWTService` — access/refresh/service/virtual/password-reset/email-verification/magic-link/admin-reauth tokens; refresh rotation with fail-closed revocation persistence; token hashing |
| Sessions | `app/core/AuthNZ/session_manager.py` | `SessionManager` — server-side sessions, Fernet at-rest token encryption, Redis or ephemeral KV |
| Blacklist / lockout | `token_blacklist.py`, `lockout_tracker.py` | JTI revocation with strict/non-strict propagation; composite client+login lockout keys |
| Passwords / MFA | `password_service.py`, `mfa_service.py` | Argon2id with transparent rehash, history, breached list, registration codes; TOTP with Fernet-encrypted secrets and backup codes |
| API keys | `api_key_manager.py`, `api_key_crypto.py`, `virtual_keys.py` | `tldw_<id>.<secret>` format (dot separator); PBKDF2-KDF hashes with legacy HMAC fallback for dual-secret rotations; scopes read/write/admin/service; usage recording; virtual-key budgets |
| RBAC | `rbac.py`, `rbac_seed.py`, `permissions.py`, `org_rbac.py`, `platform_admin.py` | effective-permission resolution; seeded roles (admin/user/viewer/reviewer/moderator); scoped org/team overlay; the single `PLATFORM_ADMIN_ROLES` source of truth |
| Single user | `single_user_session.py`, `ip_allowlist.py` | opaque cookie sessions (fail-closed validation); trusted-proxy-aware client IP; IP allowlists |
| CSRF | `csrf_protection.py` | double-submit cookie; optional HMAC binding to the user |
| User persistence | `User_DB_Handling.py`, `repos/` (41 modules) | `verify_jwt_and_fetch_user` (incl. scoped-token enforcement), `authenticate_api_key_user`, RBAC enrichment; repos are the only persistence boundary |
| BYOK | `byok_*.py`, `user_provider_secrets.py`, `provider_credential_runtime.py` | enablement/allowlists, encrypted envelopes, runtime resolution, rotation |
| Auth routes | `app/api/v1/endpoints/auth.py` | router prefix `/auth` under `/api/v1`: login, refresh, logout, MFA flows, register, password reset, sessions CRUD, single-user cookie session |
| Admin surface | `app/api/v1/endpoints/admin/__init__.py` (32 submodules; 33 files including `__init__.py`) + `app/services/admin_*_service.py` (~40) | router gated by `Depends(RequireRole("admin"))`; see the pattern below |
| Billing | `app/core/Billing/` | `BillingEnforcer` plan-limit enforcement (OSS: limits, not payments); `PlanTier`/`get_plan_limits`; Stripe only for injected clients; per-user quotas live in UserProfiles `limits.*` (ADR-064) |
| Governance / Audit | `app/core/Governance/`, `app/core/Audit/` | policy-rule resolution (allow/warn/require_approval/deny) feeding MCP and ACP; `UnifiedAuditService` with PII detection, risk scoring, JSONL fallback queue, 90-day retention |

## Multi-user login → session → permission check (dataflow)

1. `POST /api/v1/auth/login` (rate-limited) resolves the client IP through trusted proxies only and checks the composite client+login lockout (429 with `Retry-After`).
2. User fetch by normalized identifier; unknown user and wrong password return the same generic 401 (no enumeration), both recording failures.
3. Argon2id verification returns `(valid, needs_rehash)` — valid-but-old hashes upgrade transparently.
4. If MFA is enabled (multi-user + Postgres only), a temporary session plus an ephemeral token returns **202 MFAChallengeResponse**; `/auth/mfa/login` consumes it, verifies TOTP then backup code.
5. Session first, tokens second: the session row is created with placeholders, real JWTs are minted carrying `session_id` and scope claims, then the session is updated.
6. Per-request: the resolver prefers a Bearer JWT (2-dot heuristic); a non-JWT bearer is an API key (and in single-user mode, *any* bearer is the API key). The JWT path enforces issuer/audience, impersonation claims, **scoped-claim enforcement** (a scoped token used on a route without `require_token_scope` is 403), session validity, and blacklist; the user is fetched and RBAC-enriched onto `request.state.auth`.
7. Authorization deps: `require_permissions("x","y")` is AND (admin-bypass claims short-circuit); `RequireRole("admin")` is OR.

## API key / single-user request (dataflow)

1. `X-API-KEY: tldw_<id>.<secret>` → parse the embedded key id → active-row fetch → PBKDF2-KDF hash verify (legacy HMAC candidates for rotating secrets) → status/expiry/IP-allowlist/scope checks → usage recorded.
2. Single-user mode routes through the same API-key flow and asserts the resolved id equals `SINGLE_USER_FIXED_ID`; the single-user IP allowlist applies.
3. Optionally, `POST /auth/single-user/session` mints an opaque cookie session after verifying an explicit `X-API-KEY` (with CSRF enabled — 503 otherwise); validation fails closed; logout revokes exactly that session. Single-user `/login` returns the configured `SINGLE_USER_API_KEY` itself as both tokens.

## The admin-services pattern

Every admin module shares one shape: an `AuthPrincipal` input; org scoping via `admin_scope_service` (platform-admin check + `enforce_admin_org_access`); persistence through an AuthNZ repo (no raw SQL in services); audit events via `admin_audit_service` deferred until after commit; background work through Jobs queues (ADR-003) or APScheduler date jobs. Notable modules: users (with a versioned-profile write gateway), api-keys, audit, backup scheduler+jobs, BYOK secrets + validation runs, budgets, config profiles, data ops/retention, GDPR-style data-subject requests, guardrails, LLM provider overrides, crypto maintenance rotation (active-execute uniqueness guard), monitoring alerts, registration codes, RBAC admin, maintenance state (consumed by the auth gate), webhooks (legacy vs canonical route selection). `admin_service.py` is a retired 60-line shim.

### BYOK validation (admin → Jobs worker)

An admin-scoped validation run rows as queued → the worker enqueues a Jobs payload (idempotency key `byok-validation:<run_id>`) → the worker claims it, pages through shared and per-user provider secrets, decrypting each envelope (undecryptable rows count as errors, never fatal) → per-provider semaphores test credentials against the provider with a server config snapshot for fallback base-URL/model → failures are redacted before persistence → the run completes with counts. The backup pipeline is the same shape (schedule rows → APScheduler → Jobs worker).

## Boundaries

- Header parsing lives only in the resolver and auth deps; endpoints use `Depends(get_auth_principal)` / `RequireRole` / `require_permissions` (the legacy `require_admin`/`require_role` shims are retired).
- Persistence only through AuthNZ repos; the 41 repo modules are the boundary.
- Scoped tokens cannot expand privileges: scope claims are refused on routes without scope enforcement.
- The MCP admin predicate is deliberately narrower than platform admin (ADR-052): only `*` permission or platform-admin roles.
- Maintenance-mode gating runs inside `get_auth_principal` — after authentication, before authorization deps.
- Auth never blocks on audit writes — except registration, which rolls the user back and returns 503 if its mandatory audit write fails.

## Failure behaviors (verified)

| Condition | Behavior |
| --- | --- |
| Missing credentials | 401 with `WWW-Authenticate: Bearer` |
| Lockout tripped | 429 + `Retry-After` (client-pair and per-username) |
| Rotated/revoked refresh | Fail-closed: rotation requires confirmed revocation persistence |
| Blacklist DB failure | Strict paths (refresh, service tokens) propagate 401 |
| Single-user cookie invalid | Fails closed as unauthenticated |
| Maintenance mode, non-admin | 503 `maintenance_mode` |
| Service token from non-loopback IP | 403 |
| Scoped token on unenforced route | 403 "Scoped token requires endpoint scope enforcement" |
| Stale `X-TLDW-Expected-User-ID` | 412 `request_config_scope_changed` |
| TEST_MODE in production | 500 unless explicitly allowed |
| Enterprise flags without multi_user+Postgres | Fail-closed unsupported matrix |

## Config keys (selection; `AuthNZ/settings.py` is authoritative)

`AUTH_MODE` (single_user default), `DATABASE_URL` (SQLite default), `JWT_SECRET_KEY` (≥32 chars; required for multi-user), `JWT_ALGORITHM` + optional RS/ES keypairs and secondary keys for rotation, `ACCESS_TOKEN_EXPIRE_MINUTES` (30), `REFRESH_TOKEN_EXPIRE_DAYS` (7), `PASSWORD_MIN_LENGTH` (10), `ARGON2_*` costs, `MAX_LOGIN_ATTEMPTS` / `LOCKOUT_DURATION_MINUTES`, `ENABLE_REGISTRATION` (false) / `REQUIRE_REGISTRATION_CODE` (true), `SESSION_ENCRYPTION_KEY` (else a 0600 persisted key file), `SINGLE_USER_API_KEY` (≥16 chars; stricter in production), `SINGLE_USER_ALLOWED_IPS`, `SERVICE_TOKEN_ALLOWED_IPS` (empty = loopback-only), `REDIS_URL`, `CSRF_ENABLED` / `CSRF_BIND_TO_USER`, `BYOK_ENABLED` + allowlists + encryption keys, `ORG_RBAC_*`, and the fail-closed enterprise flags (`AUTH_FEDERATION_ENABLED`, `SECRET_BACKENDS_ENABLED`, `MCP_CREDENTIAL_BROKER_ENABLED`).

## Governing decisions

ADR-017 (`Docs/ADR/017-scoped-org-team-rbac-core-semantics.md`), ADR-019 (`019-security-request-edge-middleware.md`), ADR-044 (`044-cookie-session-governance-owner-preflight.md`), ADR-052 (`052-mcp-admin-claims.md`), ADR-056 (`056-resource-governor-safety-net.md`), ADR-064 (`064-usage-quotas-per-user-limits.md`); adjacent: ADR-003 (Jobs vs Scheduler placement), ADR-018 (resource governance). Current references: `Docs/AuthNZ/AUTHNZ_PERMISSION_MATRIX.md`, the module README. Tests: `tldw_Server_API/tests/AuthNZ*` (Postgres fixtures may auto-start Docker unless `TLDW_TEST_NO_DOCKER=1`).

## Verified gotchas

1. A legacy boolean `is_admin` on a principal is deliberately **not** a bypass — only role `admin` or admin permission claims count.
2. `require_permissions` is AND; `require_roles` is OR — mixed-up assumptions are the classic review finding.
3. In single-user mode any Bearer credential is silently treated as the API key, even a JWT-shaped one.
4. MFA requires Postgres — SQLite deployments treat MFA as disabled during login without error.
5. New `tldw_` keys verify against slow PBKDF2-KDF hashes; legacy keys need the HMAC candidate fallback for dual-secret rotations.
6. `principal_id` is a pseudonymous hash for logs/metrics — never raw ids.
7. The chatbook client sends its token as `X-API-KEY` with redirects refused so the key can never be forwarded on a hop; a few endpoints need both header and bearer because OpenAPI security schemes gate the handler.

## Related docs

- [app-shell.md](./app-shell.md) — where the auth middleware and routers mount
- [database-layer.md](./database-layer.md) — the users DB and migrations
- [jobs-background.md](./jobs-background.md) — the Jobs queues admin workers ride on
