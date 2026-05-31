# AuthNZ Code Guide (Developers)

This guide orients project developers to the AuthNZ module: what’s in it, how it works, and how to work with it when building or extending the server.

See also: `tldw_Server_API/app/core/AuthNZ/README.md` and `Docs/Code_Documentation/AuthNZ-Developer-Guide.md` for complementary overviews.

## Scope & Goals
- Dual-mode authentication: single-user (API key) and multi-user (JWT) with a unified dependency pattern for endpoints.
- Authorization via roles/permissions (RBAC) with org/team context.
- Sessions, token rotation/blacklist, API keys and virtual keys, rate limits/quotas, CSRF, and security headers.
- Works with SQLite (default) and PostgreSQL (production); optional Redis acceleration for sessions/rate limits.

## Quick Map
- Core settings: `tldw_Server_API/app/core/AuthNZ/settings.py`
- DB pool + migrations glue: `tldw_Server_API/app/core/AuthNZ/database.py`
- JWT service: `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- Sessions + blacklist: `tldw_Server_API/app/core/AuthNZ/session_manager.py`, `tldw_Server_API/app/core/AuthNZ/token_blacklist.py`
- API keys + Virtual keys: `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`, `tldw_Server_API/app/core/AuthNZ/virtual_keys.py`
- RBAC/orgs/teams/permissions: `tldw_Server_API/app/core/AuthNZ/permissions.py`, `tldw_Server_API/app/core/AuthNZ/rbac.py`, `tldw_Server_API/app/core/AuthNZ/orgs_teams.py`, `tldw_Server_API/app/core/AuthNZ/privilege_catalog.py`
- Guardrails/middleware: `tldw_Server_API/app/core/AuthNZ/rate_limiter.py`, `tldw_Server_API/app/core/AuthNZ/llm_budget_middleware.py`, `tldw_Server_API/app/core/AuthNZ/llm_budget_guard.py`, `tldw_Server_API/app/core/AuthNZ/csrf_protection.py`, `tldw_Server_API/app/core/AuthNZ/security_headers.py`, `tldw_Server_API/app/core/AuthNZ/usage_logging_middleware.py`
- Auth flows/support: `tldw_Server_API/app/core/AuthNZ/password_service.py`, `tldw_Server_API/app/core/AuthNZ/mfa_service.py`, `tldw_Server_API/app/core/AuthNZ/email_service.py`
- Endpoint DI (use these): `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Auth endpoints: `tldw_Server_API/app/api/v1/endpoints/auth.py`
- Admin + RBAC mgmt: `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py` plus split modules in `tldw_Server_API/app/api/v1/endpoints/admin/` (`admin_user.py`, `admin_api_keys.py`, `admin_profiles.py`, `admin_sessions_mfa.py`, `admin_byok.py`, `admin_llm_providers.py`, `admin_orgs.py`, `admin_settings.py`, `admin_registration.py`, `admin_system.py`, `admin_usage.py`, `admin_budgets.py`, `admin_tools.py`, `admin_personalization.py`, `admin_network.py`), along with `.../users.py`, `.../privileges.py`, `.../register.py`
- Debug helpers: `tldw_Server_API/app/api/v1/endpoints/authnz_debug.py`

## New endpoints checklist (AuthNZ + RG)

- Always authenticate via `get_auth_principal` and enforce authorization with `RequirePermission(...)` / `RequireRole(...)` (and `require_service_principal()` for service-only routes). `require_admin`/`require_role` API-dependency shims are removed; avoid raw `request.state.user_id` checks or ad-hoc `is_single_user_mode()` branches for access control.
- For latency/cost-sensitive or user-facing endpoints (chat, audio, embeddings, evaluations, MCP, workflows, tools, media ingestion, etc.), verify whether they should be governed by Resource Governor:
  - If yes, ensure there is a policy entry (`resource_governor_policies.yaml` or DB-backed `rg_policies`) and a `route_map` entry mapping the endpoint’s path/tag to an appropriate policy id.
  - Confirm tests exercise the RG-enforced behavior where limits are important (429/Retry-After headers, tokens/streams limits, etc.).

## Modes, Profiles & Request Flow

### Modes vs Profiles (AUTH_MODE today, PROFILE + flags going forward)

- `AUTH_MODE` currently distinguishes:
  - `single_user` – fixed API key, synthetic admin user, SQLite-backed by default.
  - `multi_user` – username/password + MFA + JWT, sessions, RBAC, orgs/teams, API keys.
- In code, helpers `is_single_user_mode()` / `is_multi_user_mode()` read `AUTH_MODE` via `get_settings()`, but their use is now restricted to coordination/UX and a small number of operational knobs (startup banners, WebUI hints, warm-ups, quota toggles). They are **deprecated as authorization primitives**.
- An explicit `PROFILE` setting (see `Docs/Product/Completed/User-Unification-PRD.md`) and a small set of feature/behavior flags now describe deployment behaviour more precisely:
  - `PROFILE=local-single-user` (or legacy alias `single_user`).
  - `PROFILE=multi-user-postgres`.
  - Optional `PROFILE=multi-user-sqlite` for small dev setups.
  - Auth-adjacent behaviour flags:
    - `EMBEDDINGS_TENANT_RPS_PROFILE_AWARE` – embeddings tenant RPS quotas driven by profile and principals explicitly tagged as single-user (subject `"single_user"` via `is_single_user_principal`) vs legacy `AUTH_MODE`.
    - `MCP_SINGLE_USER_COMPAT_SHIM` – MCP single-user API-key compatibility shim vs always using the multi-user API key manager.
- New code should treat mode/profile/flags as **coordination and guardrail-tuning inputs** (bootstrap, banners, WebUI hints, quotas), not as authorization shortcuts. Auth decisions must use `AuthPrincipal` claims via the dependencies below and MUST NOT branch on `is_single_user_mode()` / `is_multi_user_mode()` or `AUTH_MODE` to grant or bypass permissions.

Admin role semantics (claims-first):
- The `admin` role claim is interpreted consistently across profiles: a principal with `roles=["admin", ...]` is treated as having both admin- and user-level access regardless of `AUTH_MODE`/`PROFILE`.
- Helpers such as `check_role`, `RequireRole("admin")`, and `RequirePermission(...)` rely on claims (`principal.roles`, `principal.permissions`, `principal.is_admin`) rather than re-reading mode, so new endpoints should always gate admin/control surfaces through these claim-first dependencies instead of adding new mode checks.

Recommended combinations (v0.1):
- Local single-user desktop: `AUTH_MODE=single_user`, `PROFILE=local-single-user` (default SQLite users DB).
- Multi-user with Postgres: `AUTH_MODE=multi_user`, `PROFILE=multi-user-postgres`, `DATABASE_URL=postgres://...`.
- Multi-user with SQLite (dev only): `AUTH_MODE=multi_user`, `PROFILE=multi-user-sqlite`, `DATABASE_URL=sqlite:///./Databases/users.db`.

### Single-User Mode (X-API-KEY)
- Configure `AUTH_MODE=single_user` and `SINGLE_USER_API_KEY` (and eventually `PROFILE=local-single-user`).
- `get_current_user` and `get_request_user` accept either:
  - `X-API-KEY: <key>`
  - `Authorization: Bearer <key>` (for OpenAI-compatible clients).
- Optional IP allowlist via `SINGLE_USER_ALLOWED_IPS`.
- JWT/session/blacklist are bypassed; the principal is treated as an admin-style user whose permissions are governed by claims:
  - Single-user claims and permissions are exercised and constrained by `tldw_Server_API/tests/AuthNZ/integration/test_single_user_claims_permissions.py`.

### Multi-User Mode (JWT)
- Configure `AUTH_MODE=multi_user` and `JWT_SECRET_KEY` (or RS/ES keys).
- Login issues `access` and `refresh` tokens; sessions persisted; refresh rotation enabled by default.
- Bearer `Authorization` is required for protected endpoints unless an API key is used as an alternative path.
- Token revocation checks the blacklist; sessions track activity, IP, UA.

### API Keys & Virtual Keys
- API keys provide non-JWT authentication; can be scoped, rotated, expire, and audited.
- Virtual keys (JWT) are short-lived scoped tokens minted by authenticated users for automation/integrations. Validation works in both modes when JWT is configured. In single-user mode the JWT service derives a surrogate secret from `SINGLE_USER_API_KEY`, so bearer JWTs can be validated; API keys remain the simplest option when operating single-user.

Note on single-user JWTs:
- In single-user mode, `get_current_user` does not accept arbitrary JWTs; it only accepts `SINGLE_USER_API_KEY` via `X-API-KEY` or as Bearer (see `tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_current_user`). Virtual JWTs can still be verified and scoped via `TokenScopeGuard`, but they do not authenticate a user in single-user mode.

JWT virtual keys support additional constraints via claims (enforced with `auth_deps.TokenScopeGuard`, not in `get_current_user`):
- `allowed_endpoints`: list of endpoint codes (e.g., `chat.completions`)
- `allowed_methods`: HTTP methods allowlist (e.g., `["POST"]`)
- `allowed_paths`: path prefixes allowlist (e.g., `["/api/v1/chat/"]`)
- `count_as` + `max_calls` or `max_runs`: simple per-token quotas
- `schedule_id` + scheduler match options (when used with workflow scheduling)

### Virtual Keys: API Keys vs JWTs

- Virtual API Keys (DB-backed)
  - Created/stored via `api_key_manager.py` in `api_keys` (may be flagged `is_virtual`).
  - Authenticate with `X-API-KEY` (or Bearer); validated by `APIKeyManager.validate_api_key`.
  - Budgets/allowlists enforced by `llm_budget_middleware.py`/`llm_budget_guard.py` (day/month token/usd, allowed endpoints/providers/models, IP allowlists, optional per-key rate_limit).
  - Rotate/revoke; only HMAC-SHA256 key hashes stored (no plaintext).
  - Best for longer-lived integrations/service accounts.

- Virtual JWTs (short-lived tokens)
  - Minted with `JWTService.create_virtual_access_token(...)`; not stored server-side.
  - Authenticate with Bearer; claim-level enforcement is applied via `auth_deps.TokenScopeGuard` (allowed_endpoints/methods/paths, quotas via `count_as` + `max_calls`/`max_runs`, optional `schedule_id`).
  - Work in both modes (single-user derives a surrogate secret; multi-user uses configured JWT secrets/keys).
  - Best for ephemeral, scoped automation (e.g., scheduled workflows) where rotation and narrow scope are required.

## Using Dependencies in Endpoints

Prefer the shared dependencies from `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`. Endpoint code should import the PascalCase factories (`RequirePermission`, `RequireRole`, and `TokenScopeGuard`); the lowercase names remain implementation/backcompat factories and are covered directly by AuthNZ unit tests.

Typical patterns:

```python
from fastapi import APIRouter, Depends
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_current_user,           # Resolves single-user, API-key, or JWT
    get_current_active_user,    # Adds active status check
    check_rate_limit,           # Diagnostics-only ingress compatibility helper
    RequirePermission,          # Claim-first permission gate
)

router = APIRouter(prefix="/example", tags=["examples"])

@router.get("/protected")
async def protected_route(user=Depends(get_current_user)):
    return {"hello": user.get("username")}

@router.post(
    "/write",
    dependencies=[Depends(RequirePermission("media.update"))],
)
async def write_route(user=Depends(get_current_active_user)):
    return {"ok": True}

@router.get("/limited", dependencies=[Depends(check_rate_limit)])
async def limited_route(user=Depends(get_current_user)):
    return {"ok": True}
```

Scoped token/key enforcement:
- Use `TokenScopeGuard(scope, endpoint_id=..., count_as=...)` to enforce virtual-key constraints on a route. It validates JWT claims when a bearer is present and applies equivalent metadata-based checks when only `X-API-KEY` is provided (allowed endpoints/methods/paths and optional per-key quotas).
 - This dependency is additive and does not replace `get_current_user`; use both when a route must authenticate and enforce scoped constraints.

Example (scoped automation endpoint):
```python
from fastapi import Depends
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_current_user, TokenScopeGuard
)

@router.post("/workflows/run", dependencies=[Depends(TokenScopeGuard(
    scope="workflows",
    endpoint_id="workflows.run",
    count_as="run",
))])
async def run_workflow(user=Depends(get_current_user)):
    return {"ok": True}
```

Notes:
- `get_current_user` handles single-user mode first, then API key, then JWT.
- `check_rate_limit` dependency is diagnostics-only (it does not enforce fallback 429s); see Rate Limiting below for enforcement paths.

Legacy compatibility DI (existing endpoints only):
- `get_optional_current_user` – legacy shim, kept for older optional-auth routes that
  need to behave differently when a user is present. New endpoints should prefer
  claim-first patterns (e.g., `AuthPrincipal` + explicit permissions/roles) rather
  than introducing fresh optional-auth behaviors.

Examples (modern, claim-first):
```python
from fastapi import Depends
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_optional_current_user,
    get_auth_principal,
    RequirePermission,
    RequireRole,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

@router.get("/maybe-auth")
async def maybe_auth(user=Depends(get_optional_current_user)):
    if user:
        return {"hello": user.get("username")}
    return {"hello": "anonymous"}

@router.get("/admin-only")
async def admin_only(principal: AuthPrincipal = Depends(RequireRole("admin"))):
    return {"ok": True, "as": "admin", "id": principal.user_id}

@router.get("/media-read")
async def media_read(principal: AuthPrincipal = Depends(RequirePermission("media.read"))):
    return {"ok": True, "id": principal.user_id}
```

## Sessions, Tokens, Revocation

- `session_manager.py` encrypts token material at rest using Fernet.
- Encryption key precedence: explicit `SESSION_ENCRYPTION_KEY` → persisted file (secure 0600) → derived keys from configured secrets (fallback) → generated in TEST_MODE.
  - Persisted key preferred path: `tldw_Server_API/Config_Files/session_encryption.key`; legacy fallback: project root `Config_Files/session_encryption.key`. Set `SESSION_KEY_STORAGE=project` to force legacy storage. In default API mode, valid legacy keys are migrated to the API path. Symlinks are handled safely and file ownership/permissions are validated.
- Blacklist checks (by JTI) protect against replay after logout/rotation; see `token_blacklist.py`.
- Redis (optional via `REDIS_URL`) accelerates lookups and counters.

Key calls you’ll see in endpoints/services:
- `JWTService.create_access_token(...)`, `create_refresh_token(...)`, `decode_access_token(...)`.
- `SessionManager.create_session(...)`, `update_session_tokens(...)`, `revoke_session(...)`.

## API Keys & Virtual Keys

- `api_key_manager.py` creates, rotates, revokes, and validates keys; stores HMAC-SHA256 hash of the presented key.
- Keys support: expiry, usage counts, IP allowlists, rate limits, audit log, and optional LLM usage budgets and allowlists.
- `auth_deps.get_current_user` can authenticate via `X-API-KEY` when no Bearer token is present.
- Virtual keys: use `POST /api/v1/auth/virtual-key` or programmatically via `JWTService.create_virtual_access_token(...)`.
  - Virtual keys are enforced via scoped claims (e.g., `scope`, allowed endpoints/methods/paths) with helpers in `auth_deps.TokenScopeGuard` and budget checks in `llm_budget_guard.py`.

Example: create and validate an API key
```python
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager

mgr = await get_api_key_manager()
rec = await mgr.create_api_key(user_id=1, name="integration", scope="write", expires_in_days=30)
# Persist the cleartext key somewhere safe on the client side
assert await mgr.validate_api_key(api_key=rec["key"])  # returns key info (includes user_id), not a full user dict
```

Notes:
- `validate_api_key(...)` returns key metadata (including `user_id`, scope, budgets, etc.), not a full user record. Endpoints that authenticate via API key obtain user details separately (see `get_current_user`).

## RBAC, Roles, Permissions, Orgs/Teams

- Permissions helpers: `tldw_Server_API/app/core/AuthNZ/permissions.py`.
- Org/team context: DI attaches `request.state.user_id`, plus memberships from `orgs_teams.list_memberships_for_user` and a content scope token via `scope_context.set_scope` for downstream DB access.
- **Modern pattern (preferred)**: use claim-first dependencies from `auth_deps`:
  - `get_auth_principal` → returns `AuthPrincipal` with `roles`/`permissions` claims.
  - `RequirePermission("perm")` / `RequireRole("role")` → enforce claims and return the principal.
  - `require_service_principal()` → enforces `principal.kind == "service"` for internal-only/service endpoints.
  - Representative usage exists on media, RAG, notes graph, evaluations CRUD endpoints, sandbox admin views, and selected workflows surfaces.
  - `get_org_policy_from_principal` → resolves an organization policy in a **principal-first** way for org-scoped features (for example, connectors/admin policy and sources):
    - Preference order:
      1. First `org_id` in `principal.org_ids` (claim-first).
      2. Synthetic `org_id=1` only for principals explicitly tagged as `subject="single_user"`.
    - This helper is the preferred org-policy dependency for new org-scoped endpoints; avoid reimplementing org selection logic from `request.state` or mode flags.
- **HTTP status semantics (AuthNZ dependencies)**:
  - `get_auth_principal`:
    - Returns an `AuthPrincipal` when credentials are valid.
    - Raises **401 Unauthorized** when credentials are missing or invalid, with `WWW-Authenticate: Bearer` and a stable detail string (e.g., `"Not authenticated (provide Bearer token or X-API-KEY)"` in multi-user mode).
  - `get_current_user`:
    - Returns a user-shaped dict when credentials are valid.
    - Raises **401 Unauthorized** for missing/invalid credentials with detail containing `"Authentication required"` and `WWW-Authenticate: Bearer`.
  - `RequirePermission` / `RequireRole`:
    - Propagate **401** from `get_auth_principal` when no principal can be resolved.
    - Raise **403 Forbidden** when a principal is present but lacks required claims:
      - `RequirePermission` → `detail="Permission denied. Required: <perm-list>"`.
      - `RequireRole` → `detail="Access denied. Required role(s): <role-list>"`.
    - These 403 payload shapes are treated as part of the public surface for claim-first/admin routes.
  - `require_service_principal`:
    - Depends on `get_auth_principal` for authentication (401 on missing/invalid credentials).
    - Raises **403 Forbidden** with `detail="Service principal required"` when a non-service principal calls a service-only route.
    - Returns the underlying `AuthPrincipal` unchanged when `principal.kind == "service"`.
- **Removed legacy helpers (historical only)**:
  - Earlier versions exposed FastAPI dependencies `PermissionChecker`, `RoleChecker`, `AnyPermissionChecker`, and `AllPermissionsChecker` from `permissions.py`. These have now been removed; endpoints must use `get_auth_principal` together with `RequirePermission` / `RequireRole` instead.
  - Existing admin and control surfaces that once used these helpers (media add, tools execute, workflows runs/events/artifacts/control/DLQ, sandbox admin) have been migrated to claim-first dependencies. Tests such as `test_media_add_permissions_claims.py`, `test_tools_permissions_claims.py`, `test_workflows_runs_permissions_claims.py`, `test_workflows_artifacts_permissions_claims.py`, `test_workflows_webhook_dlq_permissions_claims.py`, `test_workflows_control_permissions_claims.py`, and `test_sandbox_admin_permissions_claims.py` lock in the new behavior and ensure the claim-first dependencies are the true authorization gates.
  - Heavy evaluations admin paths use claim-first checks via `enforce_heavy_evaluations_admin(AuthPrincipal)` together with `RequireRole("admin")` / `RequirePermission(...)` on top of `get_auth_principal`.
  - A repository layer exists for AuthNZ and MUST be used instead of ad-hoc SQL for core tables:
    - `AuthnzUsersRepo` (`app/core/AuthNZ/repos/users_repo.py`) wraps `UsersDB` for user lookups; exercised against both SQLite and Postgres in AuthNZ tests.
    - `AuthnzApiKeysRepo` (`app/core/AuthNZ/repos/api_keys_repo.py`) centralizes `api_keys` read/write paths and is used by `APIKeyManager` and single-user bootstrap.
    - `AuthnzRbacRepo` (`app/core/AuthNZ/repos/rbac_repo.py`) fronts `UserDatabase_v2` for RBAC permission checks; higher-level helpers in `app/core/AuthNZ/rbac.py` delegate to it.
    - `AuthnzOrgsTeamsRepo` (`app/core/AuthNZ/repos/orgs_teams_repo.py`) owns organizations, teams, and membership (including default-team creation/enrollment) so `orgs_teams.py` can remain orchestration-only.

### Single-User Org Policy Fallback (Principal-Only)

Org-policy resolution now uses principal claims exclusively:

- `principal.org_ids` is authoritative when present.
- Synthetic `org_id=1` is allowed only for principals explicitly tagged with `subject="single_user"`.
- Non-single-user principals without `org_ids` receive `HTTPException(400)` with `"User has no organization memberships"`.

Tests:
- Unit tests in `tldw_Server_API/tests/AuthNZ_Unit/test_org_policy_from_principal.py` cover claim-first selection, single-user principal fallback, and legacy-flag-ignore behavior.
- HTTP-level tests in `tldw_Server_API/tests/AuthNZ_Unit/test_connectors_admin_claims.py` exercise the same behavior through connectors routes.
    - `AuthnzUsageRepo` (`app/core/AuthNZ/repos/usage_repo.py`) provides aggregate and pruning helpers for `usage_log`, `usage_daily`, `llm_usage_log`, and `llm_usage_daily`, and is used by `virtual_keys` and the AuthNZ scheduler.
    - `AuthnzRateLimitsRepo` (`app/core/AuthNZ/repos/rate_limits_repo.py`) encapsulates all DB-backed rate-limiter tables (`rate_limits`, `failed_attempts`, `account_lockouts`) and is used by `rate_limiter.RateLimiter` for counters, lockouts, and cleanup.
  - New AuthNZ code should **not** add fresh backend-specific SQL for these tables; prefer adding small, task-focused methods to the appropriate repo and calling them from business logic.
  - New code MUST NOT introduce fresh `is_single_user_mode()` branches in endpoint/business logic. Mode/profile checks are confined to a small number of coordination points (bootstrap, DB selection, and legacy compatibility helpers); authorization should flow through `AuthPrincipal` + claim-first dependencies instead.
  - New code **must not** reintroduce FastAPI dependency helpers that bypass `get_auth_principal` or duplicate `RequirePermission` / `RequireRole`. Authorization for endpoints should always flow through claim-first dependencies.
  - AuthNZ-facing documentation (usage examples, API integration guides) should treat this guide as canonical and mirror its claim-first patterns. When decorator-style helpers appear in historical examples (including the removed `*Checker` classes), they should be clearly labeled as legacy and not used for new code.

References:
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py#get_auth_principal`
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py#RequirePermission`
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py#RequireRole`

## How to Secure a New Endpoint (Checklist)

When adding a new FastAPI endpoint that needs AuthNZ and guardrails:

1. **Choose the auth principal dependency**
   - User/admin routes: depend on `get_auth_principal` (and optionally `get_current_user` when you need legacy user dicts), for example:
     - `principal: AuthPrincipal = Depends(get_auth_principal)`
     - `current_user: dict[str, Any] = Depends(get_current_user)` (only when necessary).
   - Service-only/internal routes: use `require_service_principal()`:
     - `principal: AuthPrincipal = Depends(require_service_principal)`.

2. **Define and apply permissions/roles**
   - Add or reuse a permission constant in `tldw_Server_API/app/core/AuthNZ/permissions.py`.
   - Gate the route with claim-first dependencies on the router or endpoint:
     - `dependencies=[Depends(RequirePermission(MY_PERMISSION))]`
     - and/or `dependencies=[Depends(RequireRole("admin"))]`.
   - Do **not** introduce mode-based gates or legacy admin shims; always build on `get_auth_principal` + `RequirePermission` / `RequireRole`.

3. **Wire ResourceGovernor / rate limits if needed**
   - For rate-limited endpoints, reuse shared helpers instead of new per-module limiters:
     - `Depends(check_rate_limit)` / `Depends(check_auth_rate_limit)` for generic/AuthNZ auth flows.
     - Existing RG-backed helpers for chat/embeddings/audio/evaluations/MCP when applicable.
   - If a new RG policy is required, add it to `tldw_Server_API/Config_Files/resource_governor_policies.yaml`, keep `route_map.by_path` in sync, and ensure tests cover the new policy id.

4. **Expose profile/flag behaviour via config, not auth**
   - If the endpoint is profile- or feature-specific, gate availability in the service/router layer via:
     - `PROFILE` (`get_profile()`) and
     - explicit feature flags (`ENABLE_*`, or behavior flags like `EMBEDDINGS_TENANT_RPS_PROFILE_AWARE`).
   - Do **not** use `AUTH_MODE` / `is_single_user_mode()` in the endpoint to grant or bypass permissions; mode/profile/flags are for coordination and guardrail tuning, not for auth.

5. **Add tests mirroring real principals**
   - Unit tests:
     - Override `get_auth_principal` to return principals with different `roles`/`permissions` and assert 401 vs 403 vs 200 semantics (see `tests/AuthNZ_Unit/test_auth_claim_route_level.py` and `tests/AuthNZ_Unit/test_scheduler_workflows_permissions_claims.py` as patterns).
   - Integration tests:
     - Use existing fixtures (e.g., `isolated_test_environment`, single-user bootstrap tests) to exercise JWT and API-key flows where relevant.
     - Validate that guardrails (e.g., 429 + `Retry-After`, RG headers) behave as expected for over-limit scenarios.

## Rate Limiting & Quotas

- General limiter: `rate_limiter.py` implements a token bucket over the AuthNZ `rate_limits` tables; DI provides `get_rate_limiter_dep`.
- Governance facade: `AuthGovernor` (`auth_governor.py`) is the canonical entry point for AuthNZ guardrails. It wraps the shared `RateLimiter` and virtual-key budget helpers so core flows no longer talk to those primitives directly:
  - `check_llm_budget_for_api_key(principal, api_key_id)` decorates `is_key_over_budget(...)` with `AuthPrincipal` metadata and is used by `llm_budget_guard` and `llm_budget_middleware`.
  - `check_lockout(identifier, attempt_type="login", rate_limiter=...)` and `record_auth_failure(identifier, attempt_type, rate_limiter=...)` mediate login lockout and suspicious-activity tracking and are used by the `/auth/login` endpoint.
  - `check_rate_limit(identifier, endpoint, limit=None, window_minutes=None, rate_limiter=...)` remains available for compatibility/testing and direct limiter call sites where explicitly used.
- Endpoint helpers:
  - `check_rate_limit` is diagnostics-only (no fallback 429 enforcement).
  - `check_auth_rate_limit` is diagnostics-only (no fallback 429 enforcement).
  - Route abuse protection should come from RG ingress policies and endpoint-local RG checks.
- LLM budgets: `llm_budget_middleware.py` and `llm_budget_guard.py` enforce endpoint/provider/model quotas when configured, always via `AuthGovernor.check_llm_budget_for_api_key`. Settings are `LLM_BUDGET_ENFORCE` (on/off) and `LLM_BUDGET_ENDPOINTS` (paths). Virtual key features are gated by `VIRTUAL_KEYS_ENABLED` (defaults true).

References:
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py#get_rate_limiter_dep`
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py#check_rate_limit`
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py#check_auth_rate_limit`
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py#rbac_rate_limit`
- `tldw_Server_API/app/core/AuthNZ/rate_limiter.py#RateLimiter`
- `tldw_Server_API/app/core/AuthNZ/auth_governor.py#AuthGovernor`

RBAC-aware selector (logging-only for now):
- `auth_deps.rbac_rate_limit(resource)` logs the strictest configured limit selected for a user/role-resource pair; it does not enforce yet (use `AuthGovernor`/`RateLimiter` call paths for enforcement).

Example: RBAC resource-aware logging (no enforcement)
```python
from fastapi import Depends
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_current_user, rbac_rate_limit
)

@router.get("/send", dependencies=[Depends(rbac_rate_limit("chat.send"))])
async def send_message(user=Depends(get_current_user)):
    return {"ok": True}
```

Example: custom rate limit per endpoint (IP-keyed; for per-user, pass your own identifier)
```python
from fastapi import Depends, Request, HTTPException, status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep

async def limit_10rpm(req: Request, limiter=Depends(get_rate_limiter_dep)):
    allowed, meta = await limiter.check_rate_limit(req.client.host, "example", limit=10, window_minutes=1)
    if not allowed:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Slow down")

@router.get("/tight", dependencies=[Depends(limit_10rpm)])
async def tight_route(user=Depends(get_current_user)):
    return {"ok": True}
```

Example: per-user rate limit
```python
from fastapi import Depends, HTTPException, status
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_rate_limiter_dep, get_current_user
)

async def limit_user_60rpm(user=Depends(get_current_user), limiter=Depends(get_rate_limiter_dep)):
    # Use a user-scoped identifier; choose a stable endpoint label
    identifier = f"user:{user['id']}"
    allowed, meta = await limiter.check_rate_limit(identifier, endpoint="example:user", limit=60, window_minutes=1)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=meta.get("error", "Too many requests"),
            headers={"Retry-After": str(meta.get("retry_after", 60))}
        )

@router.get("/tight-user", dependencies=[Depends(limit_user_60rpm)])
async def tight_user(user=Depends(get_current_user)):
    return {"ok": True}
```

## Expected Roles/Permissions for Resource-Governor Admin

- Resource-Governor admin and diagnostics endpoints under `/api/v1/resource-governor/*` are treated as AuthNZ admin surfaces and MUST:
  - Use claim-first dependencies: `principal: AuthPrincipal = Depends(get_auth_principal)` and `dependencies=[Depends(RequireRole("admin"))]` on the route.
  - Rely on `RequireRole("admin")` (admin role or `principal.is_admin`) as the gate; do not add new `is_single_user_mode()` / mode-specific bypasses in handlers.
  - Keep 401/403 semantics aligned with other admin endpoints (401 for missing/invalid credentials, 403 for insufficient roles), as enforced by:
    - `tldw_Server_API/tests/AuthNZ_Unit/test_resource_governor_permissions_claims.py`
    - `tldw_Server_API/tests/Resource_Governance/test_rg_capabilities_endpoint.py`
    - `tldw_Server_API/tests/Resource_Governance/test_resource_governor_endpoint.py`

## Settings & Initialization

- Central source: `settings.py` (via `get_settings()`), with env + optional project config overrides.
- Single-user requires `SINGLE_USER_API_KEY`; multi-user requires `JWT_SECRET_KEY` or asymmetric keys.
- Asymmetric JWT is supported via `JWT_PRIVATE_KEY`/`JWT_PUBLIC_KEY` (e.g., RS256/ES256); otherwise use `JWT_SECRET_KEY` for HS*.
- Initialize or migrate DB:
  - `python -m tldw_Server_API.app.core.AuthNZ.run_migrations`
  - `python -m tldw_Server_API.app.core.AuthNZ.initialize`
- DB detection: `database.py` chooses Postgres in multi-user when `DATABASE_URL` is `postgres*`; else SQLite.

Notes:
- The self-service minting endpoint `POST /api/v1/auth/virtual-key` is available only in multi-user mode. In single-user mode you can still create scoped JWTs programmatically via `JWTService.create_virtual_access_token(...)`, but API keys are usually simpler.
- Optional JWT issuer/audience enforcement is supported via `JWT_ISSUER` and `JWT_AUDIENCE`. Dual-validation during rotations is supported with `JWT_SECONDARY_SECRET` (HS) or `JWT_SECONDARY_PUBLIC_KEY` (RS/ES).
- Virtual key features can be toggled via `VIRTUAL_KEYS_ENABLED` in settings (enabled by default).
- Security alerts sinks (file/webhook/email) are configured via `SECURITY_ALERTS_*` settings; see the AuthNZ settings section for details.
- Registration toggles: `ENABLE_REGISTRATION` and `REQUIRE_REGISTRATION_CODE` control whether registration is exposed and whether codes are required.
- MFA prerequisites: enforced via `_ensure_mfa_available` — `tldw_Server_API.app.api.v1.endpoints.auth._ensure_mfa_available`
 - Virtual key feature and budget settings: `VIRTUAL_KEYS_ENABLED`, `LLM_BUDGET_ENFORCE`, `LLM_BUDGET_ENDPOINTS` — see `VIRTUAL_KEYS_ENABLED`, `LLM_BUDGET_ENFORCE`, and `LLM_BUDGET_ENDPOINTS` on `tldw_Server_API.app.core.AuthNZ.settings.Settings`

Auth Endpoints (summary):
- `POST /api/v1/auth/login` – Username/password login (multi-user) — `tldw_Server_API.app.api.v1.endpoints.auth.login`
- `POST /api/v1/auth/refresh` – Refresh JWT (multi-user) — `tldw_Server_API.app.api.v1.endpoints.auth.refresh_token`
- `POST /api/v1/auth/logout` – Logout; optional all devices (multi-user) — `tldw_Server_API.app.api.v1.endpoints.auth.logout`
- `POST /api/v1/auth/register` – Registration flow (if enabled by settings) — `tldw_Server_API.app.api.v1.endpoints.auth.register`
- `POST /api/v1/auth/forgot-password` – Send password reset email (multi-user) — `tldw_Server_API.app.api.v1.endpoints.auth.forgot_password`
- `POST /api/v1/auth/reset-password` – Reset password with token (multi-user) — `tldw_Server_API.app.api.v1.endpoints.auth.reset_password`
- `GET /api/v1/auth/verify-email` – Verify email (multi-user) — `tldw_Server_API.app.api.v1.endpoints.auth.verify_email`
- `POST /api/v1/auth/resend-verification` – Resend verification email (multi-user) — `tldw_Server_API.app.api.v1.endpoints.auth.resend_verification`
- `POST /api/v1/auth/mfa/setup` | `POST /mfa/verify` | `POST /mfa/disable` – MFA endpoints; MFA is available only in multi-user deployments with PostgreSQL — see `setup_mfa`, `verify_mfa_setup`, and `disable_mfa` in `tldw_Server_API.app.api.v1.endpoints.auth`
- `POST /api/v1/auth/virtual-key` – Mint scoped virtual JWT; multi-user only — `tldw_Server_API.app.api.v1.endpoints.auth.mint_self_virtual_key`

References:
- `tldw_Server_API.app.api.v1.endpoints.auth.mint_self_virtual_key`
- `tldw_Server_API.app.api.v1.endpoints.auth`

Operator references:
- JWT rotation runbook: `Docs/Deployment/Operations/JWT_Rotation_Runbook.md`
- AuthNZ settings/env vars (AuthNZ section): `Docs/Operations/Env_Vars.md`
- Authentication setup guide: `Docs/Published/User_Guides/Server/Authentication_Setup.md`

## Database Backends

- SQLite is the default; WAL mode enabled; migrations harmonized via `migrations.py` on startup.
- PostgreSQL is recommended for multi-user deployments; pool managed by `asyncpg`.
- `get_db_pool()` yields a singleton `DatabasePool`; use `transaction()` for writes, `acquire()` for reads.
  - `DatabasePool.execute/fetch*` normalize SQL placeholders across backends (`?` → `$1,$2,...` on Postgres). Inside explicit transactions where you work with the raw connection, prefer Postgres-style placeholders (`$1,$2,...`) for portability; SQLite shims convert `$N` to `?` automatically.

## Gotchas

- Cross-backend SQL placeholders
  - When using `get_db_transaction` (raw connections), Postgres requires `$1,$2,...` placeholders. SQLite adapter shims translate `$N` to `?`, so `$N` works on both.
  - If you aren’t in a transaction, prefer `DatabasePool.execute/fetch*` which normalizes placeholders automatically in either direction.

## Common Recipes

1) Protect a new endpoint
```python
@router.get("/secure")
async def secure(user=Depends(get_current_user)):
    return {"user_id": user["id"]}
```

2) Require a permission (claim-first)
```python
from fastapi import Depends
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequirePermission, get_current_user

@router.post("/media/update", dependencies=[Depends(RequirePermission("media.update"))])
async def update_media(user=Depends(get_current_user)):
    ...
```

3) Use a DB transaction in a handler
```python
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_db_transaction

@router.post("/do-write")
async def do_write(conn=Depends(get_db_transaction), user=Depends(get_current_user)):
    # Prefer Postgres-style placeholders for cross-backend portability
    await conn.execute("INSERT INTO example(col) VALUES($1)", "value")
    return {"ok": True}
```

4) Mint a virtual key (from an authenticated route)
```python
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.settings import get_settings

@router.post("/vk")
async def mint_vk(user=Depends(get_current_user)):
    svc = JWTService(get_settings())
    token = svc.create_virtual_access_token(
        user_id=user["id"], username=user.get("username", ""), role=user.get("role", "user"),
        scope="workflows", ttl_minutes=30, additional_claims={"allowed_endpoints": ["chat.completions"]}
    )
    return {"token": token}
```

## Testing Notes

- TEST_MODE (`TEST_MODE=1`) disables rate limiting by default and enables deterministic secrets/keys. Some DI dependencies return lightweight stubs to keep tests fast and deterministic.
- In TEST_MODE, `get_session_manager_dep` returns a stub `SessionManager` and `get_rate_limiter_dep` returns a disabled limiter stub to avoid DB/Redis work and keep tests deterministic.
- Postgres-dependent tests use a provisioned container unless `TLDW_TEST_NO_DOCKER=1`; see `tldw_Server_API/tests/AuthNZ_Postgres/` and project test README.
- Many endpoint utilities add test-only diagnostics headers for clarity (e.g., `X-TLDW-DB`, `X-TLDW-CSRF-Enabled`).
- 401 diagnostics in TEST_MODE: auth failures from `get_current_user` include `X-TLDW-Auth-Reason` and `X-TLDW-Auth-Headers` to explain why authentication failed and which headers were present (see `tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_current_user`).

## Extending AuthNZ

- Add new permissions in `permissions.py` and seed mapping in RBAC migrations/seeders.
- New admin surfaces belong under `.../endpoints/admin/` (router + shared helpers) or a feature-specific `admin_<feature>.py` module and should be guarded using claim-first dependencies:
  - Prefer `principal: AuthPrincipal = Depends(get_auth_principal)` together with `dependencies=[Depends(RequireRole("admin"))]` (and `RequirePermission(...)` where appropriate).
- For new guardrails, prefer middleware with settings gating, and expose DI helpers to keep endpoints simple.
- Keep both SQLite and Postgres code paths working; use `DatabasePool` helpers to normalize placeholders across backends when needed.

## Troubleshooting & Pitfalls

- JWT misconfig in multi-user: ensure `JWT_SECRET_KEY` length >= 32 (HS*) or set RSA/EC keys for (RS*/ES*).
- Single-user without key: set `SINGLE_USER_API_KEY` (a deterministic test key is used under TEST_MODE).
- Session key persistence failures: ensure `Config_Files/session_encryption.key` is writable and not a symlink to an invalid location.
- Virtual key JWTs do not require multi-user mode (they’re accepted in single-user when JWT is configured; a surrogate secret is derived from `SINGLE_USER_API_KEY`).
- LLM budget enforcement depends on `LLM_BUDGET_*` settings and middleware placement.
- When using `get_db_transaction` with Postgres connections, `?` placeholders are not supported. Use `$1,$2,...` or route SQL through `DatabasePool.execute`/`fetch*` for automatic normalization (outside explicit transactions).

---

If you need a deeper conceptual overview, read `tldw_Server_API/app/core/AuthNZ/README.md`. For a broader project context, see `Docs/Code_Documentation/AuthNZ-Developer-Guide.md`.
