# User Auth Dependencies PRD (v0.1)

> Status: **Implemented / Historical**
>
> This PRD describes the design and rollout plan for the unified AuthNZ dependency stack that shipped in v0.1.0. The core changes (claim-first dependencies, `AuthPrincipal`, `require_permissions` / `require_roles`, and the removal of decorator-style helpers) are now implemented and covered by tests referenced below.
>
> For up-to-date usage and examples, see:
> - `Docs/AuthNZ/AUTHNZ_USAGE_EXAMPLES.md` – current recommended patterns for endpoint protection and claim-first authorization.
> - `Docs/AuthNZ/AUTHNZ_PERMISSION_MATRIX.md` – current role/permission mappings.
> - `Docs/AuthNZ/AUTHNZ_DATABASE_CONFIG.md` – current AuthNZ DB configuration details.

## Summary

AuthNZ currently has overlapping mechanisms for:

- Attaching the current user to a request (via `User_DB_Handling` and `auth_deps`).
- Authorizing access via permissions and roles (`permissions.py`, `rbac.py`).
- Exposing current user and auth context to other modules (LLM, RAG, MCP, UI).

While the underlying RBAC model is consistent, there is duplication and ambiguity in how “current user”, “current principal”, and authorization checks are wired into the request lifecycle. This PRD proposes:

1. Making authorization checks purely claim-based (no additional RBAC DB lookups on hot paths).
2. Unifying “current user/auth” dependencies around a single `AuthPrincipal` / `AuthContext` dependency (see Principal-Governance PRD).

The goal is to make authentication/authorization dependencies predictable, efficient, and easy to use across the project.

## Related Documents

- `Docs/Product/Completed/AuthNZ-Refactor/Principal-Governance-PRD.md` – defines `AuthPrincipal` / `AuthContext` and AuthNZ guardrails.
- `Docs/Product/Completed/User-Unification-PRD.md` – defines deployment profiles (`local-single-user`, `multi-user-postgres`) and bootstrap semantics.
- `Docs/Product/Completed/AuthNZ-Refactor/Resource_Governor_PRD.md` – describes global resource governance integrated with AuthNZ via `AuthGovernor`.
  - Admin and diagnostics surfaces under `/api/v1/resource-governor/*` are now guarded via claim-first dependencies (`get_auth_principal` + `require_roles("admin")`), with existing integration tests preserved.

---

## Problems & Symptoms

### 1. Mixed Claim-Based and DB-Based Authorization

- `User_DB_Handling.verify_jwt_and_fetch_user`:
  - Decodes JWT / API key.
  - Fetches user data from AuthNZ DB.
  - Enriches with roles and effective permissions by querying RBAC tables.
- `permissions.py`:
  - Checks `user.permissions` / `user.roles` when available.
  - Falls back to DB (`get_user_database().has_permission`) for some checks.
- `rbac.py`:
  - Uses `get_configured_user_database()` to compute effective permissions and check them per user.

This leads to:

- Inconsistent behavior if roles/permissions change during a session.
- Extra DB round-trips on hot paths, even though claims were already computed at authentication time.

### 2. Multiple “Current User” Entry Points

- `User_DB_Handling.get_request_user` + mode-specific helpers.
- `auth_deps.get_current_user` and related dependencies.
- Various endpoints use slightly different dependency stacks for auth, sessions, and scopes.

Symptoms:

- Test code and services need to know which dependency to call to get “the real user”.
- Bugs where one dependency stack attaches `request.state.user_id` but another does not, leading to missing context in downstream code (usage logging, budgets, etc.).

### 3. Context Leaks and Tight Coupling

- Some modules (usage logging, LLM budget guard, resource governance) reach into `request.state` to read `user_id`, `api_key_id`, `org_ids`, `team_ids` directly.
- There is no single structured object that represents “this is the authenticated principal and their claims”.

---

## Goals

### Primary Goals

- **G1: Claim-First Authorization**
  - Make `User.permissions`, `User.roles`, and `AuthPrincipal` claims the canonical source of truth for runtime access decisions.
  - Eliminate extra DB-dependent permission checks on hot paths (except where explicitly needed, e.g., admin tooling).

- **G2: Unified Auth Dependencies**
  - Provide a small set of well-documented FastAPI dependencies:
    - `get_auth_principal` → returns `AuthPrincipal`.
    - `get_current_user` → returns `User` (wrapped principal).
    - `require_permissions`, `require_roles`, etc. built on top of claims.
  - Make all endpoints and middlewares rely on these instead of ad-hoc auth handling.
  - Resource-Governor admin/config endpoints and diagnostics routes are explicit adopters of this stack: they use `require_roles("admin")` (admin role or `principal.is_admin`) as the gate, and reuse the same 401/403 semantics as other claim-first admin surfaces (connectors admin policy, tools admin, embeddings model-management, workflows DLQ).

### Secondary Goals

- Improve test ergonomics:
  - Provide simple ways to stub `AuthPrincipal` / `User` in tests.
- Make it easy to reason about what a route needs: reading dependencies tells you exactly what auth/claims are required.

### Non-Goals (Initial Version)

- Changing external auth APIs (login, token issuance, etc.).
- Replacing non-FastAPI call sites that use decorators (those will be adapted gradually).

---

## Proposed Solution

### 0. Transitional Compatibility Layer

- Existing public dependencies are widely used and form the current compatibility surface:
  - `tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_current_user`, `get_current_active_user`, `require_admin`, `require_role`, `get_optional_current_user`.
  - `tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user`, `verify_jwt_and_fetch_user`, `verify_single_user_api_key`.
- Today, these:
  - Perform their own JWT/API-key validation and RBAC enrichment (including per-request DB lookups).
  - Write `request.state.user_id/api_key_id/org_ids/team_ids` directly, with no `AuthContext`.
  - Contain single-user-specific behavior (synthetic admin user, direct `SINGLE_USER_API_KEY` comparisons, mode-based bypasses).
- In v1, we will:
  - Keep these functions as *facades* for compatibility, but refactor their internals to delegate to `get_auth_principal` / `AuthContext`.
  - Preserve their external signatures and core HTTP semantics (401/403 behavior, key accepted headers, basic user fields) unless explicitly versioned.
  - Gradually migrate endpoints and tests toward using the new dependencies directly (`get_auth_principal`, `require_permissions`, `require_roles`), then deprecate legacy entry points once usage drops.

### 1. Claim-First Permission Checks

#### Concept

- All runtime authorization checks should be pure functions of claims already attached to the principal.
- DB access for authorization is reserved for:
  - Issuing/refreshing tokens and keys (authentication and session management).
  - Guardrail/accounting operations (rate limits, budgets) as defined in the Principal-Governance and Resource_Governor PRDs.
  - Admin operations (managing roles/permissions).

#### Implementation Sketch

- Strengthen the invariant that `User_DB_Handling` (or `get_auth_principal`) always:
  - Fetches roles and base permissions from RBAC tables.
  - Applies user-specific overrides (allow/deny).
  - Sets `user.roles`, `user.permissions`, and `user.is_admin` on the principal.

- Update `permissions.py`:
  - `check_permission(user, permission)`:
    - First consults `user.permissions` (and possibly `user.is_admin` for implicit permissions).
    - Does not hit the DB in the common case.
  - `check_role(user, role)`:
    - Consults `user.roles` claim.
  - Remove or gate the DB fallback behind a debug or admin-only flag, then phase it out.

- Update/limit usage of `rbac.py` on hot paths:
  - Keep `rbac` for admin APIs (e.g., listing effective permissions, debugging).
  - Avoid calling it from per-request permission checks.

### 2. Unified Auth Dependencies

#### Concept

- Define a single “front door” dependency for auth:
  - `get_auth_principal(request: Request) -> AuthPrincipal`.
- Build all other auth-related dependencies on top of it.

#### Implementation Sketch

- Introduce `get_auth_principal` in AuthNZ (implementation aligned with Principal-Governance PRD).
  - It:
    - Handles credential detection (Bearer, API key, single-user key).
    - Validates via JWT/Session/API-key managers.
    - Hydrates user data and RBAC claims.
    - Attaches to `request.state.auth`.
    - Behaves consistently across deployment profiles:
      - In `local-single-user`, the bootstrapped admin user (`SINGLE_USER_FIXED_ID`) is represented as a normal user principal with admin role and full claims.
      - In `multi-user-postgres`, principals are derived from the authenticated user or service identity using the same code paths.
      - Dependencies do not special-case `AUTH_MODE` or profiles for authorization; claims on `AuthPrincipal` determine access.

- Provide derivative dependencies:
  - `get_current_user(principal: AuthPrincipal = Depends(get_auth_principal)) -> User`.
  - `require_permissions(perms: list[str])`:
    - A dependency factory that asserts the principal/user has required permissions.
  - `require_roles(roles: list[str])`.

- Migrate existing dependencies:
  - `User_DB_Handling.get_request_user`, `auth_deps.get_current_user`, and any per-endpoint auth dependencies to call `get_auth_principal` or its wrappers.

### Profiles & endpoint gating

- Endpoint availability (e.g., user registration/creation) is governed by deployment profile and configuration as defined in `User-Unification-PRD.md`:
  - In `local-single-user`, additional user creation is forbidden as a hard constraint; routes that create users MUST be disabled or return errors regardless of the caller’s admin status.
  - In `multi-user-postgres`, user-management routes may be enabled and guarded by `require_permissions` / `require_roles`.
- Auth dependencies themselves remain profile-agnostic:
  - They always return an `AuthPrincipal` derived from the current credentials.
  - Profile checks are applied at routing/service level when deciding whether a route should exist or succeed.

### 3. Principal Kinds & Service Tokens

- `AuthPrincipal` includes a `kind` (or `subject_type`) field that describes the principal type, such as:
  - `user` – end-users authenticated via JWT/session.
  - `api_key` – API-key-based access mapped to a user.
  - `service` – internal service tokens or automation identities.
  - `anonymous` – unauthenticated/public access where allowed.
- Dependencies behave as follows:
  - `get_auth_principal` always returns an `AuthPrincipal` with a populated `kind`, and user-centric fields (`user_id`) are only present when applicable (e.g., `user`, `api_key`).
  - `get_current_user` and any user-centric dependencies REQUIRE a principal with user context; if invoked when `kind` is `service` or `anonymous` (or when `user_id` is missing), they MUST block the call with an appropriate 403 error rather than fabricating a user.
  - Future service-specific dependencies (e.g., `require_service_principal`) MAY be introduced for internal endpoints that expect `kind=service`.

### 4. Clean Request-State Usage

- Standardize on:
  - `request.state.auth` as the canonical `AuthContext` (principal + request metadata).
  - `request.state.auth.principal` as the canonical `AuthPrincipal`.
  - `request.state.user_id`, `api_key_id`, `org_ids`, `team_ids` preserved for backwards compatibility but always derived from `request.state.auth.principal`.
- Update middlewares that currently access `request.state` directly:
  - `UsageLoggingMiddleware` uses `request.state.auth` to derive `user_id`, `key_id`, org/team context.
  - `llm_budget_guard` uses `AuthPrincipal` to discover the key and quotas.

- PII and logging:
  - `AuthPrincipal` / `AuthContext` MUST NOT store raw secrets (JWTs, API keys) or other high-sensitivity fields.
  - When logging or emitting metrics from dependencies, use stable, non-PII identifiers (e.g., principal_id, hashed IDs), aligning with `PII_REDACT_LOGS` and `USAGE_LOG_DISABLE_META`.

### 5. Error Semantics (401 vs 403)

- `get_auth_principal`:
  - When credentials are missing or invalid, raises an unauthenticated error that maps to HTTP 401 with appropriate `WWW-Authenticate` headers where applicable.
  - The error payload (status, error code, message) is treated as part of the public API surface and SHOULD remain stable or be explicitly versioned.
- `require_permissions` / `require_roles`:
  - Assume a principal is already present (they depend on `get_auth_principal`); they do not perform their own authn.
  - When the principal lacks required permissions/roles, raise a forbidden error that maps to HTTP 403.
  - The 403 payload SHOULD include which permission(s) or role(s) failed, and that detail structure is considered part of the API surface once shipped.

---

## Scope

### In-Scope (v1)

- Define `AuthPrincipal` and `get_auth_principal` (may share implementation and file with Principal-Governance PRD).
- Update:
  - `permissions.py` to be claim-first and reduce DB lookups.
  - `User_DB_Handling` / `auth_deps` to rely on `get_auth_principal`.
- Introduce `require_permissions` / `require_roles` dependencies and use them in a selection of high-value endpoints (e.g., admin, media management).
- Ensure both deployment profiles (`local-single-user`, `multi-user-postgres`) use the same dependencies and claim semantics:
  - `local-single-user` maps the bootstrapped admin user to an `AuthPrincipal` with admin claims.
  - No dependency re-introduces `is_single_user_mode()` branching for authorization.

### Out of Scope (v1)

- Rewriting all endpoints to use the new dependencies immediately; focus on AuthNZ + a few representative routes.
- Removing all decorators in favor of dependencies; we may continue to support both.

---

## Risks & Mitigations

- **Risk: Stale claims after role/permission changes.**
  - Mitigation:
    - Keep claim TTL aligned with access-token lifetime.
    - For critical operations, optionally re-validate certain claims server-side (e.g., check “still admin”).

- **Risk: Hidden reliance on DB-based checks in existing code.**
  - Mitigation:
    - Instrument or temporarily log when DB fallbacks are used.
    - Add tests to cover key permission/role scenarios.

- **Risk: Test breakage where tests assume direct DB changes immediately affect permissions.**
  - Mitigation:
    - Document that changes take effect on next login/token issue/refresh.
    - Provide test helpers that issue new tokens after RBAC changes.

---

## Milestones & Phasing

### Phase 1: AuthPrincipal + Claim Invariants

- Implement `AuthPrincipal` and `get_auth_principal`.
- Ensure `User_DB_Handling` sets roles/permissions consistently.
- Add tests verifying:
  - A user’s permissions match DB configuration at login time.
  - `user.permissions` and `user.roles` are present and correct for JWT and API-key paths.
  - In the `local-single-user` profile, the bootstrapped admin user (`SINGLE_USER_FIXED_ID`) is represented as an `AuthPrincipal` with admin role and full claims, and no special-case `AUTH_MODE` logic is required in dependencies.
  - `AuthPrincipal.kind` is correctly set for different credential types (`user`, `api_key`, `service`, `anonymous`), and user-centric fields are only present when appropriate (e.g., single-user principals use `kind="user"` with `subject="single_user"` while still flowing through API-key-style paths via `token_type="api_key"`).

- **Status (v0.1)**: Done — validated by:
  - `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_state_consistency.py`
  - `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_jwt_happy_path.py`
  - `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_api_key_happy_path.py`
  - Domain-specific principal/state invariant suites:
    - Media/RAG: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_media_rag_invariants.py`
    - Tools execute: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_tools_invariants.py`
    - Evaluations list: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_evaluations_invariants.py`

### Phase 2: Claim-First Permissions

- Update `permissions.py` to rely on claims first.
- Ensure `is_single_user_mode()` is not used as an “allow-all” shortcut:
  - When `user.permissions` / `user.roles` are present, they are authoritative in both single-user and multi-user profiles (no DB lookup, no mode-based bypass).
  - When claims are absent, `check_permission` / `check_role` fall back to the configured `UserDatabase` for both profiles instead of auto-allowing in single-user mode.
- Add tests covering:
  - Multiple roles and overlapping permissions.
  - User-specific allow/deny overrides.
  - Admin implied permissions.
  - Single-user principals (`AuthPrincipal.kind="user"` with `subject="single_user"`) with insufficient claims receiving 403 from `require_permissions` / `require_roles`, matching multi-user semantics.

- **Status (v0.1)**: Done — covered by `tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py` and `tldw_Server_API/tests/AuthNZ/integration/test_rbac_admin_endpoints.py`.

### Phase 3: Unified Dependencies & Adoption

- Implement `require_permissions` / `require_roles` dependencies.
- Migrate a set of key endpoints (auth admin, media admin, RAG admin) to use them.
- Update `UsageLoggingMiddleware` and `llm_budget_guard` to use `AuthPrincipal`.
  - Metrics/admin, Resource-Governor admin, and selected chat/Prompt Studio surfaces are part of this adoption and are now **claim-first, tested**:
    - `POST /api/v1/metrics/reset` is gated via `require_roles("admin")` (claim-first) and covered by HTTP-level permissions tests in `tldw_Server_API/tests/AuthNZ_Unit/test_metrics_permissions_claims.py`.
    - Resource-Governor admin and diagnostics endpoints (`/api/v1/resource-governor/policy*`, `/api/v1/resource-governor/diag/*`) are gated via `get_auth_principal` + `require_roles("admin")`, with behavior validated by `tldw_Server_API/tests/AuthNZ_Unit/test_resource_governor_permissions_claims.py` and the `Resource_Governance` integration suite.
    - Chat slash-command discovery (`GET /api/v1/chat/commands`) filters commands through the centralized command authorization decision whenever slash commands are enabled, and the async command router enforces declared per-command permissions without any legacy flag gate; this is locked in by `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py` and `tldw_Server_API/tests/Chat_NEW/integration/test_chat_commands_endpoint.py`.
    - Prompt Studio dependencies (`get_prompt_studio_user`) build `user_context.is_admin` and `user_context.permissions` strictly from `User` claims (`roles`, `permissions`, `is_admin`) instead of `AUTH_MODE`; HTTP-level tests in `tldw_Server_API/tests/AuthNZ_Unit/test_prompt_studio_user_claims.py` and `tldw_Server_API/tests/prompt_studio/unit/test_prompt_studio_deps_headers.py` cover claim propagation and 401 behavior.
    - Media and RAG endpoints (`/api/v1/media/add`, `/api/v1/media/process-videos`, `/api/v1/media/process-web-scraping`, `/api/v1/rag/search*`) use `require_permissions` and are covered by RAG/media permissions tests on both SQLite and Postgres.
    - Evaluations CRUD and runs routes use `require_permissions(EVALS_READ)` / `require_permissions(EVALS_MANAGE)` and are covered by `test_evaluations_permissions_claims.py`.
    - Tools execution (`/api/v1/tools/execute`) requires `require_permissions("tools.execute:*")` and participates in the principal/state invariants suite.
    - Embeddings v5 admin and maintenance utilities use `require_roles("admin")` + `require_permissions(SYSTEM_CONFIGURE)` and `require_permissions(EMBEDDINGS_ADMIN)` and are covered by embeddings admin/model-management tests.
- Add route-level tests covering:
  - Endpoints that require a user principal, ensuring `get_current_user` / `require_permissions` fail with 403 when invoked with a `service` or `anonymous` principal lacking `user_id`.
  - Service-only endpoints (if any) that rely on `AuthPrincipal.kind=service` without requiring user-centric fields.

- **Status (v0.1)**: Done — core admin/chat/Prompt Studio/media/RAG/evaluations/tools/embeddings flows are claim-first and tested (see tests cited above), and remaining legacy helpers are explicitly documented as compatibility shims to be cleaned up in post-v0.1 iterations.

### Phase 4: Cleanup & Documentation

- Remove or deprecate DB-based permission checks on hot paths.
- Update AuthNZ README and API integration guide to:
  - Document how to use the new dependencies.
  - Provide examples for custom routes.

- **Status (v0.1)**: Done — primary guides are updated (see `Docs/Code_Documentation/Guides/AuthNZ_Code_Guide.md` and route-level claim tests in `tldw_Server_API/tests/AuthNZ_Unit/test_auth_claim_route_level.py`); any remaining documentation tweaks and legacy shim cleanup are tracked as post-v0.1 tech debt.

---

## Open Questions

- How should we handle cross-service calls where another service presents an internal token—should those produce a “service principal” with its own permissions?
  - No, Services should not present internal tokens across modules. They should pass/rely on the AuthPrincipal, since it is the ultimate source of truth.
- Do we want a formal notion of “scopes” (beyond permissions) for token issuance, or are permissions sufficient?
  - What would the difference be?
- How aggressively should we back-migrate existing endpoints vs. allowing both patterns for a time?
  - Plan for a full migration with a short compatibility window (e.g., one release), then retire legacy patterns.

---

## Success Criteria

- Most authorization checks in the critical path no longer hit the DB; they use claims on `AuthPrincipal` / `User`.
- There is a single, documented way to get the current authenticated principal/user in FastAPI endpoints.
- Middlewares and services that need auth context rely on `AuthPrincipal` rather than ad-hoc `request.state` fields.

## Verification & Regression Slice

- **Primary test modules relied on (v0.1)**:
  - Permissions and claim-first behavior: `tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py`.
  - Metrics admin and AuthNZ claim gates: `tldw_Server_API/tests/AuthNZ_Unit/test_metrics_permissions_claims.py`.
  - Resource-Governor admin/diag claim gates: `tldw_Server_API/tests/AuthNZ_Unit/test_resource_governor_permissions_claims.py` and `tldw_Server_API/tests/Resource_Governance/integration`.
  - Chat slash commands (claim-first routing): `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py` and `tldw_Server_API/tests/Chat_NEW/integration/test_chat_commands_endpoint.py`.
  - Prompt Studio dependencies and claims: `tldw_Server_API/tests/AuthNZ_Unit/test_prompt_studio_user_claims.py` and `tldw_Server_API/tests/prompt_studio/unit/test_prompt_studio_deps_headers.py`.
- **Recommended regression slice (SQLite/Postgres)**:
  - SQLite-focused slice: `python -m pytest tldw_Server_API/tests/AuthNZ_SQLite -m "not slow"` to exercise AuthNZ SQLite behavior (sessions, rate limits, usage, API keys).
  - Postgres-focused slice: `python -m pytest -m "not slow" tldw_Server_API/tests/AuthNZ_Postgres tldw_Server_API/tests/AuthNZ/integration` to cover AuthNZ Postgres repos and HTTP-level auth/guardrail flows.

## Implementation Plan

### Stage 1: AuthPrincipal + Claim Invariants
**Goal**: Establish `AuthPrincipal` and `get_auth_principal` as the single source of truth for authenticated identity and claims.

**Success Criteria**:
- `get_auth_principal` is used by core auth dependencies to derive current user context.
- `AuthPrincipal` always carries `roles`, `permissions`, and `is_admin` consistent with RBAC tables at authentication time.

**Tests**:
- Unit tests validating `AuthPrincipal` creation for:
  - JWT access tokens.
  - API keys (regular and virtual).
- Integration tests confirming `User` objects derived from `AuthPrincipal` contain correct roles/permissions.

**Status**: Done

**Notes**:
- `AuthPrincipal` / `AuthContext` are implemented in `tldw_Server_API/app/core/AuthNZ/principal_model.py`, and `get_auth_principal(request)` is implemented in `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py` with unit tests for single-user, JWT, and API-key flows.
- Core dependencies (`auth_deps.get_current_user`, `auth_deps.get_current_active_user`, `User_DB_Handling.verify_jwt_and_fetch_user` / `get_request_user`) now populate `request.state.auth` / `AuthPrincipal` from their resolved user context and, in multi-user flows, reuse an existing `AuthContext`/`AuthPrincipal` when present instead of re-running JWT/API-key logic, while preserving existing 401/403 semantics.
- Integration coverage includes:
  - `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_state_consistency.py` – single-user API-key flow and 401 semantics for `get_current_user` vs `get_auth_principal`, plus alignment between `get_current_user` / `get_request_user` and `AuthPrincipal`.
  - `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_jwt_happy_path.py` – full multi-user JWT flow (register → login → protected endpoint) asserting that `AuthPrincipal` aligns with `get_current_user`, `get_request_user`, and `request.state` on a real FastAPI app instance.
  - `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_api_key_happy_path.py` – multi-user API-key flow asserting that `AuthPrincipal`, `get_current_user`, `get_request_user`, and `request.state` stay in sync (including `api_key_id`) for a real FastAPI route.
  - `tldw_Server_API/tests/AuthNZ_Unit/test_authnz_invariants.py` – unit-level invariants for 401 semantics on missing credentials and wrapper behavior for `get_current_user`, `get_current_active_user`, and `get_request_user` across single-user and multi-user modes.

### Stage 2: Claim-First Permissions
**Goal**: Make runtime permission and role checks rely solely on claims in the common path.

**Success Criteria**:
- `permissions.py` does not hit the DB during typical request processing; it uses `user.permissions` and `user.roles`.
- Any DB fallback logic is either removed or gated behind explicit debug/admin paths.

**Tests**:
- Permission test matrix covering:
  - Multiple role combinations.
  - User-specific allow/deny overrides.
  - Admin implied permissions.

**Status**: Done

**Notes**:
- `permissions.py` now treats `user.permissions` / `user.roles` as authoritative when present, returning False without hitting the DB when claims exist but lack the required permission/role.
- DB-based fallbacks are retained only for caller contexts that do not provide claim lists at all (e.g., legacy user objects without `permissions` / `roles` attributes), reducing database usage on typical, claim-bearing code paths. Additional tests explicitly simulate DB unavailability when claims are present to prove that hot paths remain purely claim-based (`tests/AuthNZ_Unit/test_permissions_claim_first.py::test_check_permission_uses_claims_even_if_db_unavailable` and `::test_check_role_uses_claims_even_if_db_unavailable`).
- Route coverage expanding: admin router now enforces `require_roles("admin")` in addition to legacy `require_admin`, and media ingestion’s `/process-web-scraping` endpoint is now gated via `require_permissions(MEDIA_CREATE)` plus `rbac_rate_limit("media.create")` (no legacy `PermissionChecker`). Claim-first `require_permissions` / `require_roles` matrices and HTTP-level admin 401/403 semantics are covered in `tests/AuthNZ_Unit/test_permissions_claim_first.py` and `tests/AuthNZ/integration/test_rbac_admin_endpoints.py::test_admin_roles_require_auth_and_admin`.
- Single-user deployments now use the same claim-first semantics as multi-user for permission and role checks: `permissions.py` no longer treats `is_single_user_mode()` as an “allow-all” fallback when claims are missing. New tests in `tests/AuthNZ_Unit/test_permissions_claim_first.py` (e.g., `test_check_permission_single_user_mode_prefers_claims`, `test_check_permission_single_user_mode_without_claims_falls_back_to_db`, `test_check_role_single_user_mode_treats_admin_as_admin_and_user`, `test_check_role_single_user_mode_without_roles_falls_back_to_db`) assert that single-user admins are governed by claims and DB fallbacks rather than global mode flags.
- API-key authentication now enriches users with roles/permissions from RBAC tables (matching the JWT path) so claim-first checks succeed for X-API-KEY callers on protected routes.
- Usage logging middleware now derives `user_id` / `api_key_id` from `AuthPrincipal` when `request.state.auth` is present, falling back to legacy `request.state` attributes for compatibility, aligning usage metrics with the unified principal model.
- Additional matrix coverage has been added for the “any” / “all” helpers to ensure they also remain claim-first when claims are attached, even if the RBAC DB is unavailable:
  - `tests/AuthNZ_Unit/test_permissions_claim_first.py::test_check_any_permission_uses_claims_even_if_db_unavailable` and `::test_check_all_permissions_uses_claims_even_if_db_unavailable` validate that `check_any_permission` / `check_all_permissions` never call `get_user_database` when `user.permissions` is a list, and instead rely purely on the claim set.

### Stage 5: Legacy Admin Endpoints – Cleanup Checklist

The following admin/diagnostic endpoints still rely on legacy helpers (for example `require_admin` on user dicts) and are intentionally left as compatibility shims. New work SHOULD NOT extend these patterns; future cleanup can either migrate them fully to `AuthPrincipal` + `require_roles` / `require_permissions` or explicitly codify them as permanent exceptions.

- [x] Evaluations heavy admin helper
  - `tldw_Server_API/app/api/v1/endpoints/evaluations_auth.py:require_admin` (gated by `EVALS_HEAVY_ADMIN_ONLY`) remains available as a legacy shim for compatibility-only callsites, but heavy evaluations admin endpoints themselves now use the claim-first `enforce_heavy_evaluations_admin(AuthPrincipal)` helper alongside `require_roles("admin")` / `get_auth_principal`.
  - `tldw_Server_API/app/api/v1/endpoints/evaluations_unified.py:admin_cleanup_idempotency` and embeddings A/B test admin endpoints are wired through `require_roles("admin")` at the router plus `enforce_heavy_evaluations_admin(principal)` in the handler. Behavior is covered by both unit- and HTTP-level tests (`tldw_Server_API/tests/AuthNZ_Unit/test_evaluations_heavy_admin_claims.py` and `tldw_Server_API/tests/AuthNZ/integration/test_evaluations_permissions_claims.py`), including principals with `is_admin=True, roles=["admin"]` and principals with `is_admin=False, roles=["admin"]` to confirm that both forms are treated equivalently by the heavy-admin gate.
- [x] Embeddings v5 production admin/maintenance endpoints
  - `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py` now relies solely on claim-first gates for its admin and maintenance surfaces. Core diagnostics such as `/api/v1/embeddings/metrics` and model warmup continue to use `require_roles("admin")` + `require_permissions(SYSTEM_CONFIGURE)` as their gate, while internal maintenance utilities (compactor runs, job priority overrides, DLQ tools, stage controls, job-skip registry, ledger inspection, re-embed scheduling, and orchestrator diagnostics) are gated by the dedicated `embeddings.admin` permission (`EMBEDDINGS_ADMIN`) via `require_permissions(EMBEDDINGS_ADMIN)`.
  - Legacy helper `require_admin(current_user)` has been removed from the embeddings v5 module; new code MUST use `get_auth_principal` + `require_permissions` / `require_roles` instead of per-module admin shims. Behavior for the `embeddings.admin` gate and related model-management routes is covered by `tldw_Server_API/tests/AuthNZ_Unit/test_embeddings_admin_claims.py` and `tldw_Server_API/tests/AuthNZ_Unit/test_embeddings_model_management_permissions_claims.py`, which exercise 401/403/200 semantics for both admin principals and non-admin principals that hold `EMBEDDINGS_ADMIN`.
- [x] Flashcards import abuse‑cap overrides
  - `tldw_Server_API/app/api/v1/endpoints/flashcards.py::import_flashcards` and the file-based import route gate override parameters (`max_lines`, `max_line_length`, `max_field_length`, `max_items`) via `AuthPrincipal` claims: callers must have either the `flashcards.admin` permission (`FLASHCARDS_ADMIN`) or admin role/`is_admin` set; base imports without overrides remain available to regular authenticated users. This behavior is explicitly covered by `tldw_Server_API/tests/AuthNZ_Unit/test_flashcards_admin_permissions_claims.py`.
  - Cleanup target: this is now treated as a documented abuse‑cap exception with `FLASHCARDS_ADMIN` as the canonical gate, with `flashcards.admin` included in the Privilege Metadata Catalog so flashcards admins appear in RBAC privilege maps.
- [x] MCP unified diagnostics helper
- `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py::get_server_metrics` is now fully claim-first, gated via `require_permissions("system.logs")` on `AuthPrincipal` (with `system.logs` treated as the dedicated diagnostics permission). The `/mcp/modules/health` endpoint remains wired through the same `require_permissions("system.logs")` dependency.
- The Prometheus scrape endpoint (`/api/v1/mcp/metrics/prometheus`) has been migrated to the same claim-first guard and now depends on `require_permissions("system.logs")` instead of the legacy `require_admin_unless_public` helper. Deployments that previously relied on `MCP_PROMETHEUS_PUBLIC=1` for unauthenticated scraping must now provide credentials with `system.logs` (or admin-style) claims to Prometheus; any remaining references to the public flag are treated as deprecated and will be removed in future iterations. HTTP behavior for the diagnostics endpoints is covered by `tldw_Server_API/tests/AuthNZ_Unit/test_mcp_admin_permissions_claims.py` together with the MCP metrics endpoint tests.

### Stage 3: Unified Dependencies & Adoption
**Goal**: Provide and adopt unified auth dependencies across a set of key endpoints and middlewares.

**Success Criteria**:
- New dependencies (`get_auth_principal`, `get_current_user`, `require_permissions`, `require_roles`) are implemented and documented.
- Representative admin and media/RAG endpoints use `require_permissions` / `require_roles`.
- `UsageLoggingMiddleware` and `llm_budget_guard` derive context from `AuthPrincipal`.

**Tests**:
- Route-level tests ensuring that:
  - Required permissions/roles are enforced correctly.
  - Usage logging and budget guard see the expected principal data.

**Status**: Done (with documented legacy surfaces)

**Notes**:
- New claim-first FastAPI dependencies (`get_auth_principal`, `require_permissions`, `require_roles`) are implemented in `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py` with unit tests.
- A representative media endpoint (`/api/v1/media/add`) and an admin endpoint (`/api/v1/metrics/reset`) now also enforce claims using these dependencies alongside their existing auth checks.
- `llm_budget_guard` has been updated to consume `AuthPrincipal` (via `AuthContext` / `request.state.auth`) when consulting governance for LLM virtual-key budgets.
- RAG search endpoints (`/api/v1/rag/search`, `/search/stream`, `/simple`, `/batch`) and media processing (`/api/v1/media/process-videos`, `/api/v1/media/process-web-scraping`, `/api/v1/media/add`) now depend on `require_permissions`, with integration coverage (skipped when Postgres is unavailable) in `tests/AuthNZ/integration/test_rag_media_permissions_claims.py` validating 401/403 semantics for JWT and API-key flows.
- Claim-first route-level semantics are additionally covered by dedicated tests that stub `get_auth_principal` and exercise `require_permissions` / `require_roles` under different principal kinds, including `service` and `anonymous` (`tests/AuthNZ_Unit/test_auth_claim_route_level.py`), ensuring clear 401 vs 403 behavior when principals are missing or lack required claims.
- Notes graph endpoints now adopt `require_permissions` for claim-first enforcement in addition to token scopes: `tldw_Server_API/app/api/v1/endpoints/notes_graph.py` uses `require_permissions(NOTES_GRAPH_READ)` / `require_permissions(NOTES_GRAPH_WRITE)` alongside `require_token_scope("notes", ...)` and rate limiting. Integration coverage is provided by `tldw_Server_API/tests/Notes_NEW/integration/test_notes_graph_rbac.py`, which:
  - Asserts that a virtual key with the wrong `scope` receives 403 for both read (`GET /api/v1/notes/graph`) and write (`POST /api/v1/notes/{note_id}/links`) operations.
  - Asserts that a virtual key with `scope="notes"` and a principal with appropriate graph permissions succeeds (200) for both read and write paths.
- SQLite regression coverage mirrors the claim-first HTTP semantics without Postgres by stubbing RAG/media backends in `tests/AuthNZ_SQLite/test_rag_media_permissions_sqlite.py`.
- Evaluations CRUD and runs endpoints are now wired through claim-first dependencies as a high-value administrative surface. `tldw_Server_API/app/api/v1/endpoints/evaluations_crud.py` imports `require_permissions` and uses dedicated permissions (`EVALS_READ`, `EVALS_MANAGE`) from `tldw_Server_API/app/core/AuthNZ/permissions.py`:
  - Read-oriented routes (`GET /api/v1/evaluations/`, `GET /api/v1/evaluations/{eval_id}`, and list/read run endpoints) depend on `require_permissions(EVALS_READ)` in addition to existing API-key checks.
  - Mutating routes (`POST /api/v1/evaluations/`, `PATCH /api/v1/evaluations/{eval_id}`, `DELETE /api/v1/evaluations/{eval_id}`, `POST /api/v1/evaluations/{eval_id}/runs`, and `POST /api/v1/evaluations/runs/{run_id}/cancel`) depend on `require_permissions(EVALS_MANAGE)` alongside the existing RBAC rate limits and auth helpers.
  - Route-level tests in `tldw_Server_API/tests/AuthNZ/integration/test_evaluations_permissions_claims.py` stub `get_auth_principal`, `verify_api_key`, and `get_request_user` to assert:
    - 403 responses for principals that are authenticated but lack the required `evals.read` / `evals.manage` permissions on list endpoints.
    - 200 responses for principals with appropriate evaluation permissions while the underlying evaluation service is stubbed, ensuring claim-based allow paths behave as expected without touching the real Evaluations DB.
    - The existing `require_admin` helper for heavy evaluations remains the admin gate for `/api/v1/evaluations/admin/idempotency/cleanup`, and the test file documents its behavior as an admin-only compatibility shim independent of the new `require_permissions` wiring on CRUD routes.
- Workflow scheduler admin endpoints now also participate in the unified, claim-first dependency stack:
  - `tldw_Server_API/app/core/AuthNZ/permissions.py` defines `WORKFLOWS_ADMIN = "workflows.admin"` as the canonical permission for scheduler administration.
  - `tldw_Server_API/app/api/v1/endpoints/scheduler_workflows.py::admin_rescan` now depends on both:
    - `require_token_scope("workflows", require_if_present=True, endpoint_id="scheduler.workflows.admin_rescan")` for token-scoped gating, and
    - `require_permissions(WORKFLOWS_ADMIN)` for claim-first enforcement via `AuthPrincipal`.
  - Route-level tests in `tldw_Server_API/tests/AuthNZ_Unit/test_scheduler_workflows_permissions_claims.py` build a small FastAPI app around the scheduler router and stub:
    - `get_auth_principal` to attach an `AuthContext` with different principal kinds (`user`, `service`).
    - `get_request_user` and `require_token_scope` to avoid touching real DB/scope logic.
    - `get_workflows_scheduler` to a fake scheduler that records `_rescan_once` calls.
  - The tests assert that:
    - Non-admin principals without scheduler claims receive 403 for `POST /api/v1/scheduler/workflows/admin/rescan`.
    - User and service principals with `is_admin=True` succeed (200) and trigger the fake scheduler’s rescan call, confirming that admin-style claims (via `is_admin`) remain the primary gate while also flowing through the `require_permissions` / `AuthPrincipal` stack for observability and future fine-grained permissions.

- MLX provider lifecycle endpoints are now covered by the same claim-first RBAC conventions:
  - `tldw_Server_API/app/api/v1/endpoints/mlx.py` gates `POST /api/v1/llm/providers/mlx/load`, `POST /api/v1/llm/providers/mlx/unload`, and `GET /api/v1/llm/providers/mlx/status` via `dependencies=[Depends(check_rate_limit), Depends(require_roles("admin"))]`, ensuring:
    - 401 responses when `get_auth_principal` fails (no/invalid credentials).
    - 403 responses when an authenticated principal lacks the `admin` role (and is not `principal.is_admin`).
    - Successful 2xx semantics only once the unified dependencies have admitted an admin principal.
  - Route-level tests in `tldw_Server_API/tests/AuthNZ_Unit/test_mlx_permissions_claims.py` override `get_auth_principal` and the MLX registry to assert:
    - 401 for unauthenticated requests.
    - 403 for non-admin principals.
    - 200 for admin principals, with the underlying registry stub confirming that the load operation was invoked.

- Workflows virtual-key issuance now uses both role- and permission-based guards:
  - `tldw_Server_API/app/api/v1/endpoints/workflows.py::workflows_virtual_key` is protected by:
    - `Depends(auth_deps.require_roles("admin"))` to require an admin principal, and
    - `Depends(auth_deps.require_permissions(WORKFLOWS_ADMIN))` to require the `workflows.admin` permission in addition to admin role.
  - The handler preserves the documented mode behavior by returning HTTP 400 when `AUTH_MODE != "multi_user"` (`"Virtual keys only apply in multi-user mode"`), even for privileged principals.
  - Route-level tests in `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_virtual_key_permissions_claims.py` validate:
    - 401 when `get_auth_principal` fails.
    - 403 when a non-admin/non-`WORKFLOWS_ADMIN` principal attempts to mint a virtual key in multi-user mode.
    - 400 for privileged principals when `AUTH_MODE=single_user`.
    - 200 with a stubbed JWT service when an admin principal with `WORKFLOWS_ADMIN` runs in multi-user mode, confirming that the combined role+permission gate is enforced via the unified dependencies.

- Topic monitoring admin endpoints (`/api/v1/monitoring/...`) now also enforce claim-first diagnostics permissions: `tldw_Server_API/app/api/v1/endpoints/monitoring.py` adds `dependencies=[Depends(require_permissions("system.logs"))]` on watchlist/alert/notification routes in addition to `require_admin`. Route-level tests in `tldw_Server_API/tests/AuthNZ_Unit/test_monitoring_permissions_claims.py` cover 401 (no principal), 403 (principal without `system.logs`), and 200 (admin principal) cases while stubbing the underlying monitoring service.
- MCP admin diagnostics are beginning to adopt the same pattern: `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py::get_modules_health` now depends solely on `require_permissions("system.logs")` together with the global `get_auth_principal` resolver, and route-level tests in `tldw_Server_API/tests/AuthNZ_Unit/test_mcp_admin_permissions_claims.py` exercise 401 vs 403 vs 200 behavior by overriding `get_auth_principal` for different principals.

**Remaining legacy/compatibility surfaces (explicit carve-outs)**

- Heavy evaluations admin helpers:
  - `tldw_Server_API/app/api/v1/endpoints/evaluations_auth.py::require_admin` – a helper guarded by `EVALS_HEAVY_ADMIN_ONLY` for heavy evaluations flows. New evaluations endpoints rely on `require_permissions(EVALS_READ/EVALS_MANAGE)`; this helper is retained as a legacy shim and MUST NOT be used on new routes.
  - `tldw_Server_API/app/api/v1/endpoints/evaluations_unified.py::admin_cleanup_idempotency` – uses `Depends(require_roles("admin"))` at the router plus `require_admin(current_user)` as a secondary gate to mirror legacy “heavy admin” behavior. This dual gate is treated as intentionally legacy for this admin-only cleanup path.
- Embeddings v5 production admin utilities:
  - `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py` no longer exposes a local `require_admin(current_user)` helper. All admin and maintenance endpoints are gated via claim-first dependencies: core diagnostics such as `/api/v1/embeddings/metrics` and model warmup use `require_roles("admin")` + `require_permissions(SYSTEM_CONFIGURE)`, while internal maintenance utilities (compactor runs, job priority overrides, DLQ tools, stage controls, job-skip registry, ledger inspection, re-embed scheduling, and orchestrator diagnostics) rely on the dedicated `embeddings.admin` permission (`EMBEDDINGS_ADMIN`) via `require_permissions(EMBEDDINGS_ADMIN)`. HTTP behavior for these gates is covered by `tldw_Server_API/tests/AuthNZ_Unit/test_embeddings_admin_claims.py` and `tldw_Server_API/tests/AuthNZ_Unit/test_embeddings_model_management_permissions_claims.py`.
- Flashcards import “abuse caps”:
  - `tldw_Server_API/app/api/v1/endpoints/flashcards.py::import_flashcards` and `/flashcards/import-file` require a claim‑first flashcards admin gate (`_require_flashcards_admin(principal)` → `FLASHCARDS_ADMIN` permission or admin role) only when optional query parameters attempt to override import limits (line count, field length, etc.). Base imports remain available to regular authenticated users. These checks are narrowly scoped to abuse‑cap overrides and are not a general authorization pattern.
- MCP unified endpoint:
- `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py` now relies exclusively on claim-first guards such as `require_permissions("system.logs")` for diagnostics (`/mcp/modules/health`, `/mcp/metrics`, `/mcp/metrics/prometheus`) together with `get_auth_principal`. The earlier module-local `require_admin` helper has been removed; new MCP admin/diagnostics routes must continue to use `get_auth_principal` + `require_permissions` / `require_roles` rather than introducing new per-module admin shims.

### Remaining Mode-Based Decisions

- A fresh `is_single_user_mode()` audit confirms that mode checks are limited to:
  - Coordination/governance (startup banners, WebUI API-key injection for local single-user, ChaChaNotes warm-up, backpressure/tenant RPS toggles, embedding quota defaults).
  - Core auth flows (single-user API key vs multi-user JWT/API-key) and diagnostics (e.g., MCP single-user API key acceptance), with authorization decisions driven by claims on `AuthPrincipal` / `User`.
  - Rate limiting and backpressure helpers that bypass 429s for the configured single-user API key in local/dev scenarios (`auth_deps.check_rate_limit`, `auth_deps.check_auth_rate_limit`, and `backpressure.guard_backpressure_and_quota`).
- Jobs admin domain-scoped RBAC no longer branches on `is_single_user_mode()`. `_enforce_domain_scope` / `_enforce_domain_scope_unified` in `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` are now governed purely by explicit environment toggles (`JOBS_DOMAIN_SCOPED_RBAC`, `JOBS_REQUIRE_DOMAIN_FILTER`, `JOBS_DOMAIN_ALLOWLIST_*`, `JOBS_RBAC_FORCE`) and by `AuthPrincipal`-based admin gates on the router. Earlier “single-user bypass” language for jobs admin should be treated as historical.

### Stage 4: Cleanup & Documentation
**Goal**: Remove legacy, overlapping auth dependencies and update documentation.

**Success Criteria**:
- Deprecated or redundant auth dependencies are either removed or clearly marked.
- AuthNZ README and integration guide contain examples using the unified dependencies.

**Tests**:
- Documentation lint/checks where applicable, plus smoke tests for routes updated to the new dependencies.

**Status**: Done

**Notes**:
- `Docs/Code_Documentation/Guides/AuthNZ_Code_Guide.md` explicitly documents the split between modern claim-first dependencies and legacy compatibility shims:
  - Modern pattern:
    - `get_auth_principal` → returns `AuthPrincipal` with roles/permissions.
    - `require_permissions` / `require_roles` → enforce claims and return the principal; representative usage is called out for media, RAG, notes graph, evaluations CRUD, scheduler workflows admin, and chat queue diagnostics (`system.logs`).
  - Legacy shims (now removed or explicitly marked):
    - Earlier iterations exposed FastAPI dependencies `PermissionChecker`, `RoleChecker`, `AnyPermissionChecker`, and `AllPermissionsChecker` from `permissions.py`. A repository-wide audit of `tldw_Server_API/app/api/v1/endpoints` confirmed that no FastAPI routes were using these helpers as their primary gate; they have since been removed from the codebase, and historical documentation/examples that mention them should be treated as legacy only.
    - `require_admin` in evaluations auth is documented as an admin-only gate for heavy evaluations flows, while new admin surfaces should prefer `get_auth_principal` plus `require_permissions` / `require_roles`.
    - `auth_deps.require_role` and `auth_deps.get_optional_current_user` remain available as compatibility shims for existing routes but are now explicitly marked in code and in the AuthNZ Code Guide as **not for new endpoints**. New work MUST use claim-first helpers (`get_auth_principal`, `require_permissions`, `require_roles`) instead of introducing additional usages of these legacy dependencies.
- The code guide now includes a short “securing a new route” example that shows:
  - Defining a permission constant in `permissions.py`.
  - Applying `Depends(require_permissions("your.permission"))` to an endpoint.
  - Overriding `get_auth_principal` in tests to exercise 401 vs 403 semantics, reusing patterns from `tests/AuthNZ_Unit/test_auth_claim_route_level.py` and `tests/AuthNZ_Unit/test_scheduler_workflows_permissions_claims.py`.
- Selected RBAC helpers now use `AuthnzRbacRepo`, and the admin `GET /api/v1/admin/roles/{role_id}/permissions/effective` endpoint delegates to `AuthnzRbacRepo.get_role_effective_permissions`, with behavior locked in by SQLite and Postgres-backed admin endpoint tests.
- Invariant and wrapper suites are treated as living safety nets: when new auth wrappers or admin/control surfaces are introduced, they should be accompanied by updates to `tldw_Server_API/tests/AuthNZ_Unit/test_authnz_invariants.py`, `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_state_consistency.py`, and the relevant domain-specific invariant modules so that claim-first behavior and principal/state alignment remain explicitly tested across JWT, API-key, and single-user flows.
- As of this PRD snapshot, the `*Checker` helpers have been removed from `permissions.py`; new code MUST continue to rely solely on `get_auth_principal` / `require_permissions` / `require_roles` for authorization, and any new helper should be composed on top of these primitives rather than reintroducing decorator-style gates.

### Remaining adoption checklist (mode-based logic & legacy deps)

- Mode-based coordination vs auth decisions:
  - `tldw_Server_API/app/main.py`: `is_single_user_mode()` is used only for startup banners, ChaChaNotes warm-up, and WebUI config hints (including API-key injection for local single-user); no authorization decisions depend on this flag here.
  - `tldw_Server_API/app/core/PrivilegeMaps/service.py::PrivilegeMapService._build_user_dataset`: uses `is_single_user_mode()` only to choose a default role for the synthetic single-user fallback dataset when the AuthNZ DB is empty; this is a reporting/UX concern, not a gate.
  - `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py` and `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py` branch on `is_single_user_mode()` purely to select between fixed API-key vs JWT flows; in all cases, credentials are still verified and authorization is claim-first on the resulting `User`/`AuthPrincipal`.
  - `tldw_Server_API/app/api/v1/API_Deps/backpressure.py::_is_single_user_mode_runtime` and embeddings tenant quotas in `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py` use mode/profile checks only to disable tenant-style RPS quotas in local single-user/dev scenarios; they do not bypass claim-based authorization.

- Legacy admin dependencies (`require_admin`) and callers:
  - `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py::require_admin`: remains as a compatibility shim; new endpoints must prefer `get_auth_principal` + `require_roles("admin")` / `require_permissions(...)`.
  - Moderation admin endpoints (`tldw_Server_API/app/api/v1/endpoints/moderation.py::*`) now use claim-first admin gates on the router (`Depends(require_roles("admin"))` + `Depends(require_permissions(SYSTEM_CONFIGURE))`) with no remaining `require_admin` dependency. These share the same 401/403 semantics as other admin/config panels and are covered by moderation guardrail tests.
  - Metrics and setup: `POST /api/v1/metrics/reset` (`metrics.reset_metrics`) now relies solely on `Depends(require_roles("admin"))`, and `POST /api/v1/setup/reset` (`setup.reset_setup_flags`) is guarded by `require_roles("admin")` + `require_permissions(SYSTEM_CONFIGURE)`. Both endpoints are exercised via `tests/AuthNZ_Unit/test_metrics_permissions_claims.py` and `tests/integration/test_setup_reset.py`, which override `get_auth_principal` to validate 401/403/200 behavior.
  - Evaluations: `tldw_Server_API/app/api/v1/endpoints/evaluations_auth.py::require_admin` is now wrapped by `enforce_heavy_evaluations_admin(principal)` for heavy-admin flows; any remaining direct `require_admin(user)` callsites in evaluations routes are candidates to migrate to `enforce_heavy_evaluations_admin` + claim-first guards.
  - Embeddings v5 admin utilities: core diagnostics such as `/api/v1/embeddings/metrics` and model warmup use claim-first gates (`require_roles("admin")` + `require_permissions(SYSTEM_CONFIGURE)`), with HTTP behavior locked in by `tests/AuthNZ_Unit/test_embeddings_admin_claims.py` and `test_embeddings_model_management_permissions_claims.py`. Any older `require_admin(current_user)` helper now lives only in a backup module and is not part of the supported surface.
- MCP unified diagnostics: `/mcp/modules/health`, `/mcp/metrics`, and `/mcp/metrics/prometheus` are guarded exclusively via `get_auth_principal` + `require_permissions("system.logs")` (and related claim-first helpers). There is no longer a module-local `require_admin` helper in `mcp_unified_endpoint.py`, and new MCP admin endpoints must continue to use claim-first dependencies rather than introducing decorator-style admin shortcuts.

- Request-state usage (`request.state.user_id` / `api_key_id`):
  - Writers of `request.state.user_id` and `request.state.api_key_id` are centralized in AuthNZ (`auth_principal_resolver`, `User_DB_Handling.get_request_user`, `auth_deps.get_current_user`, LLM budget guard/middleware, CSRF protection); each of these derives values from authenticated credentials and `AuthPrincipal`.
  - Consumers in other modules (for example, `tldw_Server_API/app/core/Resource_Governance/deps.py::derive_entity_key`, usage logging middleware, and selected quota helpers) must continue to treat `request.state.user_id` as a derived context field, not as an authorization decision point. New code MUST NOT introduce fresh authorization checks that gate on `request.state.user_id` or `request.state.api_key_id` directly; it should always rely on `AuthPrincipal` claims via `get_auth_principal`.
