# User Unification PRD (v0.1)

## Summary

AuthNZ supports:

- **Single-user mode**: a fixed API key (`SINGLE_USER_API_KEY`), a synthetic user, SQLite users DB.
- **Multi-user mode**: username/password + MFA → JWT access/refresh, sessions, RBAC, orgs/teams, API keys, virtual keys.
- **Multiple DB backends**: SQLite by default; Postgres for production, with many dialect-specific code paths.

Today these are implemented as separate “modes” with special cases scattered across AuthNZ, and many modules contain inline Postgres/SQLite branching and DDL. This PRD proposes:

1. Treating single-user as a constrained configuration of the multi-user model (a bootstrap user + key) instead of a separate runtime mode.
2. Consolidating backend differences behind a small set of repositories, so most of AuthNZ does not care about SQLite vs Postgres.

The goal is to reduce special-case handling, make the mental model “always multi-user under the hood”, and simplify future migrations.

## Related Documents

- `Docs/Product/Completed/AuthNZ-Refactor/Principal-Governance-PRD.md` – principal model (`AuthPrincipal` / `AuthContext`) and AuthNZ guardrails.
- `Docs/Product/Completed/AuthNZ-Refactor/User-Auth-Deps-PRD.md` – unified auth dependencies, claim-first authorization, and FastAPI wiring.
- `Docs/Product/Completed/AuthNZ-Refactor/Resource_Governor_PRD.md` – global, cross-module resource governance (`ResourceGovernor`) used by AuthNZ guardrails.

---

## Problems & Symptoms

### 1. Mode Branching (Single vs Multi User)

- `settings.AUTH_MODE` influences:
  - JWT secret/algorithm validation (`get_jwt_secret`, `validate_jwt_secret`).
  - Database URL fallback (`_apply_single_user_fallback` forcing SQLite for single-user).
  - Permission checks (`permissions.check_*` treating single-user as full allow).
  - User fetching & API-key paths (`User_DB_Handling`).
- Many functions check `is_single_user_mode()` / `is_multi_user_mode()` and diverge behavior.
- Single-user mode bypasses much of the richer auth stack:
  - No JWTs or sessions.
  - No persistent user row required.
  - Permissions effectively bypassed.

### 2. Backend-Specific Logic Everywhere

- Many modules contain `if hasattr(conn, 'fetchval')` branches:
  - `api_key_manager`, `virtual_keys`, `rate_limiter`, `orgs_teams`, `quotas`, `usage_logging_middleware`, `token_blacklist`, etc.
- Each module:
  - Issues its own `CREATE TABLE IF NOT EXISTS` statements.
  - Maintains separate SQL for Postgres vs SQLite (placeholders, JSON vs TEXT, type differences).
- Backend detection is repeated (`DatabasePool.pool` check, `$N` → `?` conversions).

### 3. Cognitive Overhead and Risk

- Contributors must think “Does this run in single-user mode? multi-user? both? Which DB?” for every change.
- Dialect branches are easy to get wrong and hard to test exhaustively.
- Schema evolution is scattered; a new table/column may be created by multiple modules with slightly different DDL.

---

## Goals

### Primary Goals

- **G1: Conceptual Unification of User Model**
  - Make single-user a deployment profile of the multi-user system:
    - Exactly one admin user.
    - Exactly one primary API key.
    - Registration/login endpoints disabled by config.
  - Keep runtime auth and RBAC code identical across profiles.

- **G2: Minimize Mode Branching in Code**
  - Reduce direct `is_single_user_mode()` checks to a handful of coordination points (bootstrap, feature gating), not per-call behavior.
  - Permissions and roles should be determined by data (claims) rather than mode.

- **G3: Backend as Implementation Detail**
  - Create a small set of repository abstractions for AuthNZ data (users, sessions, API keys, orgs/teams, RBAC, usage) to hide Postgres/SQLite specifics.
  - Move DDL and migrations out of business logic modules into centralized migrations and/or repositories.

### Secondary Goals

- Simplify local environment setup and documentation:
  - “Single-user dev mode” becomes “multi-user with one bootstrapped account over SQLite”.
- Make it possible to switch from SQLite to Postgres without touching business code.

### Non-Goals (Initial Version)

- Removing `AUTH_MODE` entirely in v1; we will deprecate its usage gradually.
- Unifying all DB access patterns across the entire project (this PRD focuses on AuthNZ).

---

## Proposed Solution

### 1. Single-User as a Bootstrap Profile

#### Concept

- Single-user deployments are just multi-user deployments seeded with:
  - One admin user in `users`.
  - One primary API key linked to that user.
  - Optional one “service” token for internal automation (e.g., scheduler).
- Registration/login endpoints and admin user-management endpoints are restricted by configuration.

#### Implementation Sketch

- Add a “bootstrap” function in `AuthNZ.initialize`:
  - Checks `AUTH_MODE` and a new setting like `SINGLE_USER_BOOTSTRAP_ENABLED`.
  - Ensures:
    - A user with a deterministic id (e.g., `SINGLE_USER_FIXED_ID`) and username (e.g., `single_user`) exists; if not, creates it.
    - A primary API key for that user exists; if not, create via `APIKeyManager` and display/record it.
    - That user has `admin` role and full permissions (via RBAC tables).
- Replace mode-specific auth logic with shared flows:
  - Single-user API key verification uses normal API-key code paths via `APIKeyManager`.
  - `AuthPrincipal` (from Principal-Governance PRD) always represents a user with claims; single-user is just a user with admin role and one API key.

#### Bootstrap semantics & idempotency

- **Existing `SINGLE_USER_API_KEY`**:
  - If `SINGLE_USER_API_KEY` is set, bootstrap MUST treat it as the canonical on-wire key.
  - Bootstrap hashes/persists this key for `SINGLE_USER_FIXED_ID` if no matching API-key row exists yet, rather than generating a new key.
  - This preserves existing single-user deployments without requiring users to rotate or rediscover their key.
- **No `SINGLE_USER_API_KEY` configured**:
  - If no key is configured, bootstrap MAY generate a new primary API key for `SINGLE_USER_FIXED_ID`.
  - Emission of the generated key (logs vs stdout vs “once-only” file) is controlled by configuration and documented.
- **Idempotency**:
  - Bootstrap is safe to run multiple times: it ensures exactly one admin user with `SINGLE_USER_FIXED_ID` and exactly one primary API key for that user.
  - Re-running bootstrap on an already bootstrapped database MUST NOT create duplicate users or keys and should be a no-op aside from validation.
- **Pre-existing data and conflicts**:
  - If multiple admin users or multiple “primary” keys for `SINGLE_USER_FIXED_ID` are detected, bootstrap SHOULD fail loudly with a clear error and guidance for remediation rather than guessing.
  - If `AUTH_MODE=single_user` is configured but the database already contains multiple active users, bootstrap logs a warning and either:
    - Treats this as a misconfiguration (fail fast, recommended default), or
    - Proceeds in a documented “mixed” mode only if explicitly allowed by configuration.

#### Mode branching reduction

- `is_single_user_mode()` becomes:
  - Primarily a configuration signal for:
    - Bootstrap logic.
    - Default database URL (still SQLite-leaning in docs).
    - Feature gating UI/docs (e.g., “no multi-tenant UI in single-user profile”).
- Permission functions no longer special-case single-user; they rely on claims on the principal.

#### Existing single-user dependencies (transitional state)

- Today, several AuthNZ components implement their own single-user mode handling:
  - `auth_deps.get_current_user` has dedicated branches for `SINGLE_USER_API_KEY` and synthetic single-user dicts.
  - `User_DB_Handling.get_request_user` constructs a dummy `User` with admin-style claims when `is_single_user_mode()` is true.
  - `verify_single_user_api_key` validates the fixed key independently of APIKeyManager.
  - `auth_principal_resolver.get_auth_principal` and `User_DB_Handling.authenticate_api_key_user` still short-circuit `SINGLE_USER_API_KEY` for legacy compatibility.
- For v1, these must be treated as compatibility layers:
  - Internally, they will be refactored to rely on the bootstrapped user + API key and `get_auth_principal` / `AuthContext`.
  - Externally, their observable behavior (headers accepted, 401/403 responses, basic user fields) remains intact while single-user mode is redefined as a bootstrap profile.
  - New code MUST NOT introduce additional `is_single_user_mode()` branches for auth; instead, it should use profile/claims-based decisions and the unified principal model.

### 2. DB Backend Unification via Repositories

#### Concept

- Introduce a small “AuthNZ repository” layer that encapsulates all SQL and dialect specifics.
- Business logic modules depend on repositories, not direct SQL or `conn.execute(...)`.

#### Repository contracts

- Repositories acquire connections via the shared `DatabasePool` abstraction; they do not open raw connections directly.
- Repositories expose narrow, task-focused methods and hide Postgres/SQLite-specific SQL, placeholders, and JSON/TYPE differences.
- Expected domain errors (e.g., uniqueness violations, missing rows) are mapped to typed AuthNZ exceptions; unexpected driver errors surface as internal errors and are logged with context.
- v1 scope is intentionally limited:
  - Users, API keys, and core RBAC tables are covered by repositories.
  - Usage/metrics repositories (e.g., for `usage_log` / `llm_usage_log`) remain out of scope for this iteration and are future work.

#### Initial repositories (examples)

- `AuthnzUsersRepo`:
  - Methods:
    - `get_user_by_id`, `get_user_by_username_or_email`.
    - `create_user`, `update_user`, `deactivate_user`.
  - Under the hood, handles Postgres vs SQLite differences.

- `AuthnzApiKeysRepo`:
  - Methods used by `APIKeyManager`:
    - `insert_api_key`, `mark_revoked`, `rotate_key`.
    - `fetch_by_hashes`, `fetch_limits_for_key`, `list_keys_for_user`.

- `AuthnzRbacRepo`:
  - Methods used by RBAC helpers and bootstrap:
    - `assign_role_to_user`, `get_user_roles`, `get_effective_permissions`.

- `AuthnzOrgsTeamsRepo`, `AuthnzUsageRepo`, etc., added as needed.

#### Refactors

- Move DDL from:
  - `api_key_manager._create_tables`.
  - `virtual_keys`, `quotas`, `rate_limiter` (AuthNZ-specific tables).
  - `orgs_teams`, `usage_logging_middleware`, and relevant pieces of `token_blacklist`.
- Into:
  - Migrations in `AuthNZ.migrations` / `pg_migrations_extra`.
  - Repository initialization helpers (for compatibility with existing migrations).

#### Backend detection

- All backend detection and placeholder conversion (`$1` vs `?`) become internal to repositories or to `DatabasePool`.
- Business logic modules stop branching on `if hasattr(conn,'fetchval')`.

### 3. Deployment Profiles

Define and document profiles rather than modes:

- **Profile: local-single-user**
  - SQLite users DB (default path).
  - Bootstrap user + API key.
  - No registration endpoints for additional users (gated by config and service layer).
  - Creating additional users is forbidden as a hard constraint:
    - User-creation endpoints are disabled.
    - Repository and/or service-layer checks reject any attempt to create users beyond `SINGLE_USER_FIXED_ID` while in this profile.

- **Profile: multi-user-postgres**
  - Postgres users DB.
  - Full registration, MFA, orgs/teams, API keys, virtual keys.

Internally, both share the same code paths; only bootstrap, DB URL, and enabled endpoints differ.

#### Profile matrix (behavior overview)

| Feature / Capability        | local-single-user                | multi-user-postgres          |
|----------------------------|----------------------------------|------------------------------|
| User registration          | Disabled                         | Enabled                      |
| Additional user creation   | Forbidden (hard constraint)      | Enabled                      |
| MFA                        | Optional (config-gated)          | Enabled / strongly recommended |
| Orgs/teams                 | Optional / commonly unused       | Enabled                      |
| Virtual keys               | Optional                         | Enabled                      |
| Multi-tenant Web UI        | Disabled / single-tenant only    | Enabled                      |

Over time, `AUTH_MODE` may be decomposed into a `PROFILE` plus more granular feature flags (`ENABLE_REGISTRATION`, `ENABLE_MFA`, etc.), but v1 keeps `AUTH_MODE` for compatibility and treats profiles as an interpretation of the existing settings.

### AUTH_MODE → Profiles + Feature Flags (Migration Plan for v1.0 & Beyond)

This section is forward-looking (targeting v1.0 or later). To make “mode” an implementation detail of a higher-level deployment profile, we will:

1. **Introduce an explicit profile setting** (backed by code + docs; wired for UX/feature gating in v1.0, no auth‑path behavior change):
   - New setting/env var: `PROFILE` with allowed values such as:
     - `local-single-user` (or legacy alias `single_user`).
     - `multi-user-postgres`.
     - `multi-user-sqlite` (optional, for small multi-user dev setups).
   - v1.0 behavior:
     - If `PROFILE` is unset, the system behaves exactly as today and infers behavior from `AUTH_MODE` + `DATABASE_URL`.
     - If `PROFILE` is set, helper functions (e.g., `is_single_user_mode`, `is_multi_user_mode`) continue to read `AUTH_MODE` but may log/emit diagnostics when `PROFILE` and `AUTH_MODE` disagree (no hard failures yet).
   - post‑v1.0 target:
     - `tldw_Server_API/app/core/AuthNZ/settings.Settings` exposes a `PROFILE` field (env `PROFILE`) and a helper `get_profile()` which returns either the explicit value or a derived profile string based on `AUTH_MODE` + `DATABASE_URL` (for example, `local-single-user`, `multi-user-postgres`, `multi-user-sqlite`). Callers should treat this primarily as a coordination/UX hint and feature-gating signal (startup logs, frontend capability hints, embeddings tenant-quota behavior); all authentication and authorization decisions are **claim-first** and use `AuthPrincipal`-based dependencies (`get_auth_principal`, `require_permissions`, `require_roles`) instead of mode checks. It is acceptable to use `PROFILE` as an additional tightening signal for deployment-specific flows (for example, disabling self-registration in `local-single-user` deployments).
     - **Security invariants**: `PROFILE` must **never** be used to bypass or relax auth decisions or to grant permissions beyond those implied by `AUTH_MODE` and claims.

2. **Add feature flags that clarify UX vs auth semantics**, without changing defaults:
   - Examples:
     - `ENABLE_REGISTRATION` (default: `True` in multi-user, `False` in single-user profile).
     - `ENABLE_MFA` (default: `False` in local-single-user, `True`/recommended in multi-user-postgres).
     - `ENABLE_ORGS_TEAMS`, `ENABLE_VIRTUAL_KEYS`, `ENABLE_JOBS_DOMAIN_SCOPING`, etc. (all default to current behavior).
     - Auth-adjacent behavior flags that steer profile-aware guardrails without reintroducing mode-based authorization:
       - `ORG_POLICY_SINGLE_USER_PRINCIPAL` – controls whether the synthetic single-user org (`org_id=1`) fallback in `get_org_policy_from_principal` is driven by `AuthPrincipal` + profile (enabled) or by legacy `is_single_user_mode()` heuristics (disabled).
       - `EMBEDDINGS_TENANT_RPS_PROFILE_AWARE` – controls whether embeddings tenant RPS quotas are enforced based on profile and principals explicitly tagged as single-user (subject `"single_user"` via `is_single_user_principal`) (enabled) or legacy `AUTH_MODE`-based checks (disabled).
       - `MCP_SINGLE_USER_COMPAT_SHIM` – controls whether MCP HTTP endpoints treat `SINGLE_USER_API_KEY` as a bootstrap admin principal in single-user deployments (enabled) or always validate `X-API-KEY` via the multi-user API key manager (disabled).
   - For `v0.1.x`, these flags gate **UX/feature surface** (show/hide endpoints, admin controls, and WebUI affordances) rather than changing core auth semantics. All flags default to values that exactly match today’s behavior.
   - Registry note: as flags evolve, track canonical definitions (defaults, scope, and migration status) in `tldw_Server_API/app/core/AuthNZ/settings.py` and optionally mirror them in a dedicated doc (e.g., `Env_Vars.md`) or an admin diagnostics endpoint.

3. **Gradually rewrite `is_single_user_mode()` callsites to profile/flags**, guided by tests:
   - Classification:
     - **Coordination/UX**: startup banners, WebUI config hints, background warm-ups, quota hints – these can switch to `PROFILE` + feature flags without impacting auth semantics.
     - **Auth-adjacent**: shortcuts in auth deps, jobs admin, embeddings quotas, MCP auth compatibility – these must be backed by explicit principal/claims tests before any behavior change and, where behavior diverges from legacy `is_single_user_mode()` logic, must be gated by dedicated env flags (as in the examples above).
   - Plan:
     - Keep `is_single_user_mode()` as a thin wrapper over `get_settings().AUTH_MODE == "single_user"` for now, but add and maintain tests that assert:
       - Single-user principal claims (roles/permissions) enforce the same constraints as multi-user admin/admin-lite principals.
       - No auth path bypasses `get_auth_principal` / `require_permissions` / `require_roles` solely due to mode; any behavior differences are expressed via profile/flag combinations and validated via tests.
     - For each auth-adjacent branch (beyond jobs-admin, which already has a principal-first path), design a claim-first alternative that takes an `AuthPrincipal` and/or `AuthContext` and:
       - Enforces the same behavior via claims (roles, permissions, feature flags).
       - Is gated by a dedicated env flag (e.g., `ORG_POLICY_SINGLE_USER_PRINCIPAL`, `EMBEDDINGS_TENANT_RPS_PROFILE_AWARE`, `MCP_SINGLE_USER_COMPAT_SHIM`) so we can test before flipping defaults and preserve a compatibility path.

4. **Document migration for existing environments (no behavior changes by default)**:
   - Existing deployments can safely upgrade by:
     - Leaving `PROFILE` unset; `AUTH_MODE` continues to drive behavior as today.
     - Optionally setting `PROFILE` to match current `AUTH_MODE`:
       - `AUTH_MODE=single_user` → `PROFILE=local-single-user`.
       - `AUTH_MODE=multi_user` + Postgres `DATABASE_URL` → `PROFILE=multi-user-postgres`.
     - Not setting any new feature flags; defaults are chosen to preserve current endpoint availability and guardrail semantics.
   - Future major versions can:
     - Deprecate direct `AUTH_MODE` reads in business logic and treat it as a thin compatibility alias for `PROFILE`.
     - Promote feature flags and `PROFILE` to the primary configuration model, with `AUTH_MODE` used only to infer defaults when missing.

---

## Scope

### In-Scope (v1)

- Add bootstrap logic and data invariant for single-user profile (user + API key + admin role).
- Refactor:
  - Single-user API key validation to use API-key flow.
  - Permission checks to rely on claims instead of `is_single_user_mode`.
  - A small number of modules to use repositories instead of inline SQL:
    - `api_key_manager`.
    - `orgs_teams` (for core operations).
    - Parts of `virtual_keys` related to key limits.

**Implementation Status (AuthNZ v0.1, internal)**:
- Note: “Done” below refers to request-time/runtime paths being repo-backed; schema/migrations and bootstrap/DDL backstops may still include inline SQL (tracked as deferred follow-up).
- Repository introduction (Stage 3) is Done (v0.1):
  - `AuthnzUsersRepo`, `AuthnzApiKeysRepo`, and `AuthnzRbacRepo` are implemented and used by `User_DB_Handling`, `APIKeyManager`, and RBAC helpers.
  - New repositories `AuthnzOrgsTeamsRepo`, `AuthnzUsageRepo`, `AuthnzRateLimitsRepo`, `AuthnzSessionsRepo`, `AuthnzTokenBlacklistRepo`, `AuthnzMfaRepo`, `AuthnzMonitoringRepo`, and `AuthnzRegistrationCodesRepo` encapsulate orgs/teams membership, usage/LLM-usage tables, AuthNZ rate-limiter storage, session persistence, token blacklist storage, MFA persistence, monitoring/audit metrics, and registration-code cleanup respectively, with cross-backend tests where applicable.
- Backend drift reduction (Stage 4) is Done (v0.1):
  - `virtual_keys` budget and usage paths delegate to `AuthnzApiKeysRepo` / `AuthnzUsageRepo` instead of embedding dialect-specific SQL.
  - `orgs_teams` is now a thin orchestration layer over `AuthnzOrgsTeamsRepo` for organization, team, and membership operations (including default-team handling).
  - `rate_limiter` uses `AuthnzRateLimitsRepo` for all DB-backed rate-limiter tables (`rate_limits`, `failed_attempts`, `account_lockouts`), and the AuthNZ scheduler prunes usage tables via `AuthnzUsageRepo`.
  - `session_manager` delegates session creation, validation, refresh, listing, and cleanup to `AuthnzSessionsRepo`, and token blacklist operations (`revoke_token`, blacklist checks, cleanup, stats, and revoke-all-tokens flows) use `AuthnzTokenBlacklistRepo` together with `AuthnzSessionsRepo` for all `token_blacklist` and `sessions` table access in logout-all-devices paths.
  - Monitoring metrics, audit pruning, and AuthNZ security dashboards are now fully backed by `AuthnzMonitoringRepo`, with SQLite and Postgres repo tests exercising metric insertion, aggregation, and alert retrieval.
- Claim-first adoption has been extended to additional high-value surfaces in line with this PRD:
  - Chat slash commands:
    - `GET /api/v1/chat/commands` now filters commands by the centralized command authorization decision whenever slash commands are enabled, without any legacy flag gate. Single-user deployments are allowed through the explicit single-user owner context rather than a broad mode shortcut.
    - The async slash-command router (`tldw_Server_API/app/core/Chat/command_router.py::async_dispatch_command`) enforces declared per-command permissions using normalized `CommandContext` metadata, claim permissions, admin context, single-user owner context, and fail-closed DB permission checks. Unit and integration tests in `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py` and `tldw_Server_API/tests/Chat_NEW/integration/test_chat_commands_endpoint.py` validate permission-denied vs allowed behavior and command filtering semantics.
  - Prompt Studio:
    - `get_prompt_studio_user` in `tldw_Server_API/app/api/v1/API_Deps/prompt_studio_deps.py` builds its `user_context` from the normalized `User` object returned by `get_request_user` (claims-first), deriving `is_admin` from `User.is_admin` / `"admin"` in `User.roles` and copying `User.permissions` directly, instead of inferring Prompt Studio admin solely from `AUTH_MODE`/`is_single_user_mode()`.
    - Unit tests in `tldw_Server_API/tests/AuthNZ_Unit/test_prompt_studio_user_claims.py` exercise admin vs non-admin principals (ensuring `is_admin` and `permissions` flow from claims), and `tldw_Server_API/tests/prompt_studio/unit/test_prompt_studio_deps_headers.py` covers header forwarding and 401 behavior when no credentials are present.
- Claim-first + principal/state invariants now cover multiple high-value domains:
  - Media and RAG: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_media_rag_invariants.py` exercises JWT and API-key flows for `/api/v1/rag/search` and `/api/v1/media/process-videos`, asserting that `AuthPrincipal`, `request.state.user_id` / `api_key_id`, and `request.state.auth.principal` stay aligned (including org/team ids where present).
  - Tools execute: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_tools_invariants.py` covers the multi-user API-key path to `/api/v1/tools/execute`, ensuring `principal.api_key_id` and `request.state.api_key_id` remain in sync and that `request.state.auth.principal` mirrors both.
  - Evaluations list: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_evaluations_invariants.py` adds a JWT happy-path invariant for `/api/v1/evaluations/`, verifying that a user with `evals.read` sees consistent identity between `AuthPrincipal`, `request.state.*`, and `request.state.auth.principal`.
  - Together with the single-user claims tests (`tldw_Server_API/tests/AuthNZ/integration/test_single_user_claims_permissions.py`) and claim-first permission tests (`tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py`), these suites form a cross-domain coverage snapshot (media, RAG, tools, evaluations) for principal/state alignment in both multi-user and single-user profiles.
- Remaining inline SQL touching selected bootstrap paths is intentionally left for later phases; it is documented in `Docs/Product/Completed/AuthNZ-Refactor-Implementation-Plan.md` as out-of-scope for this iteration (e.g., minimal schema backstops in `initialize.py` and API-key bootstrap helpers).

### Out of Scope (v1)

- Fully migrating all AuthNZ-related SQL to repositories (will be incremental).
- Changing external configuration names (`AUTH_MODE`, `DATABASE_URL`) in this iteration.

---

## Risks & Mitigations

- **Risk: Breaking existing single-user deployments that rely on current behavior.**
  - Mitigation:
    - Keep `AUTH_MODE=single_user` semantics stable at the API surface.
    - Implement bootstrap to mirror current behavior (single admin-like user + API key).
    - Feature-flag any endpoint gating until tests cover both profiles.

- **Risk: Repository layer adds indirection.**
  - Mitigation:
    - Keep repositories thin and focused on SQL/dialect concerns.
    - Co-locate them with AuthNZ code and document clear contracts.

- **Risk: Migration complexity.**
  - Mitigation:
    - Start with a small, high-impact subset (API keys, core orgs/teams, RBAC).
    - Add tests that run both SQLite and Postgres variations where possible (as in current AuthNZ tests).

- **Risk: On-wire behavior changes (auth headers, status codes, error payloads).**
  - Mitigation:
    - Explicitly test both profiles to ensure 401/403 and success responses remain compatible with current behavior.
    - Document any unavoidable changes and, where needed, gate them behind configuration or versioning.

---

## Upgrade / Migration Strategy

- **Detect legacy single-user deployments**:
  - Presence of `AUTH_MODE=single_user` and/or `SINGLE_USER_API_KEY`.
  - Existing “synthetic” single-user patterns in the users/API-key tables.
- **Migration steps for operators**:
  - Take a database backup.
  - Enable the new bootstrap flow and run AuthNZ initialization once.
  - Verify that:
    - An admin user with `SINGLE_USER_FIXED_ID` exists.
    - The stored primary API key matches `SINGLE_USER_API_KEY` (if configured).
    - RBAC tables contain the expected admin role/permissions.
  - Re-run bootstrap to confirm idempotency.
- **Mixed / unexpected states**:
  - If bootstrap detects multiple active users or conflicting admin/primary key invariants under `AUTH_MODE=single_user`, it fails fast with a clear error message and remediation guidance (e.g., which rows to inspect or archive).
  - Operators are encouraged to resolve conflicts explicitly rather than relying on implicit migrations.
- **Multi-user deployments**:
  - For `AUTH_MODE=multi_user` / `multi-user-postgres`, bootstrap is a no-op beyond validation; repositories and migrations are introduced in a way that does not change on-wire auth behavior.

---

## Milestones & Phasing

### Phase 1: Single-User Bootstrap

- Implement bootstrap script/entrypoint in AuthNZ initialize:
  - Create admin user + API key if missing.
  - Wire into docs and quickstart scripts.
- Ensure single-user profile passes existing tests (and new ones).

- **Status (v0.1)**: Done — exercised by single-user bootstrap and access flows in `tldw_Server_API/tests/AuthNZ/integration/test_single_user_claims_permissions.py`.

### Phase 2: Claims-Based Single-User Permissions

- Update permissions code to:
  - Stop special-casing `is_single_user_mode()` for allow-all.
  - Rely on claims from `AuthPrincipal` / `User` (admin role).
- Add tests verifying that the single-user admin has all needed permissions via claims.

- **Status (v0.1)**: Core claims cleanup Done — `permissions.py`, chat slash commands, and Prompt Studio now use claims-first semantics in single-user and multi-user profiles (see `tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py`, `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py`, `tldw_Server_API/tests/Chat_NEW/integration/test_chat_commands_endpoint.py`, and `tldw_Server_API/tests/AuthNZ_Unit/test_prompt_studio_user_claims.py`).

### Phase 3: Repository Introduction

- Create initial repositories (`AuthnzApiKeysRepo`, `AuthnzUsersRepo`, `AuthnzRbacRepo`) and migrate:
  - `APIKeyManager` to use `AuthnzApiKeysRepo`.
  - Some RBAC helpers to use `AuthnzRbacRepo`.
- Move DDL for API key-related tables into migrations or repositories.

- **Status (v0.1)**: Done — core repos (`AuthnzUsersRepo`, `AuthnzApiKeysRepo`, `AuthnzRbacRepo`, `AuthnzUsageRepo`, and others) back primary AuthNZ flows with SQLite/Postgres tests; remaining repo migrations are explicitly deferred to post-v0.1 iterations (see `Docs/Product/Completed/AuthNZ-Refactor-Implementation-Plan.md` for the tech-debt list).

### Phase 4: Backend Drift Reduction

- Identify and refactor 2–3 high-impact modules (e.g., `virtual_keys`, `orgs_teams`, parts of `rate_limiter` that are AuthNZ-specific) to use repositories.
- Remove duplicated Postgres/SQLite branches that are now redundant.

- **Status (v0.1)**: Done — key modules such as `virtual_keys`, `orgs_teams`, and AuthNZ rate-limiter storage use repos; remaining backend drift and mode cleanup is tracked as post-v0.1 tech debt in `Docs/Product/Completed/AuthNZ-Refactor-Implementation-Plan.md` and is addressed incrementally.

### Repo Coverage vs Inline SQL (AuthNZ v0.1)

As of the current AuthNZ refactor stage (note: “repo-backed” here refers to runtime read/write operations; migrations and bootstrap/DDL backstops may still be inline):

- Runtime flows for **users**, **RBAC**, **orgs/teams**, **sessions**, **token blacklist**, **MFA**, **monitoring**, and **registration codes** are fully repo-backed; inline SQL for these concerns is limited to migrations and bootstrap helpers.
- **API keys** and **usage/LLM-usage** are partially repo-backed: validation, listing, aggregation, and pruning use `AuthnzApiKeysRepo` / `AuthnzUsageRepo`, but `APIKeyManager` still contains backend-specific inline SQL for some runtime operations (usage logging inserts now go through `AuthnzUsageRepo`).
- Non-MFA inline SQL that uses backend detection (`hasattr(conn, 'fetch*')`) and touches core tables is explicitly marked as deferred or bootstrap-only in the AuthNZ implementation plan (e.g., Postgres bootstrap DDL in `initialize.py`, runtime API-key flows in `api_key_manager.py`).

For a detailed, per-table view across the core AuthNZ tables (users, api_keys, RBAC, orgs/teams, usage/llm_usage, rate_limits/failed_attempts/account_lockouts, sessions, token_blacklist, MFA, monitoring, registration codes), see the **“Repo Coverage Table (AuthNZ core tables)”** section in `Docs/Product/Completed/AuthNZ-Refactor-Implementation-Plan.md`.

---

## Open Questions

- Should we eventually drop `AUTH_MODE` in favor of more specific flags (e.g., `ENABLE_REGISTRATION`, `ENABLE_MFA`, `PROFILE=single_user`), or keep it as a high-level profile indicator?
  - yes, drop `AUTH_MODE` for specific flags
- How much of DB backend selection should be driven by a central `DATABASE_URL` vs. per-module URLs?
  - Everything through the central DB URL, modules should only see a single DB/be agnostic.
- Should bootstrap be idempotent-only at startup, or also available as a one-time admin command?
  - one-time admin command

---

## Success Criteria

- Single-user deployments are implemented as a constrained configuration of multi-user infrastructure; legacy `SINGLE_USER_API_KEY` compatibility shims remain for now but no new auth paths are introduced.
- The number of direct `is_single_user_mode()` checks in the codebase is significantly reduced and confined to a small set of coordination points.
- At least API-key and user-related AuthNZ modules no longer contain inline dialect branches or DDL; they use repositories instead.
- For both profiles, authentication headers, HTTP status codes (401/403), and error payloads remain compatible with current behavior, or any intentional changes are explicitly documented and tested.

## Verification

- **SQLite regression slice (AuthNZ focus)**:
  - Recommended command: `python -m pytest tldw_Server_API/tests/AuthNZ_SQLite -m "not slow"`.
  - Key suites: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_users_repo_sqlite.py`, `test_authnz_api_keys_repo_sqlite.py`, `test_authnz_usage_repo_sqlite.py`, and `test_authnz_usage_repo_insert_sqlite.py` (users, API keys, usage/LLM-usage).
- **Postgres regression slice (AuthNZ focus)**:
  - Recommended command: `python -m pytest tldw_Server_API/tests/AuthNZ_Postgres tldw_Server_API/tests/AuthNZ/integration -m "not slow"`.
  - Key suites: `tldw_Server_API/tests/AuthNZ_Postgres` (AuthNZ repos over Postgres) and `tldw_Server_API/tests/AuthNZ/integration` (JWT login flows, single-user bootstrap, AuthGovernor lockouts, LLM budgets).
- These slices, together with the unit suites referenced above (permissions, metrics admin, resource-governor permissions, chat commands, Prompt Studio), act as the practical verification layer that the unified single-user/multi-user behavior and repo-backed AuthNZ paths work consistently across SQLite and Postgres.

## Implementation Plan

### Stage 1: Single-User Bootstrap
**Goal**: Make single-user deployments a bootstrap profile of multi-user by creating an admin user and primary API key via normal AuthNZ flows.

**Success Criteria**:
- Running the AuthNZ initializer in single-user profile ensures:
  - A single admin user (`SINGLE_USER_FIXED_ID`) exists in `users`.
  - A primary API key exists and is stored/displayed according to configuration.
- Existing single-user workflows (API key access, no registration) continue to work.

**Tests**:
- Integration tests that:
  - Start in single-user profile, run bootstrap, and validate presence of the user and key.
  - Use the bootstrapped key to access representative endpoints.
  - Start from an existing single-user database with `SINGLE_USER_API_KEY` set and verify that bootstrap adopts the existing key rather than generating a new one.
  - Re-run bootstrap on an already bootstrapped database to confirm it is idempotent (no duplicate users/keys).

**Status**: Done

**Notes**:
- The async helper `bootstrap_single_user_profile()` is implemented in `tldw_Server_API/app/core/AuthNZ/initialize.py`. It:
  - Ensures the single-user admin row (`SINGLE_USER_FIXED_ID`) and RBAC seed via `ensure_single_user_rbac_seed_if_needed()` when `AUTH_MODE=single_user`.
  - Uses `APIKeyManager` and the configured `SINGLE_USER_API_KEY` to upsert a non-virtual primary API key row keyed by `key_hash`, with `scope='admin'` and `status='active'`, for both Postgres and SQLite backends.
  - Emits `logger.info` / `logger.warning` messages in addition to interactive `print` output so single-user bootstrap success/failure paths are visible in standard AuthNZ logs.
  - When invoked via the AuthNZ initializer CLI (`initialize.main()`), treats bootstrap failures as a hard error in non-`TEST_MODE` runs (exiting with a non-zero status) while continuing as a soft warning when `TEST_MODE=1` to keep offline test harnesses stable.
- SQLite regression coverage for the bootstrap path is provided by `tldw_Server_API/tests/AuthNZ_SQLite/test_single_user_bootstrap_sqlite.py`, which:
  - Configures `AUTH_MODE=single_user` and a SQLite `DATABASE_URL`, runs bootstrap twice, and asserts:
    - A single admin user row exists with `id = SINGLE_USER_FIXED_ID`, `username='single_user'`, `role='admin'`, and an active/verified status.
    - A single non-virtual `api_keys` row exists whose `key_hash` matches `hash_api_key(SINGLE_USER_API_KEY)`, with `scope='admin'` and `status='active'`.
  - Re-running bootstrap is idempotent (no duplicate user/key rows).
- Postgres-backed regression coverage for the bootstrap path is provided by `tldw_Server_API/tests/AuthNZ/integration/test_single_user_bootstrap_postgres.py`, which:
  - Uses the `isolated_test_environment` fixture to provision a per-test Postgres AuthNZ database.
  - Switches to `AUTH_MODE=single_user` (with `TEST_MODE=1`), runs `bootstrap_single_user_profile()` twice, and asserts the same single-user admin and primary key invariants as the SQLite test.
- A higher-level claims test, `tldw_Server_API/tests/AuthNZ/integration/test_single_user_claims_permissions.py`, exercises:
  - RBAC seeding via `bootstrap_single_user_profile()` on SQLite.
  - Single-user `get_request_user` + `get_auth_principal` + `require_permissions/roles` using the bootstrapped admin's API key to reach protected endpoints.
- Additional bootstrap invariants around existing deployments are covered by:
  - `tldw_Server_API/tests/AuthNZ_SQLite/test_single_user_bootstrap_sqlite.py::test_single_user_bootstrap_reuses_preseeded_primary_key`, which starts from a SQLite AuthNZ database with a pre-seeded `api_keys` row for `SINGLE_USER_API_KEY` and confirms `bootstrap_single_user_profile()` reuses and upgrades that row (instead of creating a duplicate).
  - `tldw_Server_API/tests/AuthNZ/integration/test_single_user_bootstrap_postgres.py::test_single_user_bootstrap_reuses_preseeded_primary_key_postgres`, which mirrors the same invariant on a Postgres-backed AuthNZ database via the `isolated_test_environment` fixture.

### Stage 2: Claims-Based Single-User Permissions
**Goal**: Replace mode-based permission allowances with claim-based admin role for the bootstrapped user.

**Success Criteria**:
- `permissions.py` and related checks no longer special-case `is_single_user_mode()` for allow-all.
- The bootstrapped single-user admin has full required permissions via roles/permissions claims.

**Tests**:
- Permission tests verifying:
  - Admin user in single-user profile can perform privileged operations.
  - Non-admin users (if any are created) are constrained by RBAC as expected.

**Status**: Done

**Notes**:
- `tldw_Server_API/app/core/AuthNZ/permissions.py` now:
  - Prefers `user.permissions` / `user.roles` claim lists in both single-user and multi-user modes.
  - Uses `is_single_user_mode()` only as a fallback when no claims are attached, preserving legacy behavior for callers that do not yet supply claims.
- `require_admin` in `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py` no longer special-cases `is_single_user_mode()` and instead relies on `is_admin`, `role == "admin"`, or `roles` containing `"admin"`.
- `tldw_Server_API/tests/AuthNZ/integration/test_single_user_claims_permissions.py` verifies that, in single-user mode:
  - `bootstrap_single_user_profile()` seeds the admin user and primary key on an isolated SQLite database.
  - The bootstrapped admin can call permission- and role-protected endpoints via `get_request_user`, `get_auth_principal`, `require_permissions`, and `require_roles` using the configured single-user API key.
- `tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py` includes additional coverage for single-user claim-first behavior:
  - Claims govern access when present (DB is not consulted).
  - An explicit `"admin"` role in single-user mode implies both `admin` and `user` checks while still rejecting unrelated roles.
  - Negative paths for single-user principals without sufficient claims are covered: `require_permissions` / `require_roles` return HTTP 403 for `kind="user"` principals tagged with `subject="single_user"` when they lack the required permission/role, matching multi-user semantics.
  - Additional tests ensure that when no claim lists are attached in single-user mode, `check_permission` / `check_role` fall back to the configured `UserDatabase` instead of auto-allowing, so single-user deployments share the same fallback behavior as multi-user.
- Remaining `is_single_user_mode()` shortcuts are being removed from business/endpoints in favor of claim-based behavior. MCP HTTP diagnostics and persona streaming now derive identity from validated API keys instead of checking `AUTH_MODE` directly:
  - `tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py::mcp_request`, `::mcp_request_batch`, and `::list_tools` no longer branch on `is_single_user_mode()` when computing `user_id`; they rely on `get_current_user`’s `TokenData` (which already encodes the single-user admin) and pass that through to MCP.
  - `tldw_Server_API/app/api/v1/endpoints/persona.py::persona_stream` uses `get_api_key_manager().validate_api_key(...)` to resolve `user_id` from API keys for both single-user and multi-user deployments, rather than comparing against `SINGLE_USER_API_KEY` in the endpoint. This keeps persona’s MCP calls aligned with the unified AuthNZ bootstrap/API-key behavior without introducing new mode-based shortcuts.

**Status snapshot (v0.1)**: Single-user claims cleanup (permissions, chat commands, Prompt Studio) is Done and covered by the tests above; any remaining mode-based shortcuts are treated as future cleanup.

### Mode vs Claims – Coverage Snapshot

| Surface | Claim-first status | Flag gate (if any) | Test path |
|---|---|---|---|
| Metrics admin (`POST /api/v1/metrics/reset`) | Done | — | `tldw_Server_API/tests/AuthNZ_Unit/test_metrics_permissions_claims.py` |
| Resource-Governor admin/diagnostics (`/api/v1/resource-governor/policy*`, `/api/v1/resource-governor/diag/*`) | Done | — | `tldw_Server_API/tests/AuthNZ_Unit/test_resource_governor_permissions_claims.py` (+ `Resource_Governance` suite) |
| Embeddings model management, workflows DLQ, connectors admin, tools admin, notes graph admin | Done | — | Claim tests under `tldw_Server_API/tests/AuthNZ_Unit/` + feature suites |
| Chat slash commands | Done | — | `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py`, `tldw_Server_API/tests/Chat_NEW/integration/test_chat_commands_endpoint.py` |
| Prompt Studio | Done | — | `tldw_Server_API/tests/AuthNZ_Unit/test_prompt_studio_user_claims.py`, `tldw_Server_API/tests/prompt_studio/unit/test_prompt_studio_deps_headers.py` |
| Jobs admin routes | Done (principal-first path) | `JOBS_DOMAIN_RBAC_PRINCIPAL` | `tldw_Server_API/tests/AuthNZ_Unit/test_jobs_admin_permissions_claims.py` |
| Coordination/governance-only checks (startup banners, WebUI config, warm-ups, backpressure/tenant RPS, embedding quota defaults, several diagnostics paths) | Mode-aware allowed (coordination only) | `PROFILE` + feature flags | N/A |

### Repo vs Inline-SQL Coverage (AuthNZ Tables)

Note: “Repository” here means runtime queries/operations use the repository abstraction; schema changes and initial table creation are handled via migrations and bootstrap helpers.

| Area / Tables                                      | Repository                     | Inline SQL (remaining)                                          | Notes                                                                                 |
|----------------------------------------------------|--------------------------------|------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| Users (`users`, core user lookups)                 | `AuthnzUsersRepo`              | Bootstrap DDL/migrations in `initialize.py`                      | Runtime lookups for users (id/uuid/username) are repo-backed across SQLite/Postgres. |
| RBAC (`roles`, `user_roles`, `user_permissions`)   | `AuthnzRbacRepo`               | DDL/migrations only                                              | Effective-permissions and role/override queries are centralized in the repo.         |
| API keys (`api_keys`, `api_key_audit_log`)         | `AuthnzApiKeysRepo`            | Bootstrap + limited helpers in `api_key_manager.py`              | Runtime validation, listing, primary-key upsert, creation (regular/virtual), and rotation/revocation use the repo; table creation and lightweight usage/audit helpers remain in the manager. |
| Orgs/Teams (`orgs`, `teams`, memberships)          | `AuthnzOrgsTeamsRepo`          | DDL/migrations only                                              | `orgs_teams.py` orchestrates via the repo; membership/org/team CRUD is repo-backed.  |
| Usage / Metrics (`usage_log`, `usage_daily`)       | `AuthnzUsageRepo`              | DDL/migrations/tests only                                        | Scheduler pruning, aggregation, and per-request usage inserts (via middleware) use the repo. |
| LLM Usage (`llm_usage_log`, `llm_usage_daily`)     | `AuthnzUsageRepo`              | DDL/migrations; some legacy reporting helpers                    | LLM budget guard and pruning flows use repo helpers for cross-backend correctness.   |
| Rate limits (`rate_limits`)                        | `AuthnzRateLimitsRepo`         | DDL/migrations; some legacy rate-limiter metrics wiring          | AuthNZ `RateLimiter` now reads/writes DB-backed rate limits via the repo.            |
| Lockouts (`failed_attempts`, `account_lockouts`)   | `AuthnzRateLimitsRepo`         | DDL/migrations only                                              | Login lockout and failed-attempt counters are repo-backed for SQLite/Postgres.       |
| Sessions (`sessions`)                              | `AuthnzSessionsRepo`           | Minimal bootstrap helpers in `initialize.py` / legacy harmonize  | Session creation, refresh, validation, listing, and cleanup are repo-backed.         |
| Token blacklist (`token_blacklist`)                | `AuthnzTokenBlacklistRepo`     | DDL and a small SQLite harmonization helper in `token_blacklist.py` | Blacklist inserts, checks, cleanup, and stats all go through the repo.            |
| MFA (`users` MFA columns / MFA tables)             | `AuthnzMfaRepo`                | DDL/migrations only                                              | MFA secrets, backup codes, and status rows are persisted via the repo.               |
| Monitoring / AuthNZ metrics (monitoring tables)    | `AuthnzMonitoringRepo`         | DDL/migrations only                                              | Metrics/audit pruning and dashboards use the repo for both SQLite and Postgres.      |
| Registration codes (`registration_codes`)          | `AuthnzRegistrationCodesRepo`  | DDL/migrations only                                              | Scheduler cleanup and registration-code lookups are repo-backed.                     |

### Stage 3: Repository Introduction
**Goal**: Introduce AuthNZ repositories and migrate API-key and core user/RBAC operations to them.

**Success Criteria**:
- `APIKeyManager` uses `AuthnzApiKeysRepo` for all DB interactions.
- Selected RBAC helpers use `AuthnzRbacRepo`.
- DDL for API key-related tables is centralized in migrations/repositories rather than scattered.

**Tests**:
- Unit tests for repository methods against both SQLite and Postgres.
- Failure-path tests for repository methods (uniqueness violations, missing rows) to ensure consistent exception mapping.
- Existing API-key and RBAC tests pass unchanged.

**Status**: Done (v0.1) — core repos are in use; remaining repo migrations are deferred to the next iteration (see `Docs/Product/Completed/AuthNZ-Refactor-Implementation-Plan.md`).

**Notes**:
- `AuthnzApiKeysRepo` at `tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py` now backs all runtime `api_keys` operations with focused methods, including:
  - `fetch_active_by_hash_candidates(...)` used by `APIKeyManager.validate_api_key`.
  - `list_user_keys(...)` used by `APIKeyManager.list_user_keys`.
  - `upsert_primary_key(...)` used by `bootstrap_single_user_profile()` to seed the single-user primary API key for both SQLite and Postgres backends.
  - `create_api_key_row(...)` / `create_virtual_key_row(...)` and `mark_rotated(...)` / `revoke_api_key_for_user(...)` for create/rotate/revoke flows.
  - `increment_usage(...)`, `mark_key_expired(...)`, and `insert_audit_log(...)` for usage counters, status transitions, and audit logging.
- `APIKeyManager` constructs the repository lazily via a private `_get_repo()` helper, keeping the manager API unchanged while centralizing all request-time `api_keys` queries behind the repository.
- DDL creation for `api_keys` / `api_key_audit_log` and the maintenance helper `cleanup_expired_keys` still include backend-specific SQL; they are treated as bootstrap/maintenance guardrails for this iteration and may be migrated into migrations or dedicated repository helpers in a later phase.
- A minimal `AuthnzUsersRepo` has been added at `tldw_Server_API/app/core/AuthNZ/repos/users_repo.py` wrapping the existing `UsersDB` abstraction and exposing `get_user_by_id`, `get_user_by_username`, and `get_user_by_uuid` (plus a paginated `list_users` helper) against the shared `DatabasePool`. Integration tests exercise this repo against both SQLite (`tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_users_repo_sqlite.py`) and Postgres (`tldw_Server_API/tests/AuthNZ/integration/test_authnz_users_repo_postgres.py`) backends, and the admin `GET /api/v1/admin/users/{user_id}` and `GET /api/v1/admin/users` endpoints are covered by SQLite/Postgres admin endpoint tests.
- A thin `AuthnzRbacRepo` at `tldw_Server_API/app/core/AuthNZ/repos/rbac_repo.py` now centralizes calls into `UserDatabase_v2` for RBAC permission checks. The higher-level helpers in `tldw_Server_API/app/core/AuthNZ/rbac.py` have been refactored to delegate to this repo, and `AuthnzRbacRepo.get_role_effective_permissions` backs the admin `GET /api/v1/admin/roles/{role_id}/permissions/effective` endpoint. Broader migration of RBAC/DDL logic into repository helpers remains future work for this stage.

### Stage 4: Backend Drift Reduction
**Goal**: Reduce inline dialect branching in high-impact AuthNZ modules using repositories.

**Success Criteria**:
- `virtual_keys`, `orgs_teams`, and relevant parts of the AuthNZ rate limiter use repositories instead of raw SQL with `hasattr(conn,'fetchval')` checks.
- There are fewer duplicated Postgres/SQLite query variants in AuthNZ code.

**Tests**:
- Regression tests for affected modules under both backends (where supported by current test fixtures).

**Status**: Done (v0.1) — initial repo-backed refactors for `virtual_keys`, `orgs_teams`, AuthNZ rate-limiter storage, API-key management, usage logging, and virtual-key quota counters (via `AuthnzQuotasRepo`) have landed; remaining backend drift and inline SQL is now limited primarily to Postgres bootstrap DDL in `initialize.py`, with further consolidation tracked as deferred follow-up in `Docs/Product/Completed/AuthNZ-Refactor-Implementation-Plan.md` under “Remaining inline SQL / backend detection (tech debt)”.

---

### Stage 5: Post-v0.1 Roadmap – Recommended Next Steps

#### 5.1 Inline DDL & Quota Counters → Repos/Migrations

- **Goal**: Incrementally migrate remaining inline Postgres/SQLite DDL and quota counters into repository helpers or migrations so that backends remain an implementation detail.
- **AuthNZ bootstrap DDL**:
  - Add canonical migrations (SQLite + Postgres) for the AuthNZ tables that are still primarily created via bootstrap DDL in `initialize.py` (`audit_logs`, `sessions`, `registration_codes`, RBAC tables, orgs/teams), mirroring the patterns already used for usage tables and virtual-key counters (`pg_migrations_extra.ensure_usage_tables_pg`, `ensure_virtual_key_counters_pg`).
  - Treat `initialize.py` (and the small DDL helpers in `api_key_manager.py` / `rate_limiter.py`) as guarded backstops only: keep them idempotent for one release while migrations are validated in CI, then gate or retire new inline DDL additions in favor of migrations + repo helpers. In v0.1, Postgres bootstrap for these core AuthNZ tables is already centralized in `AuthNZ.pg_migrations_extra.ensure_authnz_core_tables_pg`, and `initialize.setup_database` now delegates to this helper instead of embedding raw DDL.
- **Virtual-key quota counters** (Status: Done in v0.1):
  - `AuthnzQuotasRepo` (`tldw_Server_API/app/core/AuthNZ/repos/quotas_repo.py`) now owns `vk_jwt_counters` and `vk_api_key_counters`, centralizing dialect-specific DDL/upsert logic for SQLite and Postgres via existing migrations/backstops (`migration_023_create_virtual_key_counters`, `ensure_virtual_key_counters_pg`).
  - `tldw_Server_API/app/core/AuthNZ/quotas.py::increment_and_check_jwt_quota` and `increment_and_check_api_key_quota` delegate to the repo, and new code no longer introduces inline DDL or `hasattr(conn, "fetch*")` branches for these tables; legacy helpers such as `_ensure_tables` have been retired from runtime paths.

#### 5.2 `is_single_user_mode()` → Claim-First Alternatives

- **Goal**: For each remaining `is_single_user_mode()` in auth-adjacent code, design a claim-first alternative (using `AuthPrincipal` + profile/feature flags) and back it with tests, so that “mode” becomes configuration/UX only.
- **Audit and classification**:
  - Enumerate remaining `is_single_user_mode()` callsites in auth-adjacent modules (`tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`, `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`, `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`, `tldw_Server_API/app/main.py`, embeddings/evaluations endpoints, `tldw_Server_API/app/core/PrivilegeMaps/service.py`, MCP/unified endpoints, backpressure and tenant-RPS helpers) and classify each as either coordination/UX (startup banners, WebUI hints, warm-ups) or auth/guardrail (permissions, quotas, embeddings/MCP behavior).
  - Keep coordination/UX branches mode-aware for now (eventually driven by `PROFILE` + feature flags); focus this iteration on auth/guardrail callsites where mode currently influences permissions or quotas.
- **Principal-first replacements (Stage 4 alignment)**:
  - For each auth/guardrail callsite, design a principal-based alternative that takes `AuthPrincipal` and/or `AuthContext` together with profile/feature flags (`PROFILE`, `ENABLE_*` flags, and dedicated toggles such as `EMBEDDINGS_TENANT_RPS_PROFILE_AWARE`, `MCP_SINGLE_USER_COMPAT_SHIM`), as outlined under “Stage 4: Claim-First Dependencies & AuthContext Adoption” in `Docs/Product/Completed/AuthNZ-Refactor-Implementation-Plan.md`.
  - Wire the new path behind an explicit env flag per surface so tests can exercise both legacy (mode-driven) and principal-first behavior before defaults change; keep single-user deployments participating via bootstrapped principal claims instead of `is_single_user_mode()` shortcuts.
- **Tests and deprecation**:
  - Extend the existing claim-first HTTP and unit test matrix (metrics admin, Resource-Governor, RAG/media permissions, jobs admin, Prompt Studio, embeddings model management) to cover each newly migrated callsite, asserting that single-user profile + bootstrap principal yields the same effective permissions and quotas as today, and that multi-user SQLite/Postgres behavior remains unchanged.
  - Once coverage is in place and principal-first paths are stable, update docs and guardrails to treat `is_single_user_mode()` as a compatibility alias used only in settings/bootstrap and coordination/UX code; add lints or code-review checks to discourage new auth-path logic from branching on `AUTH_MODE` or `is_single_user_mode()` directly.
