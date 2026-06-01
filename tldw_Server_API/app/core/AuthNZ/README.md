# AuthNZ

Note: This README follows the project-wide template to help contributors quickly understand features, architecture, and how to extend the module.

## 1. Descriptive of Current Feature Set

- Purpose: Authentication and authorization for both single-user and multi-user deployments; sessions, API keys, RBAC, and usage guardrails for the entire platform.
- Capabilities:
  - Modes: single-user (`X-API-KEY`) and multi-user (JWT access/refresh).
  - Account security: Argon2id passwords, MFA/TOTP, password reset and email verification.
  - Sessions and revocation: refresh rotation, blacklist-backed revocation, session metadata.
  - Authorization: roles/permissions, org/team hierarchy, fine-grained checks and scopes.
  - Keys: user API keys (rotation, expiry, IP allowlists) and Virtual Keys (scoped + LLM budgeted).
  - Guardrails: rate limiting, quotas, lockout, CSRF protection, security headers.
  - Observability: audit hooks, usage logging, metrics, alerting.
- Inputs/Outputs:
  - Inputs: credentials (username/password, TOTP/backup codes), API keys, JWTs, CSRF cookie/header for WebUI.
  - Outputs: token responses, user/session info, success messages for reset/verify flows.
- Related Endpoints (mounted under `/api/v1`):
  - Core auth + enhanced flows (reset, verify, MFA): `tldw_Server_API/app/api/v1/endpoints/auth.py:1`
  - Admin (RBAC, orgs/teams, users): `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py:1` plus split modules in `tldw_Server_API/app/api/v1/endpoints/admin/` (`admin_user.py`, `admin_api_keys.py`, `admin_profiles.py`, `admin_sessions_mfa.py`, `admin_byok.py`, `admin_llm_providers.py`, `admin_orgs.py`, `admin_settings.py`, `admin_registration.py`, `admin_system.py`, `admin_usage.py`, `admin_budgets.py`, `admin_tools.py`, `admin_personalization.py`, `admin_network.py`), and `tldw_Server_API/app/api/v1/endpoints/users.py:1`, `tldw_Server_API/app/api/v1/endpoints/privileges.py:1`, `tldw_Server_API/app/api/v1/endpoints/register.py:1`
  - Debug helpers: `tldw_Server_API/app/api/v1/endpoints/authnz_debug.py:1`
  - Dependencies used by endpoints: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py:1`
- Related Schemas (requests/responses and admin/RBAC):
  - `tldw_Server_API/app/api/v1/schemas/auth_schemas.py:1`, `tldw_Server_API/app/api/v1/schemas/admin_schemas.py:1`, `tldw_Server_API/app/api/v1/schemas/admin_rbac_schemas.py:1`
  - `tldw_Server_API/app/api/v1/schemas/api_key_schemas.py:1`, `tldw_Server_API/app/api/v1/schemas/org_team_schemas.py:1`, `tldw_Server_API/app/api/v1/schemas/privileges.py:1`

## 2. Technical Details of Features

- Architecture & Data Flow:
  - Endpoints call service modules via DI from `API_Deps/auth_deps.py`.
  - Services include: `password_service.py`, `jwt_service.py`, `session_manager.py`, `token_blacklist.py`, `api_key_manager.py`, `virtual_keys.py`, `quotas.py`, `orgs_teams.py`, `rbac.py`, `permissions.py`.
  - Middleware and guards: `rate_limiter.py`, `llm_budget_middleware.py`, `csrf_protection.py`, `security_headers.py`, `usage_logging_middleware.py`.
  - Backends: SQLite (default) and PostgreSQL (production); optional Redis for cache/limits/blacklist.
- Modes:
  - Single-user: `X-API-KEY` is validated; optional IP allowlist; JWT stack bypassed.
  - Multi-user: username/password (+MFA) → JWT access/refresh; sessions persisted with rotation and blacklist revocation.
  - Service tokens: intended for internal use; by default only loopback clients are accepted unless `SERVICE_TOKEN_ALLOWED_IPS` is configured.
- Enterprise feature flags:
  - `AUTH_FEDERATION_ENABLED`, `SECRET_BACKENDS_ENABLED`, and `MCP_CREDENTIAL_BROKER_ENABLED` are available for the enterprise roadmap.
  - Current support matrix is fail-closed: `AUTH_MODE=multi_user` plus a PostgreSQL `DATABASE_URL` are required.
  - `MCP_CREDENTIAL_BROKER_ENABLED` also requires `SECRET_BACKENDS_ENABLED`.
  - Explicit `PROFILE` values must be enterprise-compatible (`multi-user-postgres` / `enterprise*`) or the derived helpers report the feature as unsupported.
- Key Classes/Functions:
  - `settings.py` (`get_settings`, `is_single_user_mode`) for configuration.
  - `jwt_service.JWTService` issues/verifies tokens, password reset/email verification/virtual access tokens.
  - `session_manager.SessionManager` manages session lifecycle and revocation.
  - `password_service.PasswordService` handles Argon2id hashing and strength validation.
  - `api_key_manager` and `virtual_keys` provide API/Virtual key issuance, rotation, validation, and budgets.
  - `rate_limiter.RateLimiter` enforces token-bucket limits and lockouts; `quotas` records usage.
  - `orgs_teams`, `rbac`, `permissions`, `org_rbac` provide RBAC resolution and checks.
- Data Models & DB (SQLite migrations in `migrations.py`; Postgres extras in `pg_migrations_extra.py`):
  - Core tables: `users`, `sessions`, `api_keys`, `api_key_audit_log`, `token_blacklist`, `password_history`.
  - Registration/reset: `registration_codes`, `password_reset_tokens`.
  - RBAC: `roles`, `permissions`, `role_permissions`, `user_roles`, `user_permissions`.
  - Scoped RBAC: `org_role_permissions`, `team_role_permissions`.
  - Organizations/Teams: `organizations`, `org_members`, `teams`, `team_members`.
  - Usage & budgets: `rate_limits`, `usage_log`, `usage_daily`, `llm_usage_log`, `llm_usage_daily`.
  - Virtual key extensions on `api_keys` and Postgres `tool_catalogs` tables (via `pg_migrations_extra.py`).
- Configuration (selected):
  - `AUTH_MODE`, `DATABASE_URL`, `JWT_SECRET_KEY`/`JWT_PRIVATE_KEY`, `ACCESS_TOKEN_EXPIRE_MINUTES`, `REFRESH_TOKEN_EXPIRE_DAYS`.
  - `PASSWORD_MIN_LENGTH`, Argon2 cost knobs; `REDIS_URL`; `RATE_LIMIT_*`, `MAX_LOGIN_ATTEMPTS`, `LOCKOUT_DURATION_MINUTES`.
  - `ENABLE_REGISTRATION`, `REQUIRE_REGISTRATION_CODE`, `VIRTUAL_KEYS_ENABLED`, `LLM_BUDGET_ENFORCE`, `LLM_BUDGET_ENDPOINTS`.
  - `ORG_RBAC_PROPAGATION_ENABLED`, `ORG_RBAC_SCOPE_MODE`, `ORG_RBAC_SCOPED_PERMISSION_DENYLIST`.
  - `SESSION_COOKIE_SECURE`, `CSRF_BIND_TO_USER`, `SERVICE_ACCOUNT_RATE_LIMIT`, `SINGLE_USER_API_KEY`.
  - `SERVICE_TOKEN_ALLOWED_IPS` (optional allowlist for service tokens; empty means loopback-only).
  - `BYOK_ENABLED`, `BYOK_ALLOWED_PROVIDERS`, `BYOK_ENCRYPTION_KEY`, `BYOK_SECONDARY_ENCRYPTION_KEY`.
  - Settings merge env, `.env` files, and project config via `load_comprehensive_config`.

### Session Encryption Key

- Purpose: Encrypts session tokens at rest using Fernet.
- Configure explicitly with `SESSION_ENCRYPTION_KEY` (urlsafe base64, 32-byte key when decoded). If not set, a key is persisted to disk.
- Persistence locations (searched in order):
  - Default: `tldw_Server_API/Config_Files/session_encryption.key`.
  - Legacy fallback: `PROJECT_ROOT/Config_Files/session_encryption.key`.
- Legacy storage opt-in: set `SESSION_KEY_STORAGE=project` to persist and prefer `PROJECT_ROOT/Config_Files/session_encryption.key`.
  - In API mode (default), if a valid key exists at project root and the API path is missing/invalid, the manager migrates the key to the API path (0600 perms) and logs a notice.
- Security: file must be a regular file, owned by the current user, and is written with `0600` permissions; symlinks and invalid contents are rejected.
- Concurrency & Performance:
  - Async DB paths for asyncpg/aiosqlite; Redis-backed counters when available.
  - Token/JTI blacklist checks cached; rate limits use token buckets with burst support.
- Error Handling & Security:
  - Custom exceptions in `exceptions.py`; consistent HTTP errors from endpoints.
  - Input validation in `input_validation.py`; CSRF middleware for WebUI flows.
- Admin routes should be protected via claim-first dependencies (`get_auth_principal` + `RequireRole("admin")` / `RequirePermission(...)`); legacy `require_admin`/`require_role` shims in API deps are retired.

## 3. Developer-Related/Relevant Information for Contributors

- Folder Structure:
  - Core services/utilities: `settings.py`, `database.py`, `migrations.py`, `pg_migrations_extra.py`, `initialize.py`, `run_migrations.py`.
  - Auth flows: `jwt_service.py`, `session_manager.py`, `token_blacklist.py`, `password_service.py`, `mfa_service.py`, `email_service.py`.
  - RBAC/orgs: `rbac.py`, `permissions.py`, `orgs_teams.py`, `privilege_catalog.py`.
  - Guardrails: `rate_limiter.py`, `llm_budget_middleware.py`, `csrf_protection.py`, `security_headers.py`, `usage_logging_middleware.py`, `llm_budget_guard.py`.
  - Keys/budgets: `api_key_manager.py`, `virtual_keys.py`, `quotas.py`.
  - Ops/monitoring: `monitoring.py`, `alerting.py`, `scheduler.py`.
- Extension Points:
  - Add endpoints under `app/api/v1/endpoints/` and use dependencies from `API_Deps/auth_deps.py` (`get_auth_principal`, `RequireRole`, `RequirePermission`, `TokenScopeGuard`, `check_rate_limit`).
  - Extend roles/permissions using RBAC tables; seed updates go into `migrations.py` seeding section.
  - Add budgets or allowlists by extending `virtual_keys.py`/`api_key_manager.py` and updating schema + tests.
  - Add periodic tasks in `scheduler.py` (e.g., cleanup of expired tokens/lockouts).
- Coding Patterns:
  - Prefer DI via `Depends(...)`; avoid parsing headers directly in endpoints.
  - Use `loguru` for logging; never log secrets (API keys, passwords).
  - Keep both SQLite and Postgres paths functional; feature-detect backend when required.
- Tests:
  - Locations: `tldw_Server_API/tests/AuthNZ`, `tldw_Server_API/tests/AuthNZ_SQLite`, `tldw_Server_API/tests/AuthNZ_Postgres`, plus cross-cutting `tldw_Server_API/tests/Security`.
  - Run examples:
    - `python -m pytest tldw_Server_API/tests/AuthNZ -v`
    - `python -m pytest tldw_Server_API/tests/AuthNZ_Postgres -v`
  - Postgres fixtures may auto-start Docker unless `TLDW_TEST_NO_DOCKER=1`.
  - Many tests rely on `TEST_MODE=1` to bypass heavy loops and relax FKs for usage tables.
- Local Dev Tips:
  - Migrate/initialize: `python -m tldw_Server_API.app.core.AuthNZ.run_migrations` then `python -m tldw_Server_API.app.core.AuthNZ.initialize`.
  - Switch modes by `AUTH_MODE` and rerun migrations (`migrate_to_multiuser.py` assists upgrades).
  - Helpful commands:
    - Generate registration code: `...AuthNZ.initialize --create-registration-code --max-uses 10 --expires 30`
    - Rotate JWT secrets: `...AuthNZ.initialize --rotate-jwt`
    - Inspect sessions: `...AuthNZ.initialize --list-sessions --user alice@example.com`
    - Create virtual key: `...AuthNZ.initialize --create-virtual-key --user-id 3 --day-tokens 100000 --month-usd 50 --allow-endpoints chat.completions embeddings`
- Pitfalls & Gotchas:
  - MFA endpoints require multi-user mode and PostgreSQL backend.
  - LLM budget enforcement relies on correct `LLM_BUDGET_ENDPOINTS` and middleware placement.
  - Single-user `X-API-KEY` may be additionally constrained by IP allowlist.
  - Tests may rely on relaxed foreign keys for usage tables; do not tighten without updating fixtures.
- Follow-up candidates:
  - Complete docstring cleanup for obsolete params referenced in older comments.
  - Expand integration tests for lockout/CSRF and virtual-key requeues.
  - Optional CI guard to assert presence of the three section headers in module READMEs.

---

Example Quick Start (optional)

```bash
python -m tldw_Server_API.app.core.AuthNZ.run_migrations
python -m tldw_Server_API.app.core.AuthNZ.initialize
```
