# AuthNZ And Admin Domain Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: AuthNZ and Admin
- In scope: authentication modes, JWT/API key flows, RBAC, org and tenant boundaries, admin endpoints, setup/debug surfaces, audit hooks, and tests.
- Out of scope: remediation implementation and unrelated product UX changes.

## Findings Table

| ID | Candidate ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AUDIT-2026-06-27-AUTH-001 | CANDIDATE-authnz-admin-001 | confirmed_issue | static_confirmed | medium | high | security | Admin impersonation response advertises a 15 minute TTL but mints a normal access token | open | validated |
| AUDIT-2026-06-27-AUTH-002 | CANDIDATE-authnz-admin-002 | confirmed_issue | static_confirmed | high | high | security | Impersonation actor metadata is not preserved for durable audit attribution | open | validated |
| AUDIT-2026-06-27-AUTH-003 | CANDIDATE-authnz-admin-003 | likely_risk | static_confirmed | medium | high | reliability | Admin impersonation uses SQLite placeholders through a raw PostgreSQL connection path | open | needs_reproduction |

## Index Mapping

Use finding IDs like `AUDIT-2026-06-27-AUTH-001`. Set `evidence_tier` from the report section bucket (`confirmed_issue`, `likely_risk`, or `improvement_opportunity`) and `evidence_strength` from the schema allowed values. Set `source_report` to `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/authnz-admin.md`, set `owner_domain` to this report owner, and include `affected_paths`, `recommendation`, `status`, and `validation_status` in each detailed finding.

## Confirmed Issues

### AUDIT-2026-06-27-AUTH-001 / CANDIDATE-authnz-admin-001 - Admin impersonation response advertises a 15 minute TTL but mints a normal access token

- Severity: medium
- Confidence: high
- Category: security
- Source report: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/authnz-admin.md`
- Owner domain: AuthNZ and Admin
- Affected paths:
  - `tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py`
  - `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
  - `tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py`
- Evidence:
  - `admin_impersonation.py:20-34` defines `_IMPERSONATION_TTL_MINUTES = 15` and returns `expires_in_minutes` from that constant.
  - `admin_impersonation.py:49-53` documents the token as a short TTL 15 minute token.
  - `admin_impersonation.py:97-105` calls `jwt_svc.create_access_token(...)` without passing a TTL override.
  - `jwt_service.py:193-205` sets `exp` from `settings.ACCESS_TOKEN_EXPIRE_MINUTES`, so the issued impersonation token follows the normal access-token lifetime rather than the advertised impersonation lifetime.
  - `test_admin_impersonation.py:36-44` asserts only the response default. `test_admin_impersonation.py:89-94` asserts the impersonation claims, but no test decodes the JWT and verifies its `exp`.
- Impact: Admins and clients receive a response saying the impersonation token expires after 15 minutes, while the actual token can remain valid for the globally configured access-token lifetime. If that global lifetime is longer, impersonation exposure and revocation expectations are materially weaker than the endpoint contract.
- Recommendation: Add an explicit impersonation token creation path that accepts the 15 minute TTL, marks the token as an impersonation token, and returns the actual expiry. Add a regression test that decodes the token and verifies `exp - iat` matches the impersonation TTL.

### AUDIT-2026-06-27-AUTH-002 / CANDIDATE-authnz-admin-002 - Impersonation actor metadata is not preserved for durable audit attribution

- Severity: high
- Confidence: high
- Category: security
- Source report: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/authnz-admin.md`
- Owner domain: AuthNZ and Admin
- Affected paths:
  - `tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py`
  - `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
  - `tldw_Server_API/app/services/admin_users_service.py`
- Evidence:
  - `admin_impersonation.py:4-6` states that the `impersonated_by` claim is for full audit traceability.
  - `admin_impersonation.py:101-104` adds `impersonated_by` and `impersonation` claims to the JWT.
  - `admin_impersonation.py:107-111` records token creation with a process log line only; it does not call the unified admin audit service or require a privileged-action reauthentication step.
  - `User_DB_Handling.py:537-557` decodes the access token and extracts subject/org/team scope claims, but not impersonation metadata.
  - `User_DB_Handling.py:885-914` builds `AuthPrincipal` and `AuthContext` as the target user, with no `impersonated_by` or `impersonation` fields preserved for downstream audit hooks.
  - Comparable high-risk user administration operations call `verify_privileged_action` and emit durable audit events, for example `admin_users_service.py:258-266` and `admin_users_service.py:348-357`.
- Impact: Actions performed with an impersonation token are authenticated as the target user, while the original admin actor is dropped from the request auth context. Downstream audit events can therefore attribute actions to the target account rather than the admin who initiated impersonation, undermining the endpoint's stated traceability and incident response value.
- Recommendation: Treat impersonation token issuance as a privileged admin action: require step-up reauthentication or an admin reauth token, emit a durable mandatory audit event on issuance, and propagate impersonation metadata into `AuthContext`/request state so all downstream audit events can include both actor and subject.

## Likely Risks

### AUDIT-2026-06-27-AUTH-003 / CANDIDATE-authnz-admin-003 - Admin impersonation uses SQLite placeholders through a raw PostgreSQL connection path

- Severity: medium
- Confidence: high
- Category: reliability
- Source report: `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/authnz-admin.md`
- Owner domain: AuthNZ and Admin
- Affected paths:
  - `tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py`
  - `tldw_Server_API/app/core/AuthNZ/database.py`
  - `tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py`
- Evidence:
  - `admin_impersonation.py:60-66` and `admin_impersonation.py:84-90` use `pool.acquire()` and execute SQL with `?` placeholders directly on the acquired connection.
  - `database.py:642-658` shows `DatabasePool.acquire()` yields the raw asyncpg connection when the AuthNZ backend is PostgreSQL.
  - `database.py:686-710` performs `?` to `$N` conversion only in helper methods such as `DatabasePool.execute()` and `DatabasePool.fetchone()`, not in raw `pool.acquire()` callers.
  - `test_admin_impersonation.py:60-70` mocks a generic async connection and JWT service; it does not exercise a PostgreSQL-backed pool or placeholder conversion.
- Impact: In multi-user deployments backed by PostgreSQL, the impersonation endpoint can fail at runtime when asyncpg receives SQLite-style placeholders, returning a 500 instead of issuing a token or returning a domain error.
- Recommendation: Replace the raw connection calls with repository methods or `DatabasePool.fetchone()` so placeholder normalization is centralized. Add a backend-agnostic unit test or PostgreSQL fixture test for the impersonation user and role lookups.

## Improvement Opportunities

No separate improvement opportunities beyond the targeted regression tests recommended in the findings above.

## Coverage And Evidence

### Files Inspected

- `Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/endpoint-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/backend-test-inventory.txt`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- `tldw_Server_API/app/api/v1/API_Deps/setup_deps.py`
- `tldw_Server_API/app/api/v1/endpoints/auth.py`
- `tldw_Server_API/app/api/v1/endpoints/authnz_debug.py`
- `tldw_Server_API/app/api/v1/endpoints/setup.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_api_keys.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_orgs.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_user.py`
- `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`
- `tldw_Server_API/app/core/AuthNZ/database.py`
- `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- `tldw_Server_API/app/services/admin_api_keys_service.py`
- `tldw_Server_API/app/services/admin_orgs_service.py`
- `tldw_Server_API/app/services/admin_scope_service.py`
- `tldw_Server_API/app/services/admin_users_service.py`
- `tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py`
- Representative tests under `tldw_Server_API/tests/AuthNZ*` and `tldw_Server_API/tests/Admin` from the audit test inventory.

### Tests Or Scans Run

- `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py -q`
  - Result: 5 passed, 29 warnings.
- Reviewed existing audit baseline: `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/bandit-app-summary.txt`
  - Result in provided baseline: 4,818 Bandit results, 26 medium, 0 high.

Additional review commands run:

- `sed -n '1,220p' Docs/superpowers/reviews/2026-06-27-repo-audit/domains/authnz-admin.md`
- `sed -n '1,240p' Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `rg -n "router = APIRouter|include_router\\(|Depends\\(RequireRole" tldw_Server_API/app/api/v1/endpoints/admin/__init__.py`
- `rg -n "verify_privileged_action|emit_admin_account_audit_event|AuditEventType" tldw_Server_API/app/services/admin_users_service.py tldw_Server_API/app/api/v1/endpoints/admin/admin_user.py tldw_Server_API/app/services/admin_api_keys_service.py tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py`
- `rg -n "authnz-debug|require_debug_roles|security|require_local_setup_access|_require_setup_write_access" tldw_Server_API/app/api/v1/endpoints/authnz_debug.py tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/api/v1/API_Deps/setup_deps.py`
- `nl -ba tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py | sed -n '1,140p'`
- `nl -ba tldw_Server_API/app/core/AuthNZ/jwt_service.py | sed -n '180,225p'`
- `nl -ba tldw_Server_API/app/core/AuthNZ/database.py | sed -n '635,715p'`
- `nl -ba tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py | sed -n '520,575p'`
- `nl -ba tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py | sed -n '875,925p'`
- `nl -ba tldw_Server_API/app/api/v1/endpoints/admin/__init__.py | sed -n '105,148p'`
- `nl -ba tldw_Server_API/app/services/admin_users_service.py | sed -n '248,270p'`
- `nl -ba tldw_Server_API/app/services/admin_users_service.py | sed -n '340,360p'`
- `nl -ba tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py | sed -n '1,130p'`
- Backlog tracking setup: `backlog task create "Conduct AuthNZ/Admin domain audit report" ...`

### Blocked Or Unverified Areas

- No production/source code was edited.
- No dependencies were installed, services started, Docker used, or network access attempted.
- PostgreSQL behavior for CANDIDATE-authnz-admin-003 was not runtime reproduced because the domain review rules prohibit starting services or Docker. The source path is still statically confirmed.
- Full AuthNZ/Admin test suite and full Bandit scan were not rerun for this domain report; the existing comprehensive audit Bandit summary was reviewed, and one focused impersonation unit test file was run.
- Setup and debug surfaces were inspected statically. No live server probing was performed.

### Evidence Notes

- The parent admin router applies `Depends(RequireRole("admin"))` to all included admin routers, including impersonation (`admin/__init__.py:109-144`). The findings above do not claim missing authentication; they focus on token lifetime, audit attribution, and PostgreSQL reliability inside the authenticated admin impersonation flow.
- Setup endpoints expose selected unauthenticated OpenAPI surfaces, but they are guarded by local setup access dependencies and write-state checks (`setup.py:582-603`, `setup.py:2352-2382`; `setup_deps.py:347`). No setup bypass candidate was confirmed in this pass.
- AuthNZ debug endpoints are limited to single-user mode or `super_admin`/`owner` roles (`authnz_debug.py:20-42`, `authnz_debug.py:103-125`). No debug endpoint exposure candidate was confirmed in this pass.
