# AuthNZ Module Review Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved AuthNZ module review and deliver one evidence-backed, remediation-oriented report covering security, correctness, maintainability, test gaps, and explicit blind spots across the scoped AuthNZ core.

**Architecture:** This is a read-first, risk-first review plan. Execution starts by locking the current worktree baseline, then inspects the highest-risk AuthNZ primitives, then key and secret handling, then authorization and governance logic, then migrations and backend-divergence paths, and only after that runs the smallest targeted test slices needed to confirm or weaken candidate findings. No repository source changes are expected during execution; the deliverable is the final in-session report.

**Tech Stack:** Python 3, pytest, git, rg, find, sed, Markdown

---

## Scope Lock

Keep these decisions fixed during execution:

- review the current working tree by default, not only `HEAD`
- keep code scope inside `tldw_Server_API/app/core/AuthNZ`
- use `tldw_Server_API/tests/AuthNZ` as primary evidence and selectively pull from `AuthNZ_SQLite`, `AuthNZ_Postgres`, `AuthNZ_Unit`, and `AuthNZ_Federation` only when those tests directly validate scoped core behavior
- do not broaden into endpoint or dependency wiring review even when a test exercises that surface
- separate `Confirmed finding`, `Probable risk`, and `Improvement`
- do not modify repository source files during the review itself
- do not run broad blanket suites; use the smallest targeted verification needed to answer a concrete question
- keep blind spots explicit instead of implying unreviewed areas are safe
- do not reuse `Docs/superpowers/plans/2026-03-23-authnz-sequential-review.md` as a task source because it intentionally broadens beyond the approved boundary

## Review File Map

**No repository files should be modified during execution.**

**Spec and plan inputs:**
- `Docs/superpowers/specs/2026-04-07-authnz-module-review-design.md`
- `Docs/superpowers/plans/2026-04-07-authnz-module-review-execution-plan.md`

**Primary implementation files to inspect first:**
- `tldw_Server_API/app/core/AuthNZ/README.md`
- `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- `tldw_Server_API/app/core/AuthNZ/session_manager.py`
- `tldw_Server_API/app/core/AuthNZ/token_blacklist.py`
- `tldw_Server_API/app/core/AuthNZ/password_service.py`
- `tldw_Server_API/app/core/AuthNZ/mfa_service.py`
- `tldw_Server_API/app/core/AuthNZ/lockout_tracker.py`
- `tldw_Server_API/app/core/AuthNZ/auth_governor.py`
- `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
- `tldw_Server_API/app/core/AuthNZ/api_key_crypto.py`
- `tldw_Server_API/app/core/AuthNZ/crypto_utils.py`
- `tldw_Server_API/app/core/AuthNZ/key_resolution.py`
- `tldw_Server_API/app/core/AuthNZ/ip_allowlist.py`
- `tldw_Server_API/app/core/AuthNZ/virtual_keys.py`
- `tldw_Server_API/app/core/AuthNZ/byok_helpers.py`
- `tldw_Server_API/app/core/AuthNZ/byok_runtime.py`
- `tldw_Server_API/app/core/AuthNZ/byok_rotation.py`
- `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`
- `tldw_Server_API/app/core/AuthNZ/principal_model.py`
- `tldw_Server_API/app/core/AuthNZ/permissions.py`
- `tldw_Server_API/app/core/AuthNZ/rbac.py`
- `tldw_Server_API/app/core/AuthNZ/org_rbac.py`
- `tldw_Server_API/app/core/AuthNZ/orgs_teams.py`
- `tldw_Server_API/app/core/AuthNZ/quotas.py`
- `tldw_Server_API/app/core/AuthNZ/rate_limiter.py`
- `tldw_Server_API/app/core/AuthNZ/llm_budget_guard.py`
- `tldw_Server_API/app/core/AuthNZ/llm_budget_middleware.py`
- `tldw_Server_API/app/core/AuthNZ/settings.py`
- `tldw_Server_API/app/core/AuthNZ/database.py`
- `tldw_Server_API/app/core/AuthNZ/migrations.py`
- `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- `tldw_Server_API/app/core/AuthNZ/initialize.py`
- `tldw_Server_API/app/core/AuthNZ/migrate_to_multiuser.py`

**Representative repository files to inspect:**
- `tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/token_blacklist_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/users_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/mfa_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/rbac_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/quotas_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/user_provider_secrets_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/org_provider_secrets_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/byok_oauth_state_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/identity_provider_repo.py`

**High-value tests to inspect and selectively run:**
- `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service_rs256.py`
- `tldw_Server_API/tests/AuthNZ/property/test_jwt_service_property.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_session_revocation_blacklist.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_token_metadata.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_session_refresh_cache.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_token_blacklist_basic.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_mfa_service.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_real_rate_limiter.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py`
- `tldw_Server_API/tests/AuthNZ_Postgres/test_auth_enhanced_mfa.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_api_key_manager_validation.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_api_key_crypto.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_limits_unit.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_enforcement_unit.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_byok_crypto.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_byok_helpers.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_byok_base_url_validation.py`
- `tldw_Server_API/tests/AuthNZ_Federation/test_local_secret_backend.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_api_key_rotation_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_allowlists_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_virtual_keys_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_byok_rotation_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_auth_principal_resolver.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_auth_jwt_membership_validation.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_org_policy_from_principal.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_org_deps.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_state_consistency.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_jwt_happy_path.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_api_key_happy_path.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_single_user_claims_permissions.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_rbac_effective_permissions.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_claims_rbac_overlaps_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_org_rbac_scoped_permissions_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_quota_enforcement_http_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_Postgres/test_role_effective_permissions_pg.py`
- `tldw_Server_API/tests/AuthNZ_Postgres/test_orgs_teams_pg.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_database_pool_fetchone_sqlite_fallback.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_usage_truthiness.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_migrate_to_multiuser_review_fixes.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_api_keys_repo_backend_selection.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_sessions_repo_backend_selection.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_token_blacklist_repo_backend_selection.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_quotas_repo_backend_selection.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_orgs_teams_repo_backend_selection.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_api_keys_repo_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_sessions_repo_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_token_blacklist_repo_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_quotas_repo_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_rate_limits_repo_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_orgs_teams_repo_sqlite.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_api_keys_repo_postgres.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_sessions_repo_postgres.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_token_blacklist_repo_postgres.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_quotas_repo_postgres.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_rate_limits_repo_postgres.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_orgs_teams_repo_postgres.py`

**Scratch artifacts allowed during execution:**
- `/tmp/authnz_review_notes.md`
- `/tmp/authnz_identity_pytest.log`
- `/tmp/authnz_keys_pytest.log`
- `/tmp/authnz_authz_pytest.log`
- `/tmp/authnz_backend_pytest.log`

## Stage Overview

## Stage 1: Baseline and Report Contract
**Goal:** Lock the worktree baseline, confirm the exact AuthNZ review surface, and fix the final report structure before deep reading starts.
**Success Criteria:** The dirty-worktree note, scope boundary, hotspot order, test inventory, and final output template are fixed.
**Tests:** No pytest execution in this stage.
**Status:** Not Started

## Stage 2: Identity, Session, and Login Security Pass
**Goal:** Inspect JWT, session, revocation, password, MFA, and lockout behavior first.
**Success Criteria:** Candidate findings are recorded with exact file references and evidence type, and weaker concerns are clearly separated from confirmed issues.
**Tests:** Read and later run the smallest identity- and session-focused test slice needed to validate or weaken claims.
**Status:** Not Started

## Stage 3: Keys, Secret Material, and Provider-Credential Pass
**Goal:** Inspect API keys, virtual keys, allowlists, BYOK, secret storage, and key resolution paths.
**Success Criteria:** Key issuance, validation, rotation, scoping, encryption, and provider-secret handling assumptions are traced end to end.
**Tests:** Read and later run the smallest key- and secret-focused test slice needed to validate or weaken claims.
**Status:** Not Started

## Stage 4: Authorization, Governance, and Budget Pass
**Goal:** Inspect principal resolution, RBAC, org or team scoping, rate limits, quotas, and budget enforcement.
**Success Criteria:** Permission derivation, scoping rules, quota enforcement, and governance fail-open or fail-closed paths are traced with enough evidence to support confirmed findings or explicit probable-risk labels.
**Tests:** Read and later run the smallest authorization- and quota-focused test slice needed to validate or weaken claims.
**Status:** Not Started

## Stage 5: Persistence, Migrations, and Backend-Divergence Pass
**Goal:** Inspect database, migration, initialization, and representative repo behavior for schema, routing, and backend-specific correctness risks.
**Success Criteria:** SQLite or Postgres divergence, migration assumptions, repo routing, and persistence invariants are reviewed with direct code evidence and representative test coverage checks.
**Tests:** Read and later run targeted repo and migration tests, including backend-specific slices only when they answer a concrete claim.
**Status:** Not Started

## Stage 6: Targeted Verification and Final Synthesis
**Goal:** Run the selected verification slices, reconcile code and test evidence, and produce the final remediation-oriented report.
**Success Criteria:** Every major claim is tied to code inspection, test inspection, or executed verification, and the final report follows the approved section contract.
**Tests:** Only the selected slices needed to confirm or weaken candidate findings.
**Status:** Not Started

### Task 1: Lock the Baseline and Final Output Contract

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/superpowers/specs/2026-04-07-authnz-module-review-design.md`
- Inspect: `Docs/superpowers/plans/2026-04-07-authnz-module-review-execution-plan.md`
- Inspect: `tldw_Server_API/app/core/AuthNZ`
- Inspect: `tldw_Server_API/tests/AuthNZ`
- Inspect: `tldw_Server_API/tests/AuthNZ_SQLite`
- Inspect: `tldw_Server_API/tests/AuthNZ_Postgres`
- Inspect: `tldw_Server_API/tests/AuthNZ_Unit`
- Inspect: `tldw_Server_API/tests/AuthNZ_Federation`
- Test: none

- [ ] **Step 1: Capture the dirty-worktree baseline**

Run:
```bash
git status --short
```

Expected: a list of uncommitted files, including whether scoped AuthNZ files already differ from committed history.

- [ ] **Step 2: Record the commit baseline used for the review**

Run:
```bash
git rev-parse --short HEAD
```

Expected: one short commit hash to cite when a finding depends on committed behavior rather than only local edits.

- [ ] **Step 3: Enumerate the scoped AuthNZ code surface**

Run:
```bash
find tldw_Server_API/app/core/AuthNZ -maxdepth 2 -type f | sort
```

Expected: the full scoped file inventory, including migrations, secret backends, federation files, and repos.

- [ ] **Step 4: Enumerate the AuthNZ-specific test surface**

Run:
```bash
find tldw_Server_API/tests/AuthNZ tldw_Server_API/tests/AuthNZ_SQLite tldw_Server_API/tests/AuthNZ_Postgres tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/AuthNZ_Federation -maxdepth 2 -type f | sort
```

Expected: a test inventory that makes backend-specific coverage visible before any verification choices are made.

- [ ] **Step 5: Fix the final response contract before reading deeply**

Use this exact final structure:
```markdown
## Security
- severity-ordered findings with classification, confidence, remediation size, and file references

## Correctness
- severity-ordered findings with the same evidence contract

## Maintainability
- only structural issues that materially increase defect risk

## Test Gaps
- missing or weak coverage ordered by impact

## Blind Spots / Not Reviewed
- meaningful excluded surfaces, skipped backend slices, or unresolved evidence limits
```

- [ ] **Step 6: Record the hotspot reading order**

Read in this order unless strong evidence forces reprioritization:
1. `jwt_service.py`
2. `session_manager.py`
3. `token_blacklist.py`
4. `password_service.py`
5. `mfa_service.py`
6. `api_key_manager.py`
7. `auth_principal_resolver.py`
8. `permissions.py`
9. `settings.py`
10. `migrations.py`

Expected: the first pass stays risk-first instead of drifting into low-signal files.

### Task 2: Execute the Identity, Session, and Login Security Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/AuthNZ/README.md`
- Inspect: `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/session_manager.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/token_blacklist.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/password_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/mfa_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/lockout_tracker.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/auth_governor.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service_rs256.py`
- Test: `tldw_Server_API/tests/AuthNZ/property/test_jwt_service_property.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_revocation_blacklist.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_token_metadata.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_refresh_cache.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_token_blacklist_basic.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_mfa_service.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_real_rate_limiter.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py`
- Test: `tldw_Server_API/tests/AuthNZ_Postgres/test_auth_enhanced_mfa.py`

- [ ] **Step 1: Read the operator-facing AuthNZ guide first**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/README.md
```

Expected: intended contracts for auth modes, session encryption, revocation, MFA, lockout, and service-token behavior.

- [ ] **Step 2: Locate the identity and session landmarks before reading full sections**

Run:
```bash
rg -n "class JWTService|create_.*token|decode_.*token|verify|refresh|jti|kid|aud|scope" tldw_Server_API/app/core/AuthNZ/jwt_service.py
rg -n "class SessionManager|create_session|refresh|rotate|revoke|cleanup|Fernet|encrypt|decrypt" tldw_Server_API/app/core/AuthNZ/session_manager.py
rg -n "blacklist|revoke|is_revoked|count|cleanup" tldw_Server_API/app/core/AuthNZ/token_blacklist.py
rg -n "hash_password|verify_password|argon|strength|totp|backup|lockout|attempt" tldw_Server_API/app/core/AuthNZ/password_service.py tldw_Server_API/app/core/AuthNZ/mfa_service.py tldw_Server_API/app/core/AuthNZ/lockout_tracker.py tldw_Server_API/app/core/AuthNZ/auth_governor.py
```

Expected: a stable map of the high-risk code paths to read in full.

- [ ] **Step 3: Read the code in state-machine order**

Trace and note:
- token issuance and decode rules
- refresh rotation and revocation behavior
- session storage, encryption, and cleanup assumptions
- blacklist checks and cache assumptions
- password hash and verify semantics
- MFA enrollment, verify, and recovery behavior
- lockout and login-governor interactions

Expected: a candidate finding list that distinguishes confirmed bugs from suspicious but unproven flows.

- [ ] **Step 4: Inspect the tests before running them**

Run:
```bash
rg -n "^def test_|^async def test_" \
  tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py \
  tldw_Server_API/tests/AuthNZ/unit/test_jwt_service_rs256.py \
  tldw_Server_API/tests/AuthNZ/property/test_jwt_service_property.py \
  tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py \
  tldw_Server_API/tests/AuthNZ/unit/test_session_revocation_blacklist.py \
  tldw_Server_API/tests/AuthNZ/unit/test_session_manager_token_metadata.py \
  tldw_Server_API/tests/AuthNZ/unit/test_session_refresh_cache.py \
  tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py \
  tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py \
  tldw_Server_API/tests/AuthNZ/integration/test_token_blacklist_basic.py \
  tldw_Server_API/tests/AuthNZ/integration/test_mfa_service.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_real_rate_limiter.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py \
  tldw_Server_API/tests/AuthNZ_Postgres/test_auth_enhanced_mfa.py
```

Expected: a concrete list of invariants already asserted and obvious coverage gaps.

- [ ] **Step 5: Search for suspicious patterns in the identity and session slice**

Run:
```bash
rg -n "except Exception|return None|pass$|TODO|FIXME|create_task|run_coroutine_threadsafe|warning\\(|error\\(" \
  tldw_Server_API/app/core/AuthNZ/jwt_service.py \
  tldw_Server_API/app/core/AuthNZ/session_manager.py \
  tldw_Server_API/app/core/AuthNZ/token_blacklist.py \
  tldw_Server_API/app/core/AuthNZ/mfa_service.py \
  tldw_Server_API/app/core/AuthNZ/auth_governor.py
```

Expected: a short list of branches for manual inspection, not automatic findings.

### Task 3: Execute the Keys, Secret Material, and Provider-Credential Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/api_key_crypto.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/crypto_utils.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/key_resolution.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/ip_allowlist.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/virtual_keys.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/byok_helpers.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/byok_runtime.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/byok_rotation.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/secret_backends/base.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/secret_backends/local_encrypted.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/user_provider_secrets_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/org_provider_secrets_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/byok_oauth_state_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/identity_provider_repo.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_api_key_manager_validation.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_api_key_crypto.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_limits_unit.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_enforcement_unit.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_byok_crypto.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_byok_helpers.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_byok_base_url_validation.py`
- Test: `tldw_Server_API/tests/AuthNZ_Federation/test_local_secret_backend.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_api_key_rotation_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_allowlists_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_virtual_keys_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_byok_rotation_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_Postgres/test_allowlists_pg.py`
- Test: `tldw_Server_API/tests/AuthNZ_Postgres/test_allowlists_budget_402_pg.py`
- Test: `tldw_Server_API/tests/AuthNZ_Postgres/test_virtual_keys_pg.py`

- [ ] **Step 1: Locate the key and secret-management landmarks**

Run:
```bash
rg -n "create_api_key|rotate|revoke|validate|scope|allowlist|prefix|hash" tldw_Server_API/app/core/AuthNZ/api_key_manager.py tldw_Server_API/app/core/AuthNZ/api_key_crypto.py
rg -n "virtual key|budget|endpoint|provider|scope|spend|limit" tldw_Server_API/app/core/AuthNZ/virtual_keys.py tldw_Server_API/app/core/AuthNZ/key_resolution.py
rg -n "BYOK|encrypt|decrypt|secret|backend|provider|oauth|state|rotation" \
  tldw_Server_API/app/core/AuthNZ/byok_helpers.py \
  tldw_Server_API/app/core/AuthNZ/byok_runtime.py \
  tldw_Server_API/app/core/AuthNZ/byok_rotation.py \
  tldw_Server_API/app/core/AuthNZ/secret_backends/base.py \
  tldw_Server_API/app/core/AuthNZ/secret_backends/local_encrypted.py
```

Expected: a reading map for issuance, validation, secret handling, and provider-credential flows.

- [ ] **Step 2: Read the code in trust-boundary order**

Trace and note:
- how raw keys are generated, transformed, and validated
- where scopes and endpoint restrictions are enforced
- how allowlists interact with API keys and virtual keys
- how BYOK secrets are stored, encrypted, rotated, and resolved
- which code paths assume backend-specific capabilities

Expected: a candidate finding list that separates direct key-handling defects from softer hardening advice.

- [ ] **Step 3: Inspect the tests before selecting verification**

Run:
```bash
rg -n "^def test_|^async def test_" \
  tldw_Server_API/tests/AuthNZ/unit/test_api_key_manager_validation.py \
  tldw_Server_API/tests/AuthNZ/unit/test_api_key_crypto.py \
  tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py \
  tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_limits_unit.py \
  tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_enforcement_unit.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_byok_crypto.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_byok_helpers.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_byok_base_url_validation.py \
  tldw_Server_API/tests/AuthNZ_Federation/test_local_secret_backend.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_api_key_rotation_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_allowlists_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_virtual_keys_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_byok_rotation_sqlite.py
```

Expected: a concrete list of already-covered invariants and backend-specific blind spots.

- [ ] **Step 4: Search for suspicious patterns in the key and secret slice**

Run:
```bash
rg -n "except Exception|return None|pass$|TODO|FIXME|base64|b64|encrypt|decrypt|plaintext|warning\\(|error\\(" \
  tldw_Server_API/app/core/AuthNZ/api_key_manager.py \
  tldw_Server_API/app/core/AuthNZ/api_key_crypto.py \
  tldw_Server_API/app/core/AuthNZ/crypto_utils.py \
  tldw_Server_API/app/core/AuthNZ/virtual_keys.py \
  tldw_Server_API/app/core/AuthNZ/byok_helpers.py \
  tldw_Server_API/app/core/AuthNZ/byok_runtime.py \
  tldw_Server_API/app/core/AuthNZ/byok_rotation.py \
  tldw_Server_API/app/core/AuthNZ/secret_backends/local_encrypted.py
```

Expected: a short list of branches to inspect manually.

### Task 4: Execute the Authorization, Governance, and Budget Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/principal_model.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/permissions.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/rbac.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/org_rbac.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/orgs_teams.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/privilege_catalog.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/quotas.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/rate_limiter.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/llm_budget_guard.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/llm_budget_middleware.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/rbac_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/quotas_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_principal_resolver.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_jwt_membership_validation.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_org_policy_from_principal.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_org_deps.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_state_consistency.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_jwt_happy_path.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_api_key_happy_path.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_single_user_claims_permissions.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_rbac_effective_permissions.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_resource_governor_invariants.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_llm_budget_invariants.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_claims_rbac_overlaps_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_org_rbac_scoped_permissions_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_quota_enforcement_http_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_Postgres/test_role_effective_permissions_pg.py`
- Test: `tldw_Server_API/tests/AuthNZ_Postgres/test_orgs_teams_pg.py`

- [ ] **Step 1: Locate the authorization and governance landmarks**

Run:
```bash
rg -n "get_auth_principal|principal|membership|single_user|multi_user|service token|api key|jwt" tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py tldw_Server_API/app/core/AuthNZ/principal_model.py
rg -n "require_|permission|role|rbac|org|team|inherit|scope|deny|allow" tldw_Server_API/app/core/AuthNZ/permissions.py tldw_Server_API/app/core/AuthNZ/rbac.py tldw_Server_API/app/core/AuthNZ/org_rbac.py tldw_Server_API/app/core/AuthNZ/orgs_teams.py tldw_Server_API/app/core/AuthNZ/privilege_catalog.py
rg -n "quota|budget|limit|burst|token bucket|refund|daily|monthly|spend" tldw_Server_API/app/core/AuthNZ/quotas.py tldw_Server_API/app/core/AuthNZ/rate_limiter.py tldw_Server_API/app/core/AuthNZ/llm_budget_guard.py tldw_Server_API/app/core/AuthNZ/llm_budget_middleware.py
```

Expected: a reading map for principal derivation, permission resolution, and enforcement behavior.

- [ ] **Step 2: Read the code in authorization-flow order**

Trace and note:
- principal construction and membership validation
- single-user versus multi-user branching
- permission and role resolution
- org or team scope propagation and overlap behavior
- rate limiting, quotas, and budget enforcement order
- any fail-open paths when enforcement state is unavailable

Expected: candidate findings with enough evidence to separate privilege bugs from performance or maintainability concerns.

- [ ] **Step 3: Inspect the tests before selecting verification**

Run:
```bash
rg -n "^def test_|^async def test_" \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_principal_resolver.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_jwt_membership_validation.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_org_policy_from_principal.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_org_deps.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_state_consistency.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_jwt_happy_path.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_api_key_happy_path.py \
  tldw_Server_API/tests/AuthNZ/integration/test_single_user_claims_permissions.py \
  tldw_Server_API/tests/AuthNZ/integration/test_rbac_effective_permissions.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_resource_governor_invariants.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_llm_budget_invariants.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_claims_rbac_overlaps_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_org_rbac_scoped_permissions_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_quota_enforcement_http_sqlite.py
```

Expected: a concrete map of what the suite asserts about privilege derivation, scoping, and enforcement.

- [ ] **Step 4: Search for suspicious patterns in the authorization slice**

Run:
```bash
rg -n "except Exception|return None|pass$|TODO|FIXME|if not .*return|warning\\(|error\\(" \
  tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py \
  tldw_Server_API/app/core/AuthNZ/permissions.py \
  tldw_Server_API/app/core/AuthNZ/rbac.py \
  tldw_Server_API/app/core/AuthNZ/org_rbac.py \
  tldw_Server_API/app/core/AuthNZ/quotas.py \
  tldw_Server_API/app/core/AuthNZ/rate_limiter.py \
  tldw_Server_API/app/core/AuthNZ/llm_budget_guard.py
```

Expected: a shortlist of branches that deserve manual scrutiny before any verification is run.

### Task 5: Execute the Persistence, Migrations, and Backend-Divergence Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/AuthNZ/database.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/db_config.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/initialize.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/migrate_to_multiuser.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/token_blacklist_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/users_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/mfa_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/rbac_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/quotas_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_database_pool_fetchone_sqlite_fallback.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_usage_truthiness.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_migrate_to_multiuser_review_fixes.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_api_keys_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_sessions_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_token_blacklist_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_quotas_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_orgs_teams_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_api_keys_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_sessions_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_token_blacklist_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_quotas_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_rate_limits_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_orgs_teams_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_api_keys_repo_postgres.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_sessions_repo_postgres.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_token_blacklist_repo_postgres.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_quotas_repo_postgres.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_rate_limits_repo_postgres.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_orgs_teams_repo_postgres.py`

- [ ] **Step 1: Locate the persistence and migration landmarks**

Run:
```bash
rg -n "sqlite|postgres|pool|connect|transaction|pragma|fallback|retry" tldw_Server_API/app/core/AuthNZ/database.py tldw_Server_API/app/core/AuthNZ/db_config.py
rg -n "CREATE TABLE|ALTER TABLE|migration|schema|version|seed|api_keys|usage|sessions|org" tldw_Server_API/app/core/AuthNZ/migrations.py tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py
rg -n "initialize|bootstrap|single_user|multi_user|migrate|repo|routing" tldw_Server_API/app/core/AuthNZ/initialize.py tldw_Server_API/app/core/AuthNZ/migrate_to_multiuser.py
```

Expected: a reading map for backend routing, schema evolution, and bootstrap assumptions.

- [ ] **Step 2: Read the code in persistence-risk order**

Trace and note:
- connection and pool behavior
- schema and migration ordering assumptions
- initialization and bootstrap side effects
- repo backend-selection logic
- SQLite or Postgres divergence that could change semantics

Expected: candidate findings that distinguish schema bugs, backend drift, and repo routing fragility.

- [ ] **Step 3: Inspect the tests before selecting verification**

Run:
```bash
rg -n "^def test_|^async def test_" \
  tldw_Server_API/tests/AuthNZ/unit/test_database_pool_fetchone_sqlite_fallback.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_usage_truthiness.py \
  tldw_Server_API/tests/AuthNZ/unit/test_migrate_to_multiuser_review_fixes.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_api_keys_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_sessions_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_token_blacklist_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_quotas_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_orgs_teams_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_api_keys_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_sessions_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_token_blacklist_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_quotas_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_rate_limits_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_orgs_teams_repo_sqlite.py
```

Expected: a map of which backend-routing and migration invariants are already explicit and which remain implicit.

- [ ] **Step 4: Search for suspicious patterns in the persistence slice**

Run:
```bash
rg -n "except Exception|return None|pass$|TODO|FIXME|ALTER TABLE|DROP |IF NOT EXISTS|warning\\(|error\\(" \
  tldw_Server_API/app/core/AuthNZ/database.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/AuthNZ/initialize.py \
  tldw_Server_API/app/core/AuthNZ/migrate_to_multiuser.py \
  tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py \
  tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py \
  tldw_Server_API/app/core/AuthNZ/repos/token_blacklist_repo.py \
  tldw_Server_API/app/core/AuthNZ/repos/quotas_repo.py \
  tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py
```

Expected: a shortlist of branches for manual scrutiny, not automatic findings.

### Task 6: Run Targeted Verification and Produce the Final Review

**Files:**
- Create: none
- Modify: none
- Inspect: `/tmp/authnz_identity_pytest.log`
- Inspect: `/tmp/authnz_keys_pytest.log`
- Inspect: `/tmp/authnz_authz_pytest.log`
- Inspect: `/tmp/authnz_backend_pytest.log`
- Test: selected slices from Tasks 2-5 only

- [ ] **Step 1: Run the portable identity and session verification slice**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py \
  tldw_Server_API/tests/AuthNZ/unit/test_jwt_service_rs256.py \
  tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py \
  tldw_Server_API/tests/AuthNZ/unit/test_session_revocation_blacklist.py \
  tldw_Server_API/tests/AuthNZ/unit/test_session_manager_token_metadata.py \
  tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py \
  tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py \
  tldw_Server_API/tests/AuthNZ/integration/test_token_blacklist_basic.py \
  tldw_Server_API/tests/AuthNZ/integration/test_mfa_service.py -v | tee /tmp/authnz_identity_pytest.log
```

Expected: the portable slice passes; failures are triaged as either real candidate findings or environment-sensitive noise.

- [ ] **Step 2: Run the portable keys and secrets verification slice**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_api_key_manager_validation.py \
  tldw_Server_API/tests/AuthNZ/unit/test_api_key_crypto.py \
  tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py \
  tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_limits_unit.py \
  tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_enforcement_unit.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_byok_crypto.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_byok_helpers.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py \
  tldw_Server_API/tests/AuthNZ_Federation/test_local_secret_backend.py -v | tee /tmp/authnz_keys_pytest.log
```

Expected: the portable key-management slice passes or produces failures that materially strengthen a concrete finding.

- [ ] **Step 3: Run the portable authorization and budget verification slice**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_principal_resolver.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_jwt_membership_validation.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_org_policy_from_principal.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_state_consistency.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_jwt_happy_path.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_api_key_happy_path.py \
  tldw_Server_API/tests/AuthNZ/integration/test_single_user_claims_permissions.py \
  tldw_Server_API/tests/AuthNZ/integration/test_rbac_effective_permissions.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_claims_rbac_overlaps_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_org_rbac_scoped_permissions_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_quota_enforcement_http_sqlite.py -v | tee /tmp/authnz_authz_pytest.log
```

Expected: the portable governance slice passes or narrows a disputed claim.

- [ ] **Step 4: Run backend-specific verification only if a current claim depends on it**

Run one or more of these only when needed:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ_Postgres/test_auth_enhanced_mfa.py -v | tee /tmp/authnz_backend_pytest.log
python -m pytest tldw_Server_API/tests/AuthNZ_Postgres/test_role_effective_permissions_pg.py -v | tee -a /tmp/authnz_backend_pytest.log
python -m pytest tldw_Server_API/tests/AuthNZ/integration/test_authnz_sessions_repo_postgres.py -v | tee -a /tmp/authnz_backend_pytest.log
python -m pytest tldw_Server_API/tests/AuthNZ/integration/test_authnz_token_blacklist_repo_postgres.py -v | tee -a /tmp/authnz_backend_pytest.log
```

Expected: tests pass or skip cleanly via the project fixtures; if the environment cannot support them, record that as a blind spot rather than overclaiming.

- [ ] **Step 5: Reconcile code, test, and runtime evidence before writing findings**

For each candidate issue, confirm:
- what exact code path supports it
- whether any test already contradicts it
- whether executed verification raised or lowered confidence
- whether the right label is `Confirmed finding`, `Probable risk`, or `Improvement`

Expected: no speculative bug survives as a confirmed finding.

- [ ] **Step 6: Write the final in-session report**

Use this exact structure and ordering:
```markdown
## Security
- confirmed findings first, then probable risks, then improvements

## Correctness
- confirmed findings first, then probable risks, then improvements

## Maintainability
- only issues that materially increase future defect risk

## Test Gaps
- highest-leverage missing or weak coverage first

## Blind Spots / Not Reviewed
- backend-specific skips, excluded endpoint wiring, and any unresolved evidence limits
```

Each reported item must include:
- short title
- classification
- severity
- confidence
- remediation size
- concise reasoning
- actionable file references

Expected: one evidence-backed, remediation-oriented review that matches the approved spec without drifting into fixes.
