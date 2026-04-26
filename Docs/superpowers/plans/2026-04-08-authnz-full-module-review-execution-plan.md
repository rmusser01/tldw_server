# AuthNZ Full Module Review Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved full AuthNZ review and deliver one findings-first, evidence-backed report plus staged review artifacts covering core AuthNZ, immediate API integration points, representative claim-first consumers, tests, and documentation contracts.

**Architecture:** This is a read-first, boundary-aware audit plan. Execution starts by locking the current workspace baseline, creating staged review artifacts under `Docs/superpowers/reviews/authnz-full-module/`, and fixing the final report contract before deep reading begins. It then inspects runtime authentication and authorization behavior, persistence and configuration safety, and test or documentation alignment, using only the smallest targeted verification needed to confirm or weaken specific claims. Narrow proof-of-fix patches are not pre-authored in this plan; if a critical localized issue is confirmed, preserve the pre-fix evidence and write a dedicated remediation plan for that exact defect instead of improvising code changes from the review plan.

**Tech Stack:** Python 3, FastAPI, pytest, git, ripgrep, sed, Markdown

---

## Scope Lock

Keep these decisions fixed during execution:

- review the current working tree by default, not only `HEAD`
- label any finding that depends on uncommitted local changes
- keep code scope centered on `tldw_Server_API/app/core/AuthNZ`, `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`, and `tldw_Server_API/app/api/v1/endpoints/auth.py`
- inspect representative admin or control-surface endpoints only when they directly use claim-first AuthNZ dependencies, are high-fan-out consumers of AuthNZ behavior, or emerge as regression hotspots from current evidence
- use `tldw_Server_API/tests/AuthNZ`, `tldw_Server_API/tests/AuthNZ_SQLite`, `tldw_Server_API/tests/AuthNZ_Postgres`, `tldw_Server_API/tests/AuthNZ_Unit`, and `tldw_Server_API/tests/AuthNZ_Federation` as the primary test evidence set
- start documentation review from the bounded seed set in the approved spec and expand only when a seed document points to another behavior-defining contract
- separate `Confirmed finding`, `Probable risk`, `Improvement`, and `Open question`
- keep pre-fix evidence in the stage artifacts before any remediation branch is considered
- do not modify repository source files during this review execution plan
- do not run broad blanket suites; use the smallest targeted verification needed to answer a concrete claim
- keep blind spots explicit instead of implying unreviewed enterprise, federation, or secret-backend paths are safe
- if a dedicated review worktree is not already available, execute this plan in the current workspace and rely on the Stage 1 baseline to distinguish pre-existing local changes from reviewed behavior

## Review File Map

**Create during execution:**
- `Docs/superpowers/reviews/authnz-full-module/README.md`
- `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage1-baseline-and-boundary-inventory.md`
- `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage2-runtime-authentication-and-authorization-analysis.md`
- `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage3-persistence-migrations-and-configuration-safety.md`
- `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage4-tests-docs-drift-and-verification-gaps.md`
- `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage5-final-synthesis-roadmap-and-patch-decisions.md`

**Spec and plan inputs:**
- `Docs/superpowers/specs/2026-04-08-authnz-full-module-review-design.md`
- `Docs/superpowers/plans/2026-04-08-authnz-full-module-review-execution-plan.md`

**Primary documentation and contract references:**
- `tldw_Server_API/app/core/AuthNZ/README.md`
- `Docs/Code_Documentation/Guides/AuthNZ_Code_Guide.md`
- `Docs/API-related/User_Registration_API_Documentation.md`
- `Docs/Operations/Env_Vars.md`
- `Docs/Getting_Started/QUICKSTART.md`
- `Docs/Getting_Started/TROUBLESHOOTING.md`

**Primary source files to inspect first:**
- `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`
- `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- `tldw_Server_API/app/core/AuthNZ/session_manager.py`
- `tldw_Server_API/app/core/AuthNZ/token_blacklist.py`
- `tldw_Server_API/app/core/AuthNZ/password_service.py`
- `tldw_Server_API/app/core/AuthNZ/mfa_service.py`
- `tldw_Server_API/app/core/AuthNZ/lockout_tracker.py`
- `tldw_Server_API/app/core/AuthNZ/auth_governor.py`
- `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
- `tldw_Server_API/app/core/AuthNZ/virtual_keys.py`
- `tldw_Server_API/app/core/AuthNZ/permissions.py`
- `tldw_Server_API/app/core/AuthNZ/rbac.py`
- `tldw_Server_API/app/core/AuthNZ/org_rbac.py`
- `tldw_Server_API/app/core/AuthNZ/orgs_teams.py`
- `tldw_Server_API/app/core/AuthNZ/settings.py`
- `tldw_Server_API/app/core/AuthNZ/database.py`
- `tldw_Server_API/app/core/AuthNZ/migrations.py`
- `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- `tldw_Server_API/app/core/AuthNZ/initialize.py`
- `tldw_Server_API/app/core/AuthNZ/migrate_to_multiuser.py`
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- `tldw_Server_API/app/api/v1/endpoints/auth.py`

**Representative integration consumers to inspect only if selected by Stage 1 criteria:**
- `tldw_Server_API/app/api/v1/endpoints/users.py`
- `tldw_Server_API/app/api/v1/endpoints/privileges.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_sessions_mfa.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_byok.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_orgs.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_registration.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_system.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_settings.py`

**Representative repository files to inspect during persistence review:**
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

**High-value existing tests to reuse during the review:**
- `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service_rs256.py`
- `tldw_Server_API/tests/AuthNZ/property/test_jwt_service_property.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_session_revocation_blacklist.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_token_metadata.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_session_refresh_cache.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_api_keys.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_auth_deps_precedence.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_jwt_happy_path.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_api_key_happy_path.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_state_consistency.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_endpoints_integration_fixed.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_real_rate_limiter.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_single_user_claims_permissions.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_rbac_effective_permissions.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_mfa_service.py`
- `tldw_Server_API/tests/AuthNZ_Postgres/test_auth_enhanced_mfa.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_auth_claim_deps.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_auth_claim_route_level.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_hardening.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_auth_principal_resolver.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_claim_first_single_user_mode_guardrail.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_admin_roles_single_user_claims.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_lockout_scope.py`
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
- `tldw_Server_API/tests/AuthNZ_SQLite/test_virtual_keys_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_byok_rotation_sqlite.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_api_keys_repo_postgres.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_sessions_repo_postgres.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_token_blacklist_repo_postgres.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_quotas_repo_postgres.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_rate_limits_repo_postgres.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_orgs_teams_repo_postgres.py`

**Scratch artifacts allowed during execution:**
- `/tmp/authnz_full_review_inventory.txt`
- `/tmp/authnz_full_runtime_pytest.log`
- `/tmp/authnz_full_persistence_pytest.log`
- `/tmp/authnz_full_contract_pytest.log`
- `/tmp/authnz_full_verification_notes.md`

## Stage Overview

## Stage 1: Baseline and Boundary Inventory
**Goal:** Lock the current workspace baseline, create stable review artifacts, inventory the exact scoped files, and fix the final findings contract before deep reading begins.
**Success Criteria:** Review artifacts exist under `Docs/superpowers/reviews/authnz-full-module/`, the baseline and inventory are recorded, the representative endpoint selection rule is operationalized, and the final response contract is frozen.
**Tests:** No pytest execution in this stage.
**Status:** Not Started

## Stage 2: Runtime Authentication and Authorization Analysis
**Goal:** Inspect credential resolution, JWT and session behavior, endpoint dependency chains, claim-first authorization, and stateful login security controls first.
**Success Criteria:** Runtime and authz candidate findings are tied to exact files, tests, and verification commands, with confirmed findings separated cleanly from probable risks.
**Tests:** Run only the smallest runtime and authz test slices listed below that materially confirm or weaken candidate findings.
**Status:** Not Started

## Stage 3: Persistence, Migrations, and Configuration Safety
**Goal:** Review settings precedence, database routing, migration safety, initialization paths, and backend-specific repository behavior.
**Success Criteria:** SQLite or PostgreSQL divergence, schema or migration hazards, and fail-open or fail-closed configuration issues are documented with direct code evidence and representative test coverage.
**Tests:** Run only the smallest persistence, repo, migration, or backend-selection test slices needed to settle specific claims.
**Status:** Not Started

## Stage 4: Tests, Docs Drift, and Verification Gaps
**Goal:** Compare the reviewed runtime behavior against documentation claims and the actual AuthNZ test surface, then identify missing, weak, or misleading coverage.
**Success Criteria:** Documentation drift and test gaps are ranked by operational or regression risk, and unverified enterprise or federation paths are explicitly downgraded when direct validation is not available.
**Tests:** Run only the narrow contract tests needed to reconcile code, docs, and existing test claims.
**Status:** Not Started

## Stage 5: Final Synthesis, Roadmap, and Patch Decisions
**Goal:** Deduplicate findings across stages, produce the final findings-first report and roadmap, and decide whether any issue warrants a separate remediation branch.
**Success Criteria:** The final review output follows the approved structure, every major claim is backed by code inspection, test inspection, verification, or an explicit confidence downgrade, and any patch candidate is either rejected or spun into a separate remediation plan with pre-fix evidence preserved.
**Tests:** No new blanket suites. Reuse earlier verification output and run only one last narrow slice if a single claim remains unresolved.
**Status:** Not Started

### Task 1: Prepare Review Artifacts and Capture the Baseline

**Files:**
- Create: `Docs/superpowers/reviews/authnz-full-module/README.md`
- Create: `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage1-baseline-and-boundary-inventory.md`
- Create: `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage2-runtime-authentication-and-authorization-analysis.md`
- Create: `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage3-persistence-migrations-and-configuration-safety.md`
- Create: `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage4-tests-docs-drift-and-verification-gaps.md`
- Create: `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage5-final-synthesis-roadmap-and-patch-decisions.md`
- Inspect: `Docs/superpowers/specs/2026-04-08-authnz-full-module-review-design.md`
- Inspect: `Docs/superpowers/plans/2026-04-08-authnz-full-module-review-execution-plan.md`
- Test: none

- [ ] **Step 1: Create the review output directory**

Run:
```bash
mkdir -p Docs/superpowers/reviews/authnz-full-module
```

Expected: the `Docs/superpowers/reviews/authnz-full-module` directory exists and no application source files change.

- [ ] **Step 2: Create one markdown file per stage with a fixed evidence template**

Use these exact title lines and section headings:
```markdown
# Stage 1: Baseline and Boundary Inventory
# Stage 2: Runtime Authentication and Authorization Analysis
# Stage 3: Persistence, Migrations, and Configuration Safety
# Stage 4: Tests, Docs Drift, and Verification Gaps
# Stage 5: Final Synthesis, Roadmap, and Patch Decisions

## Scope
## Files Reviewed
## Tests Reviewed
## Docs Reviewed
## Validation Commands
## Confirmed Findings
## Probable Risks
## Improvements
## Open Questions
## Exit Note
```

- [ ] **Step 3: Write `Docs/superpowers/reviews/authnz-full-module/README.md`**

Document:
- the stage order `1 -> 2 -> 3 -> 4 -> 5`
- the path to each stage report
- the rule that confirmed findings come before probable risks and improvements
- the rule that issues fixed later in a separate remediation branch must still remain visible in the review record
- the rule that enterprise, federation, and secret-backend paths need explicit confidence downgrades when direct verification is unavailable
- the canonical final response structure from Step 7

- [ ] **Step 4: Capture the workspace baseline**

Run:
```bash
git status --short
git rev-parse --short HEAD
git log --oneline -n 20 -- tldw_Server_API/app/core/AuthNZ tldw_Server_API/app/api/v1/API_Deps/auth_deps.py tldw_Server_API/app/api/v1/endpoints/auth.py
```

Expected: a clear baseline showing the current dirty-worktree state, the short `HEAD` hash, and the recent churn window for the scoped AuthNZ surface.

- [ ] **Step 5: Capture the exact scoped inventory**

Run:
```bash
{
  rg --files tldw_Server_API/app/core/AuthNZ
  printf '%s\n' \
    tldw_Server_API/app/api/v1/API_Deps/auth_deps.py \
    tldw_Server_API/app/api/v1/endpoints/auth.py \
    tldw_Server_API/app/core/AuthNZ/README.md \
    Docs/Code_Documentation/Guides/AuthNZ_Code_Guide.md \
    Docs/API-related/User_Registration_API_Documentation.md \
    Docs/Operations/Env_Vars.md \
    Docs/Getting_Started/QUICKSTART.md \
    Docs/Getting_Started/TROUBLESHOOTING.md
  rg --files tldw_Server_API/tests/AuthNZ tldw_Server_API/tests/AuthNZ_SQLite tldw_Server_API/tests/AuthNZ_Postgres tldw_Server_API/tests/AuthNZ_Unit tldw_Server_API/tests/AuthNZ_Federation
} | sort | tee /tmp/authnz_full_review_inventory.txt
```

Expected: one stable inventory file containing the scoped implementation, seed docs, and AuthNZ-focused test surface.

- [ ] **Step 6: Select the representative integration consumers before deep reading**

Run:
```bash
rg -n "get_auth_principal|require_permissions\\(|require_roles\\(" \
  tldw_Server_API/app/api/v1/endpoints/users.py \
  tldw_Server_API/app/api/v1/endpoints/privileges.py \
  tldw_Server_API/app/api/v1/endpoints/admin \
  | head -n 200
```

Expected: a bounded candidate list of claim-first consumers to sample later, without expanding into a general endpoint audit.

- [ ] **Step 7: Freeze the final user-facing review output contract**

Use this exact final response structure:
```markdown
## Findings
### Confirmed findings
- severity, confidence, file references, impact, and fix direction when clear

### Probable risks
- material issues not fully proven, with explicit confidence limits

### Improvements
- lower-priority hardening or maintainability suggestions

## Open Questions
- only unresolved ambiguities that materially affect confidence

## Verification
- files and docs inspected, tests run, and what remains unverified

## Remediation Roadmap
- immediate fixes
- near-term hardening
- structural refactors worth scheduling
```

- [ ] **Step 8: Write the Stage 1 baseline note**

`Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage1-baseline-and-boundary-inventory.md` must record:
- the dirty-worktree baseline from Step 4
- the short `HEAD` hash from Step 4
- the inventory from Step 5
- the candidate integration-consumer list from Step 6
- the fixed final response structure from Step 7
- the explicit statement that this review plan does not authorize source patches

- [ ] **Step 9: Commit the review scaffold**

Run:
```bash
git add Docs/superpowers/reviews/authnz-full-module Docs/superpowers/plans/2026-04-08-authnz-full-module-review-execution-plan.md
git commit -m "docs: scaffold AuthNZ full-module review artifacts"
```

Expected: one docs-only commit captures the review workspace before stage findings are added.

### Task 2: Execute Stage 2 Runtime Authentication and Authorization Analysis

**Files:**
- Modify: `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage2-runtime-authentication-and-authorization-analysis.md`
- Inspect: `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/session_manager.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/token_blacklist.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/password_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/mfa_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/lockout_tracker.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/auth_governor.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/virtual_keys.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/permissions.py`
- Inspect: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/auth.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service_rs256.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_revocation_blacklist.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_api_keys.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_auth_deps_precedence.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_claim_deps.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_claim_route_level.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_hardening.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_principal_resolver.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_jwt_happy_path.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_api_key_happy_path.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_state_consistency.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_endpoints_integration_fixed.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_real_rate_limiter.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_mfa_service.py`
- Test: `tldw_Server_API/tests/AuthNZ_Postgres/test_auth_enhanced_mfa.py`

- [ ] **Step 1: Read the runtime auth source files in one bounded pass**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/api/v1/API_Deps/auth_deps.py
sed -n '1,260p' tldw_Server_API/app/api/v1/endpoints/auth.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/permissions.py
```

Expected: the credential-resolution order, endpoint dependency chain, and claim-first enforcement entry points are visible before deeper subsystem reading begins.

- [ ] **Step 2: Read the stateful identity and token files**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/jwt_service.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/session_manager.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/token_blacklist.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/password_service.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/mfa_service.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/lockout_tracker.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/auth_governor.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/api_key_manager.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/virtual_keys.py
```

Expected: JWT handling, session lifecycle, lockout state, API-key validation, and MFA flow assumptions are visible enough to record candidate findings.

- [ ] **Step 3: Read the highest-value runtime and authz tests before running anything**

Run:
```bash
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/unit/test_session_revocation_blacklist.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_hardening.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_state_consistency.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py
```

Expected: the strongest existing runtime contracts are visible before verification commands are chosen.

- [ ] **Step 4: Write the Stage 2 artifact before executing tests**

`Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage2-runtime-authentication-and-authorization-analysis.md` must record:
- the exact files reviewed
- the tests reviewed from Steps 1 through 3
- candidate findings grouped as confirmed only when code evidence is already conclusive
- any likely runtime risks that still need verification
- the exact verification commands selected in Steps 5 and 6

- [ ] **Step 5: Run the smallest unit-level runtime verification slice**

Run:
```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py \
  tldw_Server_API/tests/AuthNZ/unit/test_jwt_service_rs256.py \
  tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py \
  tldw_Server_API/tests/AuthNZ/unit/test_session_revocation_blacklist.py \
  tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py \
  tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py \
  tldw_Server_API/tests/AuthNZ/unit/test_auth_deps_precedence.py \
  tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_claim_deps.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_claim_route_level.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_hardening.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_principal_resolver.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py \
  | tee /tmp/authnz_full_runtime_pytest.log
```

Expected: primarily `PASSED`; any `FAILED` or `SKIPPED` result must be treated as evidence to investigate, not as a pass by implication.

- [ ] **Step 6: Run the smallest integration-level runtime verification slice**

Run:
```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_jwt_happy_path.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_api_key_happy_path.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_state_consistency.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_endpoints_integration_fixed.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_real_rate_limiter.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_via_auth_governor.py \
  tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py \
  tldw_Server_API/tests/AuthNZ/integration/test_mfa_service.py \
  tldw_Server_API/tests/AuthNZ_Postgres/test_auth_enhanced_mfa.py \
  | tee -a /tmp/authnz_full_runtime_pytest.log
```

Expected: passing or explicitly skipped backend-sensitive results; any skip reason must be copied into the stage artifact as a confidence limit.

- [ ] **Step 7: Update and commit the Stage 2 artifact**

`Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage2-runtime-authentication-and-authorization-analysis.md` must be updated with:
- the executed commands from Steps 5 and 6
- the observed pass, fail, or skip outcomes
- confidence upgrades or downgrades based on those results
- the exact line-level file references supporting each finding

Run:
```bash
git add Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage2-runtime-authentication-and-authorization-analysis.md
git commit -m "docs: record AuthNZ runtime and authz review stage"
```

Expected: one docs-only commit preserves the runtime-auth stage findings before later stages reinterpret them.

### Task 3: Execute Stage 3 Persistence, Migrations, and Configuration Safety

**Files:**
- Modify: `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage3-persistence-migrations-and-configuration-safety.md`
- Inspect: `tldw_Server_API/app/core/AuthNZ/settings.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/database.py`
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
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/user_provider_secrets_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/org_provider_secrets_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/byok_oauth_state_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/identity_provider_repo.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_lockout_scope.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_usage_truthiness.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_migrate_to_multiuser_review_fixes.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_api_keys_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_sessions_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_token_blacklist_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_quotas_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_orgs_teams_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_initialize_single_user_invariant_repo_routing.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_api_keys_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_sessions_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_token_blacklist_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_quotas_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_rate_limits_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_orgs_teams_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_virtual_keys_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_byok_rotation_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_api_keys_repo_postgres.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_sessions_repo_postgres.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_token_blacklist_repo_postgres.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_quotas_repo_postgres.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_rate_limits_repo_postgres.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_orgs_teams_repo_postgres.py`

- [ ] **Step 1: Read the configuration and bootstrap files**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/settings.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/database.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/initialize.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/migrate_to_multiuser.py
```

Expected: settings precedence, mode handling, DB initialization, and migration-entry assumptions are visible before repo or schema details are judged.

- [ ] **Step 2: Read the migration and representative repository files**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/migrations.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/repos/token_blacklist_repo.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/repos/quotas_repo.py
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py
```

Expected: representative persistence routing, schema evolution, and backend-selection patterns are visible enough to classify candidate risks.

- [ ] **Step 3: Read the highest-value persistence and backend-selection tests before running them**

Run:
```bash
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/unit/test_migrate_to_multiuser_review_fixes.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/unit/test_authnz_api_keys_repo_backend_selection.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_api_keys_repo_sqlite.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/integration/test_authnz_api_keys_repo_postgres.py
```

Expected: the strongest schema and repo-routing expectations are visible before verification commands are chosen.

- [ ] **Step 4: Write the Stage 3 artifact before executing tests**

`Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage3-persistence-migrations-and-configuration-safety.md` must record:
- the exact files and docs reviewed
- any suspected fail-open or fail-closed branches
- any backend divergence questions that require verification
- the exact verification commands selected in Steps 5 and 6

- [ ] **Step 5: Run the smallest unit-level persistence verification slice**

Run:
```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_lockout_scope.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_usage_truthiness.py \
  tldw_Server_API/tests/AuthNZ/unit/test_migrate_to_multiuser_review_fixes.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_api_keys_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_sessions_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_token_blacklist_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_quotas_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_orgs_teams_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_initialize_single_user_invariant_repo_routing.py \
  | tee /tmp/authnz_full_persistence_pytest.log
```

Expected: primarily `PASSED`; any backend-selection or migration failure becomes direct evidence to investigate, not a reason to skip the area.

- [ ] **Step 6: Run the smallest backend-sensitive persistence verification slice**

Run:
```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_api_keys_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_sessions_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_token_blacklist_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_quotas_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_rate_limits_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_orgs_teams_repo_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_virtual_keys_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_byok_rotation_sqlite.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_api_keys_repo_postgres.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_sessions_repo_postgres.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_token_blacklist_repo_postgres.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_quotas_repo_postgres.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_rate_limits_repo_postgres.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_orgs_teams_repo_postgres.py \
  | tee -a /tmp/authnz_full_persistence_pytest.log
```

Expected: passing or explicitly skipped backend-sensitive results; any skip or environment limitation must be copied into the stage artifact as a confidence downgrade.

- [ ] **Step 7: Update and commit the Stage 3 artifact**

`Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage3-persistence-migrations-and-configuration-safety.md` must be updated with:
- the executed commands from Steps 5 and 6
- the observed pass, fail, or skip outcomes
- any backend-specific confidence limits
- exact line-level file references supporting each persistence finding

Run:
```bash
git add Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage3-persistence-migrations-and-configuration-safety.md
git commit -m "docs: record AuthNZ persistence and configuration review stage"
```

Expected: one docs-only commit preserves the persistence stage before test-gap synthesis begins.

### Task 4: Execute Stage 4 Tests, Docs Drift, and Verification Gaps

**Files:**
- Modify: `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage4-tests-docs-drift-and-verification-gaps.md`
- Inspect: `tldw_Server_API/app/core/AuthNZ/README.md`
- Inspect: `Docs/Code_Documentation/Guides/AuthNZ_Code_Guide.md`
- Inspect: `Docs/API-related/User_Registration_API_Documentation.md`
- Inspect: `Docs/Operations/Env_Vars.md`
- Inspect: `Docs/Getting_Started/QUICKSTART.md`
- Inspect: `Docs/Getting_Started/TROUBLESHOOTING.md`
- Inspect: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/auth.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py`
- Test: `tldw_Server_API/tests/AuthNZ/property/test_auth_endpoints_property.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_endpoints_integration.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_endpoints_integration_fixed.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_hardening.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_legacy_admin_shim_removed.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_optional_current_user_shim_removed.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_db_adapter_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_principal_resolver_mode_shim_removed.py`
- Test: `tldw_Server_API/tests/Resource_Governance/test_auth_route_map_coverage.py`

- [ ] **Step 1: Read the documentation seed set in one bounded pass**

Run:
```bash
sed -n '1,260p' tldw_Server_API/app/core/AuthNZ/README.md
sed -n '1,260p' Docs/Code_Documentation/Guides/AuthNZ_Code_Guide.md
sed -n '1,260p' Docs/API-related/User_Registration_API_Documentation.md
sed -n '1,260p' Docs/Operations/Env_Vars.md
sed -n '1,260p' Docs/Getting_Started/QUICKSTART.md
sed -n '1,260p' Docs/Getting_Started/TROUBLESHOOTING.md
```

Expected: the currently documented auth contracts are visible before code-vs-doc drift is classified.

- [ ] **Step 2: Map the doc claims back to code and test anchors**

Run:
```bash
rg -n "AUTH_MODE|single-user|single_user|multi-user|multi_user|JWT|X-API-KEY|virtual key|virtual keys|claim-first|get_auth_principal|require_permissions|require_roles" \
  tldw_Server_API/app/core/AuthNZ/README.md \
  Docs/Code_Documentation/Guides/AuthNZ_Code_Guide.md \
  Docs/API-related/User_Registration_API_Documentation.md \
  Docs/Operations/Env_Vars.md \
  Docs/Getting_Started/QUICKSTART.md \
  Docs/Getting_Started/TROUBLESHOOTING.md \
  tldw_Server_API/app/api/v1/API_Deps/auth_deps.py \
  tldw_Server_API/app/api/v1/endpoints/auth.py
```

Expected: a concrete claim map that makes documentation drift and contract mismatches actionable instead of impressionistic.

- [ ] **Step 3: Read the strongest contract and hardening tests before running them**

Run:
```bash
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/property/test_auth_endpoints_property.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/integration/test_auth_endpoints_integration.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ/integration/test_auth_endpoints_integration_fixed.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_hardening.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_legacy_admin_shim_removed.py
sed -n '1,240p' tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_optional_current_user_shim_removed.py
sed -n '1,240p' tldw_Server_API/tests/Resource_Governance/test_auth_route_map_coverage.py
```

Expected: documented endpoint and dependency behavior is cross-checked against the tests that claim to enforce it before verification commands are chosen.

- [ ] **Step 4: Write the Stage 4 artifact before executing tests**

`Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage4-tests-docs-drift-and-verification-gaps.md` must record:
- the exact docs reviewed
- candidate documentation drifts and test-gap hypotheses
- which claims are already proven by code alone
- which claims need the verification commands from Steps 5 and 6

- [ ] **Step 5: Run the smallest contract-level verification slice**

Run:
```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py \
  tldw_Server_API/tests/AuthNZ/property/test_auth_endpoints_property.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_endpoints_integration.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_endpoints_integration_fixed.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_hardening.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_legacy_admin_shim_removed.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_optional_current_user_shim_removed.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_db_adapter_backend_selection.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_principal_resolver_mode_shim_removed.py \
  | tee /tmp/authnz_full_contract_pytest.log
```

Expected: passing or explicitly skipped contract checks; any failures or skips become direct evidence for drift, not noise to ignore.

- [ ] **Step 6: Run the route-coverage verification slice**

Run:
```bash
source .venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Resource_Governance/test_auth_route_map_coverage.py | tee -a /tmp/authnz_full_contract_pytest.log
```

Expected: one targeted confirmation of route-map coverage for auth surfaces, or a clear failure that must be recorded as a contract gap.

- [ ] **Step 7: Update and commit the Stage 4 artifact**

`Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage4-tests-docs-drift-and-verification-gaps.md` must be updated with:
- the executed commands from Steps 5 and 6
- the observed pass, fail, or skip outcomes
- ranked documentation drifts
- ranked missing or weak test invariants
- explicit blind spots for enterprise, federation, or secret-backend paths that were not strongly validated

Run:
```bash
git add Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage4-tests-docs-drift-and-verification-gaps.md
git commit -m "docs: record AuthNZ docs and test-gap review stage"
```

Expected: one docs-only commit preserves the docs and coverage assessment before final synthesis begins.

### Task 5: Execute Stage 5 Final Synthesis, Roadmap, and Patch Decisions

**Files:**
- Modify: `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage5-final-synthesis-roadmap-and-patch-decisions.md`
- Inspect: `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage1-baseline-and-boundary-inventory.md`
- Inspect: `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage2-runtime-authentication-and-authorization-analysis.md`
- Inspect: `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage3-persistence-migrations-and-configuration-safety.md`
- Inspect: `Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage4-tests-docs-drift-and-verification-gaps.md`
- Test: none by default

- [ ] **Step 1: Read and deduplicate the stage findings**

Run:
```bash
sed -n '1,260p' Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage1-baseline-and-boundary-inventory.md
sed -n '1,260p' Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage2-runtime-authentication-and-authorization-analysis.md
sed -n '1,260p' Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage3-persistence-migrations-and-configuration-safety.md
sed -n '1,260p' Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage4-tests-docs-drift-and-verification-gaps.md
```

Expected: one consolidated set of findings without duplicate issues or conflicting severity labels.

- [ ] **Step 2: Write the Stage 5 synthesis artifact**

`Docs/superpowers/reviews/authnz-full-module/2026-04-08-stage5-final-synthesis-roadmap-and-patch-decisions.md` must contain:
```markdown
# Stage 5 Final Synthesis

## Highest-Confidence Findings
## Probable Risks
## Improvements
## Open Questions
## Verification Summary
## Remediation Roadmap
## Patch-Gate Decisions
## Blind Spots / Not Reviewed
```

- [ ] **Step 3: Bucket the remediation roadmap**

Use these exact roadmap buckets:
- `Immediate fixes`
- `Near-term hardening`
- `Structural refactors worth scheduling`

Every roadmap item must name the exact file or file group affected and state why the item belongs in that bucket.

- [ ] **Step 4: Make the patch-gate decision explicitly**

Use this exact decision matrix:
```markdown
- Not confirmed: no remediation branch
- Confirmed but broad or product-dependent: roadmap only
- Confirmed, critical, localized, and verifiable: preserve pre-fix evidence and write a separate remediation plan before any code change
```

Expected: the review does not silently drift into patching. Any code-fix branch is either rejected or broken into a fresh remediation plan with pre-fix evidence preserved.

- [ ] **Step 5: Run one last verification command only if a single claim remains unresolved**

Reuse exactly one already-defined command from Task 2 Step 5, Task 2 Step 6, Task 3 Step 5, Task 3 Step 6, Task 4 Step 5, or Task 4 Step 6. If no unresolved claim remains, record `No additional verification required` in the Stage 5 artifact instead of running more tests.

Expected: no blanket reruns. Either one unresolved claim is settled, or the decision to stop is explicit.

- [ ] **Step 6: Commit the final review artifacts**

Run:
```bash
git add Docs/superpowers/reviews/authnz-full-module
git commit -m "docs: finalize AuthNZ full-module review artifacts"
```

Expected: one docs-only commit preserves the final staged review record.

## Self-Review Checklist

- Verify that every stage from the approved spec maps to a task in this plan.
- Verify that the stage artifact path and filenames match the approved review workspace contract.
- Verify that the plan never authorizes ad hoc source patches during the review.
- Verify that the final response structure in Task 1 matches the approved design.
- Verify that every verification command is targeted and uses the project virtual environment.
- Verify that enterprise, federation, and secret-backend blind spots are explicitly preserved if direct validation is weak.
