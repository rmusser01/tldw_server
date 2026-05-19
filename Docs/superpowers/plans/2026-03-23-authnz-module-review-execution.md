# AuthNZ Module Review Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved AuthNZ module review and deliver one consolidated, evidence-backed review covering security, correctness, maintainability, and test gaps for `tldw_Server_API/app/core/AuthNZ` and `tldw_Server_API/tests/AuthNZ`.

**Architecture:** This is a read-first, risk-first review plan. Execution stays inside the AuthNZ module and its test suite, inspects the most security-sensitive primitives first, and then broadens to full module coverage before synthesizing one final findings report. No source changes are expected during execution; the deliverable is the final review output in-session.

**Tech Stack:** Python 3, pytest, ripgrep, git, Markdown

---

## Review File Map

**No repository files should be modified during execution.**

**Primary implementation files to inspect during the review:**
- `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- `tldw_Server_API/app/core/AuthNZ/session_manager.py`
- `tldw_Server_API/app/core/AuthNZ/token_blacklist.py`
- `tldw_Server_API/app/core/AuthNZ/password_service.py`
- `tldw_Server_API/app/core/AuthNZ/mfa_service.py`
- `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
- `tldw_Server_API/app/core/AuthNZ/api_key_crypto.py`
- `tldw_Server_API/app/core/AuthNZ/virtual_keys.py`
- `tldw_Server_API/app/core/AuthNZ/quotas.py`
- `tldw_Server_API/app/core/AuthNZ/rate_limiter.py`
- `tldw_Server_API/app/core/AuthNZ/lockout_tracker.py`
- `tldw_Server_API/app/core/AuthNZ/auth_governor.py`
- `tldw_Server_API/app/core/AuthNZ/rbac.py`
- `tldw_Server_API/app/core/AuthNZ/permissions.py`
- `tldw_Server_API/app/core/AuthNZ/org_rbac.py`
- `tldw_Server_API/app/core/AuthNZ/orgs_teams.py`
- `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`
- `tldw_Server_API/app/core/AuthNZ/principal_model.py`
- `tldw_Server_API/app/core/AuthNZ/settings.py`
- `tldw_Server_API/app/core/AuthNZ/database.py`
- `tldw_Server_API/app/core/AuthNZ/db_config.py`
- `tldw_Server_API/app/core/AuthNZ/migrations.py`
- `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- `tldw_Server_API/app/core/AuthNZ/initialize.py`
- `tldw_Server_API/app/core/AuthNZ/run_migrations.py`
- `tldw_Server_API/app/core/AuthNZ/migrate_to_multiuser.py`
- `tldw_Server_API/app/core/AuthNZ/startup_integrity.py`
- `tldw_Server_API/app/core/AuthNZ/rg_startup_guard.py`
- `tldw_Server_API/app/core/AuthNZ/security_headers.py`
- `tldw_Server_API/app/core/AuthNZ/csrf_protection.py`
- `tldw_Server_API/app/core/AuthNZ/ip_allowlist.py`
- `tldw_Server_API/app/core/AuthNZ/key_resolution.py`
- `tldw_Server_API/app/core/AuthNZ/crypto_utils.py`
- `tldw_Server_API/app/core/AuthNZ/byok_config.py`
- `tldw_Server_API/app/core/AuthNZ/byok_helpers.py`
- `tldw_Server_API/app/core/AuthNZ/byok_rotation.py`
- `tldw_Server_API/app/core/AuthNZ/byok_runtime.py`
- `tldw_Server_API/app/core/AuthNZ/user_provider_secrets.py`
- `tldw_Server_API/app/core/AuthNZ/secret_backends/local_encrypted.py`
- `tldw_Server_API/app/core/AuthNZ/secret_backends/registry.py`
- `tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/token_blacklist_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/users_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/mfa_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/quotas_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/rbac_repo.py`
- `tldw_Server_API/app/core/AuthNZ/repos/orgs_teams_repo.py`

**High-value tests to inspect and selectively run:**
- `tldw_Server_API/tests/AuthNZ/property/test_jwt_service_property.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_jwt_dual_key_rotation.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_token_metadata.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_api_key_crypto.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_limits_unit.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_enforcement_unit.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_lockout_reset.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_password_service_backend_selection.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_auth_service_backend_agnostic.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_settings_guardrails.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_backend_detection_guardrail.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_startup_integrity.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_rg_startup_guard.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_usage_truthiness.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_api_key_rotation_audit.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_real_rate_limiter.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_forgot_password_flow_integration.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_verify_email_flow_integration.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_magic_link_flow_integration.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_mfa_service.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_integration_flow_extended.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_comprehensive.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_rbac_effective_permissions.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_orgs_teams_repo_postgres.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_quotas_repo_postgres.py`

## Stage Overview

## Stage 1: Inventory and Review Setup
**Goal:** Confirm the exact AuthNZ file/test surface and build the review checklist before deep reading.
**Success Criteria:** The scoped files, test buckets, high-risk primitives, and report structure are fixed so later passes do not drift into excluded endpoint or app wiring code.
**Tests:** No test execution in this stage.
**Status:** Not Started

## Stage 2: Security Pass
**Goal:** Identify vulnerabilities, unsafe defaults, fail-open behavior, secret-handling problems, and privilege-boundary risks inside the AuthNZ module.
**Success Criteria:** High-risk primitives are inspected first, targeted tests are read and selectively run, and all security findings are captured with severity, confidence, and file references.
**Tests:** JWT/session/key/security unit and integration tests listed below.
**Status:** Not Started

## Stage 3: Correctness Pass
**Goal:** Find behavioral bugs, inconsistent state transitions, backend parity issues, and broken assumptions in authentication, sessions, quotas, and repo logic.
**Success Criteria:** Core flows are traced end to end within the module, targeted tests are used to validate expectations, and correctness findings are separated from security concerns.
**Tests:** Flow, session, MFA, rate-limit, and backend-related tests listed below.
**Status:** Not Started

## Stage 4: Maintainability Pass
**Goal:** Identify design debt, confusing module boundaries, migration/config complexity, and brittle patterns that raise future change risk.
**Success Criteria:** Large or cross-cutting files, repeated logic, backend branching, and operational guardrails are inspected and documented as maintainability issues or improvements.
**Tests:** Settings, migrations, startup, and backend guardrail tests listed below.
**Status:** Not Started

## Stage 5: Test-Gap Pass and Final Synthesis
**Goal:** Map test coverage against the module surface, identify missing invariants, and produce the final consolidated review.
**Success Criteria:** Missing or weak tests are prioritized by risk reduction value and the final report is delivered in the approved four-section format.
**Tests:** Reuse prior test inventory; run additional focused tests only if needed to confirm a disputed claim.
**Status:** Not Started

### Task 1: Lock the Review Surface and Checklist

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/AuthNZ`
- Inspect: `tldw_Server_API/tests/AuthNZ`
- Test: none

- [ ] **Step 1: Enumerate the scoped file surface**

Run:
```bash
source .venv/bin/activate
rg --files tldw_Server_API/app/core/AuthNZ tldw_Server_API/tests/AuthNZ | sort
```

Expected: a stable list of AuthNZ implementation and test files, with no endpoint paths outside the approved scope.

- [ ] **Step 2: Enumerate the highest-risk implementation files**

Run:
```bash
source .venv/bin/activate
printf '%s\n' \
  tldw_Server_API/app/core/AuthNZ/jwt_service.py \
  tldw_Server_API/app/core/AuthNZ/session_manager.py \
  tldw_Server_API/app/core/AuthNZ/token_blacklist.py \
  tldw_Server_API/app/core/AuthNZ/password_service.py \
  tldw_Server_API/app/core/AuthNZ/mfa_service.py \
  tldw_Server_API/app/core/AuthNZ/api_key_manager.py \
  tldw_Server_API/app/core/AuthNZ/api_key_crypto.py \
  tldw_Server_API/app/core/AuthNZ/virtual_keys.py \
  tldw_Server_API/app/core/AuthNZ/rbac.py \
  tldw_Server_API/app/core/AuthNZ/permissions.py \
  tldw_Server_API/app/core/AuthNZ/org_rbac.py \
  tldw_Server_API/app/core/AuthNZ/orgs_teams.py \
  tldw_Server_API/app/core/AuthNZ/rate_limiter.py \
  tldw_Server_API/app/core/AuthNZ/quotas.py \
  tldw_Server_API/app/core/AuthNZ/ip_allowlist.py \
  tldw_Server_API/app/core/AuthNZ/secret_backends/local_encrypted.py
```

Expected: a hand-curated reading order for the security and correctness passes.

- [ ] **Step 3: Record the final review output format before reading deeply**

Use this structure for the final response:
```markdown
## Security
- findings ordered by severity, then lower-confidence risks, then improvements

## Correctness
- findings ordered by severity, then lower-confidence risks, then improvements

## Maintainability
- findings ordered by severity, then lower-confidence risks, then improvements

## Test Gaps
- missing invariants first, then weak/misleading tests, then lower-priority gaps
```

- [ ] **Step 4: Confirm the workspace is still safe for a read-only review**

Run:
```bash
git status --short
```

Expected: no source files in `tldw_Server_API/app/core/AuthNZ` or `tldw_Server_API/tests/AuthNZ` are modified as part of the review setup.

### Task 2: Execute the Security Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/session_manager.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/token_blacklist.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/password_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/mfa_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/api_key_crypto.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/virtual_keys.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/key_resolution.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/crypto_utils.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/ip_allowlist.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/security_headers.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/csrf_protection.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/settings.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/byok_config.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/byok_helpers.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/byok_rotation.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/byok_runtime.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/user_provider_secrets.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/secret_backends/local_encrypted.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/token_blacklist_repo.py`
- Test: `tldw_Server_API/tests/AuthNZ/property/test_jwt_service_property.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_jwt_dual_key_rotation.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_api_key_crypto.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_limits_unit.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_enforcement_unit.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_api_key_rotation_audit.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_real_rate_limiter.py`

- [ ] **Step 1: Read the token, session, and key lifecycle files in order**

Read and trace:
- token issuance and verification
- refresh rotation and revocation
- session persistence and encrypted storage
- API key hashing, lookup, rotation, and revocation
- virtual key scope and limit enforcement

- [ ] **Step 2: Search for fail-open or hazardous patterns in the security surface**

Run:
```bash
source .venv/bin/activate
rg -n "except Exception|return True|allow|fallback|TEST_MODE|DEBUG|insecure|TODO|FIXME" \
  tldw_Server_API/app/core/AuthNZ/jwt_service.py \
  tldw_Server_API/app/core/AuthNZ/session_manager.py \
  tldw_Server_API/app/core/AuthNZ/token_blacklist.py \
  tldw_Server_API/app/core/AuthNZ/password_service.py \
  tldw_Server_API/app/core/AuthNZ/mfa_service.py \
  tldw_Server_API/app/core/AuthNZ/api_key_manager.py \
  tldw_Server_API/app/core/AuthNZ/virtual_keys.py \
  tldw_Server_API/app/core/AuthNZ/ip_allowlist.py \
  tldw_Server_API/app/core/AuthNZ/secret_backends/local_encrypted.py
```

Expected: a short list of suspicious branches or guardrails to inspect manually, not a final finding list by itself.

- [ ] **Step 3: Read the targeted security tests and note the invariants they protect**

Capture:
- what the test asserts
- whether the assertion is a direct security invariant or an indirect regression signal
- what important neighboring behavior is still untested

- [ ] **Step 4: Run the focused security verification suite**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/property/test_jwt_service_property.py \
  tldw_Server_API/tests/AuthNZ/unit/test_jwt_dual_key_rotation.py \
  tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py \
  tldw_Server_API/tests/AuthNZ/unit/test_api_key_crypto.py \
  tldw_Server_API/tests/AuthNZ/unit/test_ip_allowlist.py \
  tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_limits_unit.py \
  tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_enforcement_unit.py -v
```

Expected: the focused unit/property suite collects and passes; any failure is investigated as either environment noise or evidence supporting a finding.

- [ ] **Step 5: Run an integration spot-check only if a security claim depends on lifecycle behavior**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py \
  tldw_Server_API/tests/AuthNZ/integration/test_api_key_rotation_audit.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_login_lockout_real_rate_limiter.py -v
```

Expected: integration tests either pass or expose a concrete lifecycle assumption worth documenting; do not widen scope to fix unrelated harness issues.

- [ ] **Step 6: Draft the Security section before moving on**

For each item record:
- severity
- confidence
- why it matters
- exact file and line references
- whether it is a confirmed finding, probable risk, or improvement

### Task 3: Execute the Correctness Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/session_manager.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/password_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/mfa_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/token_blacklist.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/quotas.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/rate_limiter.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/lockout_tracker.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/auth_governor.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/input_validation.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/username_utils.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/database.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/db_config.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/users_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/sessions_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/mfa_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/quotas_repo.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_token_metadata.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_password_service_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_auth_service_backend_agnostic.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_lockout_reset.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_forgot_password_flow_integration.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_verify_email_flow_integration.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_magic_link_flow_integration.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_mfa_service.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_integration_flow_extended.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_comprehensive.py`

- [ ] **Step 1: Trace state transitions inside the core auth services**

Trace:
- token creation, refresh, and revocation
- session creation, update, and deletion
- password hashing, validation, reset, and history rules
- MFA enrollment, validation, backup, and disable flows
- quota, lockout, and governor state updates

- [ ] **Step 2: Inspect backend branching and repo contracts**

Read the repo and database helpers and note:
- where SQLite and Postgres behavior diverge
- where methods return different shapes or defaults by backend
- where missing transactions or error handling could create correctness bugs

- [ ] **Step 3: Read the targeted correctness tests before running anything**

For each test, note:
- the behavior under test
- the assumptions it bakes in
- whether nearby branches appear untested

- [ ] **Step 4: Run the focused correctness verification suite**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_session_manager_token_metadata.py \
  tldw_Server_API/tests/AuthNZ/unit/test_password_service_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/unit/test_auth_service_backend_agnostic.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_lockout_reset.py \
  tldw_Server_API/tests/AuthNZ/integration/test_forgot_password_flow_integration.py \
  tldw_Server_API/tests/AuthNZ/integration/test_verify_email_flow_integration.py \
  tldw_Server_API/tests/AuthNZ/integration/test_magic_link_flow_integration.py \
  tldw_Server_API/tests/AuthNZ/integration/test_mfa_service.py -v
```

Expected: targeted flow tests collect; passing tests strengthen the behavioral baseline, while meaningful failures become evidence for correctness findings.

- [ ] **Step 5: Use one broad integration pair only if needed to confirm cross-flow consistency**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_integration_flow_extended.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_comprehensive.py -v
```

Expected: broad integration tests provide confirmation for disputed flow claims; if the environment blocks them, record that limitation instead of broadening scope.

- [ ] **Step 6: Draft the Correctness section before moving on**

Separate:
- concrete bugs
- likely but unproven correctness risks
- lower-priority cleanup or simplification opportunities

### Task 4: Execute the Maintainability Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/AuthNZ/settings.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/database.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/db_config.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/initialize.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/run_migrations.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/migrate_to_multiuser.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/startup_integrity.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/rg_startup_guard.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/monitoring.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/alerting.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/scheduler.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/__init__.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/repos/*.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_settings_guardrails.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_backend_detection_guardrail.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_startup_integrity.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_rg_startup_guard.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_usage_truthiness.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_migrate_to_multiuser_review_fixes.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_db_setup.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_backends_pytest.py`

- [ ] **Step 1: Identify oversized or cross-cutting files**

Run:
```bash
source .venv/bin/activate
wc -l \
  tldw_Server_API/app/core/AuthNZ/*.py \
  tldw_Server_API/app/core/AuthNZ/repos/*.py | sort -nr | sed -n '1,25p'
```

Expected: a ranked list of the largest AuthNZ files, useful for spotting files likely carrying too many responsibilities.

- [ ] **Step 2: Search for maintainability risk markers**

Run:
```bash
source .venv/bin/activate
rg -n "except Exception|pass$|TODO|FIXME|pragma: no cover|os.getenv|load_.*config|if .*postgres|if .*sqlite" \
  tldw_Server_API/app/core/AuthNZ
```

Expected: candidate hotspots for brittle branching, broad exception handling, or configuration sprawl that need manual inspection.

- [ ] **Step 3: Read the settings, migration, startup, and repo layers with boundary questions in mind**

Check:
- whether responsibilities are mixed across files
- whether configuration loading is predictable
- whether backend-specific behavior is centralized or scattered
- whether migration logic is easy to reason about and verify

- [ ] **Step 4: Read the targeted guardrail and migration tests**

Focus on:
- what structural guarantees are already locked in
- what design assumptions remain implicit
- what long-term maintenance risks the tests do not catch

- [ ] **Step 5: Run the focused maintainability verification suite**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_settings_guardrails.py \
  tldw_Server_API/tests/AuthNZ/unit/test_backend_detection_guardrail.py \
  tldw_Server_API/tests/AuthNZ/unit/test_startup_integrity.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rg_startup_guard.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_usage_truthiness.py \
  tldw_Server_API/tests/AuthNZ/unit/test_migrate_to_multiuser_review_fixes.py -v
```

Expected: structural guardrail tests collect and pass; failures indicate either genuine maintainability regressions or environment assumptions worth documenting.

- [ ] **Step 6: Use backend/setup integration tests only if a maintainability claim depends on them**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/integration/test_db_setup.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_backends_pytest.py -v
```

Expected: only use these tests when they help confirm a backend-parity or setup-complexity claim; otherwise skip to stay within review scope and time.

- [ ] **Step 7: Draft the Maintainability section before moving on**

Separate:
- concrete maintainability defects
- probable long-term risks
- improvement opportunities that would materially simplify the module

### Task 5: Execute the Test-Gap Pass and Final Synthesis

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/AuthNZ`
- Inspect: `tldw_Server_API/tests/AuthNZ`
- Test: reuse targeted tests from Tasks 2-4 only if needed

- [ ] **Step 1: Compare the module surface to the tests you actually inspected**

Run:
```bash
source .venv/bin/activate
rg --files tldw_Server_API/app/core/AuthNZ | rg '\.py$' | sort
```

Expected: a complete implementation inventory to compare against the tests already read.

- [ ] **Step 2: Identify modules with weak or indirect coverage**

Run:
```bash
source .venv/bin/activate
rg -l "JWTService|SessionManager|PasswordService|MFA|RateLimiter|VirtualKey|RBAC|Quota|Org|Team|Blacklist|BYOK|secret_backend" \
  tldw_Server_API/tests/AuthNZ | sort
```

Expected: a rough coverage map showing where direct tests cluster and where important areas may have only indirect or no obvious coverage.

- [ ] **Step 3: Write the prioritized missing-invariant list**

Prioritize gaps by:
- security risk reduction
- regression risk reduction
- likelihood of catching backend-parity drift
- likelihood of clarifying currently ambiguous behavior

- [ ] **Step 4: Re-run only the smallest additional test needed to settle any disputed finding**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_jwt_dual_key_rotation.py -v
```

Expected: rerun the smallest already-inspected test file tied to the disputed claim; if `test_jwt_dual_key_rotation.py` is not the right one, substitute another previously inspected single test file rather than broadening the suite.

- [ ] **Step 5: Synthesize the final review in the approved four-section order**

The final response must:
- start with findings, not summary
- order issues by severity within each section
- include exact file references
- distinguish confirmed findings from probable risks and improvements
- explicitly say when a section has no confirmed issues

- [ ] **Step 6: Add a short residual-risk note**

Record:
- what was not runtime-verified
- where environment limitations prevented stronger confirmation
- any AuthNZ areas that remain inherently lower-confidence despite careful reading

## Self-Check Before Delivery

- [ ] No findings rely on files outside `tldw_Server_API/app/core/AuthNZ` or `tldw_Server_API/tests/AuthNZ`.
- [ ] Security, correctness, maintainability, and test gaps are kept separate.
- [ ] Severity and confidence are both present where needed.
- [ ] Lower-confidence concerns are not stated as confirmed bugs.
- [ ] The final response leads with findings and file references, not background summary.
- [ ] Any test command actually run is recorded accurately in the final response if it materially affected conclusions.
