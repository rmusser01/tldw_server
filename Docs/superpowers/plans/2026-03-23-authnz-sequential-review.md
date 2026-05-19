# AuthNZ Sequential Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Conduct a sequential, subsystem-by-subsystem AuthNZ review and produce one evidence-backed findings document per stage plus a final remediation synthesis.

**Architecture:** This is a read-first, stage-gated review plan. Each stage inspects a bounded set of source files, checks the matching tests and runtime assumptions, records findings before suggesting fixes, and only then proceeds to the next stage. Review outputs live under `Docs/superpowers/reviews/authnz/` so later remediation work can reference stable findings instead of chat history.

**Tech Stack:** Python 3, FastAPI, pytest, ripgrep, git, Markdown

---

## Review File Map

**Create during execution:**
- `Docs/superpowers/reviews/authnz/README.md`
- `Docs/superpowers/reviews/authnz/2026-03-23-stage1-boundary-deps-review.md`
- `Docs/superpowers/reviews/authnz/2026-03-23-stage2-login-session-review.md`
- `Docs/superpowers/reviews/authnz/2026-03-23-stage3-keys-budget-review.md`
- `Docs/superpowers/reviews/authnz/2026-03-23-stage4-rbac-admin-review.md`
- `Docs/superpowers/reviews/authnz/2026-03-23-stage5-authnz-synthesis.md`

**Primary source files to inspect:**
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`
- `tldw_Server_API/app/core/AuthNZ/principal_model.py`
- `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- `tldw_Server_API/app/api/v1/endpoints/auth.py`
- `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- `tldw_Server_API/app/core/AuthNZ/session_manager.py`
- `tldw_Server_API/app/core/AuthNZ/password_service.py`
- `tldw_Server_API/app/core/AuthNZ/mfa_service.py`
- `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
- `tldw_Server_API/app/core/AuthNZ/virtual_keys.py`
- `tldw_Server_API/app/core/AuthNZ/llm_budget_middleware.py`
- `tldw_Server_API/app/core/AuthNZ/llm_budget_guard.py`
- `tldw_Server_API/app/core/AuthNZ/quotas.py`
- `tldw_Server_API/app/core/AuthNZ/rbac.py`
- `tldw_Server_API/app/core/AuthNZ/permissions.py`
- `tldw_Server_API/app/core/AuthNZ/org_rbac.py`
- `tldw_Server_API/app/core/AuthNZ/orgs_teams.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_rbac.py`

**High-value existing tests to reuse during the review:**
- `tldw_Server_API/tests/AuthNZ/unit/test_auth_deps_precedence.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_test_mode_runtime_guard.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_hardening.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_auth_principal_resolver.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_session_revocation_blacklist.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_csrf_binding.py`
- `tldw_Server_API/tests/AuthNZ_Postgres/test_auth_enhanced_mfa.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_api_key_manager_validation.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_limits_unit.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_enforcement_unit.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_llm_budget_middleware.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_llm_budget_guard_state_failure.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_api_key_rotation_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_llm_budget_402_sqlite.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_org_rbac_scoped_permissions_sqlite.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_rbac_admin_endpoints.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_rbac_effective_permissions.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_admin_roles_single_user_claims.py`

## Stage Overview

## Stage 1: Boundary and Dependency Review
**Goal:** Validate the auth boundary before trusting downstream behavior.
**Success Criteria:** Token/header precedence, mode gating, test-mode behavior, principal construction, and fallback logic are all traced and documented with concrete findings or an explicit “no findings” conclusion.
**Tests:** `test_auth_deps_precedence.py`, `test_test_mode_runtime_guard.py`, `test_auth_deps_hardening.py`, `test_auth_principal_resolver.py`
**Status:** Not Started

## Stage 2: Login, Session, JWT, and MFA Review
**Goal:** Validate the main authentication state machine.
**Success Criteria:** Login, refresh, revocation, password reset, email verification, session handling, and MFA transitions are each reviewed with evidence and any state-machine inconsistencies are documented.
**Tests:** `test_auth_endpoints_extended.py`, `test_jwt_service.py`, `test_session_manager_configured_key.py`, `test_session_revocation_blacklist.py`, `test_csrf_binding.py`, `test_auth_enhanced_mfa.py`
**Status:** Not Started

## Stage 3: API Key, Virtual Key, Budget, and Quota Review
**Goal:** Validate programmatic access issuance and enforcement.
**Success Criteria:** Key lifecycle, scope enforcement, allowlists, quota accounting, and budget failure behavior are each reviewed with concrete findings or explicit clearance notes.
**Tests:** `test_api_key_manager_validation.py`, `test_virtual_keys_limits_unit.py`, `test_virtual_keys_enforcement_unit.py`, `test_llm_budget_middleware.py`, `test_llm_budget_guard_state_failure.py`, `test_api_key_rotation_sqlite.py`, `test_llm_budget_402_sqlite.py`
**Status:** Not Started

## Stage 4: RBAC, Org, Team, and Admin Authorization Review
**Goal:** Validate permission resolution and admin protection.
**Success Criteria:** Claim-first guards, scoped permission semantics, org/team inheritance, override behavior, and admin endpoint consistency are each reviewed and documented.
**Tests:** `test_org_rbac_scoped_permissions_sqlite.py`, `test_rbac_admin_endpoints.py`, `test_rbac_effective_permissions.py`, `test_permissions_claim_first.py`, `test_admin_roles_single_user_claims.py`
**Status:** Not Started

## Stage 5: Whole-Surface Synthesis
**Goal:** Consolidate prior findings into one prioritized remediation sequence.
**Success Criteria:** Duplicates are removed, cross-cutting risks are grouped, and the final document provides a ranked fix order with rationale.
**Tests:** Re-run only the smallest set of tests needed to confirm any disputed findings from earlier stages.
**Status:** Not Started

### Task 1: Prepare Review Artifacts and Inventory

**Files:**
- Create: `Docs/superpowers/reviews/authnz/README.md`
- Create: `Docs/superpowers/reviews/authnz/2026-03-23-stage1-boundary-deps-review.md`
- Create: `Docs/superpowers/reviews/authnz/2026-03-23-stage2-login-session-review.md`
- Create: `Docs/superpowers/reviews/authnz/2026-03-23-stage3-keys-budget-review.md`
- Create: `Docs/superpowers/reviews/authnz/2026-03-23-stage4-rbac-admin-review.md`
- Create: `Docs/superpowers/reviews/authnz/2026-03-23-stage5-authnz-synthesis.md`
- Modify: `Docs/superpowers/plans/2026-03-23-authnz-sequential-review.md`
- Test: none

- [ ] **Step 1: Create the review output directory**

Run:
```bash
mkdir -p Docs/superpowers/reviews/authnz
```

Expected: the directory exists with no other workspace changes.

- [ ] **Step 2: Create one markdown file per stage with a fixed findings template**

Each file should contain these headings:
```markdown
# Stage N Title

## Scope
## Code Paths Reviewed
## Tests Reviewed
## Validation Commands
## Findings
## Coverage Gaps
## Improvements
## Exit Note
```

- [ ] **Step 3: Write `Docs/superpowers/reviews/authnz/README.md`**

Document:
- the stage order `D -> A -> B -> C -> E`
- the path of each stage report
- the rule that findings must be written before fixes are proposed

- [ ] **Step 4: Verify the workspace is in a safe starting state**

Run:
```bash
git status --short
```

Expected: only the new review-doc paths appear, or the tree is still clean if files have not been created yet.

- [ ] **Step 5: Commit the review scaffold**

Run:
```bash
git add Docs/superpowers/reviews/authnz Docs/superpowers/plans/2026-03-23-authnz-sequential-review.md
git commit -m "docs: scaffold authnz review artifacts"
```

Expected: one docs-only commit capturing the review workspace.

### Task 2: Execute Stage 1 Boundary and Dependency Review

**Files:**
- Modify: `Docs/superpowers/reviews/authnz/2026-03-23-stage1-boundary-deps-review.md`
- Inspect: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/principal_model.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/settings.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_auth_deps_precedence.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_test_mode_runtime_guard.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_hardening.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_auth_principal_resolver.py`

- [ ] **Step 1: Read the boundary entrypoints and map the request path**

Run:
```bash
rg -n "get_auth_principal|Authorization|X-API-KEY|TEST_MODE|request.state" tldw_Server_API/app/api/v1/API_Deps/auth_deps.py tldw_Server_API/app/core/AuthNZ/auth_principal_resolver.py tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py
```

Expected: a concise map of the boundary branches to record in the stage report.

- [ ] **Step 2: Inspect single-user and multi-user precedence logic**

Read the relevant functions and record:
- which credential type wins when multiple are present
- where mode checks happen
- whether any fallback path weakens the intended boundary

- [ ] **Step 3: Inspect principal construction and sanitization**

Confirm:
- sensitive fields are stripped before user objects escape dependencies
- request state caching cannot leak stale or overprivileged data across branches

- [ ] **Step 4: Review the targeted tests for expected invariants**

Read the four listed tests and copy the core invariant each one protects into the stage report.

- [ ] **Step 5: Run the targeted Stage 1 tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_auth_deps_precedence.py \
  tldw_Server_API/tests/AuthNZ/unit/test_test_mode_runtime_guard.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_deps_hardening.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_auth_principal_resolver.py -v
```

Expected: tests collect and pass; any failure is triaged as either environment noise or a potential finding.

- [ ] **Step 6: Write the Stage 1 report**

Record:
- confirmed findings ordered by severity
- test gaps or ambiguous behaviors
- low-risk improvements
- an exit note stating whether later stages need to account for a boundary-level concern

- [ ] **Step 7: Commit the Stage 1 report**

Run:
```bash
git add Docs/superpowers/reviews/authnz/2026-03-23-stage1-boundary-deps-review.md
git commit -m "docs: record authnz boundary review findings"
```

Expected: one commit containing only the Stage 1 report.

### Task 3: Execute Stage 2 Login, Session, JWT, and MFA Review

**Files:**
- Modify: `Docs/superpowers/reviews/authnz/2026-03-23-stage2-login-session-review.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/auth.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/session_manager.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/password_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/mfa_service.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/csrf_protection.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_revocation_blacklist.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_csrf_binding.py`
- Test: `tldw_Server_API/tests/AuthNZ_Postgres/test_auth_enhanced_mfa.py`

- [ ] **Step 1: Map the Stage 2 state machine**

Document the transitions for:
- login
- refresh
- logout and revoke-all
- forgot-password and reset-password
- verify-email and resend-verification
- MFA setup, verify, login, and disable

- [ ] **Step 2: Inspect JWT issuance, session persistence, and revocation coupling**

Confirm:
- access and refresh tokens are differentiated correctly
- refresh rotation invalidates prior state
- blacklist or session revocation is checked on the intended paths

- [ ] **Step 3: Inspect password, CSRF, and MFA edge cases**

Capture any inconsistencies involving:
- password reset token reuse
- session encryption key behavior
- CSRF binding assumptions
- MFA bootstrap and recovery flows

- [ ] **Step 4: Review the targeted tests and note missing scenarios**

Pay special attention to:
- lockout and retry edges
- refresh race conditions
- password reset replay
- MFA bypass or partial-enrollment behavior

- [ ] **Step 5: Run the focused Stage 2 unit tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py \
  tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py \
  tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py \
  tldw_Server_API/tests/AuthNZ/unit/test_session_revocation_blacklist.py \
  tldw_Server_API/tests/AuthNZ/unit/test_csrf_binding.py -v
```

Expected: focused unit coverage passes and sharpens the state-machine review.

- [ ] **Step 6: Run MFA coverage if PostgreSQL or Docker-backed fixtures are available**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ_Postgres/test_auth_enhanced_mfa.py -v
```

Expected: pass if the fixture environment is available; otherwise note the environment limitation in the report instead of blocking the stage.

- [ ] **Step 7: Write the Stage 2 report**

Record:
- confirmed state-machine or flow bugs
- environment-limited validation that still needs follow-up
- missing tests and lower-risk hardening ideas

- [ ] **Step 8: Commit the Stage 2 report**

Run:
```bash
git add Docs/superpowers/reviews/authnz/2026-03-23-stage2-login-session-review.md
git commit -m "docs: record authnz login and session review findings"
```

Expected: one docs-only commit for Stage 2 findings.

### Task 4: Execute Stage 3 API Key, Virtual Key, Budget, and Quota Review

**Files:**
- Modify: `Docs/superpowers/reviews/authnz/2026-03-23-stage3-keys-budget-review.md`
- Inspect: `tldw_Server_API/app/core/AuthNZ/api_key_manager.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/virtual_keys.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/llm_budget_middleware.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/llm_budget_guard.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/quotas.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/ip_allowlist.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_api_key_manager_validation.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_limits_unit.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_enforcement_unit.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_llm_budget_middleware.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_llm_budget_guard_state_failure.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_api_key_rotation_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_llm_budget_402_sqlite.py`

- [ ] **Step 1: Trace the lifecycle of API keys and virtual keys**

Record how keys are:
- created
- stored or hashed
- rotated
- revoked
- matched to scopes or limits

- [ ] **Step 2: Inspect enforcement paths for scope, allowlist, and budget checks**

Focus on:
- whether enforcement happens before work is performed
- whether degraded guard state fails open or fails closed
- whether key metadata and budgets can drift from actual usage

- [ ] **Step 3: Review the targeted key and budget tests**

Document which tests cover:
- validation
- rotation
- scope enforcement
- middleware refusal behavior
- state-failure handling

- [ ] **Step 4: Run the focused Stage 3 tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ/unit/test_api_key_manager_validation.py \
  tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_limits_unit.py \
  tldw_Server_API/tests/AuthNZ/unit/test_virtual_keys_enforcement_unit.py \
  tldw_Server_API/tests/AuthNZ/unit/test_llm_budget_middleware.py \
  tldw_Server_API/tests/AuthNZ/unit/test_llm_budget_guard_state_failure.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_api_key_rotation_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_llm_budget_402_sqlite.py -v
```

Expected: targeted key and budget coverage passes, or failures become candidate findings after triage.

- [ ] **Step 5: Write the Stage 3 report**

Separate:
- confirmed enforcement or accounting bugs
- weak or missing tests
- lower-risk hardening ideas such as clearer failure semantics or audit improvements

- [ ] **Step 6: Commit the Stage 3 report**

Run:
```bash
git add Docs/superpowers/reviews/authnz/2026-03-23-stage3-keys-budget-review.md
git commit -m "docs: record authnz key and budget review findings"
```

Expected: one commit containing only Stage 3 notes.

### Task 5: Execute Stage 4 RBAC, Org, Team, and Admin Authorization Review

**Files:**
- Modify: `Docs/superpowers/reviews/authnz/2026-03-23-stage4-rbac-admin-review.md`
- Inspect: `tldw_Server_API/app/core/AuthNZ/rbac.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/permissions.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/org_rbac.py`
- Inspect: `tldw_Server_API/app/core/AuthNZ/orgs_teams.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/admin/admin_rbac.py`
- Inspect: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_org_rbac_scoped_permissions_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_rbac_admin_endpoints.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_rbac_effective_permissions.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_admin_roles_single_user_claims.py`

- [ ] **Step 1: Trace permission resolution from principal to endpoint guard**

Document:
- how claims become permissions
- how org and team scope changes the result
- where explicit deny or override semantics apply

- [ ] **Step 2: Inspect admin endpoint protection for consistency**

Check:
- whether admin routes consistently use claim-first dependencies
- whether any route relies on retired shims or weaker assumptions
- whether single-user behavior is intentionally special-cased or accidentally permissive

- [ ] **Step 3: Review the RBAC and scoped-permission tests**

Capture which invariants are already covered and which risky combinations are untested.

- [ ] **Step 4: Run the focused Stage 4 tests**

Run:
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ_SQLite/test_org_rbac_scoped_permissions_sqlite.py \
  tldw_Server_API/tests/AuthNZ/integration/test_rbac_admin_endpoints.py \
  tldw_Server_API/tests/AuthNZ/integration/test_rbac_effective_permissions.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_permissions_claim_first.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_admin_roles_single_user_claims.py -v
```

Expected: targeted RBAC coverage passes; any failure is tied back to a specific semantic risk before it is called a finding.

- [ ] **Step 5: Write the Stage 4 report**

Rank findings by:
- privilege-escalation risk
- scope confusion risk
- endpoint inconsistency
- test matrix weakness

- [ ] **Step 6: Commit the Stage 4 report**

Run:
```bash
git add Docs/superpowers/reviews/authnz/2026-03-23-stage4-rbac-admin-review.md
git commit -m "docs: record authnz rbac review findings"
```

Expected: one docs-only commit for Stage 4.

### Task 6: Execute Stage 5 Whole-Surface Synthesis

**Files:**
- Modify: `Docs/superpowers/reviews/authnz/2026-03-23-stage5-authnz-synthesis.md`
- Inspect: `Docs/superpowers/reviews/authnz/2026-03-23-stage1-boundary-deps-review.md`
- Inspect: `Docs/superpowers/reviews/authnz/2026-03-23-stage2-login-session-review.md`
- Inspect: `Docs/superpowers/reviews/authnz/2026-03-23-stage3-keys-budget-review.md`
- Inspect: `Docs/superpowers/reviews/authnz/2026-03-23-stage4-rbac-admin-review.md`
- Inspect: `Docs/superpowers/specs/2026-03-23-authnz-review-design.md`
- Test: re-run only disputed or high-signal targeted tests from earlier stages

- [ ] **Step 1: Aggregate all confirmed findings**

Build one master list grouped by:
- boundary wiring
- auth state machine
- programmatic access and budgets
- authorization semantics
- cross-cutting configuration or backend parity

- [ ] **Step 2: Remove duplicate findings and merge related root causes**

Do not count the same issue twice when it appears in multiple stages.

- [ ] **Step 3: Rank the issues by severity and leverage**

Use:
- exploitable security risk
- correctness impact
- blast radius
- ease of safe remediation

- [ ] **Step 4: Define the recommended remediation order**

The final sequence should identify:
- what to fix first
- what tests should be added before or alongside each fix
- which issues can wait until after foundational work lands

- [ ] **Step 5: Re-run only the smallest disputed test set if needed**

Run:
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_auth_deps_precedence.py -v
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py -v
python -m pytest tldw_Server_API/tests/AuthNZ/unit/test_api_key_manager_validation.py -v
python -m pytest tldw_Server_API/tests/AuthNZ/integration/test_rbac_effective_permissions.py -v
```

Expected: run only the one command that matches the disputed finding; do not run all four unless all four are directly relevant.

- [ ] **Step 6: Write the synthesis report**

The final document must include:
- a deduplicated issue list
- severity and leverage ranking
- the recommended remediation order
- the first slice to fix before broader cleanup

- [ ] **Step 7: Commit the synthesis report**

Run:
```bash
git add Docs/superpowers/reviews/authnz/2026-03-23-stage5-authnz-synthesis.md
git commit -m "docs: add authnz review synthesis"
```

Expected: a final docs-only commit containing the ranked remediation view.

### Task 7: Close Out and Handoff

**Files:**
- Modify: `Docs/superpowers/plans/2026-03-23-authnz-sequential-review.md`
- Inspect: `Docs/superpowers/reviews/authnz/README.md`
- Inspect: `Docs/superpowers/reviews/authnz/2026-03-23-stage5-authnz-synthesis.md`
- Test: none

- [ ] **Step 1: Mark each stage status in this plan**

Update each Stage Overview entry from `Not Started` to the correct final status.

- [ ] **Step 2: Verify the review artifact set is complete**

Run:
```bash
ls -1 Docs/superpowers/reviews/authnz
```

Expected: the README, four stage reports, and the synthesis report are present.

- [ ] **Step 3: Summarize the review outcome for the user**

The handoff summary should include:
- the highest-severity findings
- any blocked validation due to environment limits
- the recommended first remediation slice

- [ ] **Step 4: Commit the final plan status update if changed**

Skip this step if the plan file status lines were not updated.

Run:
```bash
git add Docs/superpowers/plans/2026-03-23-authnz-sequential-review.md
git commit -m "docs: finalize authnz review plan status"
```

Expected: no-op if the plan file was not updated; otherwise a final docs-only commit.
