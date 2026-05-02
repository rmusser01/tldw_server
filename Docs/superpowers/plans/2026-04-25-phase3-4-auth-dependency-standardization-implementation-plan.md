# Phase 3.4 Auth Dependency Standardization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking. Do not replace auth dependencies route-wide until an inventory and migration map exist.

**Goal:** Standardize API v1 authentication and authorization dependencies around the claim-first `AuthPrincipal` / `AuthContext` model, while keeping single-user, test-mode, API-key, role, permission, org/team, billing, and setup-access behavior stable.

**Architecture:** Inventory dependency usage first, then migrate route families to a small set of explicit dependency aliases. Keep `get_current_user` compatibility for legacy callers during the transition, but make new/changed routes depend on `AuthPrincipal` or specific guard aliases rather than ad hoc user dictionaries.

**Tech Stack:** FastAPI dependencies, AuthNZ `AuthPrincipal`, pytest, security regression tests, Bandit

---

## Current Inventory

Observed on 2026-04-25:

- `auth_deps.py` already contains the newer claim-first surface:
  - `get_auth_principal`
  - `require_permissions`
  - `require_roles`
  - `require_api_key_scope`
  - `require_service_principal`
  - `require_token_scope`
- Legacy compatibility remains active:
  - `get_current_user`
  - `get_current_active_user`
  - user dictionaries returned from dependencies
  - endpoint-local admin checks
- Other dependency modules add additional auth gates:
  - `billing_deps.require_within_limit`
  - `org_deps.require_org_role` and related org/team guards
  - `setup_deps.require_local_setup_access`
  - route-local guards in `prompts.py`, `setup.py`, `sandbox.py`, and admin endpoint modules
- Several endpoint modules combine `require_roles("admin")`, `require_permissions(...)`, rate limits, token scopes, and manual principal checks in different orders.
- Draft helper contract spec created: `Docs/superpowers/reviews/auth-dependencies/2026-04-25-helper-contract-spec.md`.

## Standard Dependency Surface

Preferred route dependency categories:

- `CurrentPrincipal`: resolves and returns `AuthPrincipal`.
- `RequireRole`: returns `AuthPrincipal` after role checks.
- `RequirePermission`: returns `AuthPrincipal` after permission checks.
- `RequireTokenScope`: validates API-token scope and returns `AuthPrincipal`.
- `RequireOrgRole`: validates org/team role and returns typed scope context.
- `RequireBillingLimit`: validates quota but does not replace identity resolution.
- `RequireSetupAccess`: handles local/setup-only access explicitly and remains separate from normal user auth.

Compatibility rule:

- Existing route behavior must not change during dependency cleanup.
- `get_current_user` remains available for legacy tests and route families that still require dictionary compatibility.
- No endpoint should become more permissive when auth dependencies are reordered or replaced.

## File Structure

- Create: `Docs/superpowers/reviews/auth-dependencies/2026-04-25-auth-dependency-inventory.md`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/org_deps.py` only if the inventory shows duplicated org/team role logic
- Modify: selected pilot endpoint family after inventory
- Create: `tldw_Server_API/tests/AuthNZ/test_auth_dependency_contract.py`
- Add/modify pilot endpoint tests for single-user, multi-user, API-key, role, permission, and test-mode behavior

## Task 1: Build The Dependency Inventory

- [x] Generate a route-family inventory of `Depends(...)` auth dependencies at the static reference level. See `Docs/superpowers/reviews/auth-dependencies/2026-04-25-auth-dependency-inventory.md`.
- [x] Categorize each dependency as identity, role, permission, token scope, quota, org/team scope, setup/local access, or route-local manual check.
- [x] Create a route-by-route auth map for the selected `skills` pilot. See `Docs/superpowers/reviews/phase3-pilots/2026-04-25-skills-pilot-readiness.md`.
- [x] Identify route families with raw user-dictionary signals instead of `AuthPrincipal`. See `Docs/superpowers/reviews/auth-dependencies/2026-04-25-auth-risk-scan.md`.
- [x] Identify endpoints with duplicate or inconsistent admin-check signals at the triage level. See `Docs/superpowers/reviews/auth-dependencies/2026-04-25-auth-special-route-and-admin-triage.md`; per-route inconsistency judgement remains part of each migration slice.
- [x] Identify route families where dependency ordering may matter for quota or audit side effects at the scan level. Endpoint-by-endpoint ordering confirmation remains part of each migration slice.
- [x] Record route families that are intentionally public, setup-local, webhook-secret-only, or provider-compatible at the scan level. See `Docs/superpowers/reviews/auth-dependencies/2026-04-25-auth-special-route-and-admin-triage.md`.

## Task 2: Define Stable Alias Helpers

- [x] Draft the alias/helper contract spec and capture the `require_token_scope(...)` return-type constraint. See `Docs/superpowers/reviews/auth-dependencies/2026-04-25-helper-contract-spec.md`.
- [ ] Add lightweight named aliases or documented helper factories for the standard dependency categories.
- [ ] Keep helpers thin wrappers over the existing `auth_deps.py` functions unless a real duplicate contract exists.
- [ ] Ensure all helpers return the same principal/context type consistently.
- [ ] Add contract tests for single-user and multi-user modes.
- [ ] Add tests that API-key and JWT paths populate `request.state.auth` consistently.

## Task 3: Migrate One Low-Risk Route Family

Recommended pilot candidates:

- `slides` admin-free user-owned endpoints
- `skills` endpoints
- `storage` generated-file list/detail endpoints

Selected planning candidate:

- [x] Draft the `skills` pilot execution packet for alias usage, request-state preservation, and verification gates. See `Docs/superpowers/reviews/phase3-pilots/2026-04-25-skills-pilot-execution-packet.md`.

Pilot steps:

- [ ] Replace ad hoc dependency usage with standard aliases only inside the pilot family.
- [ ] Preserve existing status codes for missing auth, invalid auth, inactive user, forbidden role, and forbidden permission.
- [ ] Preserve audit/quota side effects and request-state population.
- [ ] Add focused tests for JWT, API-key, single-user, and TEST_MODE paths where the family supports them.
- [ ] Run focused Bandit on touched Python paths.

## Task 4: Migrate Admin And Permission-Heavy Families Separately

Admin-heavy route families have higher blast radius because role and permission checks may both be intentional.

- [ ] Inventory admin routes that use `dependencies=[Depends(require_roles("admin"))]`.
- [ ] Inventory admin routes that also call service-layer platform-admin checks.
- [ ] Preserve defense-in-depth checks where services are callable outside FastAPI.
- [ ] Migrate one admin module at a time.
- [ ] Add tests proving non-admin users cannot access each migrated route.

## Task 5: Clean Up Compatibility Only After Route Migration

- [ ] Mark legacy dictionary dependencies as compatibility shims in docstrings.
- [ ] Do not remove `get_current_user` until all direct endpoint consumers are migrated or explicitly exempted.
- [ ] Update AuthNZ docs to direct new route code to `AuthPrincipal`.
- [ ] Add a lint-style inventory test only after the migration is far enough along to avoid excessive churn.

## Verification

Minimum verification before any Phase 3.4 PR:

```bash
python3 -m pytest tldw_Server_API/tests/AuthNZ/test_auth_dependency_contract.py -v
python3 -m pytest <pilot auth endpoint test files> -v
python3 -m bandit -r tldw_Server_API/app/api/v1/API_Deps <touched endpoint files>
```

Run broader AuthNZ SQLite/Postgres suites when touching shared AuthNZ resolution, request-state, org/team role, or setup-access code.

## Out Of Scope

- Changing role/permission policy semantics.
- Removing TEST_MODE support.
- Removing single-user mode.
- Replacing billing quota dependencies.
- Rewriting AuthNZ persistence or migrations.
