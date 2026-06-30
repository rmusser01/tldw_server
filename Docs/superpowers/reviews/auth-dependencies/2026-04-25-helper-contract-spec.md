# Phase 3.4 Auth Dependency Helper Contract Spec

**Date:** 2026-04-25

**Status:** Draft contract for implementation after PR #1125 and the Phase 2 bases are accepted stable.

## Purpose

Define a small, stable auth dependency surface before migrating endpoint families. The goal is to standardize new route code around `AuthPrincipal` and documented guard helpers without changing single-user, multi-user, API-key, JWT, test-mode, setup, org/team, quota, or webhook behavior.

## Current Constraints

- `auth_deps.py` already provides the claim-first primitives:
  - `get_auth_principal`
  - `require_permissions`
  - `require_roles`
  - `require_api_key_scope`
  - `require_service_principal`
  - `require_token_scope`
- `get_current_user` and `get_current_active_user` remain compatibility shims for dictionary-style route code and tests.
- `get_auth_principal` already honors legacy dependency overrides for `get_current_active_user` and `get_request_user`.
- Successful principal resolution populates `request.state.auth` with `AuthContext` and also attaches legacy state such as `_auth_user`, `user_id`, `api_key_id`, org IDs, team IDs, and active org/team IDs.
- `require_token_scope(...)` is a guard dependency that currently returns `None`, not `AuthPrincipal`.
- Some route families depend on setup-local access, webhook-secret checks, billing quotas, org/team role guards, or service-layer admin checks and should not be folded into a generic user-auth alias.

## Standard Surface

Use these categories for route signatures and dependency lists:

- `CurrentPrincipal`: resolves and returns `AuthPrincipal`.
- `CurrentUserDict`: documented legacy compatibility alias for routes that still require dictionary user payloads.
- `AdminPrincipal`: resolves and returns `AuthPrincipal` after admin role checks.
- `RequireRole`: documented factory category backed by `require_roles(...)`.
- `RequirePermission`: documented factory category backed by `require_permissions(...)`.
- `RequireApiKeyScope`: documented factory category backed by `require_api_key_scope(...)`.
- `ServicePrincipal`: resolves and returns `AuthPrincipal` after service-principal checks.
- `TokenScopeGuard`: dependency-list guard backed by `require_token_scope(...)` and returning `None`.
- `RequireSetupAccess`: remains separate from user auth.
- `RequireOrgRole`: remains separate from normal user auth and returns its existing org/team context.
- `RequireBillingLimit`: remains a quota guard and must not replace identity resolution.

Important correction:

- Do not force `require_token_scope(...)` into an `AuthPrincipal`-returning dependency in Phase 3.4. Routes that need both token-scope validation and identity should use `TokenScopeGuard` plus `CurrentPrincipal`.

## Proposed Alias Implementation

Implementation should be thin and mostly type/documentation oriented.

Recommended additions in `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`:

```python
from typing import Annotated, Any

CurrentPrincipal = Annotated[AuthPrincipal, Depends(get_auth_principal)]
CurrentUserDict = Annotated[dict[str, Any], Depends(get_current_active_user)]

_require_admin_principal = require_roles("admin")
AdminPrincipal = Annotated[AuthPrincipal, Depends(_require_admin_principal)]

ServicePrincipal = Annotated[AuthPrincipal, Depends(require_service_principal)]
```

Documented standard factories:

```python
RequireRole = require_roles
RequirePermission = require_permissions
RequireApiKeyScope = require_api_key_scope
TokenScopeGuard = require_token_scope
```

Notes:

- If uppercase factory aliases are considered too non-idiomatic for this project, keep the existing lower-case factory names and document them as the standard surface instead.
- Avoid wrapping existing factories unless a wrapper adds a real behavior guarantee.
- Keep helper names explicit enough that endpoint signatures reveal whether a dependency returns a principal or only enforces a guard.

## Request-State Invariants

Every route migrated to `CurrentPrincipal` or an auth guard must preserve these invariants:

- `request.state.auth` is an `AuthContext` after successful principal resolution.
- `request.state.auth.principal` is the same principal returned to the route when a principal is returned.
- `request.state._auth_user` remains populated for legacy paths that rely on dictionary user state.
- `request.state.user_id` matches `principal.user_id` when present.
- `request.state.api_key_id` matches `principal.api_key_id` when present.
- Org/team state remains aligned with principal org/team claims.
- TEST_MODE dependency overrides for `get_current_active_user` and `get_request_user` keep working.

## Behavior Preservation

Do not change these semantics in Phase 3.4:

- Missing auth still returns the existing `401` shape and `WWW-Authenticate` behavior.
- Invalid auth still returns the existing `401` behavior.
- Inactive users keep existing status behavior.
- Role and permission failures remain `403`.
- Admin bypass behavior in `require_roles`, `require_permissions`, `require_api_key_scope`, and `require_token_scope` remains unchanged.
- Single-user API key and bearer-as-key compatibility remains unchanged.
- API-key scope validation remains separate from JWT token-scope validation.
- Setup-local access stays separate from normal user auth.
- Webhook-secret routes stay separate from normal user auth.
- Service-layer admin checks remain in place where they protect code callable outside FastAPI.

## Route Migration Contract

For each route-family migration:

- Replace only local dependency spelling with the standard alias/factory surface.
- Do not reorder auth, billing, rate-limit, audit, or quota dependencies unless the route-family review proves ordering is irrelevant.
- Preserve dependency-list guards that exist for side effects or quota enforcement.
- Preserve service-layer admin checks as defense in depth.
- Keep `get_current_user` and `get_current_active_user` for route families that still need dictionary user payloads.
- Add focused tests before and after migration for the route family.

Recommended first pilot:

- `skills`, because the pilot-readiness map already covers response shape, pagination, auth dependencies, frontend callers, and verification targets.

## Test Matrix

Contract tests for aliases/helpers:

- `CurrentPrincipal` returns an `AuthPrincipal` in single-user mode.
- `CurrentPrincipal` returns an `AuthPrincipal` in multi-user JWT mode.
- `CurrentPrincipal` works with API-key auth.
- `CurrentPrincipal` honors `get_current_active_user` dependency overrides.
- `CurrentPrincipal` honors `get_request_user` dependency overrides.
- `AdminPrincipal` allows admin and rejects non-admin.
- `ServicePrincipal` allows service principal and rejects user principal.
- `RequirePermission` preserves AND semantics.
- `RequireRole` preserves OR semantics.
- `RequireApiKeyScope` preserves JWT bypass behavior when configured.
- `TokenScopeGuard` returns `None` on success and preserves existing failure status codes.
- Successful auth populates `request.state.auth`.
- Successful auth populates legacy `_auth_user` and `user_id` state.

Pilot tests:

- `skills` routes preserve existing unauthenticated status codes.
- `skills` routes preserve existing forbidden status codes.
- `skills` routes preserve single-user API key access.
- `skills` routes preserve multi-user JWT access.
- `skills` routes preserve TEST_MODE override behavior.
- Any dependency ordering around quotas, rate limits, or audit markers is unchanged.

## Pending Decisions

- Whether to add uppercase factory aliases or standardize documentation around the existing lower-case factory names.
- Whether `AdminPrincipal` should require the literal `admin` role only or use the existing admin-bypass helper semantics through `require_roles("admin")`.
- Whether to add a lint-style inventory test after enough endpoint families have migrated.
