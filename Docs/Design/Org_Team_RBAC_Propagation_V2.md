# Org/Team RBAC Propagation (v2)

## Summary
Introduce scoped RBAC propagation so organization/team membership roles map to permission grants and can be enforced per request using an active org/team context. The system remains backward-compatible via feature flags and defaults to legacy (global) permission resolution.

**ADR coverage:** Backfilled by `Docs/ADR/017-scoped-org-team-rbac-core-semantics.md` for implemented core scoped RBAC semantics. Owner sign-off was recorded in TASK-520. The ADR intentionally excludes the unimplemented admin mapping endpoints, resolver metrics/failure flag, and the older invalid-claim fallback behavior from accepted current-architecture claims.

## Goals
- Map org/team membership roles to permission grants without granting platform-wide admin powers.
- Support active org/team scoping for permission checks (opt-in enforcement).
- Keep single-user mode behavior unchanged.
- Preserve existing RBAC tables and permissions while adding scoped overlays.

## Non-Goals (v2)
- Full per-resource ACLs for every domain object (tracked as a follow-up once scoped RBAC is stable).
- Automatic cross-org role propagation beyond explicit membership (no implicit org-to-org role bridging).
- Replacing the global roles/permissions system; scoped RBAC is additive.

## Current State (v1)
- Global RBAC tables: `roles`, `permissions`, `role_permissions`, `user_roles`, `user_permissions`.
- Org/team membership roles stored in `org_members` and `team_members` and used by `org_deps` only.
- `AuthPrincipal.permissions` is computed from global RBAC and used by `require_permissions`.
- `request.state` includes `org_ids`/`team_ids` and optional `active_org_id`/`active_team_id` (JWT claims or membership lookup).

## Design Overview
Scoped RBAC adds a second layer of permission grants derived from org/team membership roles. These scoped grants are merged into `AuthPrincipal.permissions` based on the active org/team context and `ORG_RBAC_SCOPE_MODE`. The default mode is `require_active`, which applies scoped permissions only when an active org/team is available. Deployments should ensure every client has an active scope (for example via a default "no_org" org/team) so scoped checks are always meaningful.

### Feature Flags
- `ORG_RBAC_PROPAGATION_ENABLED` (bool, default: false)
  - Enables org/team role-to-permission propagation.
- `ORG_RBAC_SCOPE_MODE` (str, default: "require_active")
  - `union`: union scoped permissions across all org/team memberships (backward-compatible).
  - `active_only`: apply scoped permissions only for `active_org_id` and (if set) `active_team_id`. When `active_org_id` is set but `active_team_id` is not, include team-level grants for teams in the active org.
  - `require_active`: same as `active_only`, but scoped permissions are only applied when an active org/team is available.
- `ORG_RBAC_SCOPED_PERMISSION_DENYLIST` (list[str], default: admin-level prefixes/names)
  - Denylist of permission prefixes or exact names that cannot be granted via org/team roles. This defaults to admin-level capabilities (system/users/api/billing/security/etc). MCP and `tools.execute:*` remain allowed.

### Validation and Filtering
- Role-permission mappings must exclude denylisted permissions. Any mapping changes should reject denylisted entries with clear validation errors (400) and avoid partial updates.
- Runtime filtering always drops denylisted permissions before merging into `AuthPrincipal.permissions` as a safety net.
- Implementation locations:
  - Role creation/update handlers: admin RBAC endpoints in `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py` (or a dedicated admin RBAC module if split out).
  - Permission-check flow: scoped resolver in `tldw_Server_API/app/core/AuthNZ/org_rbac.py` applies denylist filtering before merging into `AuthPrincipal.permissions`; `require_permissions` in `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py` relies on the filtered set.

### Active Org/Team Resolution
Priority order for active scope:
1) JWT claims: `active_org_id`/`active_team_id`.
2) Default org/team from membership lists.

Active scope is stored in `request.state.active_org_id` / `request.state.active_team_id` and propagated to `AuthPrincipal`.

Validation and selection rules:
- If JWT claims reference an org/team the user is not a member of, ignore the invalid entry by default, log a warning with user ID and claimed scope, and fall back to the next priority source.
- Strict mode: `ORG_RBAC_STRICT_SCOPE_VALIDATION=true` rejects invalid claimed scope with `403` instead of falling back.
- Deterministic membership fallback: select the "first org/team" by role priority (`owner > admin > lead > member`), then creation timestamp (earliest first), then lexicographic ID ascending.

No request headers are used for scope selection in v2; scope is derived from JWT claims or default membership.

## Data Model
Add mapping tables for org/team membership roles to permission grants:

- `org_role_permissions`:
  - `org_role` TEXT (e.g., owner/admin/lead/member)
  - `permission_id` INTEGER
  - PK: (`org_role`, `permission_id`)
  - FK: `permission_id` -> `permissions(id)` ON DELETE CASCADE ON UPDATE CASCADE
  - Indexes: `permission_id` (reverse lookup); composite PK already covers (`org_role`, `permission_id`)
  - Constraint: `org_role` must be one of `owner|admin|lead|member` (enum or CHECK)

- `team_role_permissions`:
  - `team_role` TEXT
  - `permission_id` INTEGER
  - PK: (`team_role`, `permission_id`)
  - FK: `permission_id` -> `permissions(id)` ON DELETE CASCADE ON UPDATE CASCADE
  - Indexes: `permission_id` (reverse lookup); composite PK already covers (`team_role`, `permission_id`)
  - Constraint: `team_role` must be one of `owner|admin|lead|member` (enum or CHECK)

Optional (future, deferred for this release): scoped overrides per user within org/team:
- `org_member_permissions` (`org_id`, `user_id`, `permission_id`, `granted`, `expires_at`)
- `team_member_permissions` (`team_id`, `user_id`, `permission_id`, `granted`, `expires_at`)

### Migration Notes
- Seed default role-to-permission mappings for `owner/admin/lead/member` (aligned with current org/team role semantics) before enabling scoped propagation.
- One-time migration script should populate `org_role_permissions` / `team_role_permissions` from the current defaults so existing memberships immediately resolve to scoped permissions.
- Enforce role validity via enum/CHECK constraints during migration; reject unknown role strings so the mapping tables remain consistent.

## Permission Resolution Algorithm
1) Resolve global permissions from `user_roles` + `user_permissions` (existing behavior).
2) Resolve scoped permissions:
   - Fetch membership roles from `org_members` / `team_members`.
   - Map roles to permissions via `org_role_permissions` / `team_role_permissions`.
   - Drop denylisted permissions via `ORG_RBAC_SCOPED_PERMISSION_DENYLIST`.
3) Apply scoping mode:
   - `union`: union all scoped permissions for all memberships.
   - `active_only`: include org-level permissions for `active_org_id` and team-level permissions for `active_team_id` when set; when only `active_org_id` is set, include team-level permissions for teams in the active org.
   - `require_active`: same as `active_only`, but scoped permissions are only applied when an active scope is available.
4) Merge global + scoped permissions into `AuthPrincipal.permissions` (default union: permission granted if present in either set). A configurable merge strategy can be supported if needed (`union`, `intersection`, `scoped-overrides-global`).

### Performance and Caching
- Cache scoped permission computation (membership fetch + role-to-permission mapping + denylist filtering via `ORG_RBAC_SCOPED_PERMISSION_DENYLIST`).
- Suggested cache keys: per `user_id` for `union`, or `(user_id, active_org_id, active_team_id)` for `active_only`/`require_active` or when active scope affects output.
- TTL guidance: short-lived (for example 30-120 seconds) with optional jitter; use longer TTLs only when membership churn is rare.
- Explicitly invalidate caches when `org_role_permissions` / `team_role_permissions` mappings change or when `org_members` / `team_members` records change (and future per-user overrides, if added).
- Optionally cache computed `AuthPrincipal.permissions` to avoid per-request recompute, using the same keying, TTL, and invalidation triggers.

## API/Service Changes
- Create a scoped permission resolver in `tldw_Server_API/app/core/AuthNZ/org_rbac.py` (new module; do not use `rbac.py`).
- Update `User_DB_Handling.verify_jwt_and_fetch_user` (and API-key auth flow) to:
  - populate active org/team IDs from claims and default membership,
  - call the scoped resolver when enabled,
  - attach scoped permissions to `AuthPrincipal` and the user model.
- Error handling: if the scoped resolver hits a DB timeout, return global-only permissions and set a resolver-failure flag on `AuthPrincipal`; do not fail the request. `AuthPrincipal.resolver_failure` is server-internal only (logged + metrics), not surfaced to clients in responses or headers.
- Observability: emit `resolver_success_count`, `resolver_failure_count`, and `resolver_latency_ms` metrics for the scoped resolver. Alert on sustained failures (e.g., >5% failure rate over 5m) or sustained high latency (e.g., >1m p95 latency over 5m).
- Admin endpoints to manage role-permission mappings are required for production (canonical `/api/v1/` prefix):
  - `GET/POST/DELETE /api/v1/admin/rbac/org-roles/{role}/permissions`
  - `GET/POST/DELETE /api/v1/admin/rbac/team-roles/{role}/permissions`
  - Auth: platform-admin only (401 unauthenticated, 403 unauthorized).
  - Rate limiting enforced for all admin operations (429 on limit).
  - Request/response formats:
    - `GET`: `200 {"role":"<role>","permissions":["perm.a","perm.b"]}`
    - `POST`: body `{"permissions":["perm.a","perm.b"]}`; `200 {"role":"<role>","permissions":[...],"added":["perm.a"]}`
    - `DELETE`: body `{"permissions":["perm.a","perm.b"]}`; `200 {"role":"<role>","permissions":[...],"removed":["perm.a"]}`
  - Error responses:
    - `400` validation failures, `404` unknown role, `409` conflicts (duplicate add/remove missing), `422` schema errors, `500` unexpected errors.
  - Validation rules:
    - `role` path param must be non-empty, lower-case, and match an allowed org/team role name; pattern `[a-z0-9_-]{1,64}`.
    - `permissions` must be a non-empty list, de-duplicated, and each entry must exist in `permissions` table, not match `ORG_RBAC_SCOPED_PERMISSION_DENYLIST`, be lower-case dotted, and contain no whitespace.
- Create/update/delete of role-permission mappings must emit audit events.
- Expose scoped RBAC support via a capabilities endpoint or existing metadata so clients can detect when scoped RBAC is enabled.

## Compatibility & Rollout
- Default flags keep behavior unchanged (no scoped propagation).
- `union` mode can be used for zero-regression adoption.
- `active_only`/`require_active` can be enabled per deployment once clients set active org/team.

## Testing Plan
- Unit tests: scoped permission resolution with union vs active_only.
- Integration tests: admin endpoints update mappings; scoped permissions applied to `require_permissions`.
- Regression: single-user mode unaffected; JWT/API-key auth still works with no active scope.
- Performance tests: load tests measuring cache hit/miss rates; permission resolution latency across varying org/team sizes; DB query latency for role-to-permission lookups used by scoped resolution.
- Security tests: unauthorized access to admin endpoints; abuse of claimed scopes; permission denylist bypass attempts; privilege escalation via role-permission mapping manipulation; verify `require_permissions` enforcement under each attack case.
- Migration tests: seed default role-to-permission mappings then enable scoped propagation to confirm permissions are preserved; rollback from enabled to disabled state to ensure mappings and behaviors revert safely.

## Decisions (Resolved)
- Default scope mode: `require_active`.
- Scoped permissions: allow all except admin-level permissions (denylist).
- Active scope: JWT claims + default org/team membership; no request headers.
- MCP permissions and `tools.execute:*` are allowed in scoped grants.
