# TASK-12020.42 Authenticated User Scope Propagation

## Problem

JWT and API-key authentication validate organization and team membership, apply
scoped permissions, and store the resulting scope on `request.state` and
`AuthPrincipal`. The legacy `User` object returned by `get_request_user` does not
declare or receive those fields. Consumers such as workspace sharing therefore
observe empty membership even when authentication has already validated it.

Live PostgreSQL UAT reproduced the failure: an owner token contained `org_ids`
and the owner could list the organization, but creating an organization-scoped
workspace share returned `403` because `user.org_ids` was absent.

## Decision

Keep `AuthPrincipal` as the canonical authorization context while preserving the
validated scope on the established `User` compatibility model. Add:

- `org_ids: list[int]`
- `team_ids: list[int]`
- `active_org_id: int | None`
- `active_team_id: int | None`

Populate these fields only after membership validation and
`apply_scoped_permissions` complete. Apply the same rule to JWT and API-key
authentication. Do not derive scopes from unvalidated token claims, widen an
API key's scope, or change sharing endpoint authorization logic.

## Alternatives Considered

1. Read scope from `request.state` only in sharing. This is a smaller endpoint
   patch but leaves the returned-user contract inconsistent and does not fix
   other compatibility consumers.
2. Migrate sharing entirely to `AuthPrincipal`. This is a valid later cleanup,
   but it changes several endpoint dependencies and is too broad for the live
   UAT blocker.

## Data Flow

1. Decode and validate credentials.
2. Load current memberships and reject stale or unauthorized scope claims.
3. Resolve effective permissions and active scope.
4. Assign the effective scope to `User`, `request.state`, and `AuthPrincipal`.
5. Existing sharing code reads the same validated values from `User`.

## Error Handling

Existing behavior remains unchanged. JWT membership lookup failures and stale
claims continue to fail closed with `403`. Explicit API-key scopes outside the
current membership also continue to return `403`; an unscoped API key retains
the existing fallback behavior when membership lookup is unavailable. This
task does not change those policies. The new model defaults are empty/`None`,
so single-user and test callers remain backward compatible.

## Verification

- Focused unit tests assert JWT returned-user scope and stale-claim rejection.
- AuthNZ happy-path tests assert JWT and API-key `request_user` scope matches
  `AuthPrincipal` and `request.state`.
- Existing sharing tests retain allowed and denied scope coverage.
- Live PostgreSQL UAT repeats organization sharing, recipient listing, and
  shared-workspace metadata retrieval before the WebUI CDP walkthrough.
