# Phase 3.4 Auth Dependency Inventory

Date: 2026-04-25

Scope: `tldw_Server_API/app/api/v1`

This inventory is the starting point for Phase 3.4. It counts auth dependency references in endpoint modules and groups the visible dependency styles so route-family migration can happen in small, behavior-preserving slices.

Important caveat: these counts come from static text scanning of endpoint files. They are usage counts, not unique route counts. A single dependency factory can be referenced once in a decorator, once in a function signature, and once in a helper. Before changing any route family, build a route-by-route dependency map from the actual FastAPI decorator and function signature.

## Dependency Pattern Counts

| Pattern | Static references |
| --- | ---: |
| `get_auth_principal` | 553 |
| `rbac_rate_limit` | 362 |
| `require_permissions` | 198 |
| `check_rate_limit` | 184 |
| `require_token_scope` | 134 |
| `require_roles` | 118 |
| `require_within_limit` | 44 |
| `require_local_setup_access` | 17 |
| `require_org_admin` | 15 |
| `require_shared_audio_installer_access` | 7 |
| `require_org_membership` | 6 |
| `require_api_key_scope` | 6 |
| `require_org_owner` | 3 |
| `get_current_active_user` | 1 |

## Dependency Categories

| Category | Current patterns |
| --- | --- |
| Identity | `get_auth_principal`, legacy `get_current_active_user`, route-local user dictionary handling |
| Role and permission | `require_roles`, `require_permissions`, endpoint-local admin checks |
| Token and API-key scope | `require_token_scope`, `require_api_key_scope` |
| Rate and quota | `rbac_rate_limit`, `check_rate_limit`, `require_within_limit` |
| Org/team scope | `require_org_membership`, `require_org_owner`, `require_org_admin` |
| Setup/local access | `require_local_setup_access`, `require_shared_audio_installer_access` |

Phase 3.4 should standardize names and return types without changing these categories. Billing quota, org/team scope, and setup-local access should remain explicit layers rather than being hidden inside a generic "current user" dependency.

## Highest-Mix Route Families

| Route family | Static references | Pattern mix |
| --- | ---: | --- |
| `mcp_hub_management` | 94 | Mostly `get_auth_principal`, plus `check_rate_limit` |
| `writing_manuscripts` | 64 | `rbac_rate_limit` |
| `chat` | 59 | `get_auth_principal`, permissions, token scope, quota, rate limits |
| `slides` | 59 | Permissions plus `rbac_rate_limit` |
| `notes` | 58 | `rbac_rate_limit` |
| `claims` | 46 | Principal, roles, permissions |
| `admin/admin_ops` | 36 | Principal-heavy admin operations |
| `data_tables` | 34 | Principal, permissions, rate limits |
| `orgs` | 34 | Principal plus org membership/admin/owner gates |
| `setup` | 31 | Principal, roles, permissions, setup-local, shared-audio installer access |
| `workflows` | 27 | Roles, permissions, token scope |

These families are useful for understanding the current spread, but most are poor first pilots because dependency ordering may affect rate-limit, quota, audit, or setup behavior.

## Migration Risk Notes

- `chat` mixes identity, token scopes, quota checks, and rate limits. Treat it as a later migration after helper aliases and contract tests are stable.
- `slides` is a tempting cross-phase pilot because it appears in the response-envelope and pagination inventories, but its permission and rate-limit mix means Phase 3.4 should not use it as the very first auth cleanup unless tests already cover the relevant deny paths.
- `data_tables` has the same cross-phase appeal, but combines principal, permission, and rate-limit dependencies.
- `orgs` should be migrated separately from generic auth cleanup because org/team scope semantics are distinct from user identity semantics.
- `setup` should remain a separate access model. Local setup and shared-audio installer guards are intentionally not normal user-auth dependencies.
- `mcp_hub_management`, `notes`, and `writing_manuscripts` have high reference counts and should wait until the standard alias shape is proven elsewhere.
- Admin modules should be migrated one module at a time and keep service-layer defense-in-depth checks where those services are callable outside FastAPI.

## Pilot Recommendation

Start Phase 3.4 with a route family that has:

- few dependency styles
- no org/team scope
- no setup-local access
- no streaming or file response behavior
- focused endpoint tests for unauthorized, forbidden, and allowed paths

Recommended first candidates:

1. `skills`: smallest cross-phase candidate if its route-level auth scan confirms it has limited dependencies.
2. `storage`: candidate for generated-file list/detail endpoints after verifying file-download routes are excluded.
3. A narrow non-admin submodule with only `get_auth_principal` plus one guard, chosen after route-by-route dependency mapping.

Avoid `chat`, `setup`, `orgs`, `admin/*`, and `mcp_hub_management` for the first Phase 3.4 pilot.

## Next Action

Build a route-by-route auth map for the first candidate family before code changes. The map should capture:

- decorator-level dependencies
- signature-level dependencies
- dependency return type expected by route code
- manual role, permission, or admin checks inside the route body
- quota/rate-limit dependencies whose ordering might produce side effects
- public, setup-local, webhook-secret-only, and provider-compatible exemptions

Only after that map exists should Phase 3.4 introduce standard alias helpers or migrate a pilot family.
