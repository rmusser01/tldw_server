# MCP module authorization: capability checks instead of one admin role

Parent: TASK-13320 (comprehensive core-module code review). Scope: replace the five hand-rolled
`_is_admin` helpers in `core/MCP_unified/` with checks for the capability each operation actually
requires, and name the capabilities that do not exist yet. This document defines a bounded
correction to existing behaviour, not a new capability surface.

Status: proposed. No code change is in scope until the permission names in
[Decisions](#decisions) are approved, because every option moves a cross-user boundary.

## Problem

`core/MCP_unified/` defines `_is_admin` five times, and they disagree:

| Definition | bare-string `roles` | `permissions` count | `system.configure` | dead `is_admin` probe |
|---|---|---|---|---|
| `protocol_types.py:134` `_metadata_has_admin_claims` | yes | `*` only | no | n/a |
| `sandbox_module.py:210` | yes | yes | yes | yes |
| `kanban_module.py:1585` | no, list only | no | no | yes |
| `media_module.py:2058` | no, list only | no | no | no |
| `notes_module.py:2200` | no, list only | no | no | no |
| `mcp_discovery_module.py:283` | yes | no | no (live `user_roles` SELECT) | no |

The review filed this as "four of the five are wrong". That framing does not survive looking at
what each one gates.

### The five sites ask four different questions

| Question | Sites | What admin currently unlocks |
|---|---|---|
| May I read another user's data? | `sandbox_module.py:142`, `media_module.py:2077` | another user's sandbox session; ownership bypass on **every** media access |
| May I destroy data irreversibly? | `media_module.py:1847`, `notes_module.py:2010` | permanent hard delete |
| May I control workflow execution? | `kanban_module.py:1753,1763,1773,1802` | pause, resume, drain a board; force-reassign a claim |
| What scope should I list? | `mcp_discovery_module.py:157` | all catalogs instead of the caller's orgs and teams |

These are not four wrong answers to one question. They are four questions sharing one name. There
is no claim set that is simultaneously correct for "may read another tenant's data" and "may pause
a board", so unifying the five on any single definition necessarily makes some call sites wrong.
That is why they drifted, and why nothing in the code records which is intended.

`mcp_discovery_module.py:157` is arguably not an authorization question at all. It selects a
listing scope; a `scope` parameter with a permission check on the wider scopes expresses that
better than a boolean.

### The role is standing in for the capability

`rbac_seed.py:112` defines the admin role as `"admin": sorted(base)` — it holds every permission.
So "has the admin role" and "holds the permission this operation needs" are equivalent **by
construction** for that role. The modules check the former. That works until someone needs to grant
one capability without granting all of them, which is exactly the state the `system.configure`
divergence describes.

### The permission vocabulary exists and is unused here

`rbac_seed.py:30-53` already seeds resource-scoped names: `media.read`, `media.create`,
`media.delete`, `system.configure`, `users.manage_roles`, `claims.admin`,
`moderation.review.decide`, `notifications.control`. `MCP_unified/auth/rbac.py` defines a
`Resource` × `Action` enum, and `MCP_unified/auth/authnz_rbac.py` provides
`AuthNZRBAC.check_permission(user_id, resource, action, resource_id)` backed by the AuthNZ
database, which fails closed when the database is unavailable.

No module uses any of it for these decisions. All five read raw JWT metadata off the request
context instead, bypassing the RBAC layer one directory over. The single module that greps for
`check_permission` — `web_research_module.py:279` — is an unrelated per-URL egress hook.

### The overload is baked into the mapping, not only the helpers

Adopting the existing RBAC layer as-is would not fix this. `authnz_rbac.py:_map_to_permission`
resolves `Action.ADMIN` to `"system.configure"` for **every** resource:

```python
if action == Action.ADMIN:
    return "system.configure"
```

So `(SANDBOX, ADMIN)`, `(MEDIA, ADMIN)` and `(NOTE, ADMIN)` would all collapse onto the same
permission. `Resource` also has no `SANDBOX` or board member at all — the vocabulary cannot name
two of the four questions above. The mapping has to grow before the modules can adopt it.

### What the review over-stated

Three of TASK-13320's supporting claims did not hold up, and the corrected picture is why this is
a consistency defect rather than a live breach:

1. **The cited test does not exist.** `test_protocol_scope_enforcement.py` is not in the
   repository. It was the only evidence offered for "the protocol accepts the string shape". The
   claim is nonetheless true — `protocol_types.py:113` returns `(value,)` for a `str` — but it was
   verified against a file that is not there.

2. **The bare-string `roles` split is latent, not live.** All three producers are safe:
   `server.py:1486` and `:1592` wrap in `list()`, and `:1615` assigns `token_data.roles`, a
   Pydantic `list[str]` field which rejects a bare string rather than coercing it. A string-shaped
   `roles` claim can only reach the modules from a directly constructed `RequestContext`.

3. **The `system.configure` cross-user reach is narrow.** `_build_role_grants` gives
   `system.configure` to the admin role only; `user`, `moderator`, `viewer` and `reviewer` do not
   receive it. Single-user mode grants it (`User_DB_Handling.py:291`) but also sets
   `roles = ["admin"]` and `is_admin = True`. The set of principals holding `system.configure`
   without already being admin-by-role is empty unless one is granted deliberately as an override.

The severity is therefore lower than filed. The defect is real: five divergent answers on two
cross-user boundaries, with no recorded intent, drifting further with each new module.

### Dead code confirmed

`getattr(context, "is_admin", False)` in `sandbox_module.py:212` and `kanban_module.py:1587` is
dead against the real type — `protocol_types.py` contains zero occurrences of `is_admin`. It fires
only for the `SimpleNamespace` fixtures used in tests, so both copies are partly tested through a
branch production cannot reach.

## Decisions

Each needs approval before implementation, because each moves a boundary.

**D1. Check the capability, not the role.** Each call site asks for the permission its operation
requires. The admin role continues to pass everything, because it holds every permission —
`AuthNZRBAC.check_permission` also short-circuits on the admin role before consulting permissions,
so existing admin access is preserved by two independent mechanisms.

**D2. Name the four capabilities.** Two exist, two do not. Proposed names follow the seeded
`<domain>.<verb>` convention:

| Call site | Required capability | Status |
|---|---|---|
| `media_module.py:1847` permanent delete | `media.delete` | exists |
| `notes_module.py:2010` permanent delete | `notes.delete` | **new** |
| `media_module.py:2077` ownership bypass | `media.read_any` | **new** |
| `sandbox_module.py:142` other user's session | `sandbox.sessions.read_any` | **new** |
| `kanban_module.py` workflow control ×4 | `kanban.workflow.control` | **new** |
| `mcp_discovery_module.py:157` listing scope | `catalogs.read_any` | **new** |

`system.configure` is used for none of them. It means *may configure the system*; that it currently
grants cross-user sandbox access is the clearest symptom of the overload being corrected here.

**D3. Separate the cross-tenant capabilities from the destructive ones.** `media.delete` and
`media.read_any` are deliberately distinct: destroying your own data and reading everyone's data
are different powers, and the current code grants both with one check.

**D4. `mcp_discovery_module.py:157` stops being a boolean.** It becomes a scope selection, with
`catalogs.read_any` required for scopes beyond the caller's own memberships.

**D5. Extend the mapping before adopting it.** `_map_to_permission` gains per-resource ADMIN
entries rather than collapsing them to `system.configure`, and `Resource` gains the members needed
to name sandbox sessions and boards.

**D6. Delete the dead `is_admin` probe** and convert the fixtures that depend on it to set real
metadata claims, so those tests exercise the production path.

## The contract

One shared helper, replacing five:

```python
async def require_capability(
    context: RequestContext,
    resource: Resource,
    action: Action,
    *,
    resource_id: str | None = None,
) -> None:
    """Raise PermissionError unless the caller holds the mapped permission."""
```

Rules it must honour:

- **Fail closed.** An unmapped `(resource, action)` pair denies and logs, as `check_permission`
  already does. A database error denies.
- **No metadata fallback.** The helper does not read `roles` or `permissions` off the request
  context. That is what produced five answers; the database is the single source.
- **Admin role still passes**, via `check_permission`'s existing bypass.
- **Claim shapes are irrelevant** once metadata is no longer consulted, which retires axis 1 of the
  divergence table without a coercion rule.

## Implementation boundaries

**The sync/async boundary is the hard part.** `check_permission` is async and database-backed. Four
of the five current helpers are sync, and `kanban_module`'s are called from `_sync` methods
dispatched through `asyncio.to_thread` from the async entry points at `kanban_module.py:1748`,
`:1758`, `:1768` and `:1798`. Those `_sync` bodies cannot await.
Each such site must either resolve the capability in the async caller and pass the decision into
the thread, or move the check to the async entry point. Resolving it in the async caller is
preferred: it keeps the database call off the worker thread and makes the check visible at the
tool boundary.

**`media_module.py:2077` is on a hot path.** `_assert_media_access` runs on every media access. A
per-call database round trip is not acceptable there; the decision must be resolved once per
request and carried, not re-queried per item.

**Out of scope:** the rest of the MCP authorization surface, the `tools.execute:*` mechanism, and
any change to how the JWT is minted.

## Migration stages

1. Extend `Resource` and `_map_to_permission`; seed the new permissions and grant them to the admin
   role. No call-site change. Behaviour identical.
2. Add `require_capability` and characterization tests pinning current behaviour at all five sites.
3. Convert the two sites that are already async and not hot — `sandbox_module.py:142` and
   `mcp_discovery_module.py:157`.
4. Convert `media_module.py:1847` and `notes_module.py:2010` (destructive, low frequency).
5. Convert `kanban_module` ×4, moving the check to the async entry points.
6. Convert `media_module.py:2077` with per-request caching, and delete the five helpers.

Stages 3 to 6 are independently revertible.

## Verification and release gate

- A parity test asserting that no `_is_admin`-shaped helper remains, as a `tests/lint/` ratchet
  seeded at zero once stage 6 lands.
- Per-stage: a test that the capability is required, a test that the admin role still passes, and a
  test that an unmapped pair denies.
- The two cross-user sites get an explicit negative test: a principal holding `system.configure`
  and not the admin role must be denied at `sandbox_module.py:142`.
- Stage 1 must show no behaviour change: the full `tests/MCP_unified` suite at its pre-change
  baseline.

## Alternatives rejected

**Unify the five on one claim set.** The four options considered were: adopt the sandbox set
(widest), adopt the protocol set, roles-only (narrowest), or document only. All three code options
share the same flaw — they pick one answer for four different questions, so each leaves some call
site holding a permission that does not match its operation. Widening is the least disruptive given
that `system.configure` is effectively admin-only today, but it would grant a `*`-permission holder
ownership bypass on all media, which is a boundary move made by accident rather than by decision.

**Adopt the existing RBAC layer unchanged.** Rejected by D5: `Action.ADMIN` maps to
`system.configure` for every resource, so the five checks would collapse to one permission again,
and `Resource` cannot name sandbox sessions or boards.

**Do nothing.** The divergence has grown once per module added. Two of the five sit on cross-user
boundaries.

## Follow-ups not in scope

- `_metadata_has_admin_claims` in `protocol_types.py` still admits requests on metadata claims. It
  is a different layer with a different job (admitting a request, not authorizing an operation) and
  should be reviewed separately once the module layer no longer reads metadata.
- The `roles`/`permissions` claim-shape inconsistency becomes moot for modules under this design,
  but the protocol layer still accepts both shapes.
