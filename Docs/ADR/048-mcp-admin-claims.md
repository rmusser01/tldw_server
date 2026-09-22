# ADR-048: One MCP admin predicate, with permissions narrower than AuthNZ

**Status:** Accepted
**Date:** 2026-09-22
**Backfilled from:** not backfilled
**Decision owner:** repository owner (decided 2026-09-22 during core-module review remediation)
**Related task:** TASK-13338
**Related spec/plan:** `Docs/superpowers/reviews/2026-09-21-core-module-duplication-synthesis.md` (F37)

## Decision

MCP has **one** administrator predicate, `protocol_types.metadata_has_admin_claims`,
reached by modules through `BaseModule.caller_is_admin`.

A caller is an MCP administrator when either:

- a role claim is in AuthNZ's `_PLATFORM_ADMIN_ROLES` — `admin`, `owner`, `super_admin`
  — imported, not restated; or
- a permission claim is the `*` wildcard.

Permissions are **deliberately narrower than AuthNZ's `_ADMIN_CLAIM_PERMISSIONS`**:
`system.configure` and the `admin` permission do **not** make an MCP administrator.
That is the only intentional divergence.

## Context

Six `_is_admin` definitions had grown independently, with six different claim sets,
gating a cross-user resource check and two irreversible deletes:

| Site | roles accepted | permissions accepted | probes `context.is_admin` |
| --- | --- | --- | --- |
| `media_module` | `admin`, list only | — | — |
| `notes_module` | `admin`, list only | — | — |
| `kanban_module` | `admin`, list only | — | yes |
| `sandbox_module` | `admin`, str or list | `*`, `system.configure` | yes |
| `mcp_discovery_module` | `admin`, str or list | — | — |
| `protocol_types` | `admin` | `*` | — |
| **AuthNZ** | `admin`, `owner`, `super_admin` | `*`, `system.configure`, `admin` | — |

They disagreed in both directions.

**Under-grant.** A principal with the AuthNZ role `owner` connects; `server.py:1486`
writes `metadata["roles"] = ["owner"]`. Every MCP predicate tested the literal `admin`,
so a platform **owner** was refused permanent media delete, permanent note delete and
every kanban policy operation, while being an administrator everywhere else in the
product. Same for `super_admin`.

**Over-grant.** An API key normalised into permissions containing `system.configure`,
with no admin role, was an administrator per `sandbox_module` and therefore passed the
**cross-user** session gate at `sandbox_module.py:142`, reaching another user's sandbox
session — while `protocol_types._metadata_has_admin_claims`, the predicate MCP used for
its own trusted-claims gate, returned `False` for that same caller.

**Dead probes.** `kanban_module` and `sandbox_module` both read
`getattr(context, "is_admin", False)`. `RequestContext` defines no such attribute and
`server.py:1483-1487` drops `principal.is_admin` when building it, so the probe was
always `False`: both quietly ran the roles-only logic they appeared stricter than.

## Alternatives considered

**Full parity with AuthNZ `_claims_mark_admin`.** One predicate, zero divergence, and
the obvious choice on consistency grounds. Rejected: it would newly grant permanent
media delete, permanent note delete and kanban policy operations to any caller holding
`system.configure`, and widen the trusted-claims gate to match. A configuration
permission should not authorise destroying another user's data.

**Roles only, ignoring permissions entirely.** Strictest, and closes the over-grant most
aggressively. Rejected: `*` genuinely means every permission, and both `protocol_types`
and `sandbox_module` honour it today; dropping it is a wider breaking change than the
problem needs.

## Consequences

- Platform `owner` and `super_admin` gain permanent media delete, permanent note delete,
  kanban policy operations and sandbox cross-user access — they had these everywhere
  else already. This is the under-grant fix.
- **Breaking:** an API key whose only admin-ish claim is `system.configure` **loses**
  cross-user sandbox session access. Any deployment relying on that must grant one of
  the three platform admin roles instead. This is the over-grant fix, and it is the one
  behaviour change that can break a working configuration.
- MCP's role set is imported from AuthNZ, so adding a platform admin role there reaches
  MCP with no second edit.
- The permission divergence is load-bearing and must not be "tidied up" into parity
  without revisiting this ADR. `test_admin_claims_matrix.py` pins both halves, with the
  reason on each row.

## Follow-up

`_PLATFORM_ADMIN_ROLES` is spelled out identically in four places under `app/core`
(`AuthNZ/auth_principal_resolver.py`, `AuthNZ/byok_helpers.py`,
`Claims_Extraction/claims_service.py`, and now imported by MCP). Collapsing those is
tracked separately as TASK-13345 — the same divergence risk, one level up.
