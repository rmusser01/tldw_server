# ADR-017: Scoped Org Team RBAC Core Semantics

**Status:** Accepted
**Date:** 2026-06-04
**Backfilled from:** `Docs/Design/Org_Team_RBAC_Propagation_V2.md`
**Decision owner:** Owner sign-off via 2026-06-04 continuation instruction after TASK-520 scope summary
**Related task:** TASK-520
**Related spec/plan:** `Docs/Design/Org_Team_RBAC_Propagation_V2.md`, `Docs/ADR/inventory/2026-06-03-acp-rbac-confirmation-audit.md`

## Decision

Scoped Org/Team RBAC is an opt-in permission overlay that defaults to `require_active`, filters admin-level permissions from scoped grants, derives active scope from JWT, API-key, or membership context rather than request headers, and allows MCP/tool execution permissions in scoped grants.

## Context

The project already has global roles and permissions. Organization and team memberships also carry role information, but those membership roles should not automatically become platform-wide permissions. Scoped RBAC adds a second, feature-flagged layer that maps org/team membership roles to scoped permission grants and merges those grants into the authenticated principal only when scoped propagation is enabled.

The current implementation confirms the core semantics from the source design:

- `ORG_RBAC_PROPAGATION_ENABLED` defaults off for backward compatibility.
- `ORG_RBAC_SCOPE_MODE` defaults to `require_active`.
- Scoped grants are filtered through an admin-level denylist before they merge into principal permissions.
- JWT users derive active org/team scope from claims or default memberships.
- API-key users derive scope from the key's org/team scope or memberships.
- Request headers are not a scope source.
- MCP and `tools.execute:*` permissions are eligible scoped grants.

The accepted current behavior excludes implementation gaps identified by TASK-519: the admin mapping endpoints listed in the source design, resolver metrics/failure flags, and the older invalid-claim fallback behavior are not accepted here as implemented current architecture.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep only global RBAC | Cannot express org/team membership permissions without granting platform-wide capabilities. |
| Enable scoped propagation by default | Risks changing existing authorization behavior for deployments that have not prepared active scope data. |
| Default to union across all memberships | Can grant permissions from inactive org/team contexts and weakens tenant scoping. |
| Allow admin-level permissions in scoped grants | Creates a privilege-escalation path from org/team membership roles into platform administration. |
| Use request headers for active scope | Lets clients influence authorization scope through a weaker and easier-to-spoof channel than authenticated claims or membership-derived defaults. |
| Exclude MCP/tool permissions from scoped grants | Prevents org/team roles from authorizing bounded tool execution even though those permissions are intentionally not in the admin denylist. |

## Consequences

Scoped RBAC remains additive to global RBAC and is controlled by feature flags. Deployments can keep legacy behavior by leaving scoped propagation disabled.

When enabled, `require_active` is the default posture: scoped permissions apply only when an active org/team context is available. Invalid JWT active-scope claims currently fail closed with `403`; this ADR does not accept the source design's older default fallback behavior.

Scoped permission mapping must continue to filter denylisted admin capabilities at runtime. Admin-level permission changes, mapping endpoints, resolver observability, and fallback semantics require separate implementation and, if they become durable architecture policy, a new or superseding ADR.

MCP/tool permissions may be granted through scoped org/team mappings, including `tools.execute:*`, because they are outside the admin-level denylist.

## Follow-up

- Use this ADR as the covering record for INV-024.
- Track admin mapping endpoints, resolver metrics/failure flags, and any invalid-claim fallback change as follow-up implementation or decision work.
