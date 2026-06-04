# ACP/AuthNZ/RBAC ADR Candidate Confirmation Audit

**Date:** 2026-06-03
**Backlog:** TASK-519
**Follow-up:** TASK-520

## Scope

This audit confirms whether `INV-023` and `INV-024` are current enough to backfill as accepted ADRs. It does not create accepted ADRs; it records the evidence and limits for the next backfill task.

## Dispositions

| Inventory ID | Disposition | Next action |
| --- | --- | --- |
| INV-023 | Current governing for implemented ACP persistence. | Backfill an ADR in TASK-520 for shared ACP session/registry persistence and per-user orchestration persistence. Do not claim the older setup-guide consolidation work unless separately confirmed. |
| INV-024 | Current governing for core scoped Org/Team RBAC semantics. | Backfill an ADR in TASK-520 for feature-flagged scoped grants, `require_active` default scope mode, admin-level denylist filtering, JWT/API-key/default-membership scope sources, and MCP/tool eligibility. Do not claim missing admin mapping endpoints, resolver metrics, or the older invalid-claim fallback behavior. |

## Evidence Reviewed

### INV-023 - ACP Persistence

The source plan approved moving ACP session state to a shared SQLite database and orchestration state to per-user SQLite databases (`Docs/Plans/2026-03-08-acp-persistence-registry-expansion-design.md`). The current implementation matches the core persistence decision:

- `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py` defines `sessions`, `session_messages`, `agent_registry`, `agent_health_history`, `permission_policies`, and `permission_decisions` tables, with separate message storage and denormalized session token counters.
- `ACPSessionsDB` defaults to `Databases/acp_sessions.db`, uses thread-local SQLite connections, and runs the shared SQLite policy helper.
- `tldw_Server_API/app/services/admin_acp_sessions_service.py` delegates persistent ACP session state to `ACPSessionsDB` while preserving the public session-store API surface.
- `tldw_Server_API/app/core/DB_Management/Orchestration_DB.py` defines a per-user `OrchestrationDB` with projects, tasks, runs, reviews, workspaces, and workspace MCP servers. `OrchestrationDB.for_user()` resolves the user database directory, which defaults to `Databases/user_databases/<id>/`, and stores `orchestration.db` under that directory. Deployments may override the user DB base directory through configuration.
- `tldw_Server_API/app/core/Agent_Orchestration/orchestration_service.py` exposes `get_orchestration_db(user_id)` as an LRU-cached per-user factory, and `tldw_Server_API/app/api/v1/endpoints/agent_orchestration.py` uses that factory for workspace and project APIs.
- Regression coverage exists in `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py`, `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_integration_persistence.py`, `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py`, and `tldw_Server_API/tests/Agent_Orchestration/test_orchestration_db.py`.

Caveat: `orchestration_service.py` still contains a legacy in-memory service class and header text. Current API routes use the SQLite-backed factory, so the ADR should describe the governing persistence path and avoid treating the legacy class as the architecture.

### INV-024 - Scoped Org/Team RBAC

The source design resolved four core decisions: default `require_active` mode, deny admin-level permissions in scoped grants, derive active scope from JWT/default membership rather than request headers, and allow MCP/tool permissions in scoped grants. The current implementation supports those core decisions:

- `tldw_Server_API/app/core/AuthNZ/settings.py` defaults `ORG_RBAC_PROPAGATION_ENABLED` to `False`, `ORG_RBAC_SCOPE_MODE` to `require_active`, and defines an admin-level scoped permission denylist that does not include MCP or `tools.execute:*`.
- `tldw_Server_API/app/core/AuthNZ/migrations.py` creates and seeds `org_role_permissions` and `team_role_permissions`; `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py` carries the PostgreSQL equivalent.
- `tldw_Server_API/app/core/AuthNZ/org_rbac.py` normalizes scope mode to `require_active` by default, resolves org/team membership roles, reads scoped grants from the mapping tables, filters denylisted permissions, and merges scoped permissions with base permissions when propagation is enabled.
- `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py` reads active scope from JWT claims for JWT users, validates it against current memberships, falls back to membership-derived scope when no active claim is present, and applies scoped permissions. The API-key path derives scope from key org/team scope or memberships and also applies scoped permissions.
- `tldw_Server_API/tests/AuthNZ_SQLite/test_org_rbac_scoped_permissions_sqlite.py` covers `require_active` fallback, JWT active-org behavior, admin denylist filtering, and `tools.execute:*` eligibility through `require_permissions`.
- `Docs/Product/Completed/Orgs-virtual-keys-PRD.md` repeats the implemented follow-up semantics: `require_active`, JWT/default-membership scope, no request headers, denylisted admin permissions, MCP/tool eligibility, and active-org team permission behavior.

Caveats for TASK-520:

- The exact admin mapping endpoints listed in the design (`/api/v1/admin/rbac/org-roles/...` and `/team-roles/...`) were not found in the current API surface. The ADR should not claim those endpoints exist.
- Resolver success/failure/latency metrics and an `AuthPrincipal.resolver_failure` flag were not found. The ADR should treat them as follow-up implementation gaps, not accepted current behavior.
- The current JWT path rejects invalid active org/team claims with `403`; it does not implement the source design's default invalid-claim fallback with optional strict mode. The ADR should describe the implemented stricter behavior or omit the invalid-claim fallback.

## Follow-up

TASK-520 should draft two accepted ADRs from this audit:

1. ACP persistence: shared `acp_sessions.db` plus per-user `Databases/user_databases/<id>/orchestration.db` by default.
2. Scoped Org/Team RBAC core semantics: feature-flagged scoped permission overlays with `require_active` default and denylist-filtered grants.

Both ADRs should link this audit and the source design docs. Any missing operational surfaces should remain consequences or follow-up notes, not accepted claims.
