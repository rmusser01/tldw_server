# ADR-016: ACP Session And Orchestration Persistence

**Status:** Accepted
**Date:** 2026-06-04
**Backfilled from:** `Docs/Plans/2026-03-08-acp-persistence-registry-expansion-design.md`
**Decision owner:** Owner sign-off via 2026-06-04 continuation instruction after TASK-520 scope summary
**Related task:** TASK-520
**Related spec/plan:** `Docs/Plans/2026-03-08-acp-persistence-registry-expansion-design.md`, `Docs/ADR/inventory/2026-06-03-acp-rbac-confirmation-audit.md`

## Decision

Persist ACP session, registry, policy, health, and permission-decision state in shared `Databases/acp_sessions.db`, and persist user-owned orchestration state in per-user `Databases/user_databases/<id>/orchestration.db` by default.

## Context

ACP sessions and orchestration work cannot be governed by process-local memory if users and admins need state to survive server restarts. ACP session state is also a shared administrative surface: session cleanup, agent registry, health history, permission policies, and permission-decision audit records are global operational concerns rather than one user's private workspace data.

Orchestration projects, tasks, runs, reviews, workspaces, and workspace MCP server records are user-owned work. The implemented persistence path follows the existing per-user database convention and resolves the user database base directory through configuration, defaulting to `Databases/user_databases/<id>/`.

The accepted current behavior is the bounded persistence behavior confirmed by TASK-519. This ADR does not accept the older setup-guide consolidation claim from the source plan, and it does not treat the legacy in-memory orchestration service class as the governing architecture.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep ACP sessions and orchestration state in memory | Loses sessions, runs, reviews, registry state, and health/audit history on restart. |
| Store all ACP and orchestration state in one shared database | Mixes user-owned orchestration work with global ACP operational state and diverges from existing per-user database ownership patterns. |
| Store ACP sessions per user | Makes global admin visibility, cleanup, registry, health monitoring, and permission policy auditing harder. |
| Keep messages embedded in session rows | Allows unbounded row growth and makes fork slicing by message index harder. |
| Normalize token usage into a separate table | Adds joins to quota and listing paths where denormalized counters are sufficient. |

## Consequences

`Databases/acp_sessions.db` is the shared ACP persistence boundary for sessions, session messages, agent registry data, health history, permission policies, and permission-decision records. ACP session messages live in a separate `session_messages` table, and token usage remains denormalized on session records for efficient quota/status reads.

Per-user orchestration databases remain under the configured user database base directory, defaulting to `Databases/user_databases/<id>/orchestration.db`. Operational guidance and migrations must use the configured base directory rather than hard-coding only the shortened `user_databases/<id>` path.

SQLite settings should follow the implemented project pattern: WAL mode, foreign keys enabled, and short transactions for shared ACP writes.

Any future move to PostgreSQL, a different orchestration ownership model, or a unified ACP/orchestration persistence backend requires a superseding ADR.

## Follow-up

- Use this ADR as the covering record for INV-023.
- Keep setup-guide consolidation, broader registry UX, and any non-implemented ACP registry claims as separate follow-up decisions.
