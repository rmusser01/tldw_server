# ADR-068: Explicit PostgreSQL database recovery scope

**Status:** Proposed
**Date:** 2026-10-20
**Backfilled from:** not backfilled
**Decision owner:** Human requester chose whole database, including all users and Jobs, during C09
**Related task:** TASK-13260.278.18.83.45
**Related spec/plan:** Docs/Design/2026-10-20-c09-postgresql-database-recovery.md; [C09 selected closure](https://github.com/rmusser01/tldw_server/blob/14fa8c4a78da39af70df02c15fb2921356d7d200/Docs/Plans/IMPLEMENTATION_PLAN_uat_all_fixes_20261006.md#c09-health-backup-restore-and-maintenance-selected-closure)

## Decision

Expose PostgreSQL database recovery as an explicit global bundle scope covering every SQL object and row in the selected database, with platform-admin admission and an explicit offline overwrite handoff; do not relabel it as a per-user or AuthNZ-only operation.

## Context

ADR-020 distinguishes shared PostgreSQL content from per-user SQLite paths. C09 found that stale local SQLite files could masquerade as live PostgreSQL backups, while an AuthNZ-labelled pg_dump could include co-located content and Jobs. The owner explicitly selected whole-database recovery. Existing narrow-scope refusals remain valid.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Logical per-user export/import | The owner selected whole-database recovery. Tenant-aware export remains a separate capability. |
| Implicitly widen AuthNZ or ordinary dataset scope | Conceals effects on other users and Jobs. |
| Coordinate several independent databases and local file stores | Adds a distributed consistency contract beyond this selected database capability. |

## Consequences

The initial capability requires explicitly configured AuthNZ, PostgreSQL content and Jobs targets to resolve to the same host, port and database. Ambiguous or split deployments fail before artifacts or mutation. A versioned manifest declares the scope and contains one custom-format dump with verified size/hash. SQLite, local media/model files, vectors, cluster roles and external adapters are outside that dump and remain explicitly outside its coverage.

Use the existing DB_Management pg_dump/pg_restore owner and PostgreSQL helper. HTTP wet global import refuses; offline restore requires stopped writers/consumers, an idle-and-active session check, a bounded deadline and an explicit operator-controlled cold restart/Jobs review. Keep global retention separate. A pre-provisioned backup-only DSN may provide complete-row visibility; restore retains the restricted application owner, role/ACL portability and transaction. Do not enable partial-row dumps to bypass RLS failures. Restore only trusted operator archives; database DDL executes with the configured restore principal's rights. A separate isolated target is mandatory for UAT acceptance.

## Follow-up

Keep this record Proposed until implementation, security/source review and real separate-target validation are complete. No existing accepted ADR rationale is changed.
