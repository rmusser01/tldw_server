# ADR-086: Owner-scoped PostgreSQL Skill registry

- Status: Proposed
- Date: 2026-11-02
- Task: TASK-13260.278.18.83.45
- Governing decisions: [ADR-020](020-db-management-per-user-paths-and-content-backend.md), [ADR-080](080-chacha-postgres-checkouts-owned-by-entry-points.md)

## Context

SQLite stores each user's Skill registry in their separate content database. Shared PostgreSQL previously used one globally named registry without an owner column. A user's filesystem scan could soft-delete another user's registry row, and equal public names conflicted across accounts. Real restricted-role regressions reproduced both behaviors and foreign reads.

## Decision

Register PostgreSQL schema version91 with a nullable client_id column. Existing rows retain NULL ownership, UUIDs, versions and all other values. Set the normal session-owner default only after adding the column. New rows require the current authenticated checkout owner through forced row-level security, with the same predicate for reads and writes. Scope name uniqueness by client_id/name. Verify this catalog at initialization and registry access; do not repair authority with request-time DDL.

SQLite's per-user-file schema remains unchanged. Public Skill names and API response shapes remain unchanged.

Do not assign historical rows to whichever user starts the server, or infer historic ownership from directory paths: the old unscoped registry could already have been overwritten. Preserve unowned rows for explicit owner-reviewed recovery. Active files can be indexed as new owned entries with new registry identities; this is not a claim of full legacy identity compatibility.

Preserve unknown .purging-UUID directories when no visible registry row proves their disposition. Absence under row-level security cannot establish that a purge committed. Existing known committed-obsolete .cleanup entries retain their established cleanup contract. Ordinary unknown Trash archives and interrupted replacement directories already preserve their contents.

## Consequences

All registry name/UUID/list/count/update/CAS/Trash/restore/purge paths inherit the same owner boundary. Accounts can independently use the same Skill name. Unknown legacy ownership fails closed without deleting registry history or ambiguous archive bytes.

Legacy PostgreSQL Trash recovery requires an operator to review the owner/UUID mapping and canonical filesystem content before any controlled binding. This repair supplies no automatic ownership adoption and does not qualify that recovery procedure. That limitation remains a release/integration consideration; no live user data is rewritten during UAT qualification.

## Validation

Use real restricted PostgreSQL roles for two-owner services, same-name records, cold scans and mutations. Cover migration preservation of NULL-owned rows, missing-context write refusal, catalog drift refusal and byte-preserving archive reconciliation. Reuse SQLite controls and the exact approved Skill's live multi-user execution/restart validation.
