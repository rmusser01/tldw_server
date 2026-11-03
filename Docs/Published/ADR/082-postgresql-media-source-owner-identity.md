# ADR-082: PostgreSQL Media Source Owner Identity

**Status:** Proposed
**Date:** 2026-10-23
**Backfilled from:** not backfilled
**Decision owner:** UAT requester and implementation review; proposal prepared under authorized defect repair
**Related task:** TASK-13260.278.18.83.45
**Related spec/plan:** `Docs/Plans/IMPLEMENTATION_PLAN_uat_all_fixes_20261006.md#x01-supported-upgrade-selected-closure`

## Decision

Identify an ordinary Media source by its effective owner and stored URL in shared PostgreSQL, while retaining global uniqueness of reserved workspace-clone URLs through publication and recovery.

## Context

SQLite normally places each user's Media in a separate file. Shared PostgreSQL stores independent owners' records together and applies row-level access controls. A globally unique `Media.url` therefore prevents a second owner from storing the same filename, remote URL or generated local identifier, even when the first record is private and invisible. Real restricted-role regressions reproduce the second insertion failing with `UniqueConstraintError`; the live ingestion path reports a warning instead of persisting that owner's source.

Visibility and ingestion identity are distinct. An organization/team grant or Admin authority may allow reading a record, but must not make it an implicit target for another owner's ingestion, touch or overwrite. Intentional shared-record operations continue to address the validated record by ID.

Workspace clone publication has a separate invariant: recovery searches reserved `tldw-clone://workspace/` URLs even after promotion clears operation markers. Relaxing all URL uniqueness, or preserving it only while an operation marker is present, would weaken that proof.

[ADR-020](020-db-management-per-user-paths-and-content-backend.md) governs explicit PostgreSQL deployment and validation. This proposal adds the source-identity rule; it does not rewrite that accepted rationale or alter visibility authorization in [ADR-017](017-scoped-org-team-rbac-core-semantics.md).

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Vary uploaded filenames or rewrite stored URLs per user | Hides the common collision, changes source provenance and leaves other identity paths inconsistent. |
| Reuse any visible record with the same URL | Conflates visibility with ownership and can touch or overwrite another owner's record. |
| Remove URL uniqueness completely | Loses database-enforced same-owner identity and reserved clone recovery guarantees. |
| Keep global uniqueness only for pending operation rows | Promoted clones clear their operation markers but remain part of publication recovery. |
| Merge existing records or reassign legacy ownership | Risks data loss and changes historical identity without reliable provenance. |

## Consequences

- Existing row identity is `COALESCE(owner_user_id::text, client_id, '')`. The empty compatibility key is reserved for existing rows with neither owner nor client; migration never assigns them to a requester. New writes already require a nonempty client. With no explicit owner, a numeric client is converted exactly as insertion derives `owner_user_id`; a nonnumeric client keeps its literal client identity. An explicit owner uses the value insertion persists. New ingestion cannot enter the unresolved compatibility namespace.
- URL lookup, content-hash deduplication, concurrent insertion recheck and canonicalization use that owner namespace even when a caller omits an explicit owner argument. Stored source URLs remain unchanged.
- The database enforces same-owner URL uniqueness. Reserved workspace-clone URLs remain globally unique for pending and promoted rows, alongside the existing operation/source identity constraint.
- Only the schema owner changes this contract, inside the normal migration transaction. Preserve rows, IDs, UUIDs, URLs, histories and ID-based references. Do not merge records or use `DROP ... CASCADE`.
- Ordinary runtime construction validates the new contract and refuses an unmigrated or malformed schema without DDL. Same-owner deduplication, overwrite, deleted-record reservation and cross-worker races require explicit checks.
- SQLite retains its per-file behavior. Sharing visibility, role grants, RLS policies and intentional ID-based mutations retain their existing authorization rules.
- This proposal provides no old-binary rollback guarantee after independent same-URL rows exist. Recover through a separately verified backup/restore procedure.

## Follow-up

Implement and independently review the owner migration, matching runtime validation and all affected lookup/write callers. Validate private and visible-foreign isolation, omitted-owner behavior, same-owner replay/concurrency, legacy/deleted records, pending/promoted clone recovery and unchanged SQLite behavior. Then package and resume the genuine X01 upgrade modes. This proposed record is not a claim that those implementation or acceptance gates have passed.
