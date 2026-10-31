# ADR-038: Canonical Notes attachment registry and shared blob lifecycle

- **Status:** Accepted
- **Date:** 2026-08-11
- **Task:** TASK-13005
- **Depends on:** ADR-031, ADR-034, and ADR-037
- **Design:** `Docs/superpowers/specs/2026-08-11-notes-attachment-sync-and-blob-lifecycle-design.md`

## Context

Notes attachments are currently filename-addressed files plus mutable JSON sidecars.
They have no stable product identity, optimistic lifecycle, dataset authority, or
PostgreSQL tenancy contract. Sync v2 already has metadata-only `attachment.ref`
version 1, verified resumable blob upload/download, quota, restore completeness,
device acknowledgement, diagnostics, and retention foundations, but Notes REST
routes do not use them as one lifecycle.

Using filenames as identity would make rename destructive and content replacement
ambiguous. Storing Notes bytes in a second transport would duplicate integrity,
quota, restore, and garbage-collection policy. Treating a mutable `ref_count` as
deletion authority would make crash recovery unsafe.

## Decision

1. ChaChaNotes schema version 59 adds an owner/dataset/note-scoped canonical
   attachment registry with stable UUID identity, optimistic revision, soft-delete
   lifecycle, safe filename metadata, separate blob/object hashes, and forced
   PostgreSQL RLS.
2. `attachment.ref` adapter version 2 is the writable whole-object Notes attachment
   contract. Adapter version 1 remains readable/restorable but immutable. Devices
   negotiate supported versions per domain, omission means version 1, and durable
   pull cursors and domain acknowledgments are adapter-version scoped so capability
   upgrades cannot skip, acknowledge, or expose incompatible envelopes.
3. The existing shared Sync blob ledger/store is the sole binary authority. Notes
   stores the current expected digest, while Sync bindings preserve every immutable
   attachment-revision digest/size relationship and monotonically resolve it to one
   exact verified blob ID. A separate monotonic retention-release marker can stop a
   historical binding from protecting bytes without erasing its audit identity.
   Deduplicated blob rows are resolved by binding/blob ID, never by their legacy
   attachment provenance column; the existing `sync_attachments` table remains
   version-1 compatibility only.
   New v2 physical objects use an opaque per-dataset storage namespace. Existing
   global-digest keys are lazily verified and relocated per dataset; dataset-local
   GC never unlinks a legacy global key that another dataset row may share.
   Version-2 blob acknowledgments are keyed by immutable blob ID, not attachment ID
   or the blob row's creation provenance.
4. Rename preserves attachment and blob identity. Explicit replacement preserves
   attachment identity while advancing revision and binding a verified new digest.
   Ordinary same-name upload creates a new attachment with the compatibility suffix.
5. Note trash hides live attachments without changing their refs, revisions, or blob
   retention. Stable-ID delete and restore remain available for an owned trashed note
   without revealing the attachment through read surfaces; delete is an explicit
   tombstone and restore is explicit.
6. One-shot REST upload commits verified bytes before publishing a ref. Sync clients
   may create metadata-only/missing refs. Durable manifests and canonical
   postconditions repair cross-database crash windows without claiming distributed
   atomicity.
7. Active Notes attachment writes require generic blob transfer, the dedicated
   `SYNC_V2_ENABLE_NOTES_ATTACHMENT_SYNC` rollout gate, and ready per-dataset
   `notes_attachment_v2` state. Inactive omitted-dataset requests preserve legacy
   filesystem behavior only before canonical attachment authority is initialized;
   canonical authority never silently falls back to mutable legacy files.
8. Stable-ID APIs are authoritative; existing filename APIs are compatibility aliases.
   Canonical mutations use Idempotency-Key and optimistic ETag/If-Match. Downloads
   support one single byte range with bounded streaming. Canonical metadata reads
   remain available after initialized authority becomes read-only; content and
   mutation gates are evaluated separately.
9. Schema v59 creates structures only. Runtime bootstrap source-verifies and imports
   legacy files through the normal blob path, retains the originals as
   non-authoritative cleanup candidates, and never performs automatic destructive
   cleanup.
10. Blob deletion is authorized only by current refs, immutable bindings, device
    acknowledgement, audit/restore windows, quarantine/repair holds, and the existing
    retention workflow. The storage-backend byte-removal seam is invoked only after
    guarded blocker revalidation and an atomic `available -> deleting` fence shared
    with every binding/upload-completion writer; retry finalizes `deleted` after
    unlink and mutable counters alone never authorize deletion.
11. The active size ceiling is the minimum of the effective legacy Notes attachment
    limit and advertised Sync blob limit, and size must be positive. Blob digests are
    canonical lowercase SHA-256. Legacy inactive behavior keeps its current limit.
12. TASK-13005 is implemented as four atomic PRs: persistence contract, mutation
    lifecycle, legacy bootstrap, then restore/retention/diagnostics and rollout.

## Operational completion and failure semantics

The completed implementation applies the decisions above as one fail-closed
lifecycle:

- version-2 capability advertisement is dataset-bound and writable only when
  blob transfer, the dedicated default-off rollout gate, encryption, enrollment,
  and `notes_attachment_v2: ready` all agree;
- canonical create, rename, replace, delete, and restore are idempotent,
  optimistic mutations. Product projection and accepted-envelope history share
  the materialization fence; injected projection or binding failures roll back
  the accepted envelope rather than publishing partial authority;
- restore ordering requires the owning Note before a live attachment ref, and
  completeness is derived from immutable revision bindings and verified bytes;
- legacy bootstrap is resumable and source-verified, publishes readiness only
  after count/postcondition verification, retains source files for rollback,
  and exposes only bounded hashed diagnostics;
- historical binding release is monotonic and auditable. Physical collection
  revalidates all blockers under the dataset fence, uses
  `available -> deleting -> deleted`, and leaves retryable `deleting` state when
  byte removal cannot be proven complete; and
- diagnostics are read-only hints. Stable public errors and logs omit filenames,
  source paths, storage keys, blob bytes, payloads, and secret metadata.

No additional ADR is required for these operational details: they directly
implement Decisions 2, 3, 6, 8, 9, and 10 without changing the ownership,
storage, migration, or retention architecture selected here.

## Consequences

- Users can rename, replace, delete, restore, and synchronize an attachment without
  changing its stable identity or losing its audit history.
- Existing resumable transfer, quota, integrity, restore, diagnostics, and retention
  machinery is reused rather than duplicated.
- A note can be trashed without making its attachments collectible; restore reveals
  the same live refs.
- PostgreSQL tenants can use the same filenames and cannot enumerate or bind another
  owner's note, attachment, or blob.
- Enabling attachment Sync requires a resumable runtime bootstrap even after the
  database schema is current.
- Disabling the rollout gate after activation leaves canonical metadata readable but
  makes mutations fail closed; it does not restore legacy files as authority.
- Verified but temporarily unreferenced blobs may exist after a failed cross-database
  projection; retention handles them safely.
- Version-2 physical storage deduplicates within a dataset rather than across tenants;
  this trades some storage efficiency for a deletion boundary that can be locked and
  proven locally.
- PostgreSQL schema migration temporarily blocks affected Notes writes and requires
  explicit operational approval.
- Legacy files consume extra disk until an explicit later cleanup operation satisfies
  all retention evidence.

## Alternatives considered

### Keep filename as attachment identity

Rejected because rename becomes delete/create, same-name replacement is ambiguous,
and compatibility suffixes cannot represent durable multi-device identity.

### Store Notes attachment bytes in a new Notes-only transport

Rejected because it duplicates resumable transfer, checksums, quota, restore,
diagnostics, encryption metadata, and garbage-collection policy.

### Put all attachment metadata only in Sync storage

Rejected because inactive Notes behavior and owner-authorized product reads need a
current product authority; deriving every list/detail request from envelope history
would be slower and would split compatibility behavior.

### Keep global digest paths and scan every tenant before unlink

Rejected because a dataset-local retention transaction cannot lock all present and
future cross-tenant references without adding a second global physical-object
authority. Opaque per-dataset namespaces retain the existing backend and make the
deletion boundary explicit; legacy global objects remain untouched.

### Rewrite or delete legacy files during schema migration

Rejected because schema migration must remain bounded, transactional, and reversible.
Byte import is a resumable runtime process and originals remain rollback evidence.

### Cascade attachment tombstones when a note is trashed

Rejected because note trash is a visibility change, not attachment intent. Cascading
would create large mutation groups, release retention too early, and require
synthetic restores.

### Use `ref_count` as physical deletion authority

Rejected because counters can drift across crashes and repairs. Current refs,
immutable bindings, acknowledgements, windows, and holds provide auditable evidence.
