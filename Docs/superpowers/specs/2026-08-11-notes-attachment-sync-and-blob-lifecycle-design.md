# Notes attachment Sync and blob lifecycle design

- **Date:** 2026-08-11
- **Task:** TASK-13005
- **Status:** Approved for implementation planning
- **Depends on:** TASK-13004 and ADR-031
- **Architecture decision:** `Docs/ADR/038-canonical-notes-attachment-registry-and-blob-lifecycle.md`
- **Reviewed server baseline:** `dev` at `414e81a12aa71df97c4fad17df084aa7a78c474b`

## Purpose

Make Notes attachments participate in the same Sync v2 lifecycle as their owning
notes without creating a Notes-only binary transport. A stable attachment registry
becomes the product authority, `attachment.ref` version 2 carries whole-object
metadata, and the existing resumable Sync blob service remains the binary authority.

Before this design was recorded, the attachment baseline exposed a pre-existing
revision/CAS mismatch: public push discarded `object_revision`, while later append
CAS required that revision. The prerequisite fix is committed separately as
`2ab1e6d67f`. Verification after the fix included 19 attachment-ref tests, 110
endpoint/store tests, and the focused attachment/blob baseline. The only restricted
run failure was a test database write denial; its exact integration test passed when
rerun with the authorized worktree permission.

## Approved product boundaries

- A stable canonical lowercase UUIDv4 identifies an attachment independently of
  its filename and content digest.
- Rename preserves the attachment ID and blob binding.
- Explicit content replacement preserves the attachment ID, advances the
  attachment revision, and binds a newly verified digest after optimistic base
  validation.
- A normal upload whose requested filename is already live allocates the existing
  unique-name suffix and a new attachment ID. It is not an implicit replacement.
- Soft-deleting a note leaves its attachment refs durable, hidden, and retention
  active. Restoring the note reveals the same live refs.
- Manual attachment delete tombstones the ref. It is restored only by an explicit
  attachment restore operation.
- Physical blob deletion occurs only through the existing guarded
  retention/compaction workflow after active-reference, device-ack, audit, and
  restore-window blockers clear.
- When Sync is inactive, existing filename-based filesystem behavior remains the
  compatibility authority. When attachment Sync is active, writes fail closed
  unless attachment readiness and blob transfer are available.
- `dataset_id` is optional. Active Sync resolves omission to the user's one active
  default-personal Notes dataset. Inactive Sync preserves legacy behavior when the
  ID is omitted.
- Existing filename routes remain compatibility aliases. Stable-ID routes are the
  canonical API.
- A one-shot REST upload verifies and commits the blob before publishing a live
  attachment ref. Sync clients may publish a metadata-only or missing ref before
  uploading its blob.
- `attachment.ref` adapter version 2 is a whole-object aggregate. Version 1 remains
  readable inside Sync v2 but is immutable.

## Goals

1. Advertise Notes attachment Sync separately from generic blob transfer and from
   per-dataset readiness.
2. Preserve attachment identity, owner, dataset, note, filename, media type, size,
   digest, lifecycle, provenance, and optimistic ancestry on SQLite and PostgreSQL.
3. Reuse the existing resumable upload, checksum, quota, download, restore,
   diagnostics, and retention machinery.
4. Make upload, attach-from-upload, rename, replacement, delete, restore, note
   trash/restore, and repair idempotent or reviewable under concurrency.
5. Bootstrap legacy filesystem attachments without deleting, rekeying, or trusting
   mutable paths as canonical state.
6. Preserve tenant isolation, path safety, bounded requests, safe errors, and
   secret-free diagnostics.

## Non-goals

- Editing note Markdown when an attachment is inserted.
- Synchronizing arbitrary filesystem paths or sidecar JSON as product truth.
- Implicit content replacement based on a reused filename.
- Automatic attachment restore when its note is restored.
- Automatic attachment tombstone when its note is trashed.
- Cross-dataset aliases to one product row.
- A second blob transport, cloud-object-store implementation, antivirus engine, or
  new garbage-collection daemon.
- Hard deletion of attachment audit history or legacy source files during bootstrap.
- Making adapter-v1 attachment refs writable again.

## Authority and identity model

There are three durable authorities with distinct responsibilities:

1. **Notes attachment registry:** current product identity, ownership, filename,
   metadata, attachment revision, lifecycle, and current blob digest.
2. **Sync envelope and binding state:** immutable mutation history, optimistic
   ancestry, materialization status, and revision-to-blob bindings.
3. **Shared Sync blob store:** verified bytes, upload sessions, chunks, quota,
   availability, storage keys, device acknowledgements, and retention state.

No mutable filesystem filename or `ref_count` is authoritative. Blob liveness is
derived from current attachment refs, immutable attachment-revision bindings, and
retention policy. A storage key never appears in the Notes registry or public API.

Existing `sync_blob_objects` rows are deduplicated by `(dataset_id, payload_hash)`.
Their legacy `attachment_id` column is creation provenance only: it cannot identify
the current attachment after cross-attachment dedupe or replacement. Every v2
download, restore, availability, and retention lookup resolves the current registry
revision through the immutable binding table and then loads the bound blob ID. New
v2 code never selects a current blob by `sync_blob_objects.attachment_id`.

The current local backend's legacy physical key is global by digest even though blob
rows are dataset-scoped. Version 2 therefore gives every dataset a server-generated
opaque `storage_namespace_id` and writes new objects beneath that namespace. It is
never derived from a user path or exposed publicly. Deduplication remains within a
dataset; cross-dataset byte deduplication is deliberately not an authority boundary.
When a v2 operation encounters an existing dataset blob row whose key uses the
legacy global layout, it locks that row, verifies the old bytes, copies them to the
dataset namespace, re-verifies the copy, and CAS-updates only that row's storage key.
This lazy relocation is idempotent and leaves the global object untouched because
other dataset rows may still address it. Version-1 reads follow the updated row and
remain compatible. Task 13005 physical GC may unlink only namespace-version-2 keys;
legacy global objects remain metadata-retention candidates for a later global
inventory/migration and are never unlinked by a dataset-local decision.

The existing `sync_attachments` ciphertext table remains an adapter-v1 compatibility
surface. It is not the Notes product registry and receives no new v2 writes.

The product registry is keyed by `(client_id, dataset_id, attachment_id)`, using the
same authenticated owner column as `notes` and the other ChaChaNotes resources. It
authorizes exactly the owner's one active default-personal Notes dataset. A supplied
dataset must resolve to that same dataset; another same-owner personal/workspace
dataset is rejected. Product rows are dataset-scoped so two histories never alias a
single attachment.

The attachment UUID never changes. `file_name` is the mutable display/storage name;
`original_file_name` is the immutable bounded basename captured at creation.
`normalized_file_name` is derived with the existing path-safe extension policy,
Unicode normalization, and case-folding rules and is unique among live attachments
for one owner/dataset/note. Rename collisions are conflicts. Ordinary upload
collisions continue to allocate `-1`, `-2`, and so on before creating a new ID.

`blob_hash` is exactly `sha256:` followed by 64 lowercase hexadecimal characters at
every v2 boundary. It is the digest of verified logical bytes. `object_hash` is
the SHA-256 digest of canonical JSON containing the attachment identity, note ID,
current and original names, media type, size, blob hash, attachment revision,
creation/mutation provenance, and lifecycle fields. Blob and object hashes are never
interchangeable. Derived availability, `resolved_blob_id`, storage status, and
retention-release state are excluded from `object_hash`, so verifying late-arriving
bytes cannot create semantic attachment drift.

## Canonical product model and schema v59

ChaChaNotes schema version 59 creates an authoritative `note_attachments` registry.
The existing legacy files and `.meta.json` sidecars are not moved by the schema
migration.

| Field | Contract |
| --- | --- |
| `client_id` | Authenticated product owner; never client-selectable |
| `dataset_id` | Canonical active default-personal Notes dataset |
| `attachment_id` | Stable canonical lowercase UUIDv4 |
| `note_id` | Canonical owned Notes UUID; immutable |
| `file_name` | Mutable safe filename, maximum 180 characters |
| `normalized_file_name` | Server-derived live uniqueness key |
| `original_file_name` | Immutable safe basename, maximum 255 Unicode characters and 1,024 UTF-8 bytes |
| `content_type` | Normalized media type, maximum 255 characters |
| `size_bytes` | Positive verified logical byte count |
| `blob_hash` | Full SHA-256 logical-byte digest |
| `object_hash` | Canonical whole-object semantic digest |
| `version` | Positive attachment revision |
| `deleted`, `deleted_at`, `delete_reason` | Explicit attachment tombstone lifecycle |
| `created_at`, `last_modified`, `created_by` | Stable creation and mutation provenance |
| `source_kind` | `upload`, `sync`, or source-verified `legacy_bootstrap` |

Foreign keys use `ON DELETE RESTRICT`; a future hard-note-purge workflow must deal
with attachment history explicitly. Database checks enforce UUID shape, positive
version, bounded metadata, lifecycle coherence, and digest form. A partial unique
index protects `(client_id, dataset_id, note_id, normalized_file_name)` for
live rows.

ChaChaNotes and Sync storage are separate databases, so `dataset_id` cannot have a
cross-database foreign key. The coordinator validates dataset authority and the
revision binding under the dataset guard. The product registry stores the expected
current digest; the Sync binding/blob tables remain authoritative for blob identity,
existence, and integrity.

PostgreSQL enables and forces RLS. `USING` and `WITH CHECK` require the registry
owner to match the authenticated database owner and require the referenced note to
have the same owner. A soft-deleted note remains a valid owned identity. Application
queries keep explicit owner/dataset predicates because schema owners and service
roles may bypass RLS.

Sync storage adds an attachment-revision binding keyed by
`(dataset_id, attachment_id, attachment_revision)`. Its expected digest, size, and
establishing envelope cursor and `availability_at_acceptance` are immutable. A
metadata-only revision starts with no resolved blob ID; availability resolution may
move that pointer exactly once from
null to the verified blob whose dataset/digest/size match. It can never rebind the
revision to another digest or blob. Historical bindings are never rewritten to
follow a replacement, and a soft-deleted blob ID remains recorded for audit/repair.

The immutable identity fields are separate from two monotonic lifecycle fields:

- `resolved_blob_id` moves once from null to one matching verified blob ID; and
- `retention_released_at` moves once from null to the guarded compaction timestamp.

Null `resolved_blob_id` means pending bytes, not an invalid binding. Rename,
tombstone, and restore each create a binding for their new attachment revision. A
rename copies the prior resolved blob ID only when that exact blob remains available;
otherwise its binding stays pending. Late upload completion resolves the current
matching binding first, then a bounded cursor page of unreleased historical pending
bindings (maximum 1,000) by independent CAS; diagnostics and idempotent repair drain
additional pages. Only the binding of the current registry revision controls current
download availability. Replacement uses a new digest and therefore cannot be
resolved by a late upload for an older revision. Retention release never erases the
digest, size, cursor, or resolved blob ID; it only declares that the historical
binding no longer protects physical bytes after all audit/restore/device windows
expire. A later restore creates a new protected binding and must repair/reverify the
blob if the released historical bytes were deleted.

The v59 migration is transactional and additive:

1. Lock schema-version authority and affected Notes relations before inspection.
2. On PostgreSQL, catalog-verify relation/schema ownership and RLS state; temporarily
   relax FORCE only under the existing verified migration-owner pattern and restore
   the exact prior set before version advance.
3. Create the registry, indexes, constraints, and policies; create no attachment
   product rows.
4. Install complete ChaCha RLS after all schema ensures, verify fresh/upgraded
   parity, and advance to version 59.

It never reads arbitrary attachment bytes, rewrites legacy files, guesses mappings,
or deletes rows/files. Any schema/catalog mismatch rolls back. PostgreSQL locking
temporarily blocks affected Notes writes and requires the same explicit operational
approval used by the preceding Notes tenancy migrations.

The Sync-store upgrade is a separate transaction in its own schema authority. It
creates adapter-version cursor and acknowledgment tables keyed by
`(dataset_id, device_id, domain, adapter_version)`, seeds each existing domain cursor
and acknowledgment as adapter version 1, verifies counts and maxima, and only then
marks that migration complete. It retains the old tables for rollback compatibility
until the version-aware readers are deployed. Fresh and upgraded SQLite/PostgreSQL
schemas must produce the same rows, keys, and indexes; a partial seed rolls back.
The same migration creates v2 blob acknowledgments keyed by
`(dataset_id, device_id, blob_id)`, with the immutable digest recorded as verification
evidence. Legacy attachment-ID acknowledgments remain version-1 evidence only and
are not guessed into blob IDs during migration.

During the declared rollback window, every adapter-version-1 cursor/domain-ack write
updates old and new rows in one transaction, and new readers take the monotonic
maximum after validating identity. Startup reconciles either side upward before
serving Sync. Thus an old binary can advance only the legacy row and a later upgrade
repairs the versioned row, while a new binary keeps the legacy row current for an
old-binary rollback. Version-2 state is never projected into legacy tables; rollback
keeps v2 mutation disabled. Removing dual-write/legacy tables requires a later
explicit migration after the rollback window, not this task.

## `attachment.ref` adapter version 2

Adapter version 2 supports `upsert` and `tombstone` with `server_trusted_v1` and
rejects unknown payload fields. It is a whole-object aggregate. The envelope
`object_id` is the stable attachment UUID, `object_revision` is the resulting
product version, and the base cursor/revision/hash tuple is mandatory for every
mutation after creation.

An upsert payload contains:

- `attachment_id`
- `parent_domain: "notes.note"`
- `parent_object_id`
- `file_name`
- `original_file_name`
- `content_type`
- `size_bytes`
- `blob_hash`
- `created_at`
- `last_modified`
- `created_by`

Availability is not a v2 client payload field and unknown fields are rejected. During
preflight, the server resolves the submitted digest/size against the authenticated
blob ledger. The same Sync append transaction creates the revision binding with
immutable `availability_at_acceptance` as `available` or `metadata_only` and sets
`resolved_blob_id` when an exact verified blob already exists. The client payload/hash
and idempotency fingerprint do not depend on that storage observation, so dedupe
timing cannot change envelope identity. Current APIs and restore completeness derive
`available`, `metadata_only`, `missing`, `verify_failed`, `quarantined`, or `deleted`
from binding/blob state rather than rewriting the immutable envelope.

Restore intent is command metadata only. It is the exact
`routing_metadata.restore_intent=true` convention from ADR-031, requires the complete
current tombstone base, and never enters the canonical payload or `object_hash`.

Tombstones retain the complete canonical snapshot plus `deleted_at` and a stable
reason of at most 256 Unicode characters. Names and provenance remain protected
payload fields and never enter routing metadata or error text.

Creation provenance follows the same signed rules as `notes.link`: ordinary client
creation binds `created_at` to normalized `created_at_client` and `created_by` to
the authenticated device; server-origin creation uses the stable server-origin
device; trusted bootstrap alone preserves source-verified legacy provenance.
`last_modified` binds to the mutation envelope timestamp.

Adapter version 1 envelopes stay pullable, restorable, and auditable. New version-1
writes are rejected with a stable immutable-adapter error. Version-1 rows are not
silently rekeyed or promoted into the v59 registry. Capability negotiation tells
new clients that Notes attachment mutation requires version 2. Version-1 metadata
is not assumed to describe a legacy Notes filesystem file, because the old contract
does not provide enough source authority. If a bootstrap-allocated v2 attachment ID
already exists as a version-1 object head, bootstrap fails closed and preserves both
sources rather than replacing that head.

### Adapter-version negotiation and cursor safety

Device registration/bootstrap adds a bounded `supported_adapter_versions` map from
requested domain to a non-empty sorted set of positive versions. Omission means
version 1 only, preserving every existing Sync v2 device. The map accepts at most
100 known domains and at most 8 distinct versions per domain; duplicates, unknown
domains, zero, negative, and oversized values are rejected at the boundary.
Capabilities publish the server-supported versions and, separately, the writable
versions ready for the selected dataset. A device may request `attachment.ref`
version 2 only after that dataset reports `notes_attachment_v2=ready`; push rejects
any version the device did not advertise or the dataset did not make writable.

Pull never filters only by domain. For each selected domain it returns only adapter
versions advertised by that device, so an older Sync v2 client that requested
`attachment.ref` but omitted the new map receives version-1 envelopes only. Durable
device cursors are keyed by `(dataset_id, device_id, domain, adapter_version)`.
Version-aware pull tokens bind the negotiated version set and carry a scan watermark
for each version; a token is rejected with a stable restart-required error if the
device's version set changes. Registering version 2 starts its cursor at zero without
rewinding the version-1 cursor, so the device receives all v2 history and does not
silently skip envelopes hidden by its earlier capability set. Server scans remain
bounded and may advance a version watermark across other versions only after proving
there was no eligible envelope for that version in the scanned cursor range.
Version-1-only devices retain the existing numeric cursor format; the additive opaque
token is required only when more than one adapter-version stream is negotiated.
An encoded token is capped at 32,768 ASCII bytes, its authenticated decoded payload
at 24,576 bytes, and its watermark map at 800 domain-version entries. Oversize is
413; malformed encoding, unknown versions, or a bad signature is 400. Tokens contain
only dataset/device/domain/version IDs, integer watermarks, expiry/version fields,
and a server signature—never attachment metadata.

Device-supported version sets are monotonic-additive for an active registration;
removing a version requires revocation and re-registration. Domain acknowledgment
adds `adapter_version` with an omitted default of 1. The server accepts a monotonic
ack only through the maximum cursor it actually delivered for that exact version;
a client-supplied higher domain sequence is rejected. Retention evaluates the exact
envelope version and can be unblocked only by version-matching acks from active
devices that negotiated that version, plus a v2 blob acknowledgment when bytes are
involved. Version-2 download/restore acknowledges the authorized immutable `blob_id`,
not attachment ID or the blob row's creation provenance. Restore and GC resolve the
attachment revision binding to blob ID before evaluating these acks, so one ack
covers valid same-dataset dedupe and replacement never overwrites evidence for an
older blob. A version-1 domain/blob ack never covers filtered version-2 history.

Version-2 devices may read versions 1 and 2, but no object identity may have heads in
both contracts. A v2 create whose attachment ID already has any v1 head is an
immutable-version collision; the reverse is rejected as an immutable v1 write.
There is no automatic v1-to-v2 promotion or payload reinterpretation.

## Lifecycle and conflict rules

The attachment state is `live` or `tombstoned`. Hidden is derived, not persisted: a
live attachment whose parent note is soft-deleted is hidden on every Notes list,
detail, download, graph, and search surface. It remains an active blob reference.

Blob state transitions are explicit:

```text
uploading -> available
uploading -> verify_failed
metadata_only -> uploading -> available
missing -> uploading -> available
available -> quarantined
verify_failed -> uploading (same-digest explicit repair only)
deleted -> uploading (same-digest explicit repair only)
available -> deleting -> deleted (retention workflow only)
quarantined -> available (explicit administrative release only)
```

Automatic repair never clears quarantine. Same-digest repair retains audit history
and does not create a new attachment revision unless canonical attachment metadata
changes.

Conflict rules are:

- first creation requires no prior attachment head;
- exact replay returns the original result without a second row, envelope, revision,
  quota charge, or blob;
- rename requires exact base state, immutable note/original name, and unchanged blob
  binding;
- replacement requires exact base state and a completed verified blob; it preserves
  attachment identity and advances version/digest;
- delete requires exact base and creates an attachment tombstone without deleting
  blob bytes;
- restore requires exact tombstone base and
  `routing_metadata.restore_intent=true`;
- upsert without restore intent cannot resurrect a tombstone;
- reusing an attachment ID for another note is an immutable-identity conflict;
- a live normalized-name collision is reviewable and never silently renamed on
  PATCH/restore;
- note trash/restore does not change attachment versions or heads; and
- divergent rename/replace/delete/restore races create privacy-safe durable
  conflicts.

Conflict evidence contains attachment ID, note ID, safe codes, revisions, hashes,
and cursors only. It never contains names, local paths, storage keys, bytes,
comments, or tenant details.

## Mutation and crash-recovery flow

One-shot REST upload is blob-first:

1. Resolve owner/dataset, attachment readiness, note identity/live state, limits,
   filename, and idempotency authority before accepting a product mutation.
2. Stream to bounded staging while calculating chunk and full hashes.
3. Verify and commit the shared blob object.
4. Acquire the existing dataset-wide projection guard, reload the parent note,
   attachment/name head, and completed blob binding, and revalidate the full read
   set. The guard is never held while request bytes are streamed.
5. Persist a privacy-safe server-origin mutation manifest whose request fingerprint
   binds owner, dataset, note, normalized name, size, media type, digest, and
   idempotency key hash.
6. Append and materialize the canonical attachment-ref envelope.
7. Return success only after the registry row, revision binding, envelope status,
   and response manifest are durable.

If steps 4–7 fail, the verified unreferenced blob is retained as a bounded cleanup
candidate; retry reuses it. Product success is never reported for a pending or
failed envelope.

`POST .../from-upload` attaches an already completed verified resumable upload to a
note. Insert means this operation; it does not edit note Markdown. The upload
session is immutably bound at creation to owner, dataset, intended attachment ID,
parent note ID, digest, size, media type, create/replace intent, the validated
requested/original name for create, and the exact base tuple for replacement;
`from-upload` cannot reassign it across attachments or notes. The final create suffix
is deliberately allocated only at commit. Rename reuses the current binding.
Replacement consumes a completed upload and establishes a new immutable revision
binding. Delete/restore are metadata-only mutations.

Sync clients may append a metadata-only/missing v2 ref before bytes are available.
Later upload completion changes authenticated blob availability and restore
completeness. If attachment metadata also changes, that change uses a new envelope;
availability alone does not let a client rewrite protected metadata.

Product DB and Sync DB are separate transaction authorities. The durable manifest,
canonical postcondition, and idempotent materializer make every crash window
repairable; the design does not claim distributed atomicity. Deterministic checks
(caps, ownership, immutable fields, base state, blob verification, and name
collision) happen before product writes. Repair recognizes the exact postcondition
without advancing timestamps, revision, quota, or graph state twice.

The parent `notes.note` identity is a guarded read-set dependency for every v2
attachment mutation. After acquiring the dataset guard, the coordinator rechecks
the canonical note head and product row before append/projection. A public create,
rename, or replacement requires a live parent; attachment delete or restore may
target an owned soft-deleted parent and remains hidden. Stable-ID mutation lookup is
owner-authorized against that hidden row even though list, detail, content, graph,
and search reads continue to hide it. Note trash/restore uses the same dataset
guard, so a race has one deterministic order: attachment-first then hidden, or
note-trash-first then no new live attachment. No pre-upload note snapshot authorizes
a later product write.

## Note lifecycle, restore, and retention

Live ref upserts depend on the identity of their `notes.note` parent and order after
the selected note provider in restore plans. An attachment tombstone has no live
parent dependency. Public attachment delete and restore require an owned parent
identity; the parent may be soft-deleted, in which case the mutation is allowed by
stable ID but the attachment remains hidden.

Restore preview and ordered actions include v2 attachment refs, authenticated blob
availability, required hashes, and metadata-only/missing/quarantined status. They
preserve dataset, adapter version, and mutation-group ordering metadata. A request
may include v2 actions only when its registered `device_id` advertises version 2;
device-less and legacy-device previews remain version-1 compatible. A blob can be
restored before or after the ref, but the ref cannot become publicly downloadable
until both its registry state and verified blob are available.

Hidden live refs remain active retention blockers. Tombstoned refs stop being
current product references but their immutable revision bindings and audit history
remain. Existing dry-run/compact blockers continue to require:

- no active attachment ref;
- required device v2 blob-ID acknowledgment;
- expired audit, envelope, tombstone, and offline restore windows; and
- no quarantine or repair hold.

The existing compaction authority currently soft-deletes blob metadata without
removing bytes. The final implementation slice adds an idempotent storage-backend
cleanup seam invoked only after that guarded decision and a second validation of all
blockers. Under the dataset guard and a lock on the deduplicated blob row, compaction
atomically fences the blob `available -> deleting`. Every binding creator/resolver
and upload-completion writer acquires the same blob authority and rejects/retries a
`deleting` or `deleted` target; explicit same-digest repair may begin only after the
delete reaches `deleted`. The storage unlink happens after the durable fence. Success,
including an already-absent storage object after a crash, finalizes `deleted`;
transient failure leaves `deleting` with a safe retry record, and automatic code
never clears that fence back to available. The cleanup seam is not a separate daemon
or attachment DELETE side effect. A shared blob is retained while any live ref or
protected binding targets it, and a mutable counter alone can never authorize
deletion. Before fencing, compaction also proves the storage key belongs to that
dataset's opaque v2 namespace; a legacy global key is a stable nonretryable cleanup
blocker, never an unlink target.

## Legacy bootstrap and cleanup candidates

Attachment readiness is separate from schema readiness and from the six-domain
organization or `notes.link` states. Each eligible dataset has a resumable
`metadata.notes_attachment_v2` record with `initializing`, `ready`, or `failed`, a
stable bootstrap ID, source counts, cursor, and safe failure code.

Under the dataset-row lock, the upgrade ensures `attachment.ref` is enrolled,
creates its domain state when absent, records adapter target version 2, and creates
the stable `notes_attachment_v2` bootstrap record. Existing default-personal
datasets that already have the M1 domain are upgrade candidates, not silently ready;
datasets that enrolled only `notes.note` add this one independent domain without
reopening the six-domain organization group. During initialization, version-1 heads
remain pullable but all new attachment-ref writes fail closed. Adapter-v2 mutation
is advertised for that dataset only after source verification reaches `ready`.

Bootstrap pages the owner’s notes, including soft-deleted notes, in immutable note-ID
order and derives each confined legacy directory from that authoritative identity.
It never infers a note owner or ID by reverse-parsing a directory name. Candidates
within a note use immutable source-key order. For each candidate it:

1. validates the owned note identity and path confinement;
2. reads bounded sidecar metadata without trusting it for ownership;
3. captures source stat/digest, imports bytes through the normal verified blob path,
   and verifies the source again;
4. resolves a durable bootstrap source-map entry; on first sight it allocates and
   persists one real UUIDv4, and every resume reuses that recorded ID;
5. appends a source-verified v2 upsert without rewriting the product twice; and
6. records the owner-root-relative confined source path plus its public-safe hash as
   a non-authoritative cleanup candidate. Only the hash may enter Sync routing,
   diagnostics, or logs.

Bootstrap reads at most 200 notes per page, 1,000 attachment candidates per note,
1,000 candidates per invocation, and 64 KiB from one sidecar. Its serialized durable
checkpoint is at most 64 KiB and each source-key cursor is at most 4,096 UTF-8 bytes.
Crossing a byte/count cap fails the affected source or invocation with a stable safe
code and 413 semantics; malformed cursors/sidecars fail with 400 semantics. It never
materializes an unbounded directory or sidecar before enforcing these caps.

An unstable, missing, oversized, malformed, or ambiguous source remains untouched
and produces a safe recoverable bootstrap blocker. Resume uses the same bootstrap
ID and does not duplicate rows, blobs, quota, or envelopes. Ready is set only after
count and canonical fingerprint verification.

Legacy files are retained through rollout and rollback. Cleanup is never automatic
in this task. A later explicit retention operation may remove a candidate only after
the canonical ref/blob verify, all active devices acknowledge it, restore windows
expire, and the source still matches the recorded fingerprint.

## Capabilities, readiness, and rollout

Two global gates are distinct:

- existing `SYNC_V2_ENABLE_BLOB_TRANSFER` enables generic resumable blob transport;
- new `SYNC_V2_ENABLE_NOTES_ATTACHMENT_SYNC` enables canonical Notes attachment
  routes and adapter-v2 mutation.

Capabilities advertise adapter version 2, operations, blob-transfer features,
download ranges, effective size/chunk/quota limits, bootstrap support, and the
separate Notes attachment feature gate. The active Notes upload limit is
`min(NOTES_ATTACHMENT_MAX_BYTES, SYNC_V2_MAX_BLOB_BYTES)` using each setting’s
effective value. Inactive legacy upload retains its existing Notes limit.

Dataset resolution is:

- active Sync + omitted ID: active default-personal Notes dataset;
- active Sync + supplied ID: exactly that dataset and ready
  `notes_attachment_v2` state;
- inactive Sync + omitted ID + no canonical attachment readiness/rows: legacy
  filesystem behavior;
- inactive Sync + supplied ID: safe conflict rather than silently ignoring it.

Canonical authority is sticky once bootstrap initializes it. If the rollout gate,
blob transfer, or active Sync profile is later unavailable, canonical metadata reads
remain owner-authorized and read-only, content download reports a safe unavailable
state, and all attachment writes fail closed. The service never falls back to a
legacy directory for a dataset that already has canonical readiness or rows; doing
so would hide active-only uploads and create a second mutable authority.

Canonical list/detail routes require authenticated ownership and a dataset whose
canonical authority has completed initialization at least once; they do not require
the mutation or blob-transfer gates. Content additionally requires blob transfer,
the rollout gate, a resolved available binding, and a nonhidden live row. Mutations
require both gates plus current ready state. Before first successful initialization,
partial/initializing/failed canonical reads and every canonical mutation fail closed;
after initialization, a later disabled/failed gate leaves metadata readable but
read-only. Legacy filename routes preserve inactive behavior, but when Sync is active
they resolve to the canonical registry and use the same coordinator.

Rollout order is schema, disabled deployment, bootstrap dry-run/diagnostics,
per-dataset bootstrap, then feature enablement. Rollback disables mutation while
leaving canonical metadata readable; additive registry/binding state and legacy
files remain. Version-2 clients can continue metadata-only reads/sync when generic
blob transfer is disabled, but content download is unavailable and active Notes REST
writes fail closed.

Attachments must contain at least one logical byte. Version-2 registry, envelope,
one-shot, resumable-session, bootstrap, and replacement boundaries all reject
`size_bytes <= 0`, matching the reused blob transport and inactive legacy API.

## API contract

Existing routes stay available:

- `POST /api/v1/notes/{note_id}/attachments`
- `GET /api/v1/notes/{note_id}/attachments`
- `GET /api/v1/notes/{note_id}/attachments/{file_name}`
- `DELETE /api/v1/notes/{note_id}/attachments/{file_name}`

They gain optional dataset resolution and act as compatibility aliases under active
Sync. Canonical additive routes are:

- `GET /api/v1/notes/{note_id}/attachments/canonical`
- `POST /api/v1/notes/{note_id}/attachments/from-upload`
- `GET /api/v1/notes/{note_id}/attachments/by-id/{attachment_id}`
- `GET /api/v1/notes/{note_id}/attachments/by-id/{attachment_id}/content`
- `PATCH /api/v1/notes/{note_id}/attachments/by-id/{attachment_id}`
- `DELETE /api/v1/notes/{note_id}/attachments/by-id/{attachment_id}`
- `POST /api/v1/notes/{note_id}/attachments/by-id/{attachment_id}/restore`

Operation mapping is strict:

- The existing multipart `POST .../attachments` is one-shot create. The server
  allocates the attachment UUID and existing unique-name suffix, streams/verifies
  bytes, and publishes one v2 create.
- Resumable create/replace starts at existing `POST /api/v1/sync/blob-uploads`.
  For `domain="attachment.ref"`, its otherwise generic metadata contains one strict
  `notes_attachment_intent`. Create intent contains `intent="create"`, canonical
  `note_id`, client UUIDv4 `attachment_id` matching top-level object/attachment IDs,
  and `file_name`; the server binds the validated requested/original name, not a live
  final-name reservation. Replace intent contains
  `intent="replace"`, note/attachment IDs, and the exact base cursor, revision, and
  object hash; the server binds the current names and verifies the base before upload.
  Both session forms already bind dataset, owner/device, digest, positive size,
  content type, chunk shape, and idempotency key. Unknown intent fields are rejected.
- `POST .../from-upload` has the strict body `{ "upload_id": "<opaque-id>" }`. The ID
  is the existing server-issued visible-ASCII identifier, 1–128 characters, not a
  client UUID. The route consumes only a completed session whose immutable Notes
  intent matches the path and dataset. Under the dataset guard, create allocates the
  then-current unique suffix and persists that exact final name in the idempotency
  manifest/envelope; competing long-lived uploads therefore preserve ordinary
  same-name suffix behavior without reservations. Create forbids `If-Match`. Replace
  requires `If-Match` equal to both the session's base and the current registry ETag;
  drift is 409 before product mutation.
- `PATCH .../by-id/{attachment_id}` is rename-only with strict body
  `{ "file_name": "..." }`. Content, note, original name, digest, media type, and
  lifecycle fields are forbidden; content replacement always uses `from-upload`.
- `DELETE .../by-id/{attachment_id}` accepts an optional strict JSON body
  `{ "reason": string|null }`; restore uses the same body. Reason is at most 256
  Unicode characters. Delete requires a live exact base; restore requires a
  tombstoned exact base and sets routing restore intent server-side.

The canonical metadata response/list item contains dataset/note/attachment IDs,
current and original names, content type, positive size, lowercase blob hash,
revision, object hash, lifecycle state/timestamps/provenance, derived availability,
and the strong ETag. Mutation responses add `idempotent_replay: bool`; they never
return blob/storage keys. Canonical list defaults to live rows and may explicitly
select `state=live|tombstoned|all`; a parent-hidden row remains absent regardless of
that filter. Tombstoned metadata may therefore be discovered for explicit restore
when its parent is live, while its content remains unavailable.

Canonical `from-upload` and stable-ID mutations require an `Idempotency-Key`;
replace/update/delete/restore additionally require `If-Match` containing the current
revision/object-hash ETag. Exact replay returns the original safe response. A reused
key with different intent is 409.
Every supplied key is 1–128 visible ASCII characters and is hashed before durable
routing/storage; blank, control, non-ASCII, or oversized values are rejected before
work. Canonical keyset cursors are authenticated opaque ASCII strings of at most 512
bytes; malformed is 400 and oversized is 413.

The only accepted strong attachment ETag syntax is
`"att-<lowercase-uuid>-v<positive-revision>-<64-lowercase-object-hash>"`. `If-Match`
must contain exactly one such strong validator; wildcard, weak, or comma-separated
validators are 400. Absence on a required mutation is 428 and a well-formed stale
validator is the reviewable 409 optimistic conflict.

Compatibility filename routes do not make new headers mandatory for inactive
clients. Under active Sync they still use the canonical dataset guard, coordinator,
and append CAS. Supplying `Idempotency-Key` opts them into exact replay; omission
retains the existing same-name-new-upload and already-absent-delete behavior rather
than pretending to provide canonical replay guarantees.

Canonical listing uses attachment-ID keyset pagination, default 50 and maximum 200.
When canonical authority is active, the compatibility list never silently truncates
or paginates: it returns the complete legacy shape up to 1,000 live registry entries,
then returns `notes_attachment_list_requires_pagination` and points clients to the
canonical route. No partial result is labeled complete, and the implementation scans
at most 1,001 rows. Before canonical initialization, inactive filename/filesystem
listing preserves the existing legacy behavior and does not point to an unavailable
canonical route; its pre-existing unbounded directory semantics are not expanded or
claimed as a new bounded contract by this task.

Because the existing `GET .../attachments/{file_name}` path is dynamic, the static
`.../attachments/canonical` route is registered before it. Route-order tests prove
that `canonical` is never interpreted as a filename.

Downloads support `Range`, `Content-Range`, `Accept-Ranges: bytes`, strong ETag,
conditional requests, and bounded streaming. Filename aliases resolve owner/note/live
registry state first. Missing, tombstoned, hidden, quarantined, unauthorized, and
cross-dataset attachments all use nondisclosing 404 responses.

Only one RFC 9110 byte range is supported: `bytes=start-end`, `bytes=start-`, or
`bytes=-suffix_length` with decimal non-negative bounds and positive suffix length.
Comma-separated/multipart or malformed ranges are 400. A satisfiable range clamps
its end to `size-1` and returns 206 with exact `Content-Length` and `Content-Range`;
an unsatisfiable range returns 416 with `Content-Range: bytes */<size>`. No Range
returns streaming 200. A matching `If-None-Match` takes precedence and returns 304
without reading bytes; with `If-Range`, a strong match enables 206 and mismatch
falls back to full 200. Weak validators and multiple conditional ETags are rejected
rather than guessed.

Stable errors are:

- **400:** malformed identity, filename, media type, range, or immutable-field change;
- **404:** absent or unauthorized note/attachment/blob without enumeration;
- **409:** inactive supplied dataset, not-ready state, stale base, name collision,
  restore-required, idempotency drift, a missing/quarantined blob required by a
  mutation, or reviewable conflict;
- **413:** active effective size, chunk, list, request, or quota boundary exceeded;
- **422:** invalid digest, upload binding, or state transition;
- **428:** missing `Idempotency-Key` or `If-Match` on a route that requires it;
- **429:** existing per-user attachment/upload throttling; and
- **503:** transient append, projection, blob storage, or retention-busy state.

Errors and logs expose only safe codes, IDs already authorized to the caller,
bounded counts, retryability, and correlation IDs. They never expose filenames,
paths, storage keys, bytes, hashes not already supplied by the caller, raw DB errors,
or another tenant’s existence.

## Diagnostics and operational recovery

Diagnostics add bounded counts for registry/live/hidden/tombstoned refs, bootstrap
progress, metadata-only/missing/verify-failed/quarantined blobs, orphaned verified
blobs, active uploads, cleanup candidates, retention blockers, and failed/pending
projection. Every count is owner/dataset scoped; samples contain only stable safe IDs
and error codes. Sampling defaults to zero and is capped at 100 entries per category
and 500 total entries per response; higher requested limits are rejected with 413
rather than silently clamped.

Recovery actions are machine-readable and explicit: resume upload, retry verify,
repair projection, resolve conflict, restore attachment, restore note, release
quarantine, or wait for retention. No diagnostic endpoint mutates state.

## Verification strategy

Each implementation slice follows red-green-refactor. Required evidence includes:

- SQLite and server-free PostgreSQL fresh/v58-to-v59 schema, indexes, lock/RLS
  ordering, rollback stages, and concurrent initializer serialization;
- required live PostgreSQL two-owner registry, RLS, identical-filename, range-read,
  and cross-owner write denial evidence in CI, with an honest local skip when no DSN;
- adapter-v1 read-only compatibility and exact adapter-v2 payload vectors;
- product DB constraints, owner/dataset/note authority, filename normalization,
  revision bindings, and digest/object-hash separation;
- one-shot and resumable uploads, chunk/full verification, quota, dedupe, cancellation,
  resume, ranges, corruption, missing storage, and quarantine;
- real coordinator/materializer tests for create, same-name upload, rename, replace,
  delete, restore, metadata-only, exact replay, and crash windows;
- barrier races for rename/replace/delete/restore and duplicate-name creation;
- note trash/restore visibility with unchanged attachment heads/bindings;
- restore ordering, local inventory, missing/quarantined completeness, repair, and
  conflict resolution;
- retention dry-run/compact/physical-GC blockers, device acks, audit/restore windows,
  hidden refs, and immutable binding evidence;
- source-change, interrupted, malformed, oversized, ambiguous, resumed, and exact
  legacy bootstrap plus cleanup-candidate preservation;
- active/inactive/partial/initializing/failed feature-gate and optional-dataset paths;
- legacy filename compatibility, canonical pagination, ETag/If-Match,
  Idempotency-Key, Range, rate limit, path traversal, filename bounds, and safe errors;
- query-plan/query-count and hard-cap evidence rather than wall-clock thresholds; and
- existing Notes attachment, Sync blob, attachment-ref, restore/repair, retention,
  diagnostics, migration, and PostgreSQL contract regression suites.

Touched/new Ruff and formatter scopes, Bandit production scope, compile checks, and
diff checks are recorded exactly. Existing whole-file baselines are reported
separately. Independent spec review precedes planning; independent correctness and
security review precede completion.

## Implementation split

TASK-13005 becomes the parent work item for four atomic, dependency-ordered PRs:

1. **Contract and persistence:** adapter-v2 models/capabilities, ADR/spec, v59
   registry, Sync revision bindings, constraints, ownership, and RLS.
2. **Mutation lifecycle:** coordinator, materializer, canonical and compatibility
   APIs, one-shot/from-upload/rename/replace/delete/restore, and blob-state handling.
3. **Legacy bootstrap:** resumable source-verified import, readiness, rollback-safe
   cleanup candidates, and migration diagnostics.
4. **Restore and operations:** restore ordering/completeness, retention/physical-GC
   evidence, diagnostics, public documentation, regression gates, and final rollout
   verification.

Each child is independently testable and makes no promise that depends on an
uncreated future task. Dependencies are added only after all four child tasks exist.

## ADR check

```text
ADR required: yes
ADR path: Docs/ADR/038-canonical-notes-attachment-registry-and-blob-lifecycle.md
Reason: TASK-13005 creates durable product and Sync persistence, changes the
        attachment identity/conflict/API contract, introduces PostgreSQL RLS and
        migration rules, and defines the authority boundary between attachment
        metadata, immutable revision bindings, and shared blob bytes.
```
