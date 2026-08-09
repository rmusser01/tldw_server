# Notes organization Sync v2 design

- **Date:** 2026-08-08
- **Task:** TASK-13003
- **Status:** Approved design; implementation planning and execution remain separate gates
- **Depends on:** TASK-13002 and ADR-031
- **Architecture decision:** `Docs/ADR/032-durable-server-origin-sync-mutation-batches.md`
- **Reviewed server baseline:** `dev` at `a495e252c1319a6e44c20a259e92fa94c0107627`

## Purpose

Make Notes keywords, keyword collections, folders, and their user-visible
memberships first-class Sync v2 state. The result must preserve organization across
offline and multi-device use while keeping existing integer REST identifiers
compatible and preventing partially synchronized writes.

This task supplies the server contract needed for Chatbook Notes parity. It does
not claim full Notes parity by itself; later capability tasks cover manual links,
attachments, tasks, activity, moodboards, and Studio documents.

## Current `dev` reference

The latest reviewed `tldw_server` `dev` revision already provides the correct
single-object reference path:

- `notes.note` has a typed capability schema and strict payload validator;
- `NotesDomainAdapter` evaluates base lineage, delete/update conflicts, and exact
  tombstone restores;
- `NotesMaterializer` projects accepted envelopes into the per-user ChaChaNotes
  database and records apply failures; and
- normal note REST mutations can capture a canonical server-origin envelope.

The organization implementation will extend those patterns rather than the generic
`media.keyword` compatibility adapter. Notes has its own integer foreign keys,
hierarchies, effective memberships, optimistic versions, and per-user materialized
store.

The same review found that the Sync store has only `insert_envelope()`. Compound
organization operations therefore require ADR-032's atomic batch-append seam before
they can safely use Sync as the write authority.

Focused baseline verification on the isolated task worktree passed before design
changes:

```text
45 passed, 5 warnings in 42.76s
```

The command covered keyword storage, note folders, and server-origin Sync capture.

## Goals

1. Advertise six independently mutable, version-1 Notes organization domains.
2. Give keyword, collection, and folder resources stable opaque cross-device IDs
   while preserving integer database and REST IDs.
3. Represent memberships as deterministic first-class objects with explicit
   upsert and tombstone history.
4. Route active-Sync REST writes through the same strict adapters, canonical
   envelope store, and materializers as client-origin writes.
5. Make compound writes durable, ordered, idempotent, and resumable without
   claiming cross-database atomicity.
6. Detect hierarchy, ownership, uniqueness, delete/update, and unsynchronized
   dependency conflicts before silent data loss.
7. Include every domain in restore preview, repair, and public capability docs.

## Non-goals

- Synchronizing flashcards or `flashcard_keywords`.
- Synchronizing source-import provenance tables such as
  `note_folder_source_memberships` and `note_folder_source_keys`.
- Synchronizing FTS rows, counts, timestamps used only for local display, folder
  paths, or other derived indexes and projections.
- Replacing existing integer keyword, collection, or folder IDs in REST routes.
- Adding a compound `merge` operation to the Sync wire protocol.
- Implementing later Notes parity domains.

## Canonical domains

All six domains use `server_trusted_v1`, schema version 1, and `upsert` plus
`tombstone` operations. Payload validators reject unknown fields.

| Domain | Canonical entity ID | Upsert payload | Tombstone payload |
| --- | --- | --- | --- |
| `notes.keyword` | Keyword `sync_id` | `keyword`: non-empty string, at most 100 characters | Empty |
| `notes.keyword_link` | Deterministic link ID | `subject_type`: `note` or `conversation`; `subject_id`; `keyword_sync_id` | The same three identity fields |
| `notes.keyword_collection` | Collection `sync_id` | `name`: non-empty string, at most 255 characters; `parent_sync_id`: string or null | Empty |
| `notes.keyword_collection_link` | Deterministic link ID | `collection_sync_id`; `keyword_sync_id` | The same two identity fields |
| `notes.folder` | Folder `sync_id` | `name`: non-empty path segment, at most 500 characters; `parent_sync_id`: string or null | Empty |
| `notes.folder_link` | Deterministic link ID | `note_id`; `folder_sync_id` | The same two identity fields |

Strings follow the current REST normalization boundary: keyword, collection, and
folder names are stripped at input; accepted casing and interior characters are
preserved. Keywords and collection names retain their current case-insensitive
uniqueness rules. A materialized folder's complete derived path must be non-empty,
relative, and at most 500 characters.

For `notes.keyword_link`, `subject_id` is the existing stable note UUID when
`subject_type` is `note`, and the existing conversation string ID when it is
`conversation`. The referenced `notes.note` or `chat.conversation` domain must be
enrolled and the referenced object must belong to the same dataset owner.

Link tombstones carry their identity payload because a one-way hash cannot be
reversed to locate the integer-keyed projection. Resource tombstones resolve their
projection directly by resource `sync_id`.

## Stable identity and migration

ChaChaNotes schema version 55 will add a unique, non-null `sync_id` to `keywords`,
`keyword_collections`, and `note_folders` on SQLite and PostgreSQL. The migration
backfills every active and soft-deleted row with a freshly generated UUIDv4 string,
checks for duplicates, and creates backend-equivalent unique indexes. New rows
receive a canonical lowercase, hyphenated UUIDv4 string before insertion. Clients
treat the value as opaque and must not derive behavior from UUID structure.

The opaque `sync_id` is the only resource identity used in canonical envelopes and
cross-device references. Existing integer `id` values remain local compatibility
keys for storage, foreign keys, routes, and older clients. Materializers resolve or
create a local integer row by `sync_id`; they never treat a remote integer ID as
identity. Keyword, collection, and folder response schemas add `sync_id` as an
additive field so REST and Sync views can be correlated.

Backfill does not derive identity from an integer ID or mutable name. Those values
are not globally stable. If independently migrated stores later present different
sync IDs for case-insensitively equal names, the uniqueness conflict is reviewable;
the server does not guess that the objects are semantically identical.

Membership tables do not need stored sync-ID columns. Their canonical object IDs
are deterministic hashes of domain-tagged tuples. The exact algorithm is:

1. Build an object with `domain`, `schema_version: 1`, and `members` in the order
   shown below.
2. Encode it as UTF-8 JSON with keys sorted, no insignificant whitespace, and no
   ASCII escaping.
3. Compute SHA-256 and format the entity ID as
   `<domain>:sha256:<lowercase-hex-digest>`.

The member tuples are:

- `notes.keyword_link`: `[subject_type, subject_id, keyword_sync_id]`
- `notes.keyword_collection_link`: `[collection_sync_id, keyword_sync_id]`
- `notes.folder_link`: `[note_id, folder_sync_id]`

The normative test vectors for the algorithm are:

- `notes.keyword_link` with `["note", "note-123", "kw-456"]`:
  `notes.keyword_link:sha256:10f9eab3be80b6e439ce1bcf8fae952527bde7d7e026d0e227f0a87ada963be0`
- `notes.keyword_collection_link` with `["collection-123", "kw-456"]`:
  `notes.keyword_collection_link:sha256:e9427c2d8bc4cfa8586130bc1fcc54cf432ca6dbb3df77bab3e65033b6148199`
- `notes.folder_link` with `["note-123", "folder-456"]`:
  `notes.folder_link:sha256:9076b60d9d8476f852736928ef3661cb06d9ba55696dd4504657c753f414b670`

Domain tagging and canonical JSON avoid delimiter ambiguity and prevent identical
member strings in different relationship types from sharing an ID.

## Hierarchy and projection rules

`notes.keyword_collection` and `notes.folder` synchronize `name` plus
`parent_sync_id`. Integer `parent_id` is a local projection. Folder `path` is also a
projection and is recalculated from the canonical parent chain; it is never accepted
as a second Sync authority.

Every hierarchy mutation validates that:

- the parent exists, is active, and belongs to the same dataset owner;
- an object is not its own parent;
- walking the proposed parent chain cannot reach the object;
- the chain is finite and has no pre-existing cycle; and
- the resulting case-insensitive collection-name or complete folder-path uniqueness
  constraints hold.

A folder rename or move recalculates the paths of its active descendant subtree in
one ChaChaNotes transaction. A conflict during that projection fails the envelope
without partially rewriting the subtree.

Tombstoning a resource never silently cascades canonical state. Existing membership
objects, child parent pointers, and local join rows remain intact but may be hidden
by ordinary active-resource queries. Restoring the same resource therefore reveals
its preserved organization again, matching current soft-delete behavior. Creating
or restoring a membership against an already tombstoned resource is still a
dependency conflict, and explicit unlink or merge behavior still emits link
tombstones. Hard cleanup of dormant dependencies is outside this task.

Creating a REST folder path decomposes it into one `notes.folder` upsert for every
new segment, ordered parent before child. Existing matching segments are reused by
`sync_id`. The derived path is validated before the group is appended.

## Effective folder membership

`notes.folder_link` is the canonical user-visible note-to-folder relationship.
ChaChaNotes currently obtains that view from the union of manual memberships and
source-managed memberships. Source IDs and folder keys are ingestion bookkeeping,
not portable personal knowledge state.

For a source-managed operation, the coordinator calculates the prospective
effective membership union before committing the local provenance change:

- emit a link upsert only when the effective relationship changes from absent to
  present; and
- emit a link tombstone only when it changes from present to absent.

When the effective relationship changes, a server-origin envelope may carry a safe
origin-only provenance delta in routing metadata. The originating server applies
that delta and the canonical membership projection in one ChaChaNotes transaction;
remote materializers ignore it and apply only the canonical payload. When the
provenance change does not alter the effective relationship, it is derived local
bookkeeping and no canonical envelope is emitted. This preserves the visible
organization result across devices without making source IDs or folder keys a
second authority. Routing metadata for this purpose must not contain source content,
credentials, or filesystem paths.

## Existing-dataset bootstrap

Adding domain names to an existing dataset without seeding its current state would
make already stored organization disappear on another device. The profile
bootstrap path therefore gains an idempotent organization-group upgrade, not merely
a broader allow-list.

When a capable client requests all six domains for a default personal dataset that
does not yet have them, the server:

1. transactionally records the group as `initializing` in the Sync store so later
   server-origin batches and client pushes fail their in-transaction readiness
   check, and blocks client pulls of the initializing domains;
2. drains or repairs organization envelopes accepted before that state transition,
   so the product projection reaches the last accepted canonical head;
3. snapshots current active and soft-deleted keyword, collection, and folder rows
   plus relationship rows using their migrated sync IDs;
4. appends resource upserts parent-before-child, current relationship upserts, and
   then tombstones for resources that were already soft-deleted, in bounded
   deterministic batches so dormant relationships are preserved;
5. records each bootstrap envelope as applied only after verifying that its snapshot
   state still matches ChaChaNotes; bootstrap capture does not replay the envelope
   into an already-correct projection or transiently undelete rows;
6. resumes safely after interruption using a bootstrap ID and stable per-object
   envelope keys; and
7. marks the complete six-domain group `ready` only after every bootstrap envelope
   is verified and its resource/relationship counts match a final source scan.

The profile response exposes `initializing`, `ready`, or `failed` group state and a
safe repair summary. Reads stay available during initialization. A failed bootstrap
remains fail closed and repairable; it does not fall back to unsynchronized writes.
The bootstrap captures the active relationship set at the upgrade boundary. It does
not invent tombstones for link removals that occurred before the domain had a Sync
history. If source state changes despite the mutation gate, verification fails and
the bootstrap reconciles under the same initializing state instead of publishing an
inconsistent ready snapshot.

A trusted `bootstrap_capture` routing flag may preserve an existing dormant
relationship whose note or conversation already has a tombstone head. It is
accepted only for server-origin capture while the organization group is
`initializing`, and only after the corresponding local relationship row is verified.
It is not a client-settable restore bypass; ordinary new membership against a
tombstoned resource still conflicts.

## Capability enrollment and fail-closed writes

The capability response advertises all six domain schemas and their operations as
one Notes organization capability group. Enrollment accepts the group only as a
complete set. Existing datasets are not silently partially upgraded; a capable
client explicitly requests the complete group through the profile bootstrap upgrade
above. New capable default-personal dataset creation seeds the current organization
snapshot before reporting the group ready.

Domain selection remains device-aware. Pull requests with no explicit domain list
default to the registered device's requested/supported-domain intersection, not all
domains enrolled in the shared dataset. A legacy device therefore does not receive
new organization envelopes it did not advertise support for after another device
upgrades the dataset.

When no personal Sync dataset is active, current direct ChaChaNotes behavior remains
unchanged. When one is active:

- organization reads remain available;
- an organization mutation is allowed only when all six domains are enrolled and
  their bootstrap state is `ready`;
- a partial or absent group fails with
  `notes_organization_sync_domains_incomplete` and lists the missing domains;
- an initializing or failed bootstrap returns
  `notes_organization_sync_not_ready` with the safe group state and repair status;
- a conversation-keyword link also requires `chat.conversation`, and note keyword
  or folder links require `notes.note`; and
- `client_private_v1` retains the existing server-frontend mutation block.

No endpoint may bypass this check by directly changing canonical keyword,
collection, folder, or membership state. This includes inline note
keywords/folders, bulk note imports, collection keyword replacement, path creation,
source-managed effective folder changes, direct link routes, rename, merge, and
delete. A provenance-only source bookkeeping change that leaves the effective
folder relationship unchanged may update only the derived provenance tables through
the coordinator after the readiness check.

## Adapters and materializers

A strict Notes organization adapter family dispatches by the six typed domains. It
performs payload validation, optimistic base-lineage checks, restore validation,
delete/update conflict detection, dependency and ownership checks, link-ID
verification, uniqueness checks, and hierarchy validation before acceptance.
These decisions use canonical object heads plus earlier accepted steps in the same
planned group. Product tables are not treated as newer authority merely because a
projection is ahead or behind. Product-table checks remain necessary for migrated
legacy identity, local uniqueness constraints, and explicitly unsynchronized
dependencies such as flashcard links.

Resource updates are whole-object updates. A stale concurrent rename or parent
change conflicts rather than being field-merged. Identical current upserts and
tombstones are idempotent. Reattaching a link after its tombstone is a restore: the
upsert must set `routing_metadata.restore_intent` to boolean `true` and reference
the exact current tombstone base. A stale upsert may not resurrect membership.

Per-domain materializers project by canonical sync identity into the authenticated
user's ChaChaNotes database. They use backend-neutral persistence seams, preserve
current optimistic `version` behavior, apply one envelope transactionally, and mark
the envelope applied only after the projection commits. Reapplying an already
materialized identical envelope is a no-op success.

The factory registers strict adapters and materializers for every advertised domain;
there is no accept-anything fallback for these domains. Derived FTS data, folder
paths, counts, timestamps, integer IDs, and import provenance are rebuilt or
maintained locally.

## Durable server-origin mutation groups

ADR-032 defines the shared infrastructure. Each compound REST operation is
preflighted into an immutable ordered list of primitive envelopes. The plan is
evaluated in memory against current heads plus earlier planned steps, assigned a
group ID, step indexes, total count, and canonical plan hash, and appended through
one `insert_envelopes_atomic()` transaction before any materializer runs.

The Sync envelope store gains nullable server-side fields for mutation group ID,
step, step count, and plan hash, with a unique constraint on
`(dataset_id, mutation_group_id, mutation_step)` when the group ID is present.
Single-envelope and client-origin envelopes leave these fields null.

After append, the coordinator materializes from step zero in order. It stops at the
first failed or conflicting step and leaves later steps pending. Retry and repair
load the persisted group, verify its plan hash and step count, skip applied steps,
and resume at the first retryable non-applied step. An unresolved conflict blocks
the group until conflict resolution supplies an accepted outcome; repair does not
blindly replay it. A REST request succeeds only after every step applies. A
projection failure returns
`sync_server_origin_batch_materialization_failed` with a safe group identifier and
retryable status; it does not pretend the two databases rolled back together.

The whole group becomes pull-visible only after its single append transaction
commits, so clients never observe a canonical prefix without its suffix. Clients may
observe the complete canonical plan while the server projection is being repaired;
server cursors preserve the intended order.

When an HTTP idempotency key is supplied, its existing privacy-preserving hash is
part of the stable group identity. Reusing it with a different group plan returns
`sync_server_origin_batch_idempotency_conflict`. Without a client key, the server
uses a random group identity; persisted failed groups remain discoverable and
resumable by repair, while normal database uniqueness and optimistic bases prevent
silent duplicate semantic changes.

## REST mutation behavior

Under active complete organization Sync, endpoints build canonical plans rather
than mutating ChaChaNotes first.

- Resource create/rename/move/delete emits the corresponding resource envelope.
  Soft delete preserves existing membership and hierarchy objects; only endpoints
  whose accepted behavior explicitly changes those objects emit additional steps.
- Direct link/unlink emits one relationship upsert/tombstone.
- Replacing a note's keywords or folders diffs current canonical membership and
  emits only required link operations. Missing keyword or folder resources are
  created before their links.
- Creating a keyword collection with initial keywords creates or reuses keywords,
  then creates collection-link objects.
- Bulk note import uses one independently reportable mutation group per note so the
  existing multi-status response does not hide which note failed.
- Folder path creation emits new ancestors before descendants and links only after
  all referenced folders exist.

Inactive-Sync requests retain their current direct database paths. Reads continue
to expose current integer IDs and add opaque sync IDs for correlation.

## Keyword merge

Keyword merge remains a REST/coordinator behavior composed from ordinary domains;
there is no `merge` wire operation.

Preflight loads the source and target by integer compatibility ID, verifies both
optimistic versions and sync IDs, checks all link tables, and freezes the complete
move plan. If the source has any active `flashcard_keywords` row, active-Sync merge
fails before append with `notes_keyword_merge_unsynchronized_dependency`. Moving
those links would otherwise create unsynchronized state outside this task. Ordinary
keyword soft deletion does not move the flashcard link and therefore preserves the
current dormant relationship for an exact restore.

The persisted group order is:

1. upsert every missing target note, conversation, and collection membership;
2. tombstone every source membership represented by the synchronized link domains;
3. tombstone the source keyword.

Target memberships are materialized before source memberships are removed, and the
source keyword remains active until every membership move has applied. Existing
target memberships are reused idempotently. A failure records the exact step and
resumes from the durable group; it does not rerun a fresh query against a partially
moved source.

## Conflicts and stable errors

| Code | Meaning |
| --- | --- |
| `notes_organization_payload_invalid` | Payload fields, types, limits, or deterministic link ID are invalid. |
| `notes_organization_sync_domains_incomplete` | An active dataset lacks one or more of the six required organization domains. |
| `notes_organization_sync_not_ready` | The complete group was requested but bootstrap is initializing or failed; organization push/pull and REST writes remain blocked and repair status is available. |
| `notes_organization_dependency_missing` | A referenced resource/domain is absent, deleted, unenrolled, or owned elsewhere. |
| `notes_organization_hierarchy_conflict` | A parent assignment is self-referential, cyclic, stale, or otherwise invalid. |
| `notes_organization_name_conflict` | A keyword/collection name or derived folder path violates case-insensitive uniqueness. |
| `notes_organization_restore_target_conflict` | Restore does not reference the exact current tombstone head. |
| `notes_keyword_merge_unsynchronized_dependency` | Merge would move flashcard links outside the synchronized domain set. |
| `sync_server_origin_batch_idempotency_conflict` | A stable mutation-group identity was reused for a different plan. |
| `sync_server_origin_batch_materialization_failed` | The canonical group is durable but its product projection is incomplete. |

Conflict records contain opaque IDs, domain, lineage, and safe diagnostic metadata;
they do not copy plaintext note content or secrets.

## Restore, repair, and deletion

Restore uses `upsert` with `routing_metadata.restore_intent: true`, the exact current
tombstone base, and the complete canonical payload. Resources restore before links.
A collection or folder restore requires its parent to be active or restored earlier
in the same ordered plan. A link restore requires every referenced resource to be
active or restored earlier.

Restore preview reports counts and blockers for all six domains, including missing
parents, missing linked resources, name/path collisions, incomplete mutation
groups, and unresolved conflicts. Repair replays mutation groups in group-step order
and otherwise follows server-cursor order. It never synthesizes a missing group
suffix from current projections.

Soft-deleted rows retain sync IDs so tombstones and later exact restores address the
same object. Hard cleanup remains outside this task and must not remove identity
needed by retained Sync history.

## Testing strategy

Implementation planning must include, at minimum:

- capability-schema and strict validator tests for all operations and unknown
  fields;
- profile-upgrade bootstrap tests for populated, empty, interrupted, and failed
  datasets, including all-six enrollment and write blocking until verified ready;
- bootstrap tests for dormant relationships whose keyword, note, conversation,
  collection, or folder endpoint is already soft-deleted, with no transient restore;
- mixed-version device tests proving a legacy device's implicit pull remains limited
  to its registered requested domains after another device upgrades the dataset;
- identity migration tests for populated SQLite and PostgreSQL databases, including
  active and deleted rows, uniqueness, and legacy integer API compatibility;
- deterministic link-ID vectors shared across domain, adapter, and materializer
  tests;
- adapter conflict matrices for stale rename, delete/update, exact restore,
  reattachment, missing ownership/dependencies, duplicate names, and hierarchy
  cycles;
- soft-delete/restore tests proving resource relationships and hierarchy pointers
  survive without implicit link tombstones;
- materializer tests for every domain on SQLite and PostgreSQL, including folder
  subtree path projection and idempotent replay;
- batch-store tests proving all-or-none append, plan mismatch rejection, ordered
  cursors, partial projection failure, restart/resume, and no reapplication of
  completed steps;
- REST tests for direct CRUD, inline note organization, bulk items, collection
  replacement, folder paths, effective source membership, and all-six fail-closed
  gating;
- keyword merge failure injection at every step, existing-target membership,
  flashcard dependency rejection, and retry after restart; and
- restore preview, repair, capability documentation, authorization, cross-user
  isolation, payload-size, and no-secret fixture checks.

Focused tests run first. Full Sync v2 and ChaChaNotes regression suites, formatting,
lint/static analysis, and Bandit on touched production scope are required before
completion. PostgreSQL-specific claims require a real supported PostgreSQL test
path; SQLite-only tests cannot stand in for them.

## Rollout and compatibility

1. Land schema and identity migration support before advertising the domains.
2. Land strict schemas, adapters, materializers, bootstrap upgrade, and batch
   infrastructure behind the capability gate.
3. Advertise the six-domain group only when every registered implementation and
   repair path is available.
4. Allow clients to enroll the group atomically; do not silently enroll a subset.
5. Route REST writes through Sync only for fully enrolled, server-materializable
   datasets; otherwise fail closed with the documented error.
6. Preserve legacy integer route parameters and response fields. Additive `sync_id`
   fields must not change existing IDs or sorting.
7. Update Sync v2 API documentation and restore/repair operator guidance before
   declaring the task complete.

Rollback may stop new enrollment and server-origin organization writes, but it must
not drop sync-ID columns, mutation-group metadata, or already accepted envelopes.
Older code must tolerate additive columns. Repair remains available for durable
groups accepted before rollback.

## Security and privacy

Materialization is scoped through the authenticated user's database factory. Every
referenced note, conversation, keyword, collection, folder, and parent is resolved
inside that owner boundary. Client-supplied integer IDs are never trusted as
cross-device identity, and relationship hashes do not authorize a link.

Canonical payloads contain organization labels and opaque object references but no
note content. Logs, conflict metadata, group hashes, fixtures, and documentation
must not contain note bodies, credentials, authorization headers, raw idempotency
keys, or secret-bearing configuration.

## Acceptance mapping

| TASK-13003 criterion | Design coverage |
| --- | --- |
| Six versioned domains and operations | Canonical domains; capability enrollment |
| Stable identity, hierarchy, membership, base, ownership on both backends | Stable identity and migration; hierarchy; adapters/materializers |
| Canonical server-origin REST capture | Durable mutation groups; REST behavior; keyword merge |
| Unblock writes only with complete ready enrollment | Existing-dataset bootstrap; capability enrollment and fail-closed writes |
| Deterministic/idempotent changes or reviewable conflicts | Identity algorithm; conflict rules; ordered replay |
| Restore preview, repair, and docs | Restore, repair, rollout, and testing sections |
| Existing state survives domain activation | Existing-dataset bootstrap and migration tests |
| Legacy devices do not receive unsupported domains | Device-aware domain selection and mixed-version tests |

## ADR check

ADR required: yes

ADR path: `Docs/ADR/032-durable-server-origin-sync-mutation-batches.md`

Reason: TASK-13003 changes the Sync storage contract, cross-database mutation
boundary, recovery policy, and long-lived service interface for compound writes.
