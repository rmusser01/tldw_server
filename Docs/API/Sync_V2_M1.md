# Sync v2 M1 API Contract

Date: 2026-08-09
Status: Locked for M1 implementation, including the Notes organization extension
Scope: Server-connected Chatbook personal sync only

## Overview

Sync v2 M1 provides manual reliable sync and restore for an authenticated
user's personal Notes and Chat dataset. Chatbook local-only mode remains outside
this contract.

All endpoints are authenticated and scoped to the current user:

```text
/api/v1/sync
```

The base M1 personal profile domains are:

- `notes.note`
- `chat.conversation`
- `chat.message`
- `attachment.ref`

`attachment.ref` stores reference metadata only. Binary/blob transfer is not
part of M1.

The required `attachment.ref` payload metadata fields are `attachment_id`,
`parent_domain`, `parent_object_id`, `content_type`, `size_bytes`,
`payload_hash`, and `availability`.

## Notes attachment adapter version 2

The canonical Notes attachment lifecycle is an additive adapter version for the
existing `attachment.ref` domain. Version 1 remains readable and restorable but
is immutable. A device must explicitly advertise
`supported_adapter_versions.attachment.ref: [1, 2]` (or `[2]`) before it can
push or acknowledge version-2 envelopes. Omission means version 1. Version maps
are monotonic-additive for an active device; a refresh cannot silently remove a
previously advertised version.

Capabilities expose two separate maps:

- `supported_adapter_versions` describes schemas the server can parse;
- `writable_adapter_versions` describes schemas writable for the selected,
  owner-authorized dataset under its current rollout, encryption, enrollment,
  and bootstrap state.

The strict version-2 `upsert` payload is:

| Field | Contract |
| --- | --- |
| `attachment_id` | Canonical lowercase UUIDv4 and equal to envelope `object_id`. |
| `parent_domain` | Exactly `notes.note`. |
| `parent_object_id` | Canonical lowercase UUIDv4 of the owning note. |
| `file_name` | Canonical Notes-safe display name after NFKC normalization, extension allowlisting, and collision suffixing. |
| `original_file_name` | Safe basename, at most 255 characters and 1024 UTF-8 bytes. |
| `content_type` | Already-trimmed lowercase `type/subtype`; parameters are rejected. |
| `size_bytes` | Positive integer within the effective Notes/Sync attachment limit. |
| `blob_hash` | Canonical lowercase `sha256:<64 hex>` digest of the bytes. |
| `created_at`, `last_modified` | Normalized UTC timestamps. |
| `created_by` | Non-empty device or server-origin identity. |

Version-2 tombstones retain the same immutable identity and provenance fields
and add `deleted_at` plus an optional bounded `reason`. Unknown payload
or routing fields are rejected. The envelope `payload_hash` is the canonical
object hash over the complete normalized semantic payload, object revision, and
live/tombstone state; it is not merely the blob digest.

Pull cursors and domain acknowledgments are keyed by
`(device, dataset, domain, adapter_version)`. Mixed-version pulls use a signed,
bounded continuation token carrying per-version watermarks; legacy version-1
numeric cursors remain valid. Blob verification evidence for version 2 uses the
immutable `blob_id` plus digest in `blob_id_acks`; legacy attachment-ID evidence
is never reinterpreted as version-2 proof.

The additive Notes organization capability is one indivisible six-domain group:

- `notes.keyword`
- `notes.keyword_link`
- `notes.keyword_collection`
- `notes.keyword_collection_link`
- `notes.folder`
- `notes.folder_link`

The group is available only on datasets whose organization bootstrap is `ready`.
A dataset or device that has only the base profile remains valid.

The additive `notes.link` capability synchronizes explicit manual note-to-note
relationships. It has its own resumable `notes_link_v1` readiness record so an
already-ready six-domain organization bootstrap does not need to be reopened.
Wikilinks, backlinks, graph summaries, orphan state, projection queues, and graph
revisions are deterministic local projections and never Sync domains.

## Notes organization version 1

### Domain schemas and operations

All six domains use schema version `1`, encryption policy
`server_trusted_v1`, and exactly the operations `upsert` and `tombstone`.
Unknown payload fields are rejected.

| Domain | `upsert` payload | `tombstone` payload |
| --- | --- | --- |
| `notes.keyword` | Required `keyword`: stripped, non-empty string, maximum 100 characters. | Empty object `{}`. |
| `notes.keyword_link` | Required `subject_type`: `note` or `conversation`; required `subject_id`; required canonical UUIDv4 `keyword_sync_id`. A note subject ID is a canonical UUIDv4; a conversation subject ID is the existing stable conversation string ID. | The same three identity fields as `upsert`. |
| `notes.keyword_collection` | Required `name`: stripped, non-empty string, maximum 255 characters; optional `parent_sync_id`: canonical UUIDv4 or `null` (default `null`). | Empty object `{}`. |
| `notes.keyword_collection_link` | Required canonical UUIDv4 fields `collection_sync_id` and `keyword_sync_id`. | The same two identity fields as `upsert`. |
| `notes.folder` | Required `name`: stripped, non-empty string, maximum 500 characters; optional `parent_sync_id`: canonical UUIDv4 or `null` (default `null`). | Empty object `{}`. |
| `notes.folder_link` | Required canonical UUIDv4 fields `note_id` and `folder_sync_id`. | The same two identity fields as `upsert`. |

Resource `object_id` values for keywords, collections, and folders are canonical
lowercase, hyphenated RFC 4122 UUIDv4 strings. Existing integer database and REST
IDs remain local compatibility keys only. They are never accepted as canonical
Sync identity and never cross a device boundary.

Relationship `object_id` values are deterministic. Build this JSON object:

```json
{"domain":"<domain>","members":["<member-1>","<member-2>"],"schema_version":1}
```

Use the domain's ordered member tuple:

- `notes.keyword_link`: `[subject_type, subject_id, keyword_sync_id]`
- `notes.keyword_collection_link`: `[collection_sync_id, keyword_sync_id]`
- `notes.folder_link`: `[note_id, folder_sync_id]`

Serialize as UTF-8 JSON with keys sorted, separators `,` and `:` (no insignificant
whitespace), and non-ASCII characters left unescaped. Hash those exact bytes with
SHA-256 and format the ID as
`<domain>:sha256:<lowercase-hex-digest>`.

These full normative hash-only vectors fix byte-level interoperability. Their
short member strings are algorithm fixtures, not valid resource UUID examples:

| Domain and members | Exact canonical JSON | Result |
| --- | --- | --- |
| `notes.keyword_link`, `["note","note-123","kw-456"]` | `{"domain":"notes.keyword_link","members":["note","note-123","kw-456"],"schema_version":1}` | `notes.keyword_link:sha256:10f9eab3be80b6e439ce1bcf8fae952527bde7d7e026d0e227f0a87ada963be0` |
| `notes.keyword_collection_link`, `["collection-123","kw-456"]` | `{"domain":"notes.keyword_collection_link","members":["collection-123","kw-456"],"schema_version":1}` | `notes.keyword_collection_link:sha256:e9427c2d8bc4cfa8586130bc1fcc54cf432ca6dbb3df77bab3e65033b6148199` |
| `notes.folder_link`, `["note-123","folder-456"]` | `{"domain":"notes.folder_link","members":["note-123","folder-456"],"schema_version":1}` | `notes.folder_link:sha256:9076b60d9d8476f852736928ef3661cb06d9ba55696dd4504657c753f414b670` |

Link tombstones retain their identity fields because the hash cannot be reversed
to find a local join row. Resource tombstones use their resource `object_id` and
an empty payload.

### Enrollment, bootstrap, and readiness

The six domains are enrolled as one capability group; partial enrollment is
invalid. An existing dataset is not upgraded by merely adding domain strings.
A capable client requests all six through profile bootstrap, which:

1. records `initializing` in the Sync store and blocks organization push, pull,
   and server-origin mutation;
2. snapshots active and soft-deleted resources plus current relationships;
3. durably appends resource upserts parent-before-child, relationship upserts,
   and then existing resource tombstones;
4. verifies every captured envelope against the product snapshot without
   transiently restoring deleted resources; and
5. publishes `ready` only after all six counts and projections verify.

Interruption resumes under the same bootstrap ID. A failed verification publishes
`failed` plus a safe repair summary and remains fail closed. The trusted
`bootstrap_capture` routing flag can preserve a verified dormant relationship only
for server-origin capture while state is `initializing`; it is not accepted from a
client and is not a general restore bypass.

An organization mutation requires all six domains and `ready`. A missing or partial
group returns `notes_organization_sync_domains_incomplete`; `initializing` or
`failed` returns `notes_organization_sync_not_ready`. Dependency domains also must
be enrolled: note links require `notes.note`, and conversation keyword links require
`chat.conversation`.

Implicit pulls are device-aware. With no explicit domain filter, the server uses
the requesting device's stored requested/supported-domain intersection. Upgrading
the shared dataset from a capable device therefore does not leak organization
envelopes to a legacy device that never advertised the six domains. Explicit
unsupported or not-ready requests still fail closed.

### Durable append and resumable materialization

Per [ADR-034](../ADR/034-durable-server-origin-sync-mutation-batches.md), a
compound REST mutation is preflighted into an immutable list of primitive
envelopes. Every step carries a mutation-group ID, zero-based step, total step
count, and canonical plan hash. The complete plan is appended in one transaction
to the per-user Sync envelope store before any product write. Reusing a group ID
with different content is an idempotency conflict.

That transaction does **not** include `ChaChaNotes.db`; there is no cross-database
atomicity claim. After durable append, materializers project one step at a time in
group order. A failure or conflict stops the group and leaves later steps pending.
Retry loads the persisted plan, preserves its order, skips the applied prefix, and
resumes at the first non-applied step. If the product already has the exact intended
post-state, retry completes Sync bookkeeping without duplicating the product write.
The complete canonical plan can be pull-visible while product projection repair is
still pending.

### Hierarchy, membership, deletes, merge, and conflicts

Collection and folder `parent_sync_id` values are canonical; integer `parent_id`
and folder `path` are local projections. Parent assignment verifies same-owner
existence, active state, no self-parent, no cycle, finite ancestry, and applicable
case-insensitive name/path uniqueness. Folder moves rewrite an active descendant
subtree in one product transaction. Restore applies parents before children,
resources before dependent relationships, and required live/link operations before
tombstones. A complete stored mutation group remains one ordered unit.

Soft-deleting a resource never cascades canonical link tombstones or clears child
parent pointers. Relationships remain as dormant canonical state and become visible
again when the resource is restored. Creating a new link to an already-deleted
dependency conflicts. Hard cleanup is outside this contract.

`notes.folder_link` represents effective user-visible membership. On the originating
server the effective set is:

```text
(manual memberships UNION source memberships) MINUS Sync suppressions
```

Per [ADR-035](../ADR/035-canonical-folder-link-suppression-preserves-source-provenance.md),
a canonical upsert clears the pair's suppression and ensures the manual projection;
a canonical tombstone removes the manual projection and adds a suppression without
deleting source provenance. Source IDs, source keys, and import bookkeeping remain
local. Origin-only provenance routing metadata is honored only by the authenticated
origin server; remote materializers ignore it and apply the canonical link payload.

Resource changes are whole-object optimistic updates. Stale renames, parent changes,
deletes, hierarchy changes, ownership mismatches, and case-insensitive uniqueness
collisions are explicit conflicts rather than field merges. Restoring a tombstoned
link requires `routing_metadata.restore_intent: true` and the exact current
tombstone base.

Keyword merge is a REST coordinator operation, not a Sync wire operation. Its
persisted group first upserts missing target memberships, then tombstones source
memberships, then tombstones the source keyword. An unresolved conflict blocks that
step and every later step. Active flashcard membership blocks an active-Sync merge
with `notes_keyword_merge_unsynchronized_dependency`; flashcard links are not moved
or silently deleted.

Restore preview reports counts and safe current-state actions for all six domains.
Repair never synthesizes a missing group or skips a blocked conflict. Group repair
observability contains only `mutation_group_id`, `failing_step`, `error_code`,
`retry_result`, and `state`; it contains no names, note content, source values,
filesystem paths, credentials, idempotency keys, or raw database errors.

### Explicit non-goals

This extension does not synchronize flashcards or `flashcard_keywords`; source
provenance tables; FTS rows; local counts or timestamps; derived folder paths;
integer database IDs; or a compound `merge` wire operation. It does not make
product and Sync databases atomically commit together, and it does not grant a
remote device authority over server-local ingestion provenance.

## Notes link version 1

`notes.link` uses schema version `1`, encryption policy `server_trusted_v1`, and
exactly the operations `upsert` and `tombstone`. The envelope `object_id` is a
canonical lowercase UUIDv4 edge ID. Version 1 accepts only explicit `manual`
note-to-note links; unknown fields and arbitrary relationship types are rejected.

An upsert payload contains `source_note_id`, `target_note_id`, `type`, `directed`,
`weight`, nullable `label`, bounded canonical `properties`, `created_at`,
`last_modified`, and `created_by`. Undirected endpoint UUIDs must already be in
canonical string order on the Sync wire. Directed links preserve source/target
order. Source, target, type, and direction are immutable for an edge identity;
retargeting is a tombstone plus a new edge. A tombstone retains the complete
canonical edge snapshot plus deletion time and a bounded stable reason.

Both endpoint note identities must belong to the dataset owner. A public create
requires both notes to be live. Historical replay may retain a link whose owned
endpoint is soft-deleted: the link remains durable but is hidden from graph reads
until both endpoint notes are restored. Trashing or restoring a note emits no
incident link mutation and does not change link versions.

### Enrollment and bootstrap

`notes.link` is separate from the indivisible six-domain Notes organization group.
For the canonical default-personal Notes dataset, profile bootstrap adds the domain
under the dataset-row lock and records `metadata.notes_link_v1=initializing` with a
stable bootstrap ID. It then pages existing active and tombstoned explicit links in
edge-ID order, appends source-verified envelopes after the endpoint `notes.note`
identities, and verifies the count and canonical fingerprint before publishing
`ready`. Interruption resumes the same bootstrap; no product row is reapplied.

The six organization domains remain available during this link-only upgrade.
Link push, pull, and server-origin writes fail closed until `notes_link_v1` is
ready. Legacy devices that never request `notes.link` remain valid and do not
receive it through implicit pulls.

### Product API authority and repair

Graph/link endpoints accept an optional `dataset_id`. With active Sync, omission
selects the user's single active default-personal Notes dataset; a supplied ID must
identify that exact canonical dataset. With inactive Sync, omission preserves the
legacy product path, while supplying a dataset fails safely. Active-Sync create,
update, delete, and restore append a canonical server-origin envelope before the
product projection; update/delete/restore require `expected_version`. Stable
idempotency keys make exact retries return the same applied result.

Public list and detail routes are `GET /api/v1/notes/links` and
`GET /api/v1/notes/links/{edge_id}`. Mutations use the existing
`POST /api/v1/notes/{note_id}/links`, plus `PATCH`/`DELETE`
`/api/v1/notes/links/{edge_id}` and
`POST /api/v1/notes/links/{edge_id}/restore`. Listings use edge-ID keyset order,
default limit 50, and maximum limit 200. Full properties are returned only after
owner/dataset authorization.

Restore preview includes `notes.link` as an executable domain. A live link upsert
is ordered after providers for both endpoint `notes.note` identities, including a
provider that restores an endpoint; a link tombstone has no live endpoint
dependency. Missing providers or contradictory group/chronology constraints fail
closed with `sync_restore_plan_invalid`. Repair uses the normal exact-postcondition
materializer contract and never synthesizes derived graph state.

### Derived graph lifecycle

Manual links are canonical product/Sync state. Wikilinks and backlinks are parsed
from synchronized note content into owner-scoped local projections. Graph and
orphan reads are live-only, bounded, and revision-bound; tag/source nodes and their
membership edges remain compatible. Manual-only reads remain available during a
derived rebuild, while a derived-edge or orphan request returns retryable 503 until
the projection is current. Cache and pagination cursors bind the canonical dataset,
owner graph revision, parser version, and normalized request, so trash, restore,
link, membership, or projection changes cannot serve a stale cached page.

## Notes task and task activity version 1

`notes.task` and `notes.task_activity` are one coupled capability. Unbound
capabilities omit both. An owner-authorized, selected default-personal dataset
advertises both domains, both schemas, and adapter version `1` only after task and
activity source bootstrap, canonical capture, product authority, and repair wiring
are all ready. The supported and writable maps always include both domains or
neither. Devices must request and version-negotiate the complete pair.

Explicit enrollment requires `notes.note`, `server_trusted_v1`, and the complete
task/activity pair. It first binds the owner's `local-unbound` task graph to the
default-personal dataset under the dataset fence, then atomically enables task and
activity capture before either source scan. Task and activity bootstraps are
bounded, keyset-paged, source-fingerprinted, and resumable. A process failure after
the product bind cannot make the domains writable: retry verifies the immutable
binding, resumes the stored cursor, and publishes both domains in one final Sync
transaction. Readiness diagnostics remain server-private.

`notes.task` schema version `1` is the canonical whole-object task record. Upserts
and tombstones use immutable UUIDv4 task identity, parent note identity, canonical
revision/hash, status, description, priority, due date, estimate, recurrence
metadata, assignee, tags, custom fields, and projection status. Recurrence metadata
is synchronized; the server does not run a recurrence scheduler. Task creates,
updates, completion/reopen, delete, restore, and relink expand to a deterministic
mutation group containing the task, exactly one immutable activity, and when
linked, the derived `notes.note` checklist projection. Groups are limited to 1,000
steps and are durably appended before product materialization.

`notes.task_activity` schema version `1` is immutable event history. Ordinary
lifecycle events are derived by the task coordinator and cannot be directly
created by a client. A direct client may create only a `corrected` event whose
`corrects_activity_id` resolves to an exact live activity in the same owner,
dataset, note, and task scope; missing, foreign, and cross-task targets fail without
revealing existence. Server-origin REST/MCP activity and trusted legacy bootstrap
retain bounded provenance. Activity tombstones preserve the original identity and
require exact lineage.

Managed Markdown checklist markers are the stable link between canonical tasks and
derived note text. They carry task identity and exact last-common task/note anchors;
marker and locator caches are rebuildable. Equal edits converge automatically.
Incompatible task/note edits create privacy-safe drift and block silent overwrite
until an explicit resolution claim succeeds. Explicit/unmanaged checklist items
remain note content and are never silently adopted as canonical tasks.

Linked task tombstones and envelopes referenced by open drift are retained until
their restoration and repair windows close. Restore validates the exact current
tombstone, former note linkage, and current parent scope; relink is an explicit
authorized operation. Failures use stable task Sync error codes and never expose
readiness fingerprints, source rows, foreign IDs, or note content.

## Shared Types

### Capability Shape

The example below assumes an owner-selected dataset whose Notes attachment
bootstrap is ready and both attachment rollout gates are enabled. An unbound or
not-ready capability response omits version 2 from the writable map while still
reporting it as server-supported.

```json
{
  "protocol_version": "sync-v2-m1",
  "domains": [
    "notes.note",
    "chat.conversation",
    "chat.message",
    "attachment.ref",
    "workspaces.workspace",
    "workspaces.source_ref",
    "source_cache.entry",
    "media.item",
    "media.keyword",
    "media.keyword_link",
    "notes.keyword",
    "notes.keyword_link",
    "notes.keyword_collection",
    "notes.keyword_collection_link",
    "notes.folder",
    "notes.folder_link",
    "notes.link"
  ],
  "operations": {
    "notes.note": ["upsert", "tombstone"],
    "chat.conversation": ["upsert", "tombstone"],
    "chat.message": ["append", "tombstone"],
    "attachment.ref": ["upsert", "tombstone"],
    "workspaces.workspace": ["upsert", "tombstone"],
    "workspaces.source_ref": ["upsert", "tombstone"],
    "source_cache.entry": ["upsert", "tombstone"],
    "media.item": ["upsert", "tombstone"],
    "media.keyword": ["upsert", "tombstone"],
    "media.keyword_link": ["upsert", "tombstone"],
    "notes.keyword": ["upsert", "tombstone"],
    "notes.keyword_link": ["upsert", "tombstone"],
    "notes.keyword_collection": ["upsert", "tombstone"],
    "notes.keyword_collection_link": ["upsert", "tombstone"],
    "notes.folder": ["upsert", "tombstone"],
    "notes.folder_link": ["upsert", "tombstone"],
    "notes.link": ["upsert", "tombstone"]
  },
  "supported_adapter_versions": {
    "notes.note": [1],
    "attachment.ref": [1, 2]
  },
  "writable_adapter_versions": {
    "notes.note": [1],
    "attachment.ref": [1, 2]
  },
  "encryption": {
    "policy": "server_trusted_v1",
    "ready": true,
    "attestation": {
      "scope": "user_database_directory",
      "covers": ["Sync_v2.db", "ChaChaNotes.db"],
      "configured": true
    }
  },
  "blob_transfer": {
    "supported": false
  }
}
```

`server_trusted_v1` is ready only when the deployment attests at-rest encryption
coverage for the per-user database directory containing both the envelope store
and the materialized Notes/Chat projection store.

When the deployment has not attested that coverage, profile and capability
responses report `server_trusted_v1` as not ready:

```json
{
  "encryption": {
    "policy": "server_trusted_v1",
    "ready": false,
    "attestation": {
      "scope": "user_database_directory",
      "covers": ["Sync_v2.db", "ChaChaNotes.db"],
      "configured": false
    }
  },
  "warnings": [
    {
      "code": "sync_encryption_attestation_required",
      "message": "Sync v2 M1 requires deployment-level at-rest encryption coverage for the user database directory."
    }
  ]
}
```

`GET /api/v1/sync/profile` still returns this not-ready capability state.
`POST /api/v1/sync/profile/bootstrap` fails closed with
`sync_encryption_attestation_required` and must not create the default personal
dataset until the deployment attestation is configured.

### Envelope Shape

Client push requests and server pull responses use the same domain-neutral
shape. Server-assigned fields are omitted by clients on first push and included
by the server after acceptance.

```json
{
  "envelope_id": "srv_env_000000000101",
  "client_envelope_id": "client_env_01HZZ0NOTE001",
  "dataset_id": "ds_personal_01HZZ0",
  "device_id": "dev_chatbook_laptop",
  "client_profile_id": "chatbook_profile_main",
  "client_sequence": 17,
  "base_server_cursor": 98,
  "base_object_revision": 4,
  "base_object_hash": "sha256:base-note-hash",
  "server_cursor": 101,
  "domain": "notes.note",
  "operation": "upsert",
  "object_id": "note_7f3d",
  "parent_id": null,
  "schema_version": 1,
  "payload": {},
  "payload_hash": "sha256:payload-hash",
  "object_revision": 5,
  "created_at_client": "2026-05-23T18:12:44Z",
  "received_at_server": "2026-05-23T18:12:46Z",
  "deleted": false,
  "encryption_metadata": {
    "policy": "server_trusted_v1"
  }
}
```

Whole-object domains require `base_server_cursor`, `base_object_revision`, and
`base_object_hash` for updates and tombstones. New objects may use `null` base
fields. The server assigns canonical `server_cursor` and `object_revision`
values.

Client timestamps are normalized to one UTC ISO-8601 representation before
persistence and mutation-group hashing. This includes native timezone-aware
timestamps returned by PostgreSQL `TIMESTAMPTZ`, so reloading a stored group does
not change its immutable plan fingerprint. A `Z` suffix remains valid input.
For immutable groups stored before this normalization, validation uses a bounded
compatibility set over the complete plan: the exact timestamp spelling retained
by SQLite, and the exact UTC `Z` spelling when PostgreSQL has reloaded that value
as a native UTC timestamp. PostgreSQL `TIMESTAMPTZ` cannot recover an arbitrary
original non-UTC offset spelling after reload; the server-origin clock used by
production emitted canonical UTC, while the historical bootstrap path also used
the reconstructible UTC `Z` form. All other envelope fingerprint fields remain
strict.

`chat.message` append envelopes dedupe by stable message `object_id` plus
`payload_hash`. Matching duplicates are idempotent. A duplicate message ID with
a different payload hash creates a conflict.

## `GET /api/v1/sync/profile`

Returns the current Sync v2 profile without creating durable sync state.

### Query Parameters

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `device_id` | string | no | Existing client device ID, when known. |

### Response

```json
{
  "protocol_version": "sync-v2-m1",
  "min_supported_protocol_version": "sync-v2-m1",
  "profile_bootstrapped": true,
  "user_id": "user_123",
  "active_dataset_id": "ds_personal_01HZZ0",
  "device": {
    "device_id": "dev_chatbook_laptop",
    "registered": true,
    "client_profile_id": "chatbook_profile_main",
    "last_seen_at": "2026-05-23T18:10:00Z"
  },
  "dataset": {
    "dataset_id": "ds_personal_01HZZ0",
    "scope": "personal",
    "default_personal": true,
    "client_family": "chatbook",
    "domains": [
      "notes.note",
      "chat.conversation",
      "chat.message",
      "attachment.ref"
    ],
    "created_at": "2026-05-23T18:00:00Z"
  },
  "server_cursor": 128,
  "capabilities": {
    "protocol_version": "sync-v2-m1",
    "domains": [
      "notes.note",
      "chat.conversation",
      "chat.message",
      "attachment.ref",
      "workspaces.workspace",
      "workspaces.source_ref",
      "source_cache.entry",
      "media.item",
      "media.keyword",
      "media.keyword_link",
      "notes.keyword",
      "notes.keyword_link",
      "notes.keyword_collection",
      "notes.keyword_collection_link",
      "notes.folder",
      "notes.folder_link",
      "notes.link"
    ],
    "operations": {
      "notes.note": ["upsert", "tombstone"],
      "chat.conversation": ["upsert", "tombstone"],
      "chat.message": ["append", "tombstone"],
      "attachment.ref": ["upsert", "tombstone"],
      "workspaces.workspace": ["upsert", "tombstone"],
      "workspaces.source_ref": ["upsert", "tombstone"],
      "source_cache.entry": ["upsert", "tombstone"],
      "media.item": ["upsert", "tombstone"],
      "media.keyword": ["upsert", "tombstone"],
      "media.keyword_link": ["upsert", "tombstone"],
      "notes.keyword": ["upsert", "tombstone"],
      "notes.keyword_link": ["upsert", "tombstone"],
      "notes.keyword_collection": ["upsert", "tombstone"],
      "notes.keyword_collection_link": ["upsert", "tombstone"],
      "notes.folder": ["upsert", "tombstone"],
      "notes.folder_link": ["upsert", "tombstone"],
      "notes.link": ["upsert", "tombstone"]
    },
    "encryption": {
      "policy": "server_trusted_v1",
      "ready": true,
      "attestation": {
        "scope": "user_database_directory",
        "covers": ["Sync_v2.db", "ChaChaNotes.db"],
        "configured": true
      }
    },
    "blob_transfer": {
      "supported": false
    }
  },
  "domain_status": [
    {
      "domain": "notes.note",
      "last_server_cursor": 101,
      "envelope_count": 12,
      "pending_apply_count": 0,
      "pending_apply": 0,
      "failed_apply_count": 0,
      "unresolved_conflicts": 0,
      "last_apply_status": "applied",
      "last_apply_result": {
        "envelope_id": "srv_env_000000000101",
        "applied_at": "2026-05-23T18:12:46Z"
      },
      "repair_status": {
        "status": "healthy",
        "failed_apply_count": 0
      }
    },
    {
      "domain": "chat.conversation",
      "last_server_cursor": 110,
      "envelope_count": 4,
      "pending_apply_count": 0,
      "pending_apply": 0,
      "failed_apply_count": 0,
      "unresolved_conflicts": 0,
      "last_apply_status": "applied",
      "last_apply_result": {},
      "repair_status": {
        "status": "healthy",
        "failed_apply_count": 0
      }
    },
    {
      "domain": "chat.message",
      "last_server_cursor": 128,
      "envelope_count": 22,
      "pending_apply_count": 0,
      "pending_apply": 0,
      "failed_apply_count": 0,
      "unresolved_conflicts": 0,
      "last_apply_status": "applied",
      "last_apply_result": {},
      "repair_status": {
        "status": "healthy",
        "failed_apply_count": 0
      }
    },
    {
      "domain": "attachment.ref",
      "last_server_cursor": 126,
      "envelope_count": 3,
      "pending_apply_count": 0,
      "pending_apply": 0,
      "failed_apply_count": 0,
      "unresolved_conflicts": 0,
      "last_apply_status": "applied",
      "last_apply_result": {},
      "repair_status": {
        "status": "healthy",
        "failed_apply_count": 0
      }
    }
  ],
  "warnings": []
}
```

If no profile has been bootstrapped, `profile_bootstrapped` is `false`,
`dataset` is `null`, and `server_cursor` is `0`.

## `POST /api/v1/sync/profile/bootstrap`

Idempotently bootstraps server-connected Chatbook use for the authenticated
user. This endpoint registers or refreshes the current device and creates or
returns the default personal dataset.

### Request

```json
{
  "client_family": "chatbook",
  "mode": "offline_sync",
  "device_id": "dev_chatbook_laptop",
  "device_name": "Riley's MacBook",
  "client_profile_id": "chatbook_profile_main",
  "client_instance": {
    "app_version": "0.4.0",
    "platform": "macos"
  },
  "requested_domains": [
    "notes.note",
    "chat.conversation",
    "chat.message",
    "attachment.ref"
  ]
}
```

`mode` is `server_frontend` or `offline_sync`. If `device_id` is omitted, the
server generates one and the client must persist the returned value before
pushing envelopes.

### Response

```json
{
  "created": false,
  "device": {
    "device_id": "dev_chatbook_laptop",
    "registered": true,
    "client_profile_id": "chatbook_profile_main",
    "last_seen_at": "2026-05-23T18:12:00Z"
  },
  "dataset": {
    "dataset_id": "ds_personal_01HZZ0",
    "scope": "personal",
    "default_personal": true,
    "client_family": "chatbook",
    "domains": [
      "notes.note",
      "chat.conversation",
      "chat.message",
      "attachment.ref"
    ]
  },
  "server_cursor": 128,
  "capabilities": {
    "protocol_version": "sync-v2-m1",
    "domains": [
      "notes.note",
      "chat.conversation",
      "chat.message",
      "attachment.ref",
      "workspaces.workspace",
      "workspaces.source_ref",
      "source_cache.entry",
      "media.item",
      "media.keyword",
      "media.keyword_link",
      "notes.keyword",
      "notes.keyword_link",
      "notes.keyword_collection",
      "notes.keyword_collection_link",
      "notes.folder",
      "notes.folder_link",
      "notes.link"
    ],
    "operations": {
      "notes.note": ["upsert", "tombstone"],
      "chat.conversation": ["upsert", "tombstone"],
      "chat.message": ["append", "tombstone"],
      "attachment.ref": ["upsert", "tombstone"],
      "workspaces.workspace": ["upsert", "tombstone"],
      "workspaces.source_ref": ["upsert", "tombstone"],
      "source_cache.entry": ["upsert", "tombstone"],
      "media.item": ["upsert", "tombstone"],
      "media.keyword": ["upsert", "tombstone"],
      "media.keyword_link": ["upsert", "tombstone"],
      "notes.keyword": ["upsert", "tombstone"],
      "notes.keyword_link": ["upsert", "tombstone"],
      "notes.keyword_collection": ["upsert", "tombstone"],
      "notes.keyword_collection_link": ["upsert", "tombstone"],
      "notes.folder": ["upsert", "tombstone"],
      "notes.folder_link": ["upsert", "tombstone"],
      "notes.link": ["upsert", "tombstone"]
    },
    "encryption": {
      "policy": "server_trusted_v1",
      "ready": true,
      "attestation": {
        "scope": "user_database_directory",
        "covers": ["Sync_v2.db", "ChaChaNotes.db"],
        "configured": true
      }
    },
    "blob_transfer": {
      "supported": false
    }
  },
  "warnings": []
}
```

## `POST /api/v1/sync/push`

Accepts ordered envelopes from one device/profile, appends accepted envelopes to
`Sync_v2.db`, and materializes accepted Notes and Chat changes into
`ChaChaNotes.db`.

### Request

```json
{
  "dataset_id": "ds_personal_01HZZ0",
  "device_id": "dev_chatbook_laptop",
  "client_profile_id": "chatbook_profile_main",
  "base_server_cursor": 128,
  "envelopes": [
    {
      "client_envelope_id": "client_env_01HZZ0NOTE001",
      "client_sequence": 18,
      "base_server_cursor": 101,
      "base_object_revision": 5,
      "base_object_hash": "sha256:base-note-hash",
      "domain": "notes.note",
      "operation": "upsert",
      "object_id": "note_7f3d",
      "parent_id": null,
      "schema_version": 1,
      "payload": {
        "title": "Trip notes",
        "body": "Packed outline and research links.",
        "tags": ["travel", "research"],
        "updated_at": "2026-05-23T18:12:44Z"
      },
      "payload_hash": "sha256:note-payload-hash",
      "created_at_client": "2026-05-23T18:12:44Z",
      "deleted": false,
      "encryption_metadata": {
        "policy": "server_trusted_v1"
      }
    }
  ],
  "options": {
    "stop_on_conflict": false
  }
}
```

### Response

```json
{
  "dataset_id": "ds_personal_01HZZ0",
  "server_cursor": 129,
  "accepted": [
    {
      "client_envelope_id": "client_env_01HZZ0NOTE001",
      "envelope_id": "srv_env_000000000129",
      "server_cursor": 129,
      "object_id": "note_7f3d",
      "object_revision": 6,
      "apply_status": "applied"
    }
  ],
  "idempotent": [],
  "rejected": [],
  "conflicts": [],
  "apply_errors": []
}
```

`rejected` entries include `client_envelope_id`, `code`, and `message`.
`conflicts` entries include `conflict_id`, `domain`, `object_id`,
`server_object_revision`, `client_base_object_revision`, and a safe summary for
review.

## `GET /api/v1/sync/pull`

Returns accepted envelopes after a cursor in deterministic server-cursor order.

### Query Parameters

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `dataset_id` | string | yes | Dataset to pull from. |
| `device_id` | string | yes | Requesting device ID. |
| `cursor` | integer | yes | Last server cursor known to the client. |
| `domain` | string repeated | no | Optional domain filter. |
| `limit` | integer | no | Maximum envelopes to return. |
| `include_same_device_echoes` | boolean | no | Defaults to `false`; use `true` only for repair/debug flows. |

### Response

```json
{
  "dataset_id": "ds_personal_01HZZ0",
  "from_cursor": 128,
  "next_cursor": 131,
  "has_more": false,
  "envelopes": [
    {
      "envelope_id": "srv_env_000000000130",
      "client_envelope_id": "client_env_phone_CHATMSG001",
      "dataset_id": "ds_personal_01HZZ0",
      "device_id": "dev_chatbook_phone",
      "client_profile_id": "chatbook_profile_phone",
      "client_sequence": 44,
      "base_server_cursor": null,
      "base_object_revision": null,
      "base_object_hash": null,
      "server_cursor": 130,
      "domain": "chat.message",
      "operation": "append",
      "object_id": "msg_aa21",
      "parent_id": "conv_research",
      "schema_version": 1,
      "payload": {
        "conversation_id": "conv_research",
        "role": "user",
        "sender": "user",
        "content": "Summarize the saved notes.",
        "created_at": "2026-05-23T18:14:00Z"
      },
      "payload_hash": "sha256:message-payload-hash",
      "object_revision": 1,
      "created_at_client": "2026-05-23T18:14:00Z",
      "received_at_server": "2026-05-23T18:14:04Z",
      "deleted": false,
      "encryption_metadata": {
        "policy": "server_trusted_v1"
      }
    }
  ]
}
```

`next_cursor` is the highest server cursor scanned for the requested pull
window, not merely the last returned envelope. With
`include_same_device_echoes=false`, same-device envelopes may be scanned and
suppressed; `next_cursor` still advances past them so a client does not keep
re-reading suppressed rows. Domain filters work the same way: `next_cursor`
advances to the highest cursor scanned for the filtered request, including
non-matching domains skipped while satisfying the window. Full-profile clients
must persist the global/profile cursor only after an unfiltered pull. A
domain-filtered pull should update only per-domain cursor/status for that
filtered flow and must not advance the global profile cursor past unseen
domains unless the client is intentionally running a per-domain restore or
repair flow.

## `POST /api/v1/sync/restore/preview`

Compares server envelopes against a local inventory and returns a restore plan.
Clean profile restore uses the same endpoint with an empty `local_inventory`.
The response includes available datasets/domains, latest per-domain cursors,
one canonical `ordered_actions` plan, compatibility safe applies and tombstones,
missing blobs, attachment-reference summaries,
envelope ranges needed for local apply, and encryption/key status.

Restore planning preserves each complete stored mutation group and chronological
revisions of the same identity. Applicable resource, hierarchy, and relationship
dependencies are ordered before their consumers. Tombstones are ordered after
compatible live and relationship work, but this is not a blanket override of
history: an immutable historical tombstone group remains before a later exact
restore of the same identity. A contradictory dependency or chronology graph
fails closed with `sync_restore_plan_invalid`.

The planner enforces hard chronology, dependency, and complete-group edges, then
uses a deterministic ready-unit priority that chooses compatible live work before
an otherwise-ready tombstone. Preview is bounded to 50,000 scanned candidates,
10,000 ordered actions, and 1,000 members in any expanded mutation group. Exceeding
one of those ceilings returns HTTP 413 with
`sync_restore_candidate_limit_exceeded`, `sync_restore_action_limit_exceeded`, or
`sync_restore_group_limit_exceeded` respectively. The 1,000-member ceiling is
shared by atomic append, restore, and repair. Atomic append rejects an oversized
group before writing, while persisted-group reads fetch at most 1,001 rows and
reject an oversized legacy or corrupt group before constructing public actions or
materialization models.

Restore-preview request lists are bounded before service planning: `dataset_ids`
and `domains` accept at most 100 entries each, while `selected_object_ids`,
`selected_attachment_ids`, and `local_inventory` accept at most 10,000 entries
each. Repeated dataset IDs are deduplicated in first-seen order before access
checks and planning, so they never duplicate datasets or actions in the response.

`ordered_actions` is the only executable object-action sequence. Its zero-based
`plan_index` values are stable across the complete returned multi-dataset plan.
Each row contains only `plan_index`, `action`, `dataset_id`, `domain`, `object_id`,
`operation`, `server_cursor`, optional
`mutation_group_id`/`mutation_step`/`mutation_step_count`, and an optional stable
`code`. The required `dataset_id` disambiguates identical domain/object identities
in different datasets. A row never contains payload data, labels, content, local
paths or keys, raw errors, or routing metadata. Complete group steps have the same
opaque group ID and step count and remain adjacent in ascending step order.

The action is `apply`, `tombstone`, `noop`, or `conflict`. A conflict's safe `code`
describes its category and blocks execution of the plan. `safe_applies`,
`object_conflicts`, and `tombstones` remain compatibility category views, and the
existing counts are derived from the same ordered plan; concatenating those arrays
does not reconstruct execution order.

Classification simulates the local inventory sequentially without changing product
state. An earlier planned tombstone therefore changes the state used to classify a
later exact restore. If the initial inventory already matches that later live head,
the later action is still `apply`, not `noop`, because the preceding tombstone would
otherwise undo the final state. Likewise, when the initial inventory matches the
final planned live head, earlier historical live revisions remain executable
`apply` actions and the final head is reapplied. A divergent local object that does
not match the final planned state remains a conflict. In particular, a tombstone
may advance simulation only when the local identity is absent, already matches the
tombstone, matches the tombstone's explicit live base, was produced by an earlier
planned action, or initially matches the later final action for that same dataset
and identity. Otherwise the tombstone is a conflict and does not make later work
appear safe.

An accepted envelope whose stored apply status is `conflict` is also retained in
`ordered_actions` as a blocking `conflict` with safe code
`sync_restore_stored_apply_conflict`. A complete stored group is expanded before
classification, so neither a conflicted singleton nor a conflicted group can be
silently replaced by an older executable head. Stored raw error text is never
returned. For example:

```json
{
  "ordered_actions": [
    {
      "plan_index": 0,
      "action": "tombstone",
      "dataset_id": "dataset-example-1",
      "domain": "notes.keyword",
      "object_id": "11111111-1111-4111-8111-111111111111",
      "operation": "tombstone",
      "server_cursor": 41,
      "mutation_group_id": "server-origin-group-0001",
      "mutation_step": 0,
      "mutation_step_count": 2,
      "code": null
    },
    {
      "plan_index": 1,
      "action": "tombstone",
      "dataset_id": "dataset-example-1",
      "domain": "notes.folder",
      "object_id": "22222222-2222-4222-8222-222222222222",
      "operation": "tombstone",
      "server_cursor": 42,
      "mutation_group_id": "server-origin-group-0001",
      "mutation_step": 1,
      "mutation_step_count": 2,
      "code": null
    },
    {
      "plan_index": 2,
      "action": "apply",
      "dataset_id": "dataset-example-1",
      "domain": "notes.keyword",
      "object_id": "11111111-1111-4111-8111-111111111111",
      "operation": "upsert",
      "server_cursor": 43,
      "mutation_group_id": null,
      "mutation_step": null,
      "mutation_step_count": null,
      "code": null
    }
  ]
}
```

### Request

```json
{
  "dataset_ids": ["ds_personal_01HZZ0"],
  "device_id": "dev_chatbook_laptop",
  "client_profile_id": "chatbook_profile_main",
  "target_profile": {
    "kind": "existing"
  },
  "domains": [
    "notes.note",
    "chat.conversation",
    "chat.message",
    "attachment.ref"
  ],
  "local_inventory": [
    {
      "domain": "notes.note",
      "object_id": "note_7f3d",
      "object_revision": 4,
      "object_hash": "sha256:local-note-hash",
      "deleted": false
    }
  ],
  "attachment_availability_inventory": [
    {
      "attachment_id": "att_receipt_pdf",
      "parent_domain": "notes.note",
      "parent_object_id": "note_7f3d",
      "payload_hash": "sha256:attachment-bytes",
      "availability": "local_only"
    }
  ],
  "cursor": 0
}
```

### Response

The response example uses abbreviated detail arrays for readability:
`safe_applies`, `tombstones`, `missing_blobs`, and
`attachment_ref_summaries` show representative rows. The aggregate counts in
`summary` and `domain_plans` are authoritative for the full selected restore
scope and can exceed the number of rows shown in this example.

```json
{
  "dataset_id": "ds_personal_01HZZ0",
  "server_cursor": 131,
  "available_datasets": [
    {
      "dataset_id": "ds_personal_01HZZ0",
      "scope": "personal",
      "default_personal": true,
      "client_family": "chatbook",
      "domains": [
        "notes.note",
        "chat.conversation",
        "chat.message",
        "attachment.ref"
      ],
      "latest_cursor": 131
    }
  ],
  "available_domains": [
    {
      "domain": "notes.note",
      "object_count": 12,
      "tombstone_count": 1,
      "latest_cursor": 129
    },
    {
      "domain": "chat.conversation",
      "object_count": 4,
      "tombstone_count": 0,
      "latest_cursor": 120
    },
    {
      "domain": "chat.message",
      "object_count": 22,
      "tombstone_count": 1,
      "latest_cursor": 131
    },
    {
      "domain": "attachment.ref",
      "object_count": 3,
      "tombstone_count": 0,
      "latest_cursor": 126
    }
  ],
  "latest_cursor_by_domain": {
    "notes.note": 129,
    "chat.conversation": 120,
    "chat.message": 131,
    "attachment.ref": 126
  },
  "summary": {
    "safe_apply_count": 41,
    "conflict_count": 1,
    "tombstone_count": 2,
    "missing_attachment_blob_count": 3
  },
  "domain_plans": [
    {
      "domain": "notes.note",
      "safe_apply_count": 12,
      "conflict_count": 1,
      "tombstone_count": 1
    },
    {
      "domain": "chat.conversation",
      "safe_apply_count": 4,
      "conflict_count": 0,
      "tombstone_count": 0
    },
    {
      "domain": "chat.message",
      "safe_apply_count": 22,
      "conflict_count": 0,
      "tombstone_count": 1
    },
    {
      "domain": "attachment.ref",
      "safe_apply_count": 3,
      "conflict_count": 0,
      "tombstone_count": 0
    }
  ],
  "safe_applies": [
    {
      "domain": "chat.message",
      "object_id": "msg_aa21",
      "parent_id": "conv_research",
      "object_revision": 1,
      "object_hash": "sha256:message-append",
      "action": "apply",
      "envelope_range": {
        "from_cursor": 130,
        "to_cursor": 130
      }
    }
  ],
  "tombstones": [
    {
      "domain": "notes.note",
      "object_id": "note_archived",
      "parent_id": null,
      "deleted_at": "2026-05-23T17:40:00Z",
      "envelope_cursor": 127,
      "action": "delete_or_hide"
    }
  ],
  "missing_blobs": [
    {
      "attachment_id": "att_receipt_pdf",
      "parent_domain": "notes.note",
      "parent_object_id": "note_7f3d",
      "content_type": "application/pdf",
      "size_bytes": 48211,
      "payload_hash": "sha256:attachment-bytes",
      "availability": "missing_blob"
    }
  ],
  "attachment_ref_summaries": [
    {
      "attachment_id": "att_receipt_pdf",
      "parent_domain": "notes.note",
      "parent_object_id": "note_7f3d",
      "content_type": "application/pdf",
      "size_bytes": 48211,
      "payload_hash": "sha256:attachment-bytes",
      "availability": "missing_blob",
      "envelope_cursor": 126
    }
  ],
  "envelope_ranges": [
    {
      "domain": "chat.message",
      "from_cursor": 1,
      "to_cursor": 131,
      "count": 22,
      "purpose": "local_apply"
    }
  ],
  "conflicts": [
    {
      "conflict_id": "conf_note_7f3d_001",
      "domain": "notes.note",
      "object_id": "note_7f3d",
      "reason": "base_state_mismatch",
      "server_object_revision": 6,
      "client_object_revision": 4,
      "allowed_actions": [
        "overwrite",
        "duplicate_rename",
        "skip"
      ]
    }
  ],
  "encryption": {
    "policy": "server_trusted_v1",
    "ready": true,
    "key_status": {
      "client_unlock_required": false,
      "server_trusted_attested": true
    },
    "attestation": {
      "scope": "user_database_directory",
      "covers": ["Sync_v2.db", "ChaChaNotes.db"],
      "configured": true
    }
  },
  "warnings": [
    {
      "code": "sync_blob_transfer_not_supported",
      "domain": "attachment.ref",
      "attachment_id": "att_receipt_pdf",
      "parent_domain": "notes.note",
      "parent_object_id": "note_7f3d",
      "message": "Attachment reference can be restored, but the blob is not available through the M1 server contract."
    }
  ]
}
```

Top-level `summary` counts include all selected domains, including
`attachment.ref` metadata. Per-domain counts should add up to the top-level
count for each metric when all selected domains are represented in
`domain_plans`. The detail arrays in this example are abbreviated; production
M1 responses include all selected detail rows unless a future pagination
contract is explicitly added. Counts always describe the full selected restore
scope represented by the response.

`safe_applies` describe objects the client can apply without conflict.
`tombstones` describe objects the client should delete or hide locally.
`missing_blobs` and `attachment_ref_summaries` use the `attachment.ref`
metadata contract and always include the parent object reference. M1 reports
missing blobs unless the attachment reference availability states that the
server already has the blob through a non-M1 path.

## `POST /api/v1/sync/conflicts/resolve`

Records explicit user conflict decisions. Resolution creates new envelope or
resolution records and does not mutate historical envelopes.

### Request

```json
{
  "dataset_id": "ds_personal_01HZZ0",
  "device_id": "dev_chatbook_laptop",
  "resolutions": [
    {
      "conflict_id": "conf_note_7f3d_001",
      "action": "overwrite",
      "resolution_envelope": null
    },
    {
      "conflict_id": "conf_conv_42_001",
      "action": "duplicate_rename",
      "resolution_envelope": {
        "client_envelope_id": "client_env_DUP_CONV_001",
        "client_sequence": 19,
        "base_server_cursor": 131,
        "base_object_revision": null,
        "base_object_hash": null,
        "domain": "chat.conversation",
        "operation": "upsert",
        "object_id": "conv_42_copy",
        "parent_id": null,
        "schema_version": 1,
        "payload": {
          "title": "Research thread copy",
          "created_at": "2026-05-23T18:20:00Z",
          "updated_at": "2026-05-23T18:20:00Z"
        },
        "payload_hash": "sha256:conversation-copy-hash",
        "created_at_client": "2026-05-23T18:20:00Z",
        "deleted": false,
        "encryption_metadata": {
          "policy": "server_trusted_v1"
        }
      }
    }
  ]
}
```

### Response

```json
{
  "dataset_id": "ds_personal_01HZZ0",
  "server_cursor": 132,
  "resolved": [
    {
      "conflict_id": "conf_note_7f3d_001",
      "action": "overwrite",
      "status": "resolved"
    },
    {
      "conflict_id": "conf_conv_42_001",
      "action": "duplicate_rename",
      "status": "resolved",
      "envelope_id": "srv_env_000000000132",
      "server_cursor": 132
    }
  ],
  "rejected": []
}
```

M1 actions are:

- `overwrite`: resolve by overwriting the conflicting local object with the
  selected server object or by accepting a supplied resolution envelope for the
  same object.
- `duplicate_rename`: create a new object with a distinct object ID or
  title/name suffix when the domain supports it.
- `skip`: dismiss the conflict without applying either side.

## `POST /api/v1/sync/repair`

Replays accepted envelopes from `Sync_v2.db` into the user's materialized
`ChaChaNotes.db` projection. Repair is intended for failed apply recovery,
projection rebuilds, and administrative verification. It never rewrites
historical envelopes and does not replay unresolved conflict envelopes as
accepted changes.

The endpoint is scoped to the authenticated user's datasets. A user cannot run
repair against another user's dataset.

### Request

```json
{
  "dataset_id": "ds_personal_01HZZ0",
  "domains": [
    "notes.note",
    "chat.conversation",
    "chat.message"
  ],
  "since_cursor": 0,
  "failed_only": false,
  "limit": 500
}
```

If `domains` is empty, the server selects enrolled domains that have registered
materializers. `failed_only=true` selects accepted work that is not yet applied,
including pending-only mutation groups and single envelopes. A group resumes at
its first unapplied step and processes only its required ordered suffix.
`since_cursor` is inclusive of work after that cursor and must be non-negative.
The optional `limit` is a soft envelope limit over complete replay units: once a
mutation group is admitted, its required suffix is kept intact even if that one
group makes the response exceed `limit`. No group may exceed 1,000 members; an
oversized group fails closed with HTTP 413 and
`sync_restore_group_limit_exceeded`. `to_cursor` includes every member processed
from an admitted group, including members beyond the page cursor that discovered
the group.

### Response

```json
{
  "dataset_id": "ds_personal_01HZZ0",
  "domains": [
    "notes.note",
    "chat.conversation",
    "chat.message"
  ],
  "from_cursor": 0,
  "to_cursor": 131,
  "scanned_count": 38,
  "attempted_count": 38,
  "applied_count": 38,
  "failed_count": 0,
  "conflict_count": 0,
  "skipped_count": 0,
  "domain_results": [
    {
      "domain": "notes.note",
      "scanned_count": 12,
      "attempted_count": 12,
      "applied_count": 12,
      "failed_count": 0,
      "conflict_count": 0,
      "skipped_count": 0,
      "last_cursor": 129,
      "errors": []
    }
  ],
  "repair_status": {
    "status": "healthy",
    "failed_count": 0,
    "conflict_count": 0,
    "skipped_count": 0
  }
}
```

Envelope-level repair errors include `server_cursor`, `client_envelope_id`,
`domain`, `object_id`, a stable `error_code`, and a null message; raw exception
text is not returned. A response status of `repair_needed` means at least one
envelope failed replay, produced a conflict, or was skipped and remains pending,
including work blocked by an unavailable materializer. Only a run with no failed,
conflicting, or skipped work reports `healthy`.

Mutation-group results appear under `repair_status.mutation_groups`. For example:

```json
{
  "mutation_group_id": "server-origin-group-0001",
  "failing_step": 2,
  "error_code": "notes_organization_base_conflict",
  "retry_result": "blocked",
  "state": "conflict"
}
```

An applied prefix is never re-run merely to reach a failed suffix. A conflict is
reported at its exact step, blocks every later step, and is not converted into a
skip. The group object contains only the five fields shown above.

## Envelope Examples

### `notes.note` Upsert

```json
{
  "client_envelope_id": "client_env_NOTE_UPSERT_001",
  "client_sequence": 21,
  "base_server_cursor": 129,
  "base_object_revision": 6,
  "base_object_hash": "sha256:note-base",
  "domain": "notes.note",
  "operation": "upsert",
  "object_id": "note_7f3d",
  "parent_id": null,
  "schema_version": 1,
  "payload": {
    "title": "Trip notes",
    "content": "Updated outline.\n\n[[Packing list]]",
    "conversation_id": "conv_trip_planning",
    "message_id": "msg_source_42"
  },
  "payload_hash": "sha256:note-upsert",
  "created_at_client": "2026-05-23T18:30:00Z",
  "deleted": false,
  "encryption_metadata": {
    "policy": "server_trusted_v1"
  }
}
```

The canonical version-1 payload contains only `title`, `content`, nullable
`conversation_id`, and nullable `message_id`. Accepted title and Markdown bytes
are preserved exactly within the limits advertised by
`capabilities.domain_schemas.notes.note`; unknown fields are rejected. To
restore a deleted note, send an `upsert` with the full payload,
`routing_metadata.restore_intent` set to `true`, and a base cursor, revision,
and hash that exactly reference the current tombstone head.

### `chat.conversation` Upsert

```json
{
  "client_envelope_id": "client_env_CONV_UPSERT_001",
  "client_sequence": 22,
  "base_server_cursor": 110,
  "base_object_revision": 3,
  "base_object_hash": "sha256:conv-base",
  "domain": "chat.conversation",
  "operation": "upsert",
  "object_id": "conv_research",
  "parent_id": null,
  "schema_version": 1,
  "payload": {
    "title": "Research thread",
    "model": "gpt-4o-mini",
    "character_id": null,
    "updated_at": "2026-05-23T18:32:00Z"
  },
  "payload_hash": "sha256:conversation-upsert",
  "created_at_client": "2026-05-23T18:32:00Z",
  "deleted": false,
  "encryption_metadata": {
    "policy": "server_trusted_v1"
  }
}
```

### `chat.message` Append

```json
{
  "client_envelope_id": "client_env_MSG_APPEND_001",
  "client_sequence": 23,
  "base_server_cursor": null,
  "base_object_revision": null,
  "base_object_hash": null,
  "domain": "chat.message",
  "operation": "append",
  "object_id": "msg_aa21",
  "parent_id": "conv_research",
  "schema_version": 1,
  "payload": {
    "conversation_id": "conv_research",
    "role": "assistant",
    "sender": "assistant",
    "content": "Here is a concise summary.",
    "created_at": "2026-05-23T18:33:00Z"
  },
  "payload_hash": "sha256:message-append",
  "created_at_client": "2026-05-23T18:33:00Z",
  "deleted": false,
  "encryption_metadata": {
    "policy": "server_trusted_v1"
  }
}
```

### `attachment.ref` Upsert

```json
{
  "client_envelope_id": "client_env_ATT_REF_001",
  "client_sequence": 24,
  "base_server_cursor": null,
  "base_object_revision": null,
  "base_object_hash": null,
  "domain": "attachment.ref",
  "operation": "upsert",
  "object_id": "att_receipt_pdf",
  "parent_id": "note_7f3d",
  "schema_version": 1,
  "payload": {
    "attachment_id": "att_receipt_pdf",
    "parent_domain": "notes.note",
    "parent_object_id": "note_7f3d",
    "filename": "receipt.pdf",
    "content_type": "application/pdf",
    "size_bytes": 48211,
    "payload_hash": "sha256:attachment-bytes",
    "availability": "missing_blob"
  },
  "payload_hash": "sha256:attachment-ref-envelope",
  "created_at_client": "2026-05-23T18:34:00Z",
  "deleted": false,
  "encryption_metadata": {
    "policy": "server_trusted_v1"
  }
}
```

### Tombstone

```json
{
  "client_envelope_id": "client_env_NOTE_DELETE_001",
  "client_sequence": 25,
  "base_server_cursor": 129,
  "base_object_revision": 6,
  "base_object_hash": "sha256:note-base",
  "domain": "notes.note",
  "operation": "tombstone",
  "object_id": "note_7f3d",
  "parent_id": null,
  "schema_version": 1,
  "payload": {
    "deleted_at": "2026-05-23T18:35:00Z",
    "reason": "user_deleted"
  },
  "payload_hash": "sha256:note-tombstone",
  "created_at_client": "2026-05-23T18:35:00Z",
  "deleted": true,
  "encryption_metadata": {
    "policy": "server_trusted_v1"
  }
}
```
