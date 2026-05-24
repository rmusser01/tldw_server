# Sync v2 M2 API Contract

Date: 2026-05-23
Status: Planned for M2 implementation
Scope: Server-connected Chatbook personal sync with blob restore completeness

## Overview

Sync v2 M2 extends the M1 personal dataset contract. It keeps the same envelope
domains and adds server-held blob transfer for `attachment.ref` records plus
restore completeness reporting.

Chatbook can still run as a standalone local application and never connect to a
tldw_server instance. M2 applies only when an authenticated user chooses to use
tldw_server as a sync peer or dumb front-end backend.

All endpoints are authenticated and scoped to the current user:

```text
/api/v1/sync
```

M2 continues to store:

- the append-only Sync v2 envelope log for restore and audit;
- accepted materialized Notes/Chat state for dumb-front-end mode;
- committed blob metadata and bytes for uploaded `attachment.ref` payloads.

## Capabilities

M2 servers advertise blob transfer and quota fields through the existing
capabilities response:

```json
{
  "protocol_version": "sync-v2-m2",
  "domains": [
    "notes.note",
    "chat.conversation",
    "chat.message",
    "attachment.ref"
  ],
  "blob_transfer": {
    "supported": true,
    "resumable_upload": true,
    "resumable_download": true,
    "chunk_checksums": true,
    "full_checksum": "sha256",
    "storage_backend": "local_fs"
  },
  "quota": {
    "max_blob_bytes": 104857600,
    "max_chunk_bytes": 4194304,
    "max_active_uploads": 8,
    "user_blob_quota_bytes": 10737418240,
    "reserved_blob_bytes": 0,
    "used_blob_bytes": 0
  }
}
```

If `blob_transfer.supported` is false, clients must treat the server as M1 for
binary content and may still restore `attachment.ref` metadata.

## Upload Flow

The primary M2 upload path is resumable:

1. `POST /api/v1/sync/blob-uploads`
2. `PUT /api/v1/sync/blob-uploads/{upload_id}/chunks/{chunk_index}`
3. `GET /api/v1/sync/blob-uploads/{upload_id}`
4. `POST /api/v1/sync/blob-uploads/{upload_id}/complete`
5. `DELETE /api/v1/sync/blob-uploads/{upload_id}`

Upload creation requires the dataset, registered device, parent domain/object,
attachment ID, content type, size, full `sha256:<hex>` payload hash, chunk size,
chunk count, and idempotency key.

Chunk upload requires `dataset_id`, `offset_bytes`, and a per-chunk
`sha256:<hex>` hash. Completion succeeds only after all chunks and the full
logical blob verify.

`POST /api/v1/sync/attachments` remains a small-blob convenience wrapper around
the same blob commit path. It is not a separate storage model.

Quota accounting is DB-backed:

- session creation reserves pending bytes;
- cancellation or expiry releases reservations;
- completion moves bytes from reserved to committed usage;
- dedupe by dataset and full payload hash must not double-charge committed
  blobs.

## Download Flow

Clients restore uploaded blobs through:

1. `GET /api/v1/sync/attachments/{attachment_id}/manifest?dataset_id=...`
2. `GET /api/v1/sync/attachments/{attachment_id}?dataset_id=...`

The manifest reports attachment ID, blob ID, size, content type, full hash,
availability, server chunk size, and chunk hashes. Whole-blob download is
available for small blobs in M2. Range/chunk download can be layered over the
same manifest contract as the storage adapter grows.

All blob reads are scoped by authenticated user and dataset ownership.

## Restore Completeness

`POST /api/v1/sync/restore/preview` returns a profile-level `restore_status`
with per-domain and per-blob detail:

- `metadata_ready`: selected metadata can be restored, but required blobs are
  missing or intentionally skipped through `metadata_only`.
- `blocked_by_conflicts`: selected Note or Chat conversation metadata conflicts
  with local inventory and needs explicit user handling.
- `blob_incomplete`: metadata is safe, but required selected blobs are not all
  available or locally verified.
- `content_complete`: selected metadata and required selected blobs are
  available from the server.
- `verified_complete`: the client reports selected blobs are locally verified.

`selected_object_ids`, `selected_attachment_ids`, `metadata_only`,
`local_inventory`, and `attachment_availability` let clients preview first-time
restore, partial restore, and restore into an existing profile.

Restore preview uses server-derived blob object state when M2 blob transfer is
enabled. Client-authored `attachment.ref.availability` is not enough to mark a
blob as server-available.

## Key Recovery Readiness

M2 remains `server_trusted_v1` for trusted/self-hosted deployments. Normal
authentication unlocks restore in this mode, but restore readiness still reports
whether a dataset has an active recovery bundle.

Recovery bundle writes require:

- dataset ownership by the authenticated user;
- `key_purpose: "dataset_recovery"`;
- a registered `device_id` when supplied;
- non-empty wrapping/KDF metadata with `algorithm` or `wrapping_algorithm` plus
  `salt`;
- an active non-revoked `rotation_of_key_record_id` when a rotation pointer is
  supplied.

Restore preview emits `sync_key_recovery_missing` when a selected dataset has no
active recovery bundle. Manifest and preview readiness exclude revoked records.
HTTP validation errors and logs must not expose wrapped key material or
wrapping/KDF secrets.

## Deferred To M3

M2 intentionally defers:

- client-only encrypted datasets;
- passphrase unlock;
- device-key unlock;
- full key-rotation workflows;
- workspace/shared-dataset key rules;
- background/scheduled sync;
- cloud object-store backends;
- physical blob garbage collection beyond safe tombstone metadata and abandoned
  upload cleanup.
