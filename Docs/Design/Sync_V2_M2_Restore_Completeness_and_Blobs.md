# Sync v2 M2 Restore Completeness And Blobs

Date: 2026-05-23
Status: Planned for M2 implementation
Parent: `TASK-490.12`

## Purpose

Sync v2 M1 establishes the authenticated personal-dataset protocol for Notes,
Chat conversation metadata, append-only Chat messages, tombstones, restore
preview, and metadata-only `attachment.ref` envelopes. M2 closes the largest
restore gap: a user who signs into a new Chatbook device must be able to tell
whether their selected profile restore is metadata-only, blocked by conflicts,
or complete with verified referenced blobs.

M2 keeps the M1 product model intact:

- Chatbook can remain a standalone local application that never talks to a
  server.
- Chatbook can act as a dumb front end against server-materialized state with
  no local sync.
- Chatbook can sync selected personal data to and from the server for
  offline/remote usage.
- The server stores both the append-only Sync v2 envelope log and accepted
  materialized user state so dumb-front-end mode works immediately.

## Scope

M2 includes:

- Resumable upload and download of blobs referenced by `attachment.ref`.
- Per-user and per-dataset quota policy, including pending upload reservation.
- Full-blob and per-chunk checksum verification.
- Server-derived blob availability status in restore manifest and restore
  preview.
- Profile-level restore completeness with per-domain and per-blob details.
- Selective restore controls for metadata-only restore, object/domain filters,
  and selected blob hydration.
- Server-unlocked key recovery hardening for trusted/self-hosted deployments.
- Larger batch/resume behavior needed for restore reliability.

M2 does not include:

- Workspace datasets or shared-workspace key rules.
- Media/library/source-cache domain sync.
- Background or scheduled sync.
- Client-only encrypted datasets, passphrase unlock, device-key unlock, key
  rotation, or device authorization/revocation.
- Blob retention garbage collection beyond safe tombstone metadata and
  abandoned upload cleanup.
- Cloud object-store integrations such as S3. The design should leave a storage
  interface for later object stores, but the M2 default is local self-hosted
  storage.

## Existing M1 Baseline

M1 already has these useful extension points:

- `SyncV2Settings.supports_attachments` and attachment limits.
- Capabilities with `blob_transfer` and `max_attachment_bytes`.
- `SyncAttachmentUploadRequest` and `SyncAttachmentUploadResponse`.
- A guarded `/api/v1/sync/attachments` endpoint that currently rejects because
  M1 does not support blob transfer.
- `sync_attachments` metadata storage.
- Restore manifest and preview counts for `attachment.ref`, `missing_blobs`,
  and client-supplied local attachment availability.
- Key recovery bundle APIs and `sync_key_records`.

The M2 implementation should evolve these instead of adding a parallel Sync v3
or a separate media sync API.

## Locked M2 Decisions

| Area | Decision |
| --- | --- |
| Blob scope | M2 transfers only blobs referenced by `attachment.ref` in the authenticated user's personal dataset. |
| Encryption posture | M2 stays on `server_trusted_v1` by default. Blob bytes must live under the same attested per-user encryption scope as `Sync_v2.db` and `ChaChaNotes.db`. |
| Client-only encryption | Leave protocol fields and storage metadata flexible, but do not require `client_private_v1` for M2. Stricter passphrase/device-key/client-only modes remain M3 work. |
| Storage model | Store blob metadata in Sync v2 DB tables and blob bytes in a local per-user blob store rooted below `Databases/user_databases/<user_id>/sync_blobs/`. Use an internal storage adapter so object stores can be added later. |
| Existing endpoint | Keep `/api/v1/sync/attachments` as a small-blob convenience path or compatibility wrapper, but the primary M2 path is resumable upload sessions. |
| Integrity | Require `sha256` for the full logical blob and for each uploaded chunk. Complete only after size, chunk count, chunk hashes, and full hash verify. |
| Quotas | Reserve quota on upload-session creation, charge committed blobs after completion, release reservations on cancel/expiry, and deduplicate by dataset plus full payload hash. |
| Availability | Restore APIs must use server-derived blob object state, not only client-authored `attachment.ref.availability`. |
| Restore completeness | Report one profile-level status first, with per-domain and per-blob details underneath. A new-device restore is not complete until required selected blobs are downloaded and verified or the user explicitly chooses metadata-only restore. |
| Conflicts | Preserve M1 safety: restoring into an existing profile requires explicit handling when the same Note or Chat conversation metadata already exists locally. Chat messages remain append-only by stable message ID. |
| Tombstones | Continue soft-delete/tombstone envelopes. Tombstoned refs hide the attachment from restore selection, but physical blob GC is deferred. |

## Data Model Additions

The exact migration can be adjusted during implementation, but M2 should model
these concepts explicitly.

### `sync_blob_objects`

One row per committed logical blob.

- `blob_id`: server-generated stable ID.
- `dataset_id`, `owner_user_id`.
- `attachment_id`: client stable attachment ID when known.
- `payload_hash`: full logical blob hash, including algorithm prefix such as
  `sha256:<hex>`.
- `content_type`, `size_bytes`.
- `encryption_policy`: normally `server_trusted_v1` in M2.
- `storage_backend`: `local_fs` for M2.
- `storage_key`: opaque key returned by the blob store adapter.
- `status`: `available`, `metadata_only`, `uploading`, `verify_failed`,
  `quarantined`, `deleted`.
- `ref_count`: number of active attachment refs that point to this blob.
- `created_at`, `updated_at`, `deleted_at`.
- `metadata_json`.

### `sync_blob_upload_sessions`

One row per resumable upload attempt.

- `upload_id`.
- `dataset_id`, `owner_user_id`, `device_id`.
- `attachment_id`, `payload_hash`, `content_type`, `size_bytes`.
- `chunk_size`, `chunk_count`.
- `reserved_quota_bytes`.
- `status`: `created`, `uploading`, `complete`, `cancelled`, `expired`,
  `verify_failed`.
- `idempotency_key`.
- `expires_at`, `created_at`, `updated_at`.
- `metadata_json`.

### `sync_blob_chunks`

One row per accepted chunk.

- `upload_id`, `chunk_index`.
- `offset_bytes`, `size_bytes`.
- `chunk_hash`.
- `storage_key` or temporary part path.
- `received_at`.

The local blob store should write chunks to temporary upload paths and atomically
move the verified blob into its final location after completion. All filesystem
paths must be derived from trusted server IDs and checked for path containment.

## API Shape

Capabilities should advertise M2 support explicitly:

```json
{
  "protocol_version": "sync-v2-m2",
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
    "user_blob_quota_bytes": 10737418240
  }
}
```

Primary upload flow:

1. `POST /api/v1/sync/attachments/uploads`
   creates or resumes an upload session and reserves quota.
2. `PUT /api/v1/sync/attachments/uploads/{upload_id}/chunks/{chunk_index}`
   stores one chunk after size, offset, and checksum validation.
3. `GET /api/v1/sync/attachments/uploads/{upload_id}`
   returns uploaded/missing chunks and expiry.
4. `POST /api/v1/sync/attachments/uploads/{upload_id}/complete`
   assembles, verifies, commits, and marks the blob `available`.
5. `DELETE /api/v1/sync/attachments/uploads/{upload_id}`
   cancels the session and releases reserved quota.

Primary download flow:

1. `GET /api/v1/sync/attachments/{attachment_id}/manifest?dataset_id=...`
   returns the committed blob manifest, chunk map, hashes, and availability.
2. `GET /api/v1/sync/attachments/{attachment_id}/chunks/{chunk_index}?dataset_id=...`
   downloads one chunk for resumable clients.
3. `GET /api/v1/sync/attachments/{attachment_id}?dataset_id=...`
   may stream the whole blob for small blobs and should support HTTP range
   requests once the storage adapter can serve ranges safely.

The existing `POST /api/v1/sync/attachments` should become a small-blob wrapper
around the same validation and commit path, not a separate storage path.

## Restore Completeness Contract

Restore APIs should return a profile-level `restore_status` and per-domain
details. Suggested statuses:

- `metadata_ready`: selected Notes/Chat/attachment refs can be restored, but
  at least one selected blob is missing, skipped, or not requested.
- `blocked_by_conflicts`: a selected Note or Chat conversation metadata object
  conflicts with local inventory and needs explicit user action.
- `blob_incomplete`: metadata is safe to apply, but required selected blobs are
  not all available or not all verified locally.
- `content_complete`: selected metadata and required blobs are available on the
  server and ready for restore.
- `verified_complete`: the client reports that selected metadata applied and
  selected blobs downloaded with matching hashes.

Per-domain detail should include:

- `domain`.
- `selected_count`, `safe_apply_count`, `conflict_count`, `tombstone_count`.
- `required_blob_count`, `available_blob_count`, `missing_blob_count`,
  `verified_blob_count`.
- `status` and warnings.

Per-blob detail should include:

- `attachment_id`, `payload_hash`, `size_bytes`, `content_type`.
- `parent_domain`, `parent_object_id`.
- `server_availability`.
- `download_status` when supplied by the client.
- `required_for_restore`.
- `warnings`.

For first-time restore into an empty profile, `content_complete` can be reached
without object conflict review when local inventory is empty. For restore into
an existing profile, any same-ID Note or Chat conversation metadata mismatch
keeps the profile at `blocked_by_conflicts` until the client supplies explicit
resolution.

## Quota And Abuse Controls

M2 should enforce:

- `max_blob_bytes`.
- `max_chunk_bytes`.
- `max_active_uploads_per_user`.
- `max_pending_upload_bytes_per_user`.
- `max_committed_blob_bytes_per_user`.
- Optional per-dataset quota override.
- Upload-session expiration.
- Idempotency by `(dataset_id, device_id, idempotency_key)`.

Quota accounting must not rely only on blob files found on disk. The Sync v2 DB
is the quota ledger. Periodic reconciliation can be added later, but M2 should
include a manual repair/check helper or service method for tests.

## Key Recovery Hardening

M2 remains server-unlocked for trusted/self-hosted deployments, but restore
should be clearer and safer:

- Capabilities and restore manifest must show whether the dataset has at least
  one active recovery bundle for the current policy.
- Recovery bundle writes must validate that the authenticated user owns the
  dataset, `key_purpose` is `dataset_recovery`, `device_id` is registered to
  that user when supplied, and wrapping metadata includes a non-empty
  `algorithm` or `wrapping_algorithm` plus `salt`. Nested `wrapping` or `kdf`
  metadata may be accepted for client compatibility, but validation errors must
  stay generic.
- `rotation_of_key_record_id`, when supplied, must point to an active
  non-revoked recovery bundle for the same user and dataset. M2 does not add a
  complete key-rotation workflow; it only prevents stale or revoked rotation
  pointers from being accepted.
- Revoked recovery records must be excluded from readiness calculations.
- Logs, HTTP errors, and restore readiness responses must not include wrapped
  key material or KDF/wrapping secrets. Invalid bundles should surface as the
  generic `sync_validation_failed` API error.
- Restore preview should warn when a selected dataset has no active recovery
  record even if normal authentication currently unlocks it. The stable warning
  code is `sync_key_recovery_missing`.

Passphrase unlock, device-key unlock, client-only encryption, and full
key-rotation workflows remain M3 work.

## Testing Strategy

M2 implementation should be test-first and split across PR-sized tasks:

- Model validation tests for upload sessions, chunk manifests, quota responses,
  restore completeness responses, and key recovery status.
- Store tests for migrations, idempotent sessions, chunk writes, quota
  reservation/release, dedupe by payload hash, and tombstone effects.
- Service tests for upload completion, checksum failures, resume after partial
  upload, download manifests, selective restore, and conflict-blocked restore.
- API tests for each endpoint, including safe error messages.
- E2E restore tests that create Notes, Chat metadata/messages, attachment refs,
  upload blobs, restore into empty and non-empty local inventories, and verify
  completeness status transitions.
- Bandit over touched production paths before each implementation PR is
  considered done.

## Rollout Notes

M2 can ship behind `supports_attachments=False` until the upload/download paths,
quota accounting, and restore completeness tests pass together. Once enabled,
capabilities should move from `sync-v2-m1` to `sync-v2-m2` or advertise M2 blob
features while preserving M1 domain semantics. No existing Sync v2 clients are
assumed, so incompatible M1 preview fields can be extended in place as long as
the M1 tests are updated deliberately.
