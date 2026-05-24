# Sync v2 M3 API Contract

Date: 2026-05-23
Status: Implemented foundation with explicit deferrals
Scope: Capability-gated additions to `/api/v1/sync` for polished multi-device sync

## Overview

M3 extends the M1/M2 API without replacing it. M1/M2 clients must continue to
bootstrap profiles, push/pull envelopes, preview restore, resolve conflicts,
and transfer blobs through the existing endpoints.

M3 adds capability-gated surfaces for:

- background sync policy and status;
- device lifecycle and revocation;
- workspace datasets;
- broader domain enrollment;
- richer conflict review;
- stricter encryption modes and key rotation;
- retention, compaction, and diagnostics.

## Current Implementation Status

The current M3 foundation implements device lifecycle, device authorization,
device acknowledgments, background policy/leases/status, workspace dataset
enrollment, workspace/source-cache/media metadata domains, key rotation,
retention dry-run, guarded retention compaction, and redacted diagnostics.

Deferred beyond this foundation: conflict summary and preview-resolution
endpoints, physical blob byte deletion, destructive envelope audit-log deletion,
workspace Notes/Chat materialization, broad collaborative content editing,
passphrase/device-key unlock UX, and full client-only encrypted editing.

## Capabilities

`GET /api/v1/sync/capabilities` returns the server-supported domains,
operations, encryption policy metadata, blob transfer metadata, compatibility
flags, warnings, and coarse support booleans. The current M3 foundation does
not expose a separate `features` map; clients should gate behavior by
advertised domains, policy lists, compatibility flags, endpoint availability,
and documented server errors.

```json
{
  "protocol_version": "sync-v2-m1",
  "min_supported_protocol_version": "sync-v2-m1",
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
    "media.keyword_link"
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
    "media.keyword_link": ["upsert", "tombstone"]
  },
  "encryption_policies": [
    "server_trusted_v1"
  ],
  "supports_restore_manifest": true,
  "supports_conflicts": true,
  "supports_attachments": false,
  "compatibility_flags": {},
  "warnings": []
}
```

Servers may advertise any subset of domains and encryption policies. Clients
must not assume every documented M3 endpoint is available just because a server
supports Sync v2.

### Derived Content Domain Policy

M3 capabilities must not include transcript, summary, embedding, or evaluation
artifact domains. Those objects are not rejected as product ideas; they are
deferred until Sync v2 can distinguish source-of-truth user data from
rebuildable compute output.

Current M3 classification:

- transcripts: deferred source-of-truth candidate, especially when
  user-corrected or segmented; no transcript body sync in M3;
- summaries: generated summaries are rebuildable cache, while user-pinned or
  edited summaries are deferred source-of-truth candidates;
- embeddings: rebuildable cache only, never advertised as an M3 sync domain;
- evaluation artifacts: split and defer; future eval project/config/dataset
  and human-label domains may be source-of-truth, while generated run outputs
  need artifact metadata, retention, and redaction policy first.

Clients must treat absent derived domains as intentional capability gating. A
client that needs offline derived content in M3 can rebuild it from synced
source/media objects or attach an explicit artifact through `attachment.ref` and
the M2 blob APIs. Metadata-only domains must not carry transcript bodies,
summary text, embedding vectors, generated metric payloads, or raw derived
artifacts.

## Device Lifecycle

### `GET /api/v1/sync/devices`

Lists devices visible to the authenticated user.

Query parameters:

- `dataset_id` optional. When omitted, return all user devices and default
  personal dataset status.
- `include_revoked` optional boolean, default `false`.

Each device record includes:

- `device_id`
- `display_name`
- `client_profile_id`
- `status`: `pending_authorization`, `active`, `paused`, or `revoked`
- `registered_at`
- `last_seen_at`
- `last_push_at`
- `last_pull_at`
- `cursor_lag_by_domain`
- `conflict_count`
- `replayable_failure_count`
- `blob_incomplete_count`
- `key_recovery_available`

### `PATCH /api/v1/sync/devices/{device_id}`

Updates non-security metadata:

```json
{
  "display_name": "Riley's MacBook Pro",
  "user_label": "work laptop",
  "paused": false
}
```

Changing `paused` affects server-side policy/status only. It does not assume
the server can wake or stop a standalone Chatbook process.

### `POST /api/v1/sync/device-authorizations`

Creates an authorization request for a new or pending device when the dataset
policy requires more than normal authenticated registration:

```json
{
  "dataset_id": "ds_personal_user_1",
  "device_id": "dev_new_laptop",
  "authorization_method": "existing_device",
  "idempotency_key": "auth-dev-new-laptop"
}
```

`server_trusted_v1` deployments may activate devices during normal profile
bootstrap. `passphrase_wrapped_v1`, `device_wrapped_v1`, and
`client_private_v1` datasets may leave the device in `pending_authorization`
until an existing active device, passphrase unlock, or recovery method approves
it.

Profile and status responses expose server-front-end write compatibility at the
dataset level and per-domain level through `server_frontend_mutation_enabled`
and `server_frontend_mutation_blockers`. When a server advertises
`client_private_v1`, `/api/v1/sync/capabilities` sets
`compatibility_flags.server_frontend_client_private_mutation=false` and returns
the `sync_server_frontend_client_private_disabled` warning.

### `POST /api/v1/sync/device-authorizations/{authorization_id}/approve`

Approves a pending device authorization:

```json
{
  "dataset_id": "ds_personal_user_1",
  "approving_device_id": "dev_existing_laptop",
  "idempotency_key": "approve-dev-new-laptop"
}
```

Approval must be scoped to the authenticated user, dataset, and encryption
policy. Approval responses must not include raw device keys, wrapped dataset
keys, recovery material, or passphrase-derived secrets.

### `POST /api/v1/sync/devices/{device_id}/revoke`

Revokes a device:

```json
{
  "reason": "lost_device",
  "revoke_key_records": true
}
```

Revoked devices cannot push, pull, upload, download, resolve conflicts, or
store key recovery bundles. Historical envelopes remain available for audit and
restore.

## Background Sync Policy And Status

### `GET /api/v1/sync/background-policy`

Returns server policy hints for a device/dataset:

```json
{
  "dataset_id": "ds_personal_user_1",
  "device_id": "dev_phone",
  "enabled": true,
  "minimum_interval_seconds": 300,
  "backoff_floor_seconds": 60,
  "max_batch_size": 500,
  "max_blob_bytes_per_run": 104857600,
  "respect_metered_networks": true,
  "maintenance_window": null,
  "paused_reason": null,
  "pending_local_changes": false,
  "updated_at": "2026-05-23T18:00:00Z"
}
```

### `PATCH /api/v1/sync/background-policy`

Stores user/device intent:

```json
{
  "dataset_id": "ds_personal_user_1",
  "device_id": "dev_phone",
  "enabled": false,
  "paused_reason": "user_paused",
  "pending_local_changes": true
}
```

### `POST /api/v1/sync/background-leases`

Creates or refreshes a short-lived per-device sync lease so one local profile
does not run overlapping sync workers:

```json
{
  "dataset_id": "ds_personal_user_1",
  "device_id": "dev_phone",
  "lease_id": "optional-client-known-lease",
  "ttl_seconds": 120
}
```

The response includes `status` (`acquired`, `refreshed`, or `held_by_other`),
`acquired`, `lease_id`, `expires_at`, and `updated_at`. If another unexpired
lease already exists for the same dataset/device, the endpoint returns the
active lease with `acquired: false` rather than replacing it.

The lease is advisory and does not replace idempotency guarantees on push,
pull, or blob endpoints.

### `GET /api/v1/sync/background-status`

Returns profile-level and per-domain status for background sync:

- last successful push/pull;
- current lease owner and expiry;
- pending local signal reported by the client;
- server-side conflict count;
- replayable failure count;
- quota pressure;
- selected restore completeness;
- attachment blob completeness counters for `attachment.ref`.

## Device Acknowledgments

### `POST /api/v1/sync/device-acknowledgments`

Records that a device has applied or verified durable state:

```json
{
  "dataset_id": "ds_personal_user_1",
  "device_id": "dev_phone",
  "domain_acks": [
    {
      "domain": "notes.note",
      "through_server_sequence": 512,
      "applied_at": "2026-05-23T18:45:00Z"
    }
  ],
  "blob_acks": [
    {
      "attachment_id": "att_1",
      "payload_hash": "sha256:...",
      "verified_at": "2026-05-23T18:45:00Z"
    }
  ],
  "idempotency_key": "ack-dev-phone-512"
}
```

Acknowledgments are required before M3 retention/GC can safely compact or
delete data for datasets with offline devices.

## Workspace Datasets

### `POST /api/v1/sync/datasets/enroll`

Creates or enrolls a dataset:

```json
{
  "dataset_id": "ds_workspace_research",
  "scope_type": "workspace",
  "workspace_id": "workspace_123",
  "domains": ["workspaces.workspace", "workspaces.source_ref"],
  "encryption_policy": "server_trusted_v1",
  "metadata": {"mode": "offline_sync"}
}
```

Workspace enrollment must fail closed when the user lacks workspace sync
permission or when requested domains/key policy are not allowed for that
workspace.

Every existing push/pull/restore/blob/key/conflict endpoint must re-check
workspace membership when `dataset.scope_type == "workspace"`.

## Conflict Review V2

### Deferred: `GET /api/v1/sync/conflicts/summary`

Returns conflict groups with safe summaries:

- dataset;
- domain;
- object ID;
- operation;
- conflict age;
- involved devices;
- base/current/incoming revisions and hashes;
- tombstone flags;
- redacted field-level metadata when available.

### Deferred: `POST /api/v1/sync/conflicts/preview-resolution`

Previews destructive or batch decisions before applying them:

```json
{
  "dataset_id": "ds_personal_user_1",
  "device_id": "dev_laptop",
  "decisions": [
    {
      "conflict_id": "conf_1",
      "action": "overwrite",
      "resolution_envelope": {
        "domain": "notes.note",
        "operation": "upsert",
        "object_id": "note_1"
      }
    }
  ]
}
```

The existing M1 batch resolution endpoint remains the implemented apply path.
Future preview support should return expected mutations, blocked decisions, and
warnings before applying destructive changes.

## Stricter Encryption And Key Rotation

M3 extends key records with policy and epoch metadata:

- `encryption_policy`
- `key_epoch`
- `active_from_server_sequence`
- `superseded_at`
- `revoked_at`
- `wrapped_for`: `server`, `passphrase`, `device`, or `recovery`

### `POST /api/v1/sync/key-rotation/preview`

Returns affected datasets, devices, key records, retained envelope ranges, and
blockers.

### `POST /api/v1/sync/key-rotation/commit`

Creates a new key epoch and records rewrap status. It must be idempotent and
must not expose key material in responses or logs.

Client-only encrypted datasets disable server-front-end mutation for opaque
fields because the server cannot materialize content it cannot inspect. Normal
Notes/Chat API writes routed through server-origin Sync return `409` with
`sync_server_frontend_client_private_disabled` before appending an envelope.

## Retention And Diagnostics

### `POST /api/v1/sync/retention/dry-run`

Returns compaction and deletion candidates without mutating data.

Request:

```json
{
  "dataset_id": "dataset-uuid",
  "device_id": "optional-device-id",
  "domains": ["notes.note", "attachment.ref"],
  "audit_mode": true,
  "minimum_envelope_age_seconds": 0,
  "minimum_tombstone_age_seconds": 0,
  "offline_restore_window_seconds": 0,
  "limit": 1000
}
```

Response candidates are redacted. They include object/blob references and
stable blocker codes only; they never include clear payloads, ciphertext,
wrapped keys, or blob bytes.

Candidate types:

- `envelope_compaction`: superseded accepted envelope that could be compacted
  into a future snapshot after all blockers clear.
- `tombstone_prune`: tombstone envelope that could be pruned after retention
  and audit policy allow it.
- `blob_gc`: server blob metadata that could be garbage-collected only after
  active attachment refs, device verification, restore windows, and audit
  policy allow it.

Initial blocker codes:

- `retention_audit_mode`
- `retention_unacknowledged_device`
- `retention_envelope_window_active`
- `retention_tombstone_window_active`
- `retention_restore_window_active`
- `retention_active_blob_reference`
- `retention_blob_unverified_by_device`

The dry-run endpoint always reports `mutation_performed=false`.

### `POST /api/v1/sync/retention/compact`

Applies policy-permitted compaction/GC after re-running the same blocker checks
as dry-run mode. This M3 foundation keeps the envelope audit log append-only:
eligible envelope and tombstone candidates only advance per-domain compaction
checkpoints. Eligible blob-GC candidates soft-delete available blob metadata so
the blob no longer appears restore-available; physical blob byte deletion is
left to a later storage-backend GC pass.

Request:

```json
{
  "dataset_id": "dataset-uuid",
  "device_id": "optional-device-id",
  "domains": ["notes.note", "attachment.ref"],
  "confirm": true,
  "apply_envelope_compaction": true,
  "apply_tombstone_prune": true,
  "apply_blob_gc": true,
  "minimum_envelope_age_seconds": 0,
  "minimum_tombstone_age_seconds": 0,
  "offline_restore_window_seconds": 0,
  "limit": 1000
}
```

`confirm=false` never mutates and returns
`retention_confirmation_required`. If any selected candidate has blockers, the
endpoint refuses the whole apply request and returns
`retention_blocked_candidates_present`.

Response:

```json
{
  "dataset_id": "dataset-uuid",
  "dry_run": false,
  "mutation_performed": true,
  "confirmation_required": false,
  "candidate_count": 2,
  "applied_count": 1,
  "blocked_count": 0,
  "skipped_count": 1,
  "blockers": [],
  "blocker_counts": {},
  "domain_compactions": [
    {
      "domain": "notes.note",
      "through_server_sequence": 10,
      "candidate_count": 1
    }
  ],
  "blob_gc": []
}
```

Responses are redacted and must not include payloads, ciphertext, wrapped keys,
blob storage keys, or blob bytes.

### `GET /api/v1/sync/diagnostics`

Returns redacted sync health for users or admins:

- dataset/domain envelope counts;
- materialization failure counts;
- conflict counts;
- cursor lag;
- blob store health;
- active upload pressure;
- key recovery and rotation blockers;
- retention dry-run summary.

Diagnostics must not include private payloads, ciphertext blobs, wrapped keys,
KDF salts, passphrase metadata beyond algorithm identifiers, or recovery
secrets.

Initial query parameters:

- `dataset_id` (required)
- `device_id` (optional requesting-device context; diagnostics still report
  profile-level device lag)
- `retention_limit` (optional dry-run scan limit)

Initial response shape:

```json
{
  "dataset_id": "ds_personal_user_1",
  "generated_at": "2026-05-24T02:00:00Z",
  "domains": [
    {
      "domain": "notes.note",
      "envelope_count": 42,
      "object_count": 17,
      "latest_server_sequence": 104,
      "failed_apply_count": 0,
      "unresolved_conflict_count": 1
    }
  ],
  "devices": [
    {
      "device_id": "dev_phone",
      "status": "active",
      "last_seen_at": "2026-05-24T01:58:00Z",
      "domain_lag": [
        {
          "domain": "notes.note",
          "last_pulled_sequence": 100,
          "latest_server_sequence": 104,
          "lag_count": 4
        }
      ]
    }
  ],
  "blob_health": {
    "blob_object_count": 3,
    "available_blob_bytes": 120000,
    "active_upload_count": 1,
    "reserved_blob_bytes": 32000,
    "quota_limit_bytes": 104857600
  },
  "key_summary": {
    "key_record_count": 1,
    "active_key_record_count": 1,
    "revoked_key_record_count": 0,
    "superseded_key_record_count": 0,
    "rewrap_pending_count": 0,
    "recovery_available": true
  },
  "retention": {
    "dry_run": true,
    "mutation_performed": false,
    "candidate_count": 2,
    "blocked_count": 2,
    "blocker_counts": {
      "retention_audit_mode": 2
    }
  }
}
```

This first diagnostics endpoint is dataset-scoped and user-visible. Global
admin/operator aggregation, audit-event search, and destructive retention/GC
execution remain separate later M3 work.

## Compatibility Requirements

- M1/M2 endpoints remain valid.
- All M3 additions are capability-gated.
- Revoked-device checks apply to all existing write and read endpoints.
- Workspace dataset checks apply to all existing dataset-scoped endpoints.
- Existing restore preview remains the first restore safety surface; M3 may add
  richer details but must not remove M2 completeness statuses.
