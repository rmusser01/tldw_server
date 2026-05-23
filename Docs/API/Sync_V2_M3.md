# Sync v2 M3 API Contract Draft

Date: 2026-05-23
Status: Planning draft
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

## Capabilities

`GET /api/v1/sync/capabilities` adds M3 feature flags:

```json
{
  "protocol_version": "sync-v2-m3",
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
  "features": {
    "background_sync": true,
    "device_lifecycle": true,
    "device_authorization": true,
    "workspace_datasets": true,
    "device_acknowledgments": true,
    "conflict_review_v2": true,
    "strict_encryption_modes": true,
    "key_rotation": true,
    "retention_gc": false,
    "admin_diagnostics": true
  },
  "encryption_policies": [
    "server_trusted_v1",
    "passphrase_wrapped_v1",
    "device_wrapped_v1",
    "client_private_v1"
  ]
}
```

Servers may advertise any subset. Clients must not assume all M3 features are
available just because `protocol_version` starts with `sync-v2-m3`.

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
  "dataset_id": "ds_personal_user_1",
  "reason": "lost_device",
  "revoke_device_key_records": true,
  "confirm_current_device": false,
  "idempotency_key": "revoke-device-2026-05-23"
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
  "maintenance_window": null
}
```

### `PATCH /api/v1/sync/background-policy`

Stores user/device intent:

```json
{
  "dataset_id": "ds_personal_user_1",
  "device_id": "dev_phone",
  "enabled": false,
  "paused_reason": "user_paused"
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
- selected restore completeness.

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

### `GET /api/v1/sync/datasets`

Lists datasets available to the authenticated user:

- personal datasets owned by the user;
- workspace datasets where current workspace membership grants sync access.

### `POST /api/v1/sync/datasets`

Creates or enrolls a dataset:

```json
{
  "dataset_id": "ds_workspace_research",
  "scope": "workspace",
  "workspace_id": "workspace_123",
  "domains": ["workspaces.workspace", "workspaces.source_ref"],
  "encryption_policy": "server_trusted_v1",
  "mode": "offline_sync"
}
```

Workspace enrollment must fail closed when the user lacks workspace sync
permission or when requested domains/key policy are not allowed for that
workspace.

Every existing push/pull/restore/blob/key/conflict endpoint must re-check
workspace membership when `dataset.scope == "workspace"`.

## Conflict Review V2

### `GET /api/v1/sync/conflicts/summary`

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

### `POST /api/v1/sync/conflicts/preview-resolution`

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

The existing M1 batch resolution endpoint remains the apply path. Preview
returns expected mutations, blocked decisions, and warnings.

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

Client-only encrypted datasets may disable server-front-end mutation for opaque
fields because the server cannot materialize content it cannot inspect.

## Retention And Diagnostics

### `POST /api/v1/sync/retention/dry-run`

Returns compaction and deletion candidates without mutating data.

### `POST /api/v1/sync/retention/compact`

Applies policy-permitted compaction. This endpoint should remain disabled until
device acknowledgments, retention windows, and audit policy are enforced.

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

## Compatibility Requirements

- M1/M2 endpoints remain valid.
- All M3 additions are capability-gated.
- Revoked-device checks apply to all existing write and read endpoints.
- Workspace dataset checks apply to all existing dataset-scoped endpoints.
- Existing restore preview remains the first restore safety surface; M3 may add
  richer details but must not remove M2 completeness statuses.
