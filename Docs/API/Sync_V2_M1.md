# Sync v2 M1 API Contract

Date: 2026-05-23
Status: Locked for M1 implementation
Scope: Server-connected Chatbook personal sync only

## Overview

Sync v2 M1 provides manual reliable sync and restore for an authenticated
user's personal Notes and Chat dataset. Chatbook local-only mode remains outside
this contract.

All endpoints are authenticated and scoped to the current user:

```text
/api/v1/sync
```

M1 public domains are exactly:

- `notes.note`
- `chat.conversation`
- `chat.message`
- `attachment.ref`

`attachment.ref` stores reference metadata only. Binary/blob transfer is not
part of M1.

The required `attachment.ref` payload metadata fields are `attachment_id`,
`parent_domain`, `parent_object_id`, `content_type`, `size_bytes`,
`payload_hash`, and `availability`.

## Shared Types

### Capability Shape

```json
{
  "protocol_version": "sync-v2-m1",
  "domains": [
    "notes.note",
    "chat.conversation",
    "chat.message",
    "attachment.ref"
  ],
  "operations": {
    "notes.note": ["upsert", "tombstone"],
    "chat.conversation": ["upsert", "tombstone"],
    "chat.message": ["append", "tombstone"],
    "attachment.ref": ["upsert", "tombstone"]
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
  "profile_bootstrapped": true,
  "user_id": "user_123",
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
      "attachment.ref"
    ],
    "operations": {
      "notes.note": ["upsert", "tombstone"],
      "chat.conversation": ["upsert", "tombstone"],
      "chat.message": ["append", "tombstone"],
      "attachment.ref": ["upsert", "tombstone"]
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
      "pending_apply": 0,
      "unresolved_conflicts": 0,
      "last_apply_status": "applied"
    },
    {
      "domain": "chat.conversation",
      "last_server_cursor": 110,
      "pending_apply": 0,
      "unresolved_conflicts": 0,
      "last_apply_status": "applied"
    },
    {
      "domain": "chat.message",
      "last_server_cursor": 128,
      "pending_apply": 0,
      "unresolved_conflicts": 0,
      "last_apply_status": "applied"
    },
    {
      "domain": "attachment.ref",
      "last_server_cursor": 126,
      "pending_apply": 0,
      "unresolved_conflicts": 0,
      "last_apply_status": "applied"
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
      "attachment.ref"
    ],
    "operations": {
      "notes.note": ["upsert", "tombstone"],
      "chat.conversation": ["upsert", "tombstone"],
      "chat.message": ["append", "tombstone"],
      "attachment.ref": ["upsert", "tombstone"]
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
safe applies, tombstones, missing blobs, attachment-reference summaries,
envelope ranges needed for local apply, and encryption/key status.

### Request

```json
{
  "dataset_id": "ds_personal_01HZZ0",
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
    "body": "Updated outline.",
    "tags": ["travel", "research"],
    "updated_at": "2026-05-23T18:30:00Z"
  },
  "payload_hash": "sha256:note-upsert",
  "created_at_client": "2026-05-23T18:30:00Z",
  "deleted": false,
  "encryption_metadata": {
    "policy": "server_trusted_v1"
  }
}
```

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
