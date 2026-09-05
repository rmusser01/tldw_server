# Sync v2 API

Sync v2 is the canonical client sync API for Chatbook and future clients. It
uses the existing `/api/v1/sync` route family and keeps the older media-only
`/send` and `/get` endpoints available as legacy compatibility routes.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/sync/capabilities` | Discover protocol version, domains, encryption policies, limits, and feature flags. |
| `POST` | `/api/v1/sync/devices/register` | Register or refresh a client device. |
| `POST` | `/api/v1/sync/datasets/enroll` | Create or join a personal or workspace dataset. |
| `GET` | `/api/v1/sync/restore-manifest` | Preview metadata-only sync inventory before restore. |
| `POST` | `/api/v1/sync/push` | Push a batch of client envelopes. |
| `GET` | `/api/v1/sync/pull` | Pull envelopes after a cursor with optional domain filters. |
| `GET` | `/api/v1/sync/conflicts` | List unresolved, resolved, or dismissed conflicts for a dataset. |
| `POST` | `/api/v1/sync/conflicts/{conflict_id}/resolve` | Resolve or dismiss a conflict, optionally with a resolution envelope. |
| `GET` | `/api/v1/sync/keys/recovery-bundle` | Retrieve stored opaque key-recovery bundles for an accessible dataset. |
| `POST` | `/api/v1/sync/keys/recovery-bundle` | Store opaque key-recovery metadata for encrypted datasets. |
| `POST` | `/api/v1/sync/attachments` | Store or deduplicate a small client-encrypted attachment payload for an accessible dataset. |

## Envelope Shape

Each envelope is scoped by `dataset_id`, `domain`, `entity_id`, operation, and
adapter version. It may include:

- `routing_metadata`: server-readable identity and routing hints.
- `payload_clear`: only safe metadata for private datasets.
- `payload_ciphertext`: client-encrypted private payload.
- `payload_hash` and `payload_size_bytes`: integrity and accounting metadata.
- `dependencies`, `base_version`, and `entity_version`: lineage hints used by
  adapters for conflict detection.

For `client_private_v1`, clear payload fields are allowlisted. Note titles,
bodies, chat content, source-cache text, wrapped keys, and similar private
values must not appear in `payload_clear`, restore manifests, error details, or
logs.

### `notes.note` contract

The `GET /api/v1/sync/capabilities` response advertises the versioned payload
contract in `domain_schemas.notes.note`. A version-1 `server_trusted_v1` upsert
uses exactly these fields:

- required strings: `title`, `content`
- optional nullable strings: `conversation_id`, `message_id`

The server preserves accepted title and Markdown content exactly. It rejects
unknown fields and values beyond the advertised limits rather than trimming,
escaping, or truncating them. A tombstone remains the `tombstone` operation.

Restore is an `upsert` with the full canonical payload and
`routing_metadata.restore_intent: true`. Its base cursor, object revision, and
object hash must identify the current tombstone head. Ordinary upserts against
deleted notes, stale restores, and restore requests against active notes become
whole-object conflicts. Replaying the same accepted restore envelope is
idempotent.

## Restore Manifest

`GET /api/v1/sync/restore-manifest` accepts repeated `dataset_id` and `domain`
query parameters. It returns:

- dataset ID, scope, workspace ID, encryption policy, and selected domains
- approximate per-domain envelope counts
- per-domain byte estimates
- attachment availability counts and size-class summaries
- unresolved conflict counts
- key-recovery readiness
- registered devices and last-seen metadata
- applied filters

The manifest is safe to show before a user unlocks local keys. For private
datasets, dataset metadata is redacted and no payload ciphertext or wrapped key
blob is included.

## Restore Pull

After previewing the manifest, a client pulls selected domains:

```text
GET /api/v1/sync/pull?dataset_id=dataset-1&device_id=device-b&cursor=0&domain=notes&domain=chat
```

The response contains encrypted envelopes. The server does not decrypt them.
The client decrypts locally, applies through domain-specific local adapters, and
keeps conflicts visible until resolved.

## Key Recovery Bundles

`POST /api/v1/sync/keys/recovery-bundle` stores client-generated wrapped dataset
keys. The server stores the wrapped key blob and KDF metadata as opaque material
and returns only non-secret storage metadata from the write response.

`GET /api/v1/sync/keys/recovery-bundle?dataset_id=...` returns active recovery
bundle records for a dataset the authenticated user can access. Optional
`device_id` and `key_purpose` query parameters narrow the result. This endpoint
is the only Sync v2 response that returns `wrapped_key_blob` and `kdf_metadata`;
restore manifests continue to expose only `key_recovery_available`.

## Attachment Uploads

`POST /api/v1/sync/attachments` stores small opaque attachment ciphertext for
later restore hydration. The server validates that the authenticated user owns
the dataset, that the requested domain is enrolled for that dataset, and that
the payload is within the advertised `max_attachment_bytes` capability.

Requests must use `client_private_v1`; plaintext/server-trusted attachment
storage is rejected. The response returns only storage metadata:

- `attachment_id`
- `dataset_id`
- `stored`, which is `false` for an idempotent duplicate upload
- `size_bytes`
- `payload_hash`

The route does not return `payload_ciphertext`, and validation/storage errors
use sanitized messages so malformed or oversized uploads do not echo ciphertext.
Restore manifests summarize persisted attachment availability and size classes
without exposing attachment payloads.

## Conflict Policy

### Listing pages

`GET /api/v1/sync/conflicts` returns at most 20 matching conflicts per request.
Use `limit` (1–20, default 20) and `offset` (non-negative, default 0), keeping
`dataset_id`, `status`, and `domain` filters unchanged while paging. Results
are ordered by creation time, then conflict ID. Advance the offset by the number
returned until a short or empty page is received; a 20-item response is not a
complete inventory. Offset paging is not a snapshot: resolving conflicts while
filtering by status can shift later pages, so fetch before resolving or restart
at offset 0 after mutations.

A page containing Personal Context conflicts requires `device_id` and a valid
`personal_context_activation_epoch` / `personal_context_continuity_token` pair.
Proof verification covers the selected page before any conflict is returned.
Such responses wrap the list in `conflicts` alongside `dataset_id` and the
verified `personal_context_exchange`; pages without Personal Context entries
retain the bare-list response, including empty pages.

### Resolution

Domain adapters decide whether an incoming envelope is accepted, rejected, or
converted into a durable conflict. Sync v2 currently has adapters for notes,
chat, workspaces, source cache, and media compatibility.

Conflict resolution actions are:

- `accept_local`
- `accept_remote`
- `merge`
- `dismiss`

When a resolution envelope is supplied, the server validates that it targets the
same dataset, domain, and entity before storing it and marking the conflict
resolved.

## Legacy `/send` And `/get`

`POST /api/v1/sync/send` and `GET /api/v1/sync/get` remain available for the
legacy media-sync flow. They are not the long-term generic sync contract. New
clients should use Sync v2 device, dataset, envelope, pull, conflict, and
restore-manifest endpoints.

## Known Limits

- Attachment upload persistence is limited to small client-encrypted payloads
  within `max_attachment_bytes`.
- Large binary media replication remains out of V1 scope.
- Embeddings and vector stores are treated as rebuildable or reference data in
  V1.
- `client_private_v1` recovery only works if the user configured and retained a
  recovery secret capable of unwrapping the stored recovery bundle.
