# Sync

The Sync module currently contains two protocol generations:

- Legacy media sync: `/api/v1/sync/send` and `/api/v1/sync/get`, backed by the
  Media DB `sync_log`.
- Sync v2: generic device, dataset, envelope, pull, conflict, key-recovery, and
  restore-manifest APIs under the same `/api/v1/sync` route family.

Sync v2 is the forward path for Chatbook and future clients. The legacy
media-only protocol remains available during migration and is treated as a
compatibility path, not as a second long-term sync product.

## Sync v2

Sync v2 is a metadata-first, domain-adapted sync substrate. Clients register a
device, enroll or join a dataset, push envelopes, pull envelopes by cursor and
domain, list or resolve conflicts, store opaque key-recovery metadata, and fetch
a restore manifest before hydrating a new device.

Core implementation:

- `tldw_Server_API/app/core/Sync/v2/models.py`: internal protocol records.
- `tldw_Server_API/app/core/Sync/v2/store.py`: persistence facade over
  `DB_Management/Sync_DB.py`.
- `tldw_Server_API/app/core/Sync/v2/service.py`: protocol invariants,
  orchestration, restore manifests, conflict lifecycle, and key records.
- `tldw_Server_API/app/core/Sync/v2/domain_adapters/`: domain-specific
  validation and merge/conflict policy for notes, chat, workspaces,
  source-cache, and media compatibility.
- `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`: API contracts.
- `tldw_Server_API/app/api/v1/endpoints/sync.py`: legacy and v2 routes.

### Protocol Invariants

- Device IDs are scoped to the authenticated user and cannot be taken over by a
  different user.
- Dataset IDs are scoped to the owning user for personal datasets.
- Pushes are idempotent by `(dataset_id, client_envelope_id)`. Reusing the same
  ID with different content is rejected.
- Envelopes must match the push dataset and device and must target an enrolled
  domain.
- Pulls support domain filters, cursors, page limits, and echo suppression.
- Private `client_private_v1` envelopes may include only routing-safe clear
  metadata. Private title/body/content values must be encrypted client-side.
- Restore manifests are metadata-only and must not include payload ciphertext,
  wrapped key blobs, or private plaintext.
- Conflict records remain visible until the client resolves or dismisses them.

### Restore Flow

1. Client registers a device with supported domains.
2. Client enrolls a personal or workspace dataset.
3. Client pushes encrypted domain envelopes.
4. Client optionally stores a key-recovery bundle. The server stores only opaque
   wrapped key metadata and never receives the user-held recovery secret.
5. A new device registers and calls `GET /api/v1/sync/restore-manifest`.
6. The manifest returns datasets, domains, approximate counts, byte estimates,
   attachment availability, size classes, device metadata, unresolved conflict
   counts, encryption policy, and key-recovery readiness.
7. The client selects dataset/domain IDs and calls `GET /api/v1/sync/pull`.
8. The client decrypts envelopes locally and applies them through local domain
   adapters, preserving conflicts for review.

### Conflict Policy

Sync v2 does not silently overwrite divergent private content. Domain adapters
accept safe independent changes, reject malformed envelopes, or create durable
conflict records. A client resolves a conflict with:

- `accept_local`
- `accept_remote`
- `merge`
- `dismiss`

When a merge supplies a resolution envelope, the service validates and stores
that envelope before marking the conflict resolved.

### Operational Limits

`SyncV2Settings` controls batch size, pull page size, payload bytes, attachment
bytes, attachment feature support, and restore-manifest scan limits. Attachment
upload is currently feature-detectable but returns not-enabled until persistence
lands.

See `Docs/API/sync-v2.md` for endpoint details.

## 1. Descriptive of Current Feature Set

- Purpose: Keep local client state (SQLite) and the server’s user‑scoped database consistent via an append‑only `sync_log` with incremental change IDs.
- Capabilities:
  - Client engine: push local changes, pull remote changes, apply in a single transaction, and persist progress markers in a state file.
  - Server endpoints: accept client changes, apply with authoritative timestamps, return filtered deltas to the requesting client.
  - Conflict handling: last‑write‑wins (LWW) using server authoritative timestamps; idempotency for duplicate/older updates.
  - Batching and limits: configurable batch size for push/pull; efficient stream processing.
- Inputs/Outputs:
  - Inputs: `SyncLogEntry` records (entity, uuid, operation, version, timestamp, payload, client_id), client progress markers.
  - Outputs: server responses with `changes: List[SyncLogEntry]` and `latest_change_id` for the user’s log.
- Related Endpoints:
  - POST `/api/v1/sync/send` — tldw_Server_API/app/api/v1/endpoints/sync.py:83
  - GET `/api/v1/sync/get` — tldw_Server_API/app/api/v1/endpoints/sync.py:129
  - Router mount — tldw_Server_API/app/main.py:3037
- Related Schemas:
  - `SyncLogEntry` — tldw_Server_API/app/api/v1/schemas/sync_server_models.py:18
  - `ClientChangesPayload` — tldw_Server_API/app/api/v1/schemas/sync_server_models.py:50
  - `ServerChangesResponse` — tldw_Server_API/app/api/v1/schemas/sync_server_models.py:80

## 2. Technical Details of Features

- Architecture & Data Flow:
  - Client: Read local `sync_log` after `last_local_log_id_sent`, push to `/send`; then pull from `/get` after `last_server_log_id_processed`; apply in a single transaction; update progress markers.
  - Server: Validate/auth, apply incoming changes with a single authoritative timestamp for the batch, return deltas excluding entries from the same client (`client_id != requesting_client_id`).
- Key Classes/Functions:
  - Client engine and operations — tldw_Server_API/app/core/Sync/Sync_Client.py:37 (ClientSyncEngine), :102 (run_sync_cycle), :131 (_push_local_changes), :184 (_pull_and_apply_remote_changes), :237 (_apply_remote_changes_batch)
  - Server processor (sync endpoint logic) — tldw_Server_API/app/api/v1/endpoints/sync.py:200 (ServerSyncProcessor), :214 (apply_client_changes_batch)
  - Endpoints — tldw_Server_API/app/api/v1/endpoints/sync.py:83 (POST /send), :129 (GET /get)
- Dependencies:
  - `MediaDatabase` transactions and typed exceptions for conflict/db errors.
  - Central HTTP server; client uses `requests` (via Sync_Client).
- Data Models & DB:
  - `sync_log(change_id, entity, entity_uuid, operation, timestamp, client_id, version, payload)` tracked per‑user DB.
  - Entities handled include `Media`, `Keywords`, junction `MediaKeywords` with explicit link/unlink paths.
  - FTS updates are performed explicitly where triggers are disabled (see notes in Issues.md).
- Configuration:
  - Client defaults in Sync_Client.py (SERVER_API_URL, CLIENT_ID, STATE_FILE, DATABASE_PATH, SYNC_BATCH_SIZE); recommended to externalize via env or config.
  - Server mount under `/api/v1/sync` (route policy‑gated in `main.py`).
- Concurrency & Performance:
  - Server performs blocking DB ops via `asyncio.to_thread` for predictable behavior; batch all changes in a single transaction.
  - Client uses transactional apply; idempotent operations to tolerate retries.
- Error Handling:
  - Network/HTTP: robust logging; endpoints map to 500/409/4xx; client treats network errors as non‑fatal for pull when push failed.
  - Conflicts: compare versions/client_id and fall back to LWW on server timestamp; idempotency skips duplicates/older versions.
- Security:
  - Endpoints require authenticated user (per‑user DB via `get_media_db_for_user`).
  - Client TODOs: add auth headers/tokens (see Issues.md recommendations).

## 3. Developer-Related/Relevant Information for Contributors

- Folder Structure:
  - `Sync_Client.py` — client engine (state file, push/pull/apply).
  - `Issues.md` — design review and improvement backlog.
- Extension Points:
  - Add new entity handlers if schema expands; ensure payload normalization and idempotency for create/update/delete/link/unlink.
  - Consider adapter for alternative transports (e.g., queued jobs) if needed.
- Coding Patterns:
  - Keep SQL minimal and parameterized; use `MediaDatabase.transaction()` for all mutations.
  - Prefer explicit, defensive payload parsing; maintain version/client_id invariants.
- Tests:
  - No dedicated tests yet. Add unit/integration tests under `tldw_Server_API/tests/Sync/` (client state transitions, conflict resolution, endpoint round‑trip with sqlite fixtures).
- Local Dev Tips:
  - Start server; configure `SERVER_API_URL`, `CLIENT_ID`, and `DATABASE_PATH` in `Sync_Client.py` (or via env) for quick trials.
  - Exercise endpoints via `/docs`: POST `/api/v1/sync/send`, GET `/api/v1/sync/get`.
- Pitfalls & Gotchas:
  - Ensure `CLIENT_ID` uniqueness per device/instance; state file should be per‑client.
  - Be mindful of FTS sync ordering when triggers are disabled (delete/update before main; insert after).
  - Link/unlink operations for junction tables require UUID lookups; skip gracefully if parents don’t exist locally.
- Roadmap/TODOs:
  - Implement authenticated client requests; externalize configuration; add end‑to‑end tests and monitoring/metrics for sync volume and latency.
