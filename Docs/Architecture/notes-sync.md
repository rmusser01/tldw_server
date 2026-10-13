# Notes, workspaces, and Sync v2

This document describes the server side of cross-device sync: the `/api/v1/sync` surface (legacy v1 plus the v2 protocol), the envelope model with its encryption policies, per-domain adapters and materializers, conflicts, blobs, and the personal-context domains.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Routes | `app/api/v1/endpoints/sync.py` | all `/api/v1/sync/*` (capabilities, profile bootstrap, devices, datasets, push/pull, conflicts, restore, keys, blobs, legacy send/get) |
| Contracts | `app/api/v1/schemas/sync_v2_models.py` | API Pydantic models; `ConflictResolutionAction` |
| Service | `app/core/Sync/v2/service.py` | `SyncV2Service` — protocol invariants, push, pull, restore, conflicts, keys, retention, diagnostics |
| Store | `app/core/Sync/v2/store.py` | `SyncV2Store` — persistence facade with materialization/projection/conflict/retention guards |
| Models | `app/core/Sync/v2/models.py` | `SyncEnvelopeCreate`/`SyncEnvelope`/`SyncConflict`/`SyncDataset`; domain and operation literals |
| Factory | `app/core/Sync/v2/factory.py` | `sync_v2_service_for_user`, env settings, per-user DB path, workspace access checker |
| Security | `app/core/Sync/v2/security.py` | server-trusted readiness (at-rest mode + attestation), fail-closed |
| Blobs | `app/core/Sync/v2/blob_store.py` | server-held blobs, chunked resumable upload/download |
| Domains | `app/core/Sync/v2/domain_adapters/`, `materializers/` | per-domain evaluate/merge policy; server-side projection writers |
| Personal context | `v2/personal_context_*.py`, `profile.py` | PC activation/relay, profile bootstrap |
| Schema | `app/core/DB_Management/Sync_DB.py` | `sync_devices/datasets/envelopes/current_heads/device_cursors/conflicts/key_records/blob_*` DDL |
| Legacy | `app/core/Sync/Sync_Client.py`, `server_sync_processor.py` | v1 media sync (sync_log, last-write-wins) |
| Notes core | `app/core/Notes/Notes_Library.py` | `NotesInteropService` |

## Push/pull cycle (dataflow)

1. `GET /capabilities` — protocol `sync-v2-m1`, domains, adapter versions, encryption policies, limits (batch 100, pull page 100, envelope payload 256 KiB).
2. Device registration (user-scoped ids; the server-origin device id is reserved and rejected on push).
3. Dataset enrollment (personal or workspace scope; workspace access checked against the ChaCha workspace with `sync` permission).
4. `POST /push`: per-envelope validation in order — batch limit, dataset/device match, reserved personal-context routing, payload size — then the domain adapter evaluates; accepted envelopes insert atomically with a per-dataset monotonic cursor; adapter conflicts store conflict rows. Idempotency is `(dataset_id, client_envelope_id)`; reuse with different content → 409 `sync_idempotency_conflict`. Stranded tombstones recover before evaluation; accepted envelopes materialize into server projections under a guard.
5. `GET /pull`: requires an active device; intersects requested domains with device-acknowledged adapter versions; pages from the cursor with own-device echo suppression; cursors are HMAC-signed opaque tokens (1 h TTL).
6. Personal-context domains additionally require a verified active exchange (activation epoch + continuity token).
7. Conflicts: list (≤20/page) and **batched** resolution; personal-context conflicts cannot be resolved singly — exact batched review only.
8. Restore: a metadata-only manifest (counts, byte estimates, attachment availability, key readiness; 10,000 scan cap) → ordered preview → client pulls, decrypts, and applies locally.

## Envelope model and crypto

Domains cover notes (note/provenance/keyword/folder/link/task/moodboard/studio_document), chat (conversation/message), attachments, workspaces (workspace/source_ref), source_cache, media, and personal_context (manifest/scope/record/proposal/purge). Operations: `upsert | append | tombstone`. Encryption policies: `server_trusted_v1` (the M1 default), `passphrase_wrapped_v1`, `device_wrapped_v1`, `client_private_v1`; mutation groups carry `mutation_group_id/step/step_count/plan_hash`, validated on the model. Server-trusted readiness is **fail-closed**: the at-rest mode must be `encrypted_volume` or `managed_storage`, the feature flag on, and an attestation covering the user DB directory present — otherwise writes are blocked. There is no literal server-side outbox table; the client outbox model is mirrored by apply statuses, current heads, and device cursors/acks.

## Config keys (exact)

`SYNC_V2_ENABLE_BLOB_TRANSFER` (default false), `SYNC_V2_MAX_BLOB_BYTES`, `SYNC_V2_MAX_CHUNK_BYTES` (4 MiB), `SYNC_V2_MAX_ACTIVE_BLOB_UPLOADS` (8), `SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS` (86,400; 0 = no expiry, deliberate compatibility), `SYNC_V2_USER_BLOB_QUOTA_BYTES`, `SYNC_V2_AT_REST_ENCRYPTION_MODE`, `SYNC_V2_SERVER_TRUSTED_ENABLED`, `TLDW_PERSONAL_CONTEXT_MASTER_KEY` (base64, must decode to 32 bytes or PC is blocked). Per-user DB: `Databases/user_databases/<user_id>/Sync_v2.db`.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Idempotency reuse, different content | 409 `sync_idempotency_conflict` |
| Batch > 100 / payload > 256 KiB | Per-envelope rejection (`batch_limit_exceeded` / store size error) |
| `stop_on_conflict` | Later envelopes rejected `stopped_after_conflict` (retryable) |
| Resolving an already-resolved conflict | Store error unless an idempotent replay |
| PC master key missing/wrong | Capabilities report a blocker; PC endpoints fail bootstrap |
| Abandoned blob upload | Holds quota until TTL expiry (read-time predicate, ADR-048) |

## Verified gotchas

1. **Conflict actions are `overwrite | duplicate_rename | skip`** — the Sync README and parts of `Docs/API/sync-v2.md` still describe `accept_local/accept_remote/merge/dismiss` (stale vocabulary).
2. `Docs/API/sync-v2.md` documents a single-conflict resolve route; only the **batched** route exists.
3. Blob transfer exists but is capability-gated off by default.
4. No `SYNC_V2_*` keys appear in the root `Env_Vars.md` — the env reference lags code.
5. Sync handlers are sync (`def`) FastAPI functions running in the threadpool.

## Governing decisions

ADR-031 (notes capability sync domains), ADR-034 (durable server-origin mutation batches), ADR-037/039 (canonical notes link/task sync), ADR-038 (attachment registry and blob lifecycle), ADR-040 (moodboards/studio authority), ADR-048 (blob expiry as read-time predicate), ADR-065 (knowledge provenance). Living reference: `Docs/API/sync-v2.md` (with the two stale spots above).

## Related docs

- [research.md](./research.md), [database-layer.md](./database-layer.md)
- The chatbook counterpart is `server-sync.md` in the tldw_chatbook repo's architecture reference
