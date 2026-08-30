# Personal Context API

The Personal Context API exposes the server-owned copy of a user's unified
Chatbook/tldw_server profile at `/api/v1/personal-context`. Every operation is
authenticated and resolves the current user's `Personalization.db` before any
profile lookup or decryption.

## Contract

- Each authenticated user has at most one profile manifest.
- Canonical records use the shared `tldw_profile_core` schemas and the same
  immutable IDs and versions used by Chatbook.
- Workspace scopes require access to the matching user-owned server workspace.
  Workspace identifiers and labels remain encrypted server-local metadata.
- Server records must be syncable. `device_only` records belong to Chatbook and
  are rejected by this API.
- Record and scope writes advance the manifest revision transactionally.
- Mutations use expected version IDs; stale writes return HTTP 409 with
  `detail.code = "profile_version_conflict"`.
- Active records cannot share the same scope, kind, namespace, and subject.
  Collisions return HTTP 409 with
  `detail.code = "profile_semantic_key_collision"`.
- Search defaults to five results and accepts at most twenty. Canonical record
  payloads are limited to 16 KiB.
- Encrypted storage is capped at 1,000 record heads and 1,000 scope heads per
  profile so list, search, collision, and export work remain bounded.
- Pending proposals are limited to 200 per profile. Expired proposal bodies are
  replaced by content-free receipts before proposal writes as well as reads.
- Proposal history retains at most 1,000 heads; inserting another proposal
  removes the oldest terminal receipt. `GET /proposals` accepts `limit` up to
  200 and `offset` up to 1,000 so every retained receipt remains accessible.

Cross-user and unknown record IDs both return the same response:
`404 {"detail":"Personal context record not found"}`.

## Endpoints

| Area | Endpoints |
|---|---|
| Status and lifecycle | `GET /status`, `POST /manifest`, `GET /manifest` |
| Scopes | `GET /scopes`, `POST /scopes/workspace` |
| Records | `GET/POST /records`, `GET/PATCH/DELETE /records/{record_id}`, `POST /records/{record_id}/archive`, `POST /records/{record_id}/restore` |
| Agent proposals | `GET/POST /proposals`, `POST /proposals/{proposal_id}/review` |
| Server runtime | `GET/PATCH /runtime` |
| Data control | `POST /export`, `POST /purge` |

Runtime enablement is server-local and does not change Chatbook's local agent
settings. Pending proposal bodies are replaced by content-free receipts when
accepted or rejected.

## Export and deletion

Plaintext export requires the exact confirmation `EXPORT PLAINTEXT`. It may
select global or workspace scopes and includes user-only records, but excludes
keys, runtime policy, and peer-local receipts. Recovery export requires
`EXPORT RECOVERY` plus a passphrase of at least twelve characters and returns a
`scrypt-aes-256-gcm` envelope.

The server refuses `local_copy` deletion with
`server_local_copy_unsupported`; removing a device copy is a Chatbook-owned
operation. Global deletion requires `DELETE EVERYWHERE` and the current purge
generation. It advances the generation barrier transactionally, removes
readable canonical bodies and server-local runtime state, and leaves the
profile in `purge_pending` until synchronization acknowledgment is implemented.
All profile mutations return HTTP 409 with
`detail.code = "profile_purge_pending"` while that barrier is pending.

Sync endpoints and device registration are intentionally outside this API and
are introduced by the separate Personal Context synchronization contract.
