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
operation. Chatbook's **Remove local profile** deletes canonical
`PersonalContextRepository` state but does not delete the server copy,
unregister the device, or clear separate `SyncStateRepository` artifacts,
staged encrypted envelopes, or dataset staging keys. Its recovery export has no
shipped import/restore caller, and **Finish secure removal** retries canonical
profile-key cleanup only.

Global deletion requires `DELETE EVERYWHERE` and the current purge generation.
It advances the generation barrier transactionally, removes readable canonical
bodies and server-local runtime state, and leaves the profile in
`purge_pending`. The endpoint does not publish a `personal_context.purge`
envelope, and synchronization acknowledgement completion is not implemented.
All profile mutations return HTTP 409 with
`detail.code = "profile_purge_pending"` while that barrier is pending.

## REST and Sync-v2 boundary

The authenticated REST API and Sync V2 are separate surfaces over the same
canonical `PersonalContextService` and encrypted repository. Sync device
registration, capability negotiation, bootstrap, and reviewed link completion
remain under `/api/v1/sync` rather than `/api/v1/personal-context`.

The shipped first-link path supports capability negotiation, registered-device
bootstrap, a content-free reviewed reconciliation plan, link completion, and
publication of the resulting eligible Chatbook/server snapshot. Before
approval, bootstrap exchanges authentication/capability,
device-registration/public-key, display, schema/quota, and purge-generation
metadata. It also returns the server's current Sync-eligible canonical
snapshot, including record and proposal content, which Chatbook holds
transiently in memory. Durable review state and the visible plan remain
content-free, and no local Chatbook record or proposal content uploads before
approval.

After successful reviewed first linking, the peers have the same canonical
identities and bytes only for that resulting eligible snapshot. Later syncable
Chatbook mutations create encrypted local outbox entries, but no shipped
startup, background, Settings, or other production caller runs an ongoing
Personal Context sync cycle. **Overview → Manual Sync** invokes only Notes and
Chat, so those later entries remain queued locally.

The transport can validate and materialize inbound Chatbook-originated
`personal_context.manifest`, `personal_context.scope`,
`personal_context.record`, `personal_context.proposal`, and content-free
`personal_context.purge` envelopes. This is protocol capability, not a shipped
ongoing client lifecycle. The current products do not provide a dedicated
Personal Context queue/status or post-link conflict-resolution surface.

REST edits are not published to linked clients. They change the server copy
without appending a Personal Context Sync entry, so post-link edits can make the
peers diverge in either direction.

Server purge does not publish the protocol purge envelope and remains pending because acknowledgement completion is absent.

## Transport deployment boundary

Deploy non-loopback API access behind authenticated HTTPS. This is an operator
requirement, not a Personal Context API enforcement rule. Chatbook accepts
server URLs using HTTP or HTTPS. Its runtime TLS verification defaults on but
can use a custom CA or be disabled through Network settings; **Test Connection**
always uses default httpx verification instead of that saved custom/off runtime
policy.
