# Personal Context API

The Personal Context API exposes the server-owned copy of a user's unified
Chatbook/tldw_server profile at `/api/v1/personal-context`. Every operation is
authenticated and resolves the current user's `Personalization.db` before any
profile lookup or decryption.

## Contract

- Each authenticated user has at most one profile manifest.
- Canonical records use the shared `tldw_profile_core` schemas and the same
  immutable IDs and versions used by Chatbook.
- `POST /scopes/workspace` requires access to the matching user-owned server
  workspace and creates encrypted server-local workspace mapping metadata with
  the canonical scope. A canonical workspace scope received through Sync may be
  stored without that mapping and remain unbound. The stored mapping is
  resolvable for API or extension use, but no shipped canonical Personal Context
  server runtime or context-injection consumer currently calls
  `workspace_id_for_scope()` to use it.
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

Plaintext export requires the exact confirmation `EXPORT PLAINTEXT` and may
select global or workspace scopes. Recovery export requires `EXPORT RECOVERY`
plus a passphrase of at least twelve characters and returns a
`scrypt-aes-256-gcm` envelope containing all current scopes. Both modes contain
only the current canonical manifest, selected scopes, and records; they can
include user-only records. They exclude proposals, runtime policy, encrypted
server-local workspace mappings, keys, receipts, Sync state, and other
operational state. No supported server API, CLI, or application workflow imports
or restores a recovery envelope. It is not a complete or directly restorable
profile backup.

The server refuses `local_copy` deletion with
`server_local_copy_unsupported`; removing a device copy is a Chatbook-owned
operation. Chatbook's **Remove local profile** deletes canonical
`PersonalContextRepository` state but does not delete the server copy,
unregister the device, or clear separate `SyncStateRepository` artifacts,
staged encrypted envelopes, or dataset staging keys. Its recovery export has no
shipped import/restore caller, and **Finish secure removal** retries canonical
profile-key cleanup only.

Global deletion requires `DELETE EVERYWHERE` and the current purge generation.
It advances the generation barrier transactionally and retains the advanced
readable manifest head/version as the purge fence while deleting non-manifest
canonical heads and bodies plus server-local runtime state. The profile remains in
`purge_pending`. The endpoint does not publish a `personal_context.purge`
envelope, and synchronization acknowledgement completion is not implemented.
Mutations that authenticate, validate, resolve their owned objects, and reach
the existing-profile writable boundary return HTTP 409 with
`detail.code = "profile_purge_pending"`. Manifest recreation is unsupported:
surviving profile state prevents `POST /manifest` from creating a replacement.
Other authentication, request-validation, ownership, or lookup errors can occur
first, so not every request rejected while the barrier exists returns
`profile_purge_pending`.

## REST and Sync-v2 boundary

The authenticated REST API and Sync V2 are separate surfaces over the same
canonical `PersonalContextService` and encrypted repository. Sync device
registration, capability negotiation, bootstrap, and reviewed link completion
remain under `/api/v1/sync` rather than `/api/v1/personal-context`.

API-created workspace scopes and Sync-received workspace scopes also differ at
the peer-local mapping boundary. `POST /scopes/workspace` proves ownership and
atomically stores encrypted mapping metadata that APIs or extensions can
resolve. Inbound Sync materializes the canonical scope but does not invent a
server workspace binding, so that scope can remain unbound. No current API maps
an existing inbound scope, and no shipped canonical Personal Context server
runtime or context-injection consumer currently calls `workspace_id_for_scope()`
to use this mapping.

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

Eligible REST mutations now append encrypted publication batches in the same
canonical transaction. A bounded server relay installs those batches in Sync;
interrupted publication remains retryable. This does not make the currently
shipped Chatbook drain its ongoing queue.

Server purge journals its generation barrier, but remains pending because the
cross-device acknowledgement-completion lifecycle is not yet enabled.

### Ongoing activation (not yet enabled)

The server still advertises `ongoing_sync_version: 0`. Explicit version-one
bootstrap and activation-acknowledgement requests return HTTP 409 with
`personal_context_ongoing_sync_unavailable`; existing first linking is unchanged.
The implementation is preparation for the separate ongoing-sync rollout, not a
setting clients should bypass.

Once enabled, a previously completed link requests version-one bootstrap through
`POST /api/v1/sync/personal-context/bootstrap`. Its response pins one exact
eligible baseline, activation ID/digest, purge generation, publication watermark,
transport checkpoint, and continuity proof. The device must durably install that
baseline before sending the exact ID/digest and its local installation receipt to
`POST /api/v1/sync/personal-context/activation/acknowledge`.

Push, pull, conflict listing, and conflict resolution require both the current
canonical continuity proof and this device's durable acknowledgement. Copying
another device's proof or changing Sync metadata cannot activate a device.
Retries reuse exact receipts. Delivery baselines expire after 30 days; obtain a
fresh baseline rather than acknowledging an expired one. Capability downgrade
preserves stored work and acknowledgements while blocking version-one exchanges.

### User-directed conflict decisions (ongoing-sync contract)

The gated ongoing-sync contract uses the existing
`POST /api/v1/sync/conflicts/resolve` batch endpoint. Neither peer selects a
winner automatically. Each Personal Context item identifies the reviewed local
and remote envelopes and has an immutable idempotency key; requests also carry
the device's activation/continuity proof. A changed candidate requires fresh
review, not a blind retry with new IDs.

After linking, a client must not push its locally derived manifest. Such ingress
is invalid, not a user-choice conflict: the server sequences the shared manifest
from accepted semantic mutations. Initial-link reconciliation and server-issued
manifest publications remain separate supported paths.

| User choice | Action | Reviewed replacement |
| --- | --- | --- |
| Keep shared values | `skip` | None |
| Keep local values | `overwrite` | Explicit canonical target with selected local values |
| Merge | `overwrite` | Explicit canonical target with user-reviewed merged values |
| Keep both as distinct facts | `duplicate_rename` | New record ID and a noncolliding semantic key |

If two independently created record IDs claim the same semantic key, keep-local
or merge targets the established shared canonical record ID. The selected values
win because the user chose them, not because they came from a particular peer.
The losing incoming candidate is accounted for by the resolution receipt rather
than installed as a duplicate active fact. Clients must construct that explicitly
reviewed replacement; they must not expect the server to silently rewrite its ID.

The server's internal safety bound is 1,000 active conflicts per profile, not
1,000 lifetime decisions or a negotiated quota. When full, new conflicts remain
retryable without evicting existing candidates. Resolution frees an active slot
while preserving the exact decision receipt for retries.

Until rollout advertises ongoing-sync version 1, these semantics describe the
client integration contract, not an enabled end-to-end workflow.

## Transport deployment boundary

Deploy non-loopback API access behind authenticated HTTPS. This is an operator
requirement, not a Personal Context API enforcement rule. Chatbook accepts
server URLs using HTTP or HTTPS. Its runtime TLS verification defaults on but
can use a custom CA or be disabled through Network settings; **Test Connection**
always uses default httpx verification instead of that saved custom/off runtime
policy.
