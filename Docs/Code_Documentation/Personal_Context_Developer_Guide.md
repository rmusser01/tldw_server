# Personal Context developer guide

Personal Context is one canonical profile contract implemented by Chatbook and
tldw_server. The server is an authenticated home peer: it owns an encrypted
canonical copy for each user, exposes REST operations, and participates in
reviewed first-link convergence by returning the bootstrap snapshot, accepting
and materializing Chatbook's approved first-link envelopes, and recording link
completion. Its transport can validate and materialize later Personal Context
Sync V2 envelopes, but the shipped Chatbook does not invoke that transport for
later profile mutations. The [Personal Context API
reference](../API-related/Personal_Context_API.md) lists the REST surface; the
[operator guide](../User_Guides/Server/Personal_Context_Profile.md) covers
deployment and recovery workflows.

The source-only architecture authorities are the [server
design](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Design/2026-08-30-personal-context-profile-server-design.md)
and [Personal Context
ADR](https://github.com/rmusser01/tldw_server/blob/dev/backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md).

## Contract and trust boundary

- The shared package is pinned to `tldw-profile-core==0.1.0`. Its models,
  validation, fixtures, and canonical bytes are the application contract.
- The live parity and canonical-fixture tests are the authority for the current
  contract digest and exact canonical bytes. Do not copy a digest into
  documentation or use a documented value as compatibility evidence.
- Shared Core objects and Sync V2 envelopes are separate contracts. Change
  canonical models in Shared Core first; change transport domains, routing, or
  envelope behavior separately.
- Authentication and user storage resolution precede profile lookup. Each
  authenticated user has at most one manifest in that user's existing
  `Personalization.db`.
- The current Sync policy is `server_trusted_v1`. The authorized home server
  can read syncable canonical content to validate and materialize it before
  encrypting its copy at rest. This is not end-to-end encryption from the home
  server.
- A non-loopback deployment must put the authenticated API behind TLS/HTTPS so
  credentials and profile content are protected in transit. Chatbook does not
  enforce this deployment requirement: it accepts HTTP and HTTPS, and its
  runtime TLS verification may use default trust, a custom CA, or disabled
  verification. Its **Test Connection** probe always uses default httpx
  verification rather than the saved custom/off runtime policy.

### Shared contract

<!-- shared-personal-context-contract:start -->
- `tldw_profile_core` defines the versioned canonical profile object models, exact canonical bytes, interview/tool contracts, serialization, and validation used by both peers. Sync-v2 transport envelopes are a separate contract.
- After successful reviewed first linking, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for the eligible snapshot resulting from the user-approved content-free reconciliation plan.
- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. Reviewed first linking publishes the eligible snapshot resulting from the user-approved content-free reconciliation plan. Later syncable Chatbook mutations create encrypted local outbox entries, but the current shipped app does not run an ongoing Personal Context sync cycle, so those post-link changes remain queued locally. Purge production and distribution are not wired end to end.
- Each peer retains its own at-rest ciphertext and keys, local database rows, runtime permissions, conflict-review metadata, acknowledgement tracking, and other operational state.
<!-- shared-personal-context-contract:end -->

| Published during successful reviewed first linking when eligible | Not published by the shipped ongoing application lifecycle |
| --- | --- |
| Canonical manifest in the snapshot resulting from the user-approved content-free reconciliation plan | Later syncable Chatbook mutations: encrypted outbox entries are created but no shipped ongoing Personal Context caller sends them |
| Required global and linked-workspace scopes in that snapshot | Ordinary server REST mutations: the server copy changes but no Personal Context Sync entry publishes them to Chatbook |
| Eligible record heads, tombstones, and proposal review state selected by reconciliation, including approved interview answer content after it becomes a canonical record payload | Device-only or non-syncable records |
| Exact canonical object identities, versions, and bytes for those eligible objects | Runtime agent authority grants, tool availability, local workspace mappings, and enablement |
| — | Peer-local at-rest encryption/recovery keys, local undo data, caches, ciphertext, database row identities, conflict-review metadata, acknowledgement tracking, and other operational state |
| — | Encrypted interview draft and transcript objects are not Sync payloads as such; adaptive interview requests still send prior raw answers to the configured provider, while approved answer content may become a syncable canonical record as described at left |

## Storage, key custody, and encryption

The server extends the authenticated user's `Personalization.db`; Sync storage
is transport history and binding state, not the canonical application store.
Immutable object versions, current heads, optimistic compare-and-set checks,
semantic uniqueness, and purge fences remain behind the repository and service
boundaries.

The key hierarchy has three levels:

1. The explicitly configured 32-byte server root key wraps independent random
   per-profile encryption and integrity keys.
2. The profile encryption key wraps a fresh random data-encryption key for each
   immutable object version.
3. That object key encrypts the canonical bytes with associated data binding
   the profile, object type, object identity, version, and envelope schema.

Missing, malformed, changed, or orphaned root/profile key material locks the
profile and never causes replacement keys to be generated. Integrity is checked
before canonical model parsing. Each peer owns its own at-rest keys and
ciphertext; those keys are never synchronized.

The server separately owns the Sync integrity key used to authenticate
canonical transport bytes. During bootstrap, it wraps that key for the
authenticated registered Chatbook device. The wrapped key record is Sync key
distribution, not profile-content synchronization and not transfer of either
peer's at-rest key custody.

## Component ownership

All paths below are relative to the repository root.

| Component | Responsibility |
| --- | --- |
| `tldw_Server_API/app/api/v1/API_Deps/personal_context_deps.py` | Authenticated, per-user `PersonalContextService` dependency assembly and workspace ownership binding. |
| `tldw_Server_API/app/api/v1/schemas/personal_context.py` | Strict HTTP request and bounded response schemas. |
| `tldw_Server_API/app/api/v1/endpoints/personal_context.py` | REST-to-service translation and content-free error mapping. |
| `tldw_Server_API/app/core/DB_Management/Personal_Context_Key_Store.py` | Real root/profile key-custody owner, including strict root-key loading and wrapped profile keys. |
| `tldw_Server_API/app/core/Personalization/personal_context_key_provider.py` | Compatibility re-export of the database-owned key provider; it does not own custody. |
| `tldw_Server_API/app/core/Personalization/personal_context_crypto.py` | Authenticated object envelopes and symmetric key-wrapping primitives. |
| `tldw_Server_API/app/core/Personalization/personal_context_repository.py` | Stable service import for the database-owned repository that manages immutable versions, heads, and fences. |
| `tldw_Server_API/app/core/DB_Management/Personal_Context_Repository.py` | Real canonical object SQL owner, including encryption, version/head CAS, semantic uniqueness, key rotation, and purge fencing. |
| `tldw_Server_API/app/core/Personalization/personal_context_service.py` | Canonical authenticated business boundary for reads, mutations, bootstrap snapshots, inbound Sync projection, exports, and purge. |
| `tldw_Server_API/app/core/Personalization/personal_context_export.py` | Explicit-confirmation plaintext export and passphrase-encrypted recovery export helpers. |
| `tldw_Server_API/app/core/Personalization/personal_context_runtime_policy.py` | Encrypted peer-local enablement and workspace mapping metadata models; their presence does not imply a shipped runtime consumer. |
| `tldw_Server_API/app/core/Sync/v2/profile.py` | Capability-gated bootstrap, dataset binding, registered-device integrity-key wrapping, and reviewed link completion. |
| `tldw_Server_API/app/core/Sync/v2/domain_adapters/personal_context.py` | Whole-object transport validation, HMAC verification, lineage checks, and encryption before Sync persistence. |
| `tldw_Server_API/app/core/Sync/v2/materializers/personal_context.py` | Inbound accepted-envelope projection through the authenticated owner service, with content-free failure/conflict outcomes. |

Endpoints, agents, Sync code, migration or compatibility paths, and future
publishers must never access Personal Context profile tables directly. They use
`PersonalContextService`, which delegates durable invariants to the repository.

## REST flow

`authentication -> PersonalContextService -> encrypted repository -> response`

The dependency resolves exactly one authenticated user's database and
workspace-access check. The endpoint validates the HTTP shape, the service
enforces profile authority and lifecycle rules, and the repository performs the
encrypted transactional read or write. Expected version IDs provide optimistic
concurrency; unknown and cross-user opaque IDs receive the same not-found
response.

REST runtime policy and exports are server-local operations. REST record and
proposal mutations change the canonical server copy, but no server-origin
publisher currently appends those edits to the linked Personal Context Sync
streams.

`POST /scopes/workspace` is stricter than inbound canonical scope
materialization. The REST path proves that the authenticated user owns the
server workspace and atomically stores encrypted `WorkspaceRuntimePolicy`
mapping metadata with the new scope. The Sync apply path can accept a canonical
workspace scope without creating or guessing that peer-local mapping. Such an
unbound scope remains canonical storage. `workspace_id_for_scope()` can resolve
stored mapping metadata for API or extension use, but no shipped canonical
Personal Context server runtime or context-injection consumer currently calls
it. There is no current API for mapping an existing inbound scope; future
integration work must add an explicit mapping workflow rather than infer one
from canonical scope identity. The `load_companion_context()` builder and its
Companion and Persona endpoint callers use the separate companion system and
tables. They are not canonical Personal Context consumers and do not establish
a canonical Personal Context runtime or context-injection path.

Both plaintext and recovery exports serialize the same narrow snapshot shape:
the current manifest, selected scopes, and records. Recovery mode includes all
current scopes and records, then passphrase-encrypts that snapshot. Neither mode
includes proposals, runtime policy, encrypted local workspace mappings, keys,
receipts, Sync state, or other operational state. No supported server API, CLI,
or production caller imports or restores the recovery envelope. Treat it as a
protected export artifact, not a complete or directly restorable profile
backup.

## Sync and bootstrap flow

`capability negotiation -> registered device -> bootstrap snapshot/wrapped integrity key -> content-free reviewed Chatbook plan -> approval/completion -> first-link publication`

Sync V2 negotiates all five domains:

- `personal_context.manifest`
- `personal_context.scope`
- `personal_context.record`
- `personal_context.proposal`
- `personal_context.purge`

Bootstrap requires negotiated schema and quotas, an authorized registered
device, server key custody, and a stable canonical snapshot. The server binds
opaque profile/authority/generation state in the Sync dataset and returns the
canonical Sync-eligible snapshot—including record and proposal content—with a
device-wrapped server-owned integrity key. Before approval, bootstrap also
exchanges authentication/capability, device-registration/public-key, display,
schema/quota, and purge-generation metadata. Chatbook holds the remote content
transiently in memory while its durable link state and visible reconciliation
plan remain content-free. The plan presents identifiers, versions, counts,
outcomes, and local/server choices rather than profile values. No local record
or proposal content uploads before approval.

Approval permits link completion and publication of the resulting eligible
snapshot. Adapters can authenticate canonical bytes, identity, schema,
syncability, size, purge generation, and base lineage; accepted envelopes are
encrypted in Sync persistence and materialized through the per-user
`PersonalContextService`. Version divergence can become generic content-free
Sync conflict metadata. That is protocol capability, not proof of a shipped
ongoing lifecycle: later Chatbook mutations enter its encrypted local outbox,
but no startup, background, Settings, or other production caller runs a
Personal Context `sync_once()` cycle. **Overview → Manual Sync** invokes only
Notes and Chat.

## Current limitations and conflict boundaries

Reviewed first-link reconciliation handles first-link semantic collisions
before completion. Its durable state and visible plan are content-free even
though bootstrap has already downloaded the server's eligible content snapshot
into transient Chatbook memory.

The shipped client has no ongoing Personal Context sync caller, dedicated
Personal Context status/outbox surface, or dedicated post-link conflict
resolver. Later syncable Chatbook mutations remain queued locally. Generic
Sync conflict metadata is a transport capability, not a current user workflow.

REST edits are not published to linked clients. They update the server
canonical copy without appending a Personal Context Sync entry, so post-link
editing can make the peers diverge in either direction.

Server purge does not publish the protocol purge envelope, and acknowledgement
completion is absent.

The `personal_context.purge` domain, adapter validation, and inbound service
projection exist. The REST purge endpoint only advances the server-local
canonical generation fence, retains the advanced readable manifest head/version
as that fence, deletes non-manifest canonical heads and bodies plus runtime
state, and leaves the profile in `purge_pending`. A mutation returns
`profile_purge_pending` only after authentication, request validation, ownership
or object resolution, and entry into the existing-profile writable boundary.
Manifest recreation is unsupported because surviving profile state prevents a
replacement, while earlier gates may return their own errors. There is no
shipped server producer/distributor for a purge envelope and no device
acknowledgement-completion path.

## Future-client integration boundaries

A future client must not infer a complete lifecycle from the presence of Sync
domains or bootstrap endpoints. Client work owns capability negotiation and
incompatibility handling, an explicit ongoing Personal Context caller, durable
queue/status UX, and conflict review/resolution UX.

Companion server work separately owns publishing server-origin REST mutations,
producing and distributing purge envelopes, and tracking device
acknowledgements. Completing the purge acknowledgement lifecycle is shared
cross-peer work: clients must consume and acknowledge the barrier, while the
server must aggregate those acknowledgements and finish the lifecycle. Neither
side should document post-link convergence or completed purge until its own
responsibilities and the shared handshake are implemented and verified.

Chatbook's current interview boundary is also relevant to compatible clients.
Fixed mode generates questions locally and makes no model call; its encrypted
draft/transcript objects remain peer-local. Adaptive mode calls the configured
default Console provider without tools and sends the interview audience,
allowed topics, attempt number, eligible agent-visible records from the exact
selected scope, and—after the first answer—all prior answered turns including
raw answer text. The UI reveals the actual provider and model only after the
first provider response finishes. Approved answers can become ordinary
canonical record payloads governed by their record visibility and syncability.

Chatbook's **Remove local profile** deletes canonical
`PersonalContextRepository` state, including its `encrypted_outbox`, but does
not delete the server copy or unregister the device. It leaves separate
`SyncStateRepository` link/profile state, staged `sync_v2_local_outbox`
envelopes, remote heads/cursors, conflict reviews/receipts, and possibly dataset
staging keys. Its recovery export includes canonical heads, including
device-only records, but no shipped production caller imports it. Repository
rows are removed before canonical profile-key deletion; if key cleanup fails,
**Finish secure removal** retries that key cleanup only. New clients must define
and test these cleanup and recovery boundaries explicitly rather than assuming
that canonical-profile deletion covers every transport artifact.

## Privacy and diagnostics

Treat canonical profile bodies, semantic keys, proposal content, exports, and
key material as secrets. Never log canonical plaintext, durable at-rest
ciphertext, key material, or raw cryptographic exception values. Never return
internal ciphertext or raw cryptographic exception values. Authorized success
responses and explicit exports may return only the requested canonical data;
error and diagnostic boundaries translate cryptographic, key-custody,
integrity, and storage failures into sanitized, stable, content-free error
codes and messages without embedding raw exception values. An explicitly
requested recovery export is a separate passphrase-encrypted export artifact,
not exposure of internal at-rest ciphertext. It contains only the current
manifest, scopes, and records; excludes proposals, runtime policy, local
workspace mappings, and other operational state; and has no supported server
import or restore path.

Keep real or user-derived plaintext out of logs, diagnostics, Sync outbox or
routing metadata, exception text, temporary artifacts, and unencrypted test
fixtures. Deliberate synthetic canonical and conformance fixtures under Shared
Core are permitted because they prove exact shared bytes; never populate them
with production or user-derived content. Persist only the minimum opaque routing
and version metadata needed by each store. Tombstones and terminal proposal
receipts must remain content-free.

## Extension checklist

1. Decide whether the change affects the shared contract or only one peer.
2. Make shared canonical object changes in `tldw_profile_core` first; change Sync transport separately.
3. Preserve canonical identities and explicit syncability.
4. Route through the owning service; never access profile tables directly.
5. Enforce authority, scope, expiry, visibility, and secret-rejection rules.
6. Keep plaintext, ciphertext, keys, and raw cryptographic exception values out of logs, diagnostics, and outbox metadata; never return internal ciphertext or raw cryptographic exception values; keep real or user-derived plaintext out of unencrypted fixtures, while permitting deliberate synthetic canonical/conformance fixtures in Shared Core.
7. Add parity/conformance coverage in both repositories.
8. Add peer-specific migration, repository, service, API/UI, and recovery tests.
9. Update the governing ADR for storage, ownership, encryption, Sync, or authority changes.
10. Update both documentation sets whenever the shared contract changes.

## Test map

| Suite | Contract covered |
| --- | --- |
| `packages/tldw_profile_core/tests/tldw_profile_core/test_public_contract.py` | Direct Shared Core public models, canonical serialization, schema export, semantic validation, and tool contract. |
| `tldw_Server_API/tests/Personalization/test_personal_context_contract.py` | Vendored package digest parity, supported Python floor, and cross-runtime canonical bytes/integrity fixture. This live suite, not a copied digest, is the compatibility authority. |
| `tldw_Server_API/tests/Personalization/test_personal_context_auth_boundary.py` | Authentication before storage access and indistinguishable cross-user/unknown object responses. |
| `tldw_Server_API/tests/Personalization/test_personal_context_crypto.py` | Fresh object keys and nonces, associated-data binding, exact key sizes, and sanitized authentication failures. |
| `tldw_Server_API/tests/Personalization/test_personal_context_endpoints.py` | REST bounds, strict requests, typed conflicts, runtime, export confirmations, purge fencing, proposal pagination, and router registration. |
| `tldw_Server_API/tests/Personalization/test_personal_context_key_custody.py` | Independent wrapped profile keys, strict root-key handling, fail-closed locking, and absence of unwrapped keys at rest. |
| `tldw_Server_API/tests/Personalization/test_personal_context_repository.py` | Per-user schema and transactions, encrypted versions/heads, optimistic concurrency, content-free deletion, tamper detection, purge fences, and key rotation. |
| `tldw_Server_API/tests/Personalization/test_personal_context_service.py` | Service-owned authority, lifecycle, bootstrap, inbound Sync application, quotas, proposal review, export, and purge behavior. |
| `tldw_Server_API/tests/Personalization/test_personal_context_plaintext_canary.py` | Canonical bodies stay out of the database, sidecars, and logs; rejected content is shredded and integrity failures do not disclose plaintext. |
| `tldw_Server_API/tests/Personalization/integration/test_personal_context_composed_app.py` | Production route composition, modeled responses, authentication, and rate limiting. |
| `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py` | Capability/schema/quota gates, registered-device wrapping, stable reviewed bootstrap, binding, completion, pre-link upload blocking, idempotency, and privacy. |
| `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_materializer.py` | Authorized inbound projection through the owning service and content-free conflict/failure mapping. |
| `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py` | Canonical transport integrity, optimistic lineage, encrypted Sync history, replay idempotency, pull visibility, and fail-closed key custody. |
