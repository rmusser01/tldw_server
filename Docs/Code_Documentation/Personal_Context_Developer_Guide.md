# Personal Context developer guide

Personal Context is one canonical profile contract implemented by Chatbook and
tldw_server. The server is an authenticated home peer: it owns an encrypted
canonical copy for each user, exposes REST operations, and accepts linked
Chatbook changes through Sync V2. The [Personal Context API
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
  credentials and profile content are protected in transit.

### Shared contract

<!-- shared-personal-context-contract:start -->
- `tldw_profile_core` defines the versioned canonical profile object models, exact canonical bytes, interview/tool contracts, serialization, and validation used by both peers. Sync-v2 transport envelopes are a separate contract.
- After a successful reviewed link, Chatbook and tldw_server converge on the same canonical manifest, scope, record, proposal, and version identities and bytes for eligible shared objects.
- Sync V2 defines the `personal_context.manifest`, `personal_context.scope`, `personal_context.record`, `personal_context.proposal`, and content-free `personal_context.purge` domains. The current linked flow publishes eligible Chatbook-originated manifest, scope, record, and proposal changes; purge production and distribution are not wired end to end.
- Each peer retains its own at-rest ciphertext and keys, local database rows, runtime permissions, conflict-review metadata, acknowledgement tracking, and other operational state.
<!-- shared-personal-context-contract:end -->

| Shared through the current linked flow when eligible | Remains peer-local or is not currently published |
| --- | --- |
| Canonical manifest after successful reviewed linking | Peer-local at-rest encryption and recovery keys |
| Required global and linked-workspace scope objects | Raw interview answers and unfinished drafts |
| Records and tombstones whose controls permit synchronization | Runtime agent authority grants and tool availability |
| Eligible proposals and their canonical review state | Device-only records or records marked non-syncable |
| Exact canonical object identities, versions, and bytes for eligible shared objects | Local undo history, caches, ciphertext, database row identities, and other operational metadata |
| — | Conflict-review objects and acknowledgement tracking |

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
| `tldw_Server_API/app/core/Personalization/personal_context_runtime_policy.py` | Encrypted server-local runtime enablement and workspace mapping models. |
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

## Sync and bootstrap flow

`capability negotiation -> registered device -> reviewed Chatbook plan -> snapshot/wrapped integrity key -> completion -> inbound validation/materialization`

Sync V2 negotiates all five domains:

- `personal_context.manifest`
- `personal_context.scope`
- `personal_context.record`
- `personal_context.proposal`
- `personal_context.purge`

Bootstrap requires negotiated schema and quotas, an authorized registered
device, server key custody, and a stable canonical snapshot. The server binds
opaque profile/authority/generation state in the Sync dataset and returns the
canonical snapshot with a device-wrapped server-owned integrity key. Chatbook
reviews identity and semantic reconciliation before presenting the exact
bootstrap cursor for completion. Upload remains blocked until that narrow
completion transition succeeds.

After linking, adapters authenticate canonical bytes, identity, schema,
syncability, size, purge generation, and base lineage. Accepted envelopes are
encrypted in Sync persistence and materialized through the per-user
`PersonalContextService`; version divergence becomes generic content-free Sync
conflict metadata.

## Current limitations and conflict boundaries

Reviewed first-link reconciliation handles first-link semantic collisions before completion.

No dedicated post-link semantic-collision resolver exists.

REST edits are not published to linked clients.

Server purge does not publish the protocol purge envelope, and acknowledgement completion is absent.

The `personal_context.purge` domain, adapter validation, and inbound service
projection exist. The REST purge endpoint only advances the server-local
canonical generation fence, deletes readable server bodies and runtime state,
blocks later mutations, and leaves the profile in `purge_pending`. There is no
shipped server producer/distributor for a purge envelope and no device
acknowledgement-completion path.

## Privacy and diagnostics

Treat canonical profile bodies, semantic keys, proposal content, exports, and
key material as secrets. Keep plaintext out of logs, diagnostics, Sync outbox
or routing metadata, exception text, temporary artifacts, and unencrypted test
fixtures. Persist only the minimum opaque routing and version metadata needed
by each store. Tombstones and terminal proposal receipts must remain
content-free.

## Extension checklist

1. Decide whether the change affects the shared contract or only one peer.
2. Make shared canonical object changes in `tldw_profile_core` first; change Sync transport separately.
3. Preserve canonical identities and explicit syncability.
4. Route through the owning service; never access profile tables directly.
5. Enforce authority, scope, expiry, visibility, and secret-rejection rules.
6. Keep plaintext out of logs, diagnostics, outbox metadata, and unencrypted fixtures.
7. Add parity/conformance coverage in both repositories.
8. Add peer-specific migration, repository, service, API/UI, and recovery tests.
9. Update the governing ADR for storage, ownership, encryption, Sync, or authority changes.
10. Update both documentation sets whenever the shared contract changes.

## Test map

| Suite | Contract covered |
| --- | --- |
| `packages/tldw_profile_core/tests/tldw_profile_core/test_public_contract.py` | Direct Shared Core public models, canonical serialization, schema export, semantic validation, and tool contract. |
| `tldw_Server_API/tests/Personalization/test_personal_context_contract.py` | Vendored package digest parity, supported Python floor, and cross-runtime canonical bytes/integrity fixture. This live suite, not a copied digest, is the compatibility authority. |
| `tldw_Server_API/tests/Personalization/test_personal_context_endpoints.py` | REST bounds, strict requests, typed conflicts, runtime, export confirmations, purge fencing, proposal pagination, and router registration. |
| `tldw_Server_API/tests/Personalization/test_personal_context_key_custody.py` | Independent wrapped profile keys, strict root-key handling, fail-closed locking, and absence of unwrapped keys at rest. |
| `tldw_Server_API/tests/Personalization/integration/test_personal_context_composed_app.py` | Production route composition, modeled responses, authentication, and rate limiting. |
| `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_bootstrap.py` | Capability/schema/quota gates, registered-device wrapping, stable reviewed bootstrap, binding, completion, pre-link upload blocking, idempotency, and privacy. |
| `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_materializer.py` | Authorized inbound projection through the owning service and content-free conflict/failure mapping. |
| `tldw_Server_API/tests/Sync/test_sync_v2_personal_context_transport.py` | Canonical transport integrity, optimistic lineage, encrypted Sync history, replay idempotency, pull visibility, and fail-closed key custody. |
