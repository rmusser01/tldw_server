# Sync v2 M3 Polished Multi-Device Design

Date: 2026-05-23
Status: Planning gate
Scope: Server-connected Chatbook sync after M1 personal sync and M2 blob restore completeness

## Context

M1 established the durable Sync v2 shape: an append-only envelope log,
materialized Notes/Chat projections for server-front-end mode, explicit profile
bootstrap, device cursors, restore preview, conflict records, tombstones, and
server-trusted encryption readiness.

M2 added blob transfer and restore completeness: resumable upload, download
manifests, quota accounting, blob availability status, selective restore
completeness, and server-unlocked key recovery hardening.

M3 should not replace that contract. It should graduate Sync v2 from reliable
manual restore/sync into polished multi-device operation while keeping Chatbook
valid as a standalone local-only application that never connects to a server.

## Product Modes

M3 keeps the same three product modes:

- Standalone local-only Chatbook: no server identity, no server writes, no sync
  state.
- Server front-end: the user signs into tldw_server and operates directly
  against server materialized state without local sync.
- Offline sync client: the user signs into tldw_server, keeps local state, and
  syncs envelopes/blobs for offline and multi-device use.

M3 adds polish to the offline sync client mode. It must not make server
connectivity mandatory for local-only Chatbook use.

## M3 Goals

- Background/scheduled sync with transparent status and user controls.
- Device lifecycle management: list, rename, authorize, revoke, and diagnose
  devices.
- Workspace-scoped datasets with explicit permission and key policy boundaries.
- Broader domain coverage for media/library/source/cache/derived content after
  domain contracts are reviewed.
- Richer conflict review that provides safe summaries and resolution plans
  without hiding destructive choices.
- Stricter encryption modes: passphrase unlock, device-key unlock, optional
  client-only encrypted datasets or fields, and key rotation.
- Retention, compaction, per-device acknowledgment, and safe blob garbage
  collection.
- Operational observability for user-visible health, admin diagnostics, audit
  trails, and metrics without exposing private content or secret material.

## Non-Goals

- Do not require Chatbook to connect to tldw_server.
- Do not introduce breaking changes to M1/M2 clients without capability-gated
  fallbacks.
- Do not implement workspace sync before workspace permission and key rules are
  explicit.
- Do not implement destructive garbage collection until per-device
  acknowledgments and restore windows are enforced.
- Do not claim client-only encryption while server materializers require
  plaintext content for the same dataset.

## Design Principles

- Capability-gated evolution: M3 features are advertised independently so M1/M2
  clients can keep working.
- Dataset-scoped policies: sync scope, encryption mode, retention policy, and
  domain enrollment belong to a dataset, not to global server state.
- Device-scoped operations: background sync state, cursors, leases, pull
  acknowledgments, and revocation are keyed by device.
- Explicit user intent for risky actions: conflict overwrite, metadata-only
  restore, device revocation, key rotation, and GC need visible decisions.
- Server-front-end parity: accepted sync changes must still materialize into
  normal server-side state when the dataset mode requires dumb-front-end use.
- Privacy-preserving operations: logs, metrics, diagnostics, and conflict lists
  must use redacted metadata and never include wrapped keys, KDF secrets,
  ciphertext blobs, or private clear payloads.

## Workstream 1: Background Sync

M3 background sync is primarily a client scheduling and orchestration feature,
but the server must provide safe primitives:

- per-device sync policy hints: minimum interval, backoff floor, batch limits,
  metered-network guidance, and maintenance windows;
- per-device background status: last push, last pull, last successful blob
  transfer, conflict count, replayable failure count, and quota pressure;
- resumable sync leases so one Chatbook profile does not run overlapping sync
  workers against the same dataset/device;
- idempotent push/pull/blob APIs remain the source of truth;
- user-visible pause/resume intent is stored server-side when the user wants a
  device or dataset paused.

The server should not assume it can wake a standalone Chatbook client. It only
records policy, status, and safe resume state.

## Workstream 2: Device Lifecycle

M1 registered devices, and M2 required registered devices for blob/key recovery
paths. M3 turns device records into a user-visible management surface.

Required behavior:

- list all devices for the authenticated user and selected dataset;
- expose authorization state: pending, active, paused, or revoked;
- let stricter dataset policies require an existing active device or recovery
  method before a new device becomes active;
- update display name and optional user label;
- expose last seen, last push/pull, cursor lag, conflict count, blob
  completeness, active upload count, and key recovery coverage;
- revoke a device so it can no longer push, pull, upload, download, resolve
  conflicts, or store key recovery bundles;
- preserve historical envelopes from revoked devices for audit/restore;
- allow revocation to mark device-scoped key records as revoked when requested;
- require explicit confirmation for revoking the current device.

Revocation is not deletion. A revoked device remains visible in audit and
restore metadata.

## Workstream 3: Workspace Datasets

Workspace sync must be designed before implementation because it combines
authorization, materialization, conflict semantics, and key policy.

Dataset scopes:

- `personal`: current M1/M2 behavior, owned by one user.
- `workspace`: attached to a workspace and governed by workspace membership.
- `shared`: future direct sharing between users, deferred unless workspace
  sharing proves insufficient.

Workspace dataset rules:

- dataset enrollment requires a workspace role with sync permission;
- every push/pull/restore/blob/key operation re-checks current workspace
  membership;
- server-front-end materialization writes into workspace-owned projections, not
  the user's personal ChaChaNotes profile;
- personal and workspace object IDs must not alias each other;
- conflict records are scoped to workspace dataset and visible only to users
  with review permission;
- key policy is workspace-specific and may differ from personal datasets.

Initial workspace domains should be limited to workspace metadata and source
references before broader collaborative Notes/Chat behavior is enabled.

## Workstream 4: Broader Domains

M3 should add domains in tiers:

1. `workspaces.workspace` and `workspaces.source_ref`: workspace structure and
   stable references.
2. `source_cache.entry`: cached source metadata, content hashes, and provenance.
3. `media.item`, `media.keyword`, and `media.keyword_link`: media library
   metadata and tags.
4. Derived content domains such as transcripts, summaries, embeddings, and
   evaluation artifacts only after source-of-truth ownership is clear.

### Derived Content Reassessment

M3 should not advertise derived content domains. Source cache and media metadata
give enough stable anchors for restore, while the derived artifacts need a
separate ownership model because some are user-authored knowledge and others are
rebuildable compute output.

| Content class | M3 decision | Rationale | Later promotion path |
| --- | --- | --- | --- |
| Transcripts | Deferred source-of-truth candidate | Generated STT text may later be corrected by the user, split into segments, or tied to media blobs. Syncing transcript bodies before stable segment IDs and edit ownership exist would create hard conflicts and large private payloads. | Promote after transcript segment identity, user-edited vs generated provenance, tombstones, and restore conflict review are defined. Metadata may reference transcript availability, but bodies stay out of M3 media metadata. |
| Summaries | Deferred source-of-truth candidate only when user-pinned or edited | Model-generated summaries are rebuildable from source text, prompt, model, and parameters; user-pinned or edited summaries become personal knowledge. | Add a future summary domain with input hashes, prompt/model provenance, pinned/edited flags, and whole-object conflict review. Unpinned generated summaries remain cache. |
| Embeddings | Rebuildable cache | Embeddings are opaque, model-specific indexes over content already represented elsewhere. Syncing vectors would add storage and privacy risk without making restore more faithful. | Do not promote as a source-of-truth domain. Rebuild per device/server from synced content and model configuration; store only diagnostic/index status. |
| Evaluation artifacts | Split and defer | Evaluation projects, datasets, human labels, and run configs can be user-authored source-of-truth. Generated run outputs and metrics are derived artifacts tied to model versions and execution environment. | Later split into explicit eval config/label domains and generated run reference/artifact metadata. Generated outputs need retention, redaction, and blob policy before sync. |

Derived domains must meet the same admission bar as promoted M3 domains before
implementation:

- stable object identity and parent/source lineage;
- clear generated vs user-authored ownership;
- payload hash and base-state conflict rules;
- tombstone semantics;
- materializer ownership for personal and workspace datasets;
- restore preview behavior that distinguishes apply, skip, rebuild, and
  manual-conflict cases;
- redaction rules for diagnostics and conflict summaries;
- encryption policy that prevents server-front-end mode from exposing opaque
  client-private content.

Until those gates are met, derived content may be represented only through
existing anchors: `source_cache.entry` provenance, `media.*` metadata,
`attachment.ref` metadata, and M2 blob transfer for explicitly referenced
artifacts. These anchors must not smuggle transcript bodies, summary text,
vectors, or evaluation result payloads into metadata-only envelopes.

Each domain needs:

- stable object identity;
- payload hash and base-state conflict rules;
- tombstone semantics;
- projection/materializer ownership;
- restore inventory mapping;
- redaction policy for diagnostics and conflict summaries;
- property or integration tests for idempotency and stale-base conflicts.

## Workstream 5: Conflict Review

M1 supports whole-object Notes and conversation conflicts plus append-only
message merge. M3 should improve review UX through API shape, not hidden server
merges.

Server responsibilities:

- list conflicts with stable sorting, age, dataset, domain, object ID, and safe
  summary metadata;
- expose conflict detail with local/remote/base hashes, revision numbers,
  tombstone flags, and field-level metadata diffs where the server is allowed
  to inspect plaintext;
- support batch decisions with per-conflict idempotency;
- preview destructive resolution plans before applying them;
- keep `skip`, `overwrite`, and `duplicate_rename` semantics explicit.

Client-only encrypted datasets cannot use plaintext diff summaries from the
server. Those datasets should return hashes and metadata only, leaving diff
rendering to the client.

## Workstream 6: Stricter Encryption And Key Rotation

M3 must support the existing `server_trusted_v1` posture while adding stricter
dataset policies.

Candidate policies:

- `server_trusted_v1`: current trusted/self-hosted default. The server can
  materialize private data and normal authentication unlocks use.
- `passphrase_wrapped_v1`: dataset keys are wrapped by a user passphrase. The
  server stores wrapped material and unlock metadata but cannot unlock without
  client-provided passphrase-derived material.
- `device_wrapped_v1`: dataset keys are wrapped for trusted devices. Adding a
  device requires an existing trusted device or recovery flow.
- `client_private_v1`: selected fields or datasets remain opaque to the server.
  Server materializers can only operate on clear metadata.

Key rotation requirements:

- create a new active key epoch for a dataset;
- rewrap dataset keys for eligible active devices/recovery methods;
- reject new envelopes using revoked or superseded key epochs unless the
  dataset policy allows migration replay;
- preserve old key records while retained envelopes/blobs still require them;
- expose rotation status and blockers without revealing key material.

Client-only encryption is incompatible with dumb server-front-end editing for
opaque fields. The API must advertise this tradeoff directly.

## Workstream 7: Retention, Compaction, And Garbage Collection

M3 can reduce unbounded growth only after the server knows which devices have
seen which durable state.

Required primitives:

- per-device acknowledgment of applied server sequence ranges by domain;
- per-device blob verification acknowledgment for selected blobs;
- retention policy per dataset: minimum tombstone age, minimum envelope age,
  minimum offline restore window, and audit retention mode;
- compaction snapshots that summarize accepted object state without replacing
  the append-only audit log unless policy permits;
- blob GC candidates only when no active attachment refs, restore windows, or
  unacknowledged devices require the blob.

Default policy should be conservative: retain the audit log and tombstones until
explicitly configured.

## Workstream 8: Observability And Admin Diagnostics

M3 needs two levels of observability.

User-visible status:

- profile health;
- per-domain lag;
- last successful sync;
- conflicts and replayable failures;
- blob quota and completeness;
- key/recovery readiness;
- paused/revoked device status.

Admin/operator diagnostics:

- aggregate sync health metrics;
- dataset/domain envelope counts;
- failed materialization counts;
- upload session pressure;
- blob store health;
- retention/GC dry-run output;
- audit events for revocation, key rotation, conflict resolution, and
  destructive restore decisions.

Diagnostics must be redacted by default and must not log payloads, ciphertext,
wrapped key material, KDF salts, or recovery secrets.

## Implementation Order

M3 should land in this order:

1. Contract and storage readiness: API docs, schema flags, capability fields,
   and child Backlog tasks.
2. Device lifecycle and acknowledgments: list/update/revoke devices, cursor lag,
   per-device ack primitives.
3. Background sync status/policy: pause/resume, leases, status aggregation, and
   client-facing policy hints.
4. Workspace dataset design and first storage/API slice: dataset scope,
   membership checks, and workspace metadata domains.
5. Broader domain expansion: source cache and media metadata before derived
   content. Derived content is documented in M3 but remains unadvertised until
   the transcript, summary, embedding, and evaluation ownership gates are met.
6. Stricter encryption/key rotation: passphrase/device wrapped policies before
   client-private datasets.
7. Retention/GC and observability: dry-run first, then safe deletion paths.

## Success Criteria

M3 is successful when:

- a user can keep two or more Chatbook devices in sync without manual polling;
- a revoked device cannot access sync, blobs, conflicts, or key recovery APIs;
- workspace sync cannot leak data to users who lost workspace access;
- background sync state is visible and debuggable at profile and domain levels;
- conflict review supports safe batch resolution and destructive preview;
- stricter encryption policies are capability-gated and honest about
  server-front-end limitations;
- retention/GC can prove safety from device acknowledgments and policy windows;
- diagnostics explain sync health without exposing private content or secrets.
