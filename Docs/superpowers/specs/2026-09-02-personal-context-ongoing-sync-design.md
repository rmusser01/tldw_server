# Personal Context Ongoing Synchronization — Chatbook and tldw_server

**Date:** 2026-09-02

**Status:** Draft for owner review

**Applies to:** tldw_chatbook and tldw_server

**Extends:** [Unified Personal Context Profile — Chatbook and tldw_server](https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md)

## ADR check

```text
ADR required: yes
ADR path: backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md
Server mirror: Backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
Reason: This design fixes the long-lived authority, publication, retry,
conflict, deletion, capability, and cross-database durability contracts for
ongoing bidirectional Personal Context synchronization.
```

The existing Personal Context ADRs remain the governing decisions and are
amended rather than replaced. This specification resolves the ongoing-sync
details left open by the original design. If the documents conflict on this
subject, this specification controls.

## Summary

After a Chatbook profile is linked, Chatbook devices and one home tldw_server
share the same logical Personal Context records. Users may edit eligible
records from either application. Local Chatbook edits succeed offline and
queue for delivery; server edits are published into Sync V2. All linked
devices eventually pull the server-canonical result.

The server is the manifest sequencer after linking. Chatbook remains
local-first for semantic records, but its locally generated manifest changes
are speculative until the server accepts the associated semantic mutation and
publishes a canonical manifest. This avoids competing manifest histories while
preserving useful offline editing.

Synchronization is event-driven. It runs after startup, reconnection, an
eligible local mutation, workspace mapping, conflict-resolution activity, or a
user pressing **Sync now**. Failed automatic work receives a short bounded
retry sequence and then waits for another meaningful trigger. V1 adds no
permanent polling loop, push channel, or long poll.

The primary UI is the existing **Settings → My Profile** panel. It shows honest
operational state, pending work, last-attempt and last-success information,
workspace mapping needs, and conflict actions. It does not claim that a linked
profile is globally "up to date" merely because one request succeeded.

## Goals

- Keep syncable Personal Context records logically identical across Chatbook
  devices and the user's one home tldw_server.
- Allow user-authored and authorized agent-authored edits in either runtime.
- Preserve offline Chatbook edits without making Sync availability a write
  prerequisite.
- Publish direct server mutations without coupling canonical Personalization
  transactions to the separate Sync database.
- Prevent concurrent edits, key collisions, and manifest races from silently
  discarding user data.
- Make interruption recovery deterministic across every cross-database
  boundary.
- Give the user accurate, actionable status without notification noise.
- Preserve existing encryption, visibility, scope, generation, and deletion
  guarantees.
- Define a versioned contract that future clients can implement without
  copying Python internals.

## Non-goals

- Real-time delivery, WebSockets, server-sent events, push notifications, or
  long polling.
- A permanent background polling task.
- Multiple simultaneous home servers, federation, or peer-to-peer sync.
- Last-write-wins conflict handling, CRDTs, or automatic semantic merging.
- Synchronizing runtime agent grants, profile enablement, Undo history,
  interview drafts, local workspace mappings, or notification state.
- Synchronizing `device_only` records or making a scope itself device-only.
- Adding a second settings screen or a new server frontend for this phase.
- Claiming atomicity across Personal Context and Sync databases.
- Replacing the initial-link reconciliation already defined by the base
  Personal Context design.

## Product decisions

1. A linked profile has exactly one active home-server authority in V1.
2. Records are the same logical records in Chatbook and tldw_server: stable
   IDs, canonical bodies, revisions, scopes, lifecycle, and privacy controls
   survive replication.
3. Both runtimes may originate semantic record, scope, proposal, and tombstone
   mutations when local policy authorizes them.
4. After linking, the server alone sequences the shared manifest.
5. Chatbook edits commit locally before network work and queue encrypted exact-
   wire snapshots for Sync.
6. Direct server edits commit canonical state and an encrypted publication
   intent atomically in the server Personalization database.
7. Relay into the Sync database is after-commit, idempotent, and recoverable.
8. Conflicts freeze only the affected logical object. Unrelated profile data
   continues synchronizing.
9. Automatic retries are bounded; **Sync now** remains available except during
   a profile-wide blocker or a server-mandated retry window.
10. The UI notifies only when user action is required.
11. Existing Sync V2 transport and conflict endpoints are extended; a parallel
    Personal Context transport is not introduced.
12. Existing linked profiles require an explicit activation baseline before
    ongoing sync is enabled.

## Governing invariants

### Authority and identity

- The authenticated server user ID and stable server authority identify the
  home profile. A URL, display name, workspace label, or credential does not.
- Each human-visible Personal Context object keeps one canonical object ID
  across replicas.
- The server's authoritative manifest revision never moves backward.
- A client never treats its speculative manifest revision as proof that the
  home server accepted a mutation.
- Server-originated publication uses a reserved, registered home-authority
  device identity. It never impersonates the Chatbook device that happened to
  submit an earlier version.
- The reserved identity does not consume an ordinary device quota or recovery
  key slot, and authenticated clients cannot register or spoof it.

### Durability

- A Chatbook semantic mutation and its eligible Personal Context outbox rows
  commit in one Personal Context database transaction.
- A server semantic mutation, canonical manifest advance, and encrypted source
  publication rows commit in one Personalization database transaction.
- Relay acknowledges a source publication row only after the deterministic
  server-origin envelope is durable in the Sync store.
- Applying an inbound Sync object does not create an outbound echo.
- Cursor advancement occurs only across a consecutive prefix of safely applied,
  already-current, or durably conflict-pinned envelopes.
- No workflow relies on a process-local callback, timer, or service instance
  as its sole durable record of work.

### Privacy and safety

- `device_only` objects never enter a Personal Context outbox, Sync envelope,
  diagnostic payload, or server request.
- `user_only` objects may synchronize but remain unavailable to every agent
  context, search, summary, and tool path.
- Profile bodies, labels, proposal content, conflict candidates, and queued
  wire snapshots remain encrypted at rest.
- Logs, retry state, notifications, and metrics contain only bounded reason
  codes, opaque identifiers, counts, and timestamps.
- Purge generation is checked before mutation acceptance and before applying
  inbound content.

## Runtime responsibilities

### Shared Profile Core

`tldw-profile-core` continues to own canonical models, lifecycle validation,
deterministic serialization, JSON Schema, and conformance fixtures. Ongoing
sync adds no transport or database implementation to Shared Core.

The contract must expose enough schema to validate Personal Context capability
metadata, conflict-resolution extensions, and server-origin envelope metadata.
The server generates a versioned OpenAPI/JSON Schema contract artifact.
Chatbook vendors that exact artifact together with the source server commit and
checksum. Both repositories test their runtime models and client behavior
against it.

### Chatbook

Chatbook owns:

- Local encrypted canonical Personal Context storage.
- Local semantic mutations and encrypted immutable outbox snapshots.
- A coordinator that coalesces wake signals and drives Sync V2.
- A durable home-manifest checkpoint separate from speculative local state.
- Workspace mappings and per-object unmapped status.
- Durable conflict references and resolution-intent journals.
- Retry and status facts in SyncState, not in profile content.
- The existing **Settings → My Profile** status and review experience.

The app-level owner constructs or resolves the current Personal Context
service for each run. A wake signal is not permanently attached to one cached
service instance because recovery can replace that instance. Mutation code
emits a best-effort wake only after its database transaction commits; failure
to schedule work never rolls back the user's edit.

### tldw_server

tldw_server owns:

- Per-user encrypted canonical state in `Personalization.db`.
- Authoritative post-link manifest sequencing.
- An encrypted Personal Context publication outbox in the same database as
  canonical state.
- Idempotent relay from that source outbox into the existing Sync V2 store.
- A reserved home-authority device identity and server-origin envelope builder.
- Activation baselines for existing linked profiles.
- Conflict validation and the authoritative batched resolution API.
- Purge-generation advancement and rejection of stale-generation writes.

Request-scoped Personal Context services may enqueue source publications, but
they do not own the relay lifecycle. Relay is recoverable from durable state
and may be driven by after-commit work or by Sync pull.

### Sync V2

Sync V2 remains transport rather than application authority. It owns durable
per-device envelope history, cursors, registered device identity, push/pull,
and generic conflict tokens. Personal Context adapters validate and apply
canonical whole objects through each runtime's Personal Context service.

## Durable state additions

### Chatbook Personal Context database

The local outbox gains an opaque `commit_id` shared by every syncable semantic
object produced by one canonical mutation. Existing `object_type` identifies a
manifest row; no second `commit_role` field is needed.

For a commit containing syncable semantic objects:

- Each semantic object receives an encrypted immutable outbox snapshot.
- The locally derived manifest snapshot may be queued under the same
  `commit_id` for retirement accounting, but it is not independently
  authoritative after linking.
- Semantic rows become eligible before the derivative manifest row.
- The derivative manifest retires only after every syncable semantic row in
  the commit has been durably staged or otherwise terminally accounted for.

After linking, the derivative manifest row is a local commit barrier only. It
is never staged or pushed as a client-authored manifest envelope. The server's
accepted semantic mutation creates the only shared manifest advance. Initial
link reconciliation remains the sole path that publishes a client profile's
provisional manifest material.

A commit containing only `device_only` changes creates no outbox rows,
including no manifest row.

### Chatbook SyncState database

Personal Context link state gains durable operational facts:

- `last_attempt_at`
- `last_success_at`
- `last_error_code`
- `retry_not_before`
- negotiated `ongoing_sync_version`
- bounded `ongoing_sync_blockers`
- home manifest checkpoint: server cursor, manifest revision, manifest version
- activation status for pre-existing links

Generic profile timestamps are not reused: another Sync domain may update them
and make Personal Context's **Last checked** display false.

The existing Sync V2 conflict-review row remains content-free and points to a
single encrypted, protected Sync envelope containing the shared candidate.
That envelope is pinned from retention until the conflict is resolved or the
profile is purged. The ciphertext is not duplicated into the review table.

A content-free resolution journal records the selected action, expected local
and remote envelope IDs, transmission state, idempotency key, and terminal
acknowledgment. A pending offline choice may be changed or cancelled until its
first transmission. Once transmitted, the request is immutable and retried
idempotently.

Notification deduplication stores only a content-free fingerprint of the
actionable condition so restarts do not repeat the same alert.

### Server Personalization database

Every canonical server mutation writes encrypted source publication entries in
the same transaction. Source entries carry enough opaque routing and encrypted
canonical material to deterministically build Sync envelopes after commit.

Before a profile has an activated ongoing-sync link, the outbox still records
publication state. It may compact superseded entries to the latest canonical
head per object. First-link or upgrade activation stores a publication
watermark atomically with the baseline:

- Source entries at or below the watermark are represented by the baseline
  snapshot and are not replayed individually.
- Entries after the watermark are relayed normally.
- A server edit racing activation is therefore included either in the
  baseline or in the post-watermark stream, never lost between them.

Poisoned or structurally invalid source rows stop Personal Context relay for
that profile with an actionable integrity state. They are not skipped in an
infinite loop.

## End-to-end synchronization lifecycle

### Capability negotiation and activation

The server advertises:

- `ongoing_sync_version`, an integer that defaults to `0` when absent.
- `ongoing_sync_blockers`, a bounded list of machine-readable blocker codes.

Chatbook enables automatic ongoing Personal Context sync only for version `1`.
It renegotiates on startup, authentication restoration, and reconnection.
Unsupported or blocked capability leaves local profile editing available and
shows an actionable link state; it does not silently attempt a partial mode.

The server advertises version `1` only when canonical mutation publication,
reserved authority identity, pull-time recovery relay, conflict contracts,
and generation fencing are all ready.

An existing link created before this feature is not activated by capability
advertisement alone. The peers first establish an authoritative baseline and
home-manifest checkpoint while server publication is fenced by the activation
watermark. New links perform the same activation as part of successful initial
reconciliation.

### Chatbook-originated mutation

1. The authorized Personal Context service validates the local request.
2. The Personal Context repository commits the immutable canonical revision,
   local speculative manifest advance, and eligible encrypted outbox rows.
3. After commit, mutation code sends a best-effort coordinator wake.
4. The coordinator stages semantic rows through the existing cross-database
   outbox dispatcher. Receipts make stage/retry/shred idempotent.
5. The derivative local manifest is not allowed to race ahead of its semantic
   siblings.
6. Sync V2 pushes staged envelopes with immutable base metadata.
7. The server accepts, rejects, or creates a conflict under its current
   canonical object and purge generation.
8. On acceptance, the server applies the semantic object, advances the
   authoritative manifest, and writes encrypted source publication entries in
   one canonical transaction.
9. Server relay stages home-authority envelopes in Sync V2.
10. Chatbook pulls and applies the canonical server result, then advances its
    home-manifest checkpoint.

A failure at steps 3 through 10 never rolls back the local edit. The pending
count and retry state remain visible.

### Server-originated mutation

1. A server UI, API, migration, or authorized agent tool mutates only through
   the server Personal Context service.
2. Canonical state, manifest advance, and encrypted publication rows commit
   atomically in `Personalization.db`.
3. After commit, the server makes a best-effort relay attempt. Relay failure
   does not roll back the edit.
4. A later Personal Context pull performs mandatory recovery relay before it
   answers.
5. Linked clients receive the home-authority envelope and apply it through
   their local service.

Mandatory pull-time relay stages source rows until the Sync store contains one
visible post-cursor lookahead for the requested Personal Context domains, or
the source outbox is exhausted. It must test actual post-filter visibility;
blindly staging `page_size + 1` source rows is insufficient because duplicate,
excluded, or already-durable entries may not yield a client-visible envelope.

The Sync store, not the source outbox, owns per-device retention once an
envelope is durable.

### Pull and cursor advancement

One existing `sync_once()` call may return only one pull page. A coordinator
run performs its push phase once, then drains a bounded number of Personal
Context pull pages, initially 10. It must not emulate draining by repeatedly
rerunning the complete push phase. If more pull pages remain, it yields and
schedules one follow-up run rather than monopolizing the event loop.

For each page, Chatbook processes envelopes in server-cursor order. A cursor may
advance through an envelope only when it is:

- Validated and applied.
- Already represented by the same canonical head.
- Retained opaquely under the negotiated unknown-version rules.
- Converted into a durable conflict whose shared candidate envelope is pinned.

An authentication, decryption, integrity, manifest, generation, or purge
failure stops profile-wide advancement. An authenticated but invalid canonical
object may be quarantined as a per-object condition only when the server cursor
can still be recovered without accepting bad content; otherwise advancement
stops. The implementation must never advance past an unretained envelope that
would be required for later review or recovery.

### Manifest rebasing

Incoming manifest comparison uses the durable home-manifest checkpoint, not the
locally speculative manifest head. After the server publishes a canonical
manifest, Chatbook rebases its local derived view while preserving every
unaccepted semantic head and outbox row.

A manifest rebase may retire acknowledged derivative manifest work. It may not
drop semantic edits, rewrite their immutable base metadata, or reinterpret a
rejected mutation as accepted.

## Scheduling and retry behavior

The coordinator is single-flight and generation-coalesced. A wake during an
active run records that another pass is needed; it does not start a competing
run. At most one follow-up pass is scheduled for the accumulated generation.

Wake sources are:

- Application startup after services and link state are available.
- Authentication restoration or an observed offline-to-online transition.
- A successful eligible Personal Context outbox commit.
- Local Personal Context use that can reasonably discover inbound work, such
  as opening **My Profile** or building an agent-context snapshot before the
  current connection generation has completed a successful check.
- A workspace mapping change that unblocks retained objects.
- Creation or update of a conflict-resolution intent.
- **Sync now**.

V1 uses a bounded jittered retry sequence around 2 seconds, 10 seconds,
30 seconds, 2 minutes, and 5 minutes. After exhaustion, automatic work waits
for a new meaningful trigger. Edits made while connectivity is already known
to be unavailable increase the pending count but do not repeatedly re-arm an
exhausted timer sequence.

Reconnection and **Sync now** re-arm ordinary transient failures. A server
`Retry-After` or equivalent rate-limit window is persisted as
`retry_not_before` and survives restart. Manual sync does not bypass that
server instruction.

The coordinator does not install a permanent poll, long poll, socket, or timer
that lives for the process lifetime.

Activity wakes are content-free, coalesced, and suppressed after a successful
check in the current connectivity generation. They permit inbound discovery
without turning every profile read or model request into a network call.

## Failure containment

Per-object conditions allow unrelated objects to continue:

- Unmapped workspace scope.
- Ordinary edit conflict.
- Key collision.
- Authenticated invalid object that can be safely quarantined.

Profile-wide blockers stop Personal Context synchronization:

- Authentication or authorization failure.
- Missing or changed key protector.
- Payload decryption or keyed-integrity failure.
- Unsupported required capability or schema range.
- Invalid manifest, activation baseline, or purge generation.
- Purge pending.
- Poisoned publication state that prevents ordered relay.

Transport unavailability and rate limiting are retry states, not user-action
notifications unless credentials, configuration, or another explicit choice is
required.

## Conflict contract and review

Conflicts preserve both immutable candidates and freeze ordinary edits only for
the affected object. Context continues to use the last mutually acknowledged
version or omits the object when no such version exists.

The authoritative server endpoint remains the existing batched
`POST /api/v1/sync/conflicts/resolve`. The implementation must correct
Chatbook's obsolete per-conflict endpoint and action vocabulary rather than
adding a second resolution path. No ordinary Sync push with
`operation=resolve_conflict` is introduced for Personal Context.

The generic batched request adds optional expected-local and expected-remote
envelope IDs. They are required for Personal Context ongoing-sync version 1 so
the server can reject a stale review cleanly.

Conflict retention remains bounded by the negotiated Sync limits. If a runtime
cannot durably pin another candidate, it stops before advancing the cursor and
reports profile attention; it never drops the oldest unresolved candidate to
make room silently.

User-facing actions map to the server contract as follows:

- **Keep shared profile version** uses `skip`. No replacement envelope is sent.
  After server acknowledgment, Chatbook applies the already pinned shared
  candidate.
- **Keep local version** uses `overwrite` and includes the rebased canonical
  replacement envelope inside the resolution request.
- **Merge** uses `overwrite` and includes the user-reviewed merged canonical
  envelope inside the resolution request.
- **Keep both as separate records** uses `duplicate_rename`, only after explicit
  user choice, with a new object ID and a noncolliding semantic key.

The merge editor uses typed fields rather than raw canonical JSON. Restrictive
values win by default for delete, `user_only`, and other privacy-reducing
controls; the user must explicitly choose a less restrictive result.

Conflict pages contain at most 20 items. Labels are **Local version** and
**Shared profile version**, not ambiguous client/server winner language.

## Settings experience

Ongoing-sync controls extend the current `PersonalContextSettingsPanel` under
**Settings → My Profile**. That panel remains the one presentation and state
owner. The feature does not add settings to deprecated legacy settings paths.

The compact summary shows:

- Home server, with a bounded and sanitized display name.
- Primary state: **Linked**, **Queued**, **Syncing**, or **Retrying**.
- Pending semantic-object count.
- **Last checked** from `last_success_at`, or **Never**.
- An **Attention** condition when user action is required.
- **Sync now**.

Attention is orthogonal to the primary state. A profile can be **Queued** with
one conflicting object while other work continues. The interface does not use
"Up to date" because another device or server edit may exist after the last
successful check.

`last_attempt_at` changes when a network sync attempt begins.
`last_success_at` changes only after a successful eligible push/pull round,
including a successful no-op check; local edits and timer scheduling do not
change it. Because V1 is trigger-based, the UI also explains that server edits
appear on the next sync trigger rather than instantly.

Expanded details show last attempt, last success, the bounded sanitized reason
code, rate-limit retry time, activation/capability state, pending counts, and
workspace rows. Workspace rows say **Mapped** or **Needs mapping** plus record
counts. They never say a scope is "Synced," and `device_only` is presented as a
record control rather than a scope state.

**Sync now** remains enabled for per-object conflicts and unmapped workspaces.
It is disabled only for a profile-wide blocker or active server retry window;
the reason appears next to the disabled control.

Only conditions requiring a user decision produce a notification. The user can
return to a paginated conflict or workspace-mapping list from the summary.
After a modal closes, focus returns to the invoking control. Status is conveyed
by text and semantics, never color alone. No terminal-convention or global
shortcut is added.

Destructive controls remain visually separated at the bottom of the panel.

## Device removal and global deletion

### Remove this device's copy

Removing the local copy stops the coordinator, destroys local readable profile
content and keys, clears link, staged Personal Context envelopes, conflicts,
and retry state, and prevents automatic rebootstrap until the user links again.
If reachable, Chatbook unregisters the ordinary device best-effort. The shared
server profile and other devices remain unchanged.

The confirmation summarizes pending changes and unresolved reviews. For a
standalone profile it states that this may destroy the only copy.

### Delete everywhere

Delete-everywhere uses a durable, signed, idempotent, content-free purge request
stored in SyncState before readable profile keys or content are destroyed.
The editor freezes while deletion is pending. The server serializes purge
against in-flight mutations: work accepted before the purge is destroyed by
the purge transaction, and work arriving afterward with an older generation is
rejected.

The server advances the generation, destroys canonical and derived content,
publishes the content-free barrier, and retains only the minimal acknowledgment
ledger. Devices erase their replicas when they receive the barrier. A new
profile cannot be created from the old link until the initiating device has
received acknowledgment.

A delete-everywhere action initiated on tldw_server performs the same canonical
purge and barrier publication. Chatbook devices learn it through the next pull
trigger and then erase their local replicas.

Both destructive flows require the existing high-friction confirmation style;
delete-everywhere includes a typed confirmation phrase.

## Compatibility and rollout

Rollout is server-first:

1. Deploy schema and storage additions while advertising
   `ongoing_sync_version = 0`.
2. Enable atomic canonical publication and deterministic server-origin relay.
3. Enable pull-time recovery relay, conflict extensions, reserved authority
   identity, generation fencing, and activation baselines.
4. Advertise version `1` only after readiness checks pass.
5. Release Chatbook support, still gated on negotiated version `1`.
6. Activate existing links individually through a baseline and publication
   watermark.

Older Chatbook builds continue their existing initial-link behavior but do not
enter ongoing automatic sync. New Chatbook builds connected to an older server
remain locally usable and explain that ongoing sync is unavailable. Unknown
newer objects continue to follow the base design's opaque-retention rule.

The generated server contract artifact is the interop authority for current
wire details. Documentation prose explains semantics but cannot override the
versioned schema, canonical Shared Core model, or server endpoint validation.

## Observability

Existing structured logs and status surfaces are sufficient for V1; no new
telemetry stack is added. Events use opaque profile/device/object identifiers,
counts, elapsed time, attempt number, result code, and capability version.
They never include canonical bodies, semantic keys, labels, evidence spans,
raw exception messages that may contain content, or decrypted conflict data.

Required result codes distinguish at least success, nothing-to-do, offline,
rate-limited, authentication-required, capability-blocked, integrity-blocked,
purge-pending, per-object-conflict, workspace-unmapped, and relay-poisoned.

## Verification strategy

### Contract and model conformance

- The server-generated schema artifact is checked in with version and source
  commit metadata; Chatbook's vendored copy has the same checksum.
- Both applications accept valid capability, server-origin, and conflict
  fixtures and reject invalid or ambiguous variants.
- Chatbook sends the authoritative batch endpoint and server action names.
- Reserved authority identity cannot be registered or submitted by a client.

### Repository and interruption tests

Tests use real temporary Personal Context and SyncState/Sync SQLite databases,
plus real temporary server Personalization and Sync databases. They inject a
failure and restart at every cross-database boundary:

- Chatbook canonical commit before wake.
- Personal Context outbox stage before receipt.
- Receipt before source-row shredding.
- Server canonical commit before relay.
- Sync envelope durability before source acknowledgment.
- Conflict envelope pin before cursor advancement.
- Resolution transmission before acknowledgment and local apply.
- Purge-request durability before local key destruction.
- Activation baseline before and after publication watermark.

Every case must converge without duplicate semantic mutations, lost edits,
cursor skipping, or plaintext fallback.

### Multi-device integration

A two-Chatbook-device and one-server matrix covers:

- Offline local edit, reconnect, and convergence.
- Direct server edit reaching both devices, including the originating device
  after a prior client push.
- Concurrent record edits and same-key creations.
- Safe manifest rebasing with additional pending local edits.
- Workspace mapping on one device without leaking that mapping to another.
- `device_only` exclusion and `user_only` agent exclusion.
- Retry exhaustion, restart, reconnect, rate-limit persistence, and manual
  sync.
- Existing-link activation while a server edit races the baseline.
- Remove-local-copy behavior and delete-everywhere generation fencing.

### UI and accessibility

Tests mount the production `PersonalContextSettingsPanel` hierarchy and CSS at
supported terminal sizes. They cover every primary state, simultaneous
attention, long sanitized labels, disabled-action explanations, keyboard-only
conflict review, focus return, pagination, and destructive confirmations.

### Privacy evidence

Unique canary profile values are scanned across both applications' ordinary
databases, WAL files, outboxes, conflict storage, retry state, logs,
diagnostics, recovery artifacts, migration snapshots, caches, and
application-owned backups. Only encrypted or explicitly content-free artifacts
may contain related state.

This is regression evidence, not a claim that plaintext never existed in
process memory or external backups.

### Live verification

Before rollout, a production-shaped Chatbook and tldw_server run exercises the
full edit, conflict, retry, restart, and deletion flow. It uses isolated HOME,
XDG config/data/cache directories, server databases, credentials, and keyring.
The verification records fingerprints proving that the operator's real profile
and configuration were unchanged.

## Acceptance criteria

- A linked Chatbook edit succeeds offline, is visibly queued, and later
  converges to the same canonical record on the home server and another device.
- An eligible server edit is durably published and reaches every linked device
  without requiring another server edit or process-local callback to survive.
- Device-only-only mutations never create any Sync outbox row.
- A crash at any documented cross-database boundary recovers idempotently.
- Manifest sequencing cannot retire or overwrite an unaccepted semantic edit.
- Conflicts retain both encrypted candidates, freeze only the affected object,
  and resolve through the authoritative batched server contract.
- Pull cursors never advance past an object that was neither safely handled nor
  durably retained.
- Existing links activate without losing a server mutation racing the
  activation baseline.
- Automatic retries stop after the bounded sequence and restart only for a
  meaningful trigger; server retry windows survive process restart.
- **Settings → My Profile** reports accurate queued/syncing/retrying/attention,
  last-attempt, last-success, pending, and workspace-mapping state.
- User-action notifications are deduplicated across restart; routine offline
  and retry transitions do not notify.
- Remove-local-copy and delete-everywhere preserve their distinct scopes, and
  an old-generation device cannot resurrect globally deleted content.
- Contract artifacts and cross-runtime fixtures remain byte-for-byte aligned.
- Privacy canary scans find no plaintext Personal Context content in covered
  persistent artifacts.

## Documentation and implementation boundary

This document defines behavior and durable contracts. The implementation plan
must split work into dependency-ordered, single-PR Backlog tasks for the server
contract/storage foundation, server publication and activation, Chatbook state
and coordinator, conflict/UI work, and cross-runtime verification. Each task
must cite the applicable Personal Context ADR and this specification.

No production implementation begins until this specification passes the
required spec review and receives owner approval.
