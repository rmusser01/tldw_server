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
8. Conflicts freeze only the affected logical object, or both object IDs and
   semantic-key slot for a key collision. Unrelated profile data continues
   synchronizing.
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
- Server-originated publication uses the existing trusted `server-origin`-style
  pseudodevice as the home-authority identity. It is recognized internally,
  not inserted as an ordinary globally unique registered-device row, and never
  impersonates the Chatbook device that submitted an earlier version.
- The reserved identity does not consume an ordinary device quota or recovery
  key slot, and authenticated clients cannot register, submit, or spoof it.
- After ongoing-sync activation, a client-authored Personal Context envelope is
  durable ingress only. It is never pull-visible to any device.
- Only an `applied` home-authority publication is Personal Context egress.
  Pending, retryable-failed, terminally rejected, conflicted, or stale-
  generation ingress is never returned by pull.

### Durability

- A Chatbook semantic mutation and its eligible Personal Context outbox rows
  commit in one Personal Context database transaction.
- A server semantic mutation, canonical manifest advance, and encrypted source
  publication rows commit in one Personalization database transaction.
- A replay receipt in that same transaction binds client ingress identity and
  digest to the resulting canonical version and publication batch.
- Relay acknowledges a source publication row only after the deterministic
  server-origin envelope is durable in the Sync store.
- Server publication batches have durable order. Every semantic authority
  envelope is durable before the batch's manifest envelope can become visible.
- Applying an inbound Sync object does not create an outbound echo.
- A server scan watermark may pass immutable rows conclusively classified as
  permanent non-egress, including client ingress. The delivered/application
  checkpoint advances only across a consecutive prefix of authority envelopes
  that are safely applied, already-current, opaquely retained, or durably
  conflict-pinned. Those are separate facts and are never inferred from one
  another.
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

`tldw-profile-core` continues to own canonical Personal Context object models,
lifecycle validation, deterministic serialization, object JSON Schema, and
conformance fixtures. Ongoing sync adds no transport, capability, endpoint, or
database implementation to Shared Core.

tldw_server owns the wire models for Personal Context capability metadata,
conflict-resolution extensions, authority-envelope metadata, and activation.
It generates their versioned OpenAPI/JSON Schema contract artifact. Chatbook
vendors that exact artifact together with the source server commit and
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
- A replayable ingress receipt and publication-batch journal in that database.
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
per-device envelope history, cursors, ordinary registered-device identity,
push/pull, and generic conflict tokens. Personal Context adapters validate and
apply canonical whole objects through each runtime's Personal Context service.
After activation, its Personal Context pull path is role-aware: client ingress
is retained for idempotency and review but only home-authority egress is
delivered. Other Sync domains retain their existing visibility behavior.

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
- retry-budget generation, zero-based attempt index, and
  `idle`/`active`/`exhausted` state
- `next_activity_check_at`
- content-free activity work generation and last-attempted generation
- negotiated `ongoing_sync_version`
- bounded `ongoing_sync_blockers`
- negotiated continuity token and activation epoch
- home manifest checkpoint: server cursor, manifest revision, manifest version ID
- per-device activation ID, digest, and state
- privacy-cleanup requirements and acknowledgments bound to canonical versions

Generic profile timestamps are not reused: another Sync domain may update them
and make Personal Context's **Last checked** display false.

The existing Sync V2 conflict-review row remains content-free and points to a
single encrypted, protected home-authority envelope containing the shared
candidate. That envelope is pinned from retention until the conflict is
resolved or the profile is purged. The ciphertext is not duplicated into the
review table.

A content-free resolution journal records the selected action, expected local
and remote envelope IDs, transmission state, idempotency key, and terminal
acknowledgment. A pending offline choice may be changed or cancelled until its
first transmission. Once transmitted, the request is immutable and retried
idempotently.

Notification deduplication stores only a content-free fingerprint of the
actionable condition so restarts do not repeat the same alert.

### Server Personalization database

Every canonical server mutation writes encrypted source publication entries in
the same transaction. It atomically allocates a monotonic per-profile
`profile_publication_sequence` and a durable `publication_batch_id`; each row
has `batch_ordinal`, `batch_size`, and a content-free semantic, manifest, or
purge-barrier role. Semantic rows precede the manifest, while a purge barrier
is the final permissible row for its generation. Source entries carry enough
opaque routing and encrypted canonical material to deterministically build
Sync envelopes after commit. Egress envelope IDs derive from the batch ID and
ordinal, so replay cannot create a second logical publication.

Both after-commit and pull-time relay acquire the same recoverable per-profile
lease and claim only the earliest incomplete `profile_publication_sequence`.
A later batch cannot stage any row until every earlier batch is durably
`complete`, durably `covered_by_activation`, or terminalized by an explicit
purge-generation fence. The activation-covered state is bound to an activation
ID, baseline digest, and verified Sync installation receipt. A compact,
content-free `activation_covered_through_sequence` ledger preserves that
terminal proof after encrypted source-row bodies are shredded. A corrupt
ordinary batch blocks later batches with Attention; ordinary relay never skips
it.

Within the claimed batch, relay walks ordinal order. It may acknowledge rows
individually, but it cannot stage the manifest until every semantic sibling is
verified durable in Sync. A restart resumes the same deterministic batch at its
first unacknowledged ordinal. Home-authority egress is inserted as already
canonically applied transport; it is not materialized back into
`Personalization.db` as another mutation or revision.

Client ingress has a separate replay receipt in `Personalization.db`. The
receipt binds dataset, real device, client envelope ID, canonical payload
digest, purge generation, resulting canonical version, manifest revision, and
publication batch. It is created atomically with canonical application and
source publication, making replay after an uncertain Sync-state update an
equality check rather than a second mutation.

Before a profile has an activated ongoing-sync link, the outbox still records
publication state. It may compact superseded entries to the latest canonical
head per object. Activation preparation atomically stores an encrypted exact-
head baseline, its digest and ID, and a source-publication watermark at the end
of the latest fully committed `profile_publication_sequence` in
`Personalization.db`. The watermark is always a whole-batch boundary and never
splits a publication batch:

- Source batches at or below the watermark are represented by the baseline
  snapshot and are not replayed individually after the baseline is installed
  durably in Sync and those batches are marked `covered_by_activation`.
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
- A profile-scoped activation epoch and publication-continuity token when
  version `1` is ready.

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

Activation is a replayable cross-database state machine, not one transaction.
The server Personalization database owns the encrypted baseline, activation
ID, digest, generation, publication watermark, and server journal. The Sync
database owns the deterministic baseline authority envelopes, their durable
installation receipt, the ordinary device cursor, and per-device activation
acknowledgment. Chatbook SyncState owns its matching activation ID, digest,
state, and home checkpoint. The local Personal Context database owns an exact
activation-apply receipt written with the local canonical baseline.

Server preparation progresses independently:

1. `prepared`: the server commits an encrypted exact-head baseline, digest,
   purge generation, and whole-batch source watermark in one Personalization
   transaction. The watermark cannot cut through a publication batch.
2. `installed`: deterministic baseline authority envelopes and their receipt
   are durable in Sync. The server verifies that receipt before advancing its
   Personalization journal. Under the shared profile lease, one Personalization
   CAS binds the activation ID, baseline digest, and verified receipt; marks
   every incomplete source batch through the exact watermark
   `covered_by_activation`; and advances the content-free
   `activation_covered_through_sequence` ledger. Only after that commit may the
   covered encrypted source-row bodies be compacted. The baseline stays pinned
   until applicable devices acknowledge or expire.
3. `active_for_device`: Sync holds the ordinary device's activation
   acknowledgment, and the server journal has verified that exact receipt.

Chatbook progresses `required → prepared → installed → acknowledged → active`:

1. `required` means version `1` is available but this device has no current
   activation receipt. Local reads and edits remain available; syncable edits
   queue but are not pushed.
2. `prepared` means Chatbook has verified the server-installed activation ID
   and digest and saved the encrypted baseline in its existing protected
   first-link staging area, independent of its ordinary pull cursor.
3. `installed` means one local Personal Context transaction revalidated and
   reconciled that exact baseline and wrote the activation-apply receipt.
   SyncState then records the home checkpoint; on a crash between databases it
   recovers only from the exact local receipt rather than applying again.
4. `acknowledged` means the server has durably recorded this ordinary device's
   acknowledgment and returned a matching receipt.
5. `active` means Chatbook has verified and stored that receipt. Post-watermark
   authority publications then drain through ordinary ongoing pull.

Every transition uses the same activation ID and digest on replay. A server
restart resumes preparation or deterministic baseline installation; a
Chatbook restart resumes its exact staging or acknowledgment step. No restart
recreates a baseline under the same ID with different bytes. An edit racing
preparation is either in the prepared snapshot or receives a post-watermark
publication. A newly linked device receives a current baseline rather than
depending on historical source rows.

Local activation installation is a dedicated reconcile/rebase transaction,
not a wholesale replacement. For each baseline object it:

- Installs the authority version as the local head when no unaccepted local
  head exists.
- Marks an identical local head as acknowledged without creating a version.
- Retains a divergent unaccepted local head as the active local edit, stores
  the baseline version as its last acknowledged home base, and preserves the
  immutable outbox snapshot and original base metadata.
- Preserves local-only objects and every `device_only` object unchanged.
- Advances the home manifest checkpoint without treating any preserved local
  semantic head or derivative manifest barrier as accepted.

The transaction re-reads local heads before commit, so an edit made during
download/staging is classified by the same rules rather than overwritten. A
preserved divergent edit later pushes against its original acknowledged base
and either succeeds or creates an explicit conflict. Activation never retires
or rewrites it merely because the server baseline installed.

The activation epoch and publication-continuity token are random opaque values
stored durably in `Personalization.db` and bound to the activated profile,
purge generation, and installed baseline. The token remains valid only while
every canonical mutation is committed with a complete source-publication batch.
If publication journaling is unavailable, the server must either fence
canonical Personal Context writes or invalidate the token and advance the
activation epoch in the same canonical transaction before accepting a write.
A downgrade or rollback path that cannot enforce one of those choices may not
write a linked canonical profile.

Every version-1 push, pull, conflict-list, and conflict-resolution request
carries the expected activation epoch and continuity token. The server
validates them before mutation or delivery and echoes the current pair in its
response. Chatbook validates that echo before accepting a push result, applying
an envelope, resolving a conflict, or advancing either watermark. A mismatch
returns `personal_context_activation_required` and cannot partially advance
client state.

An authenticated downgrade from version `1` to `0`, a blocker response, or a
Personal Context response reporting capability mismatch immediately pauses
ongoing work fail closed. It preserves the link, local data, queued envelopes,
conflicts, retry journals, and home checkpoint. An in-flight response observed
after the mismatch cannot advance the checkpoint. A transport error is not
itself a negotiated downgrade.

When version `1` returns, Chatbook resumes directly only if the server proves
the same activation epoch and unbroken publication-continuity token. A changed
or unprovable token moves the device to `required` and runs a fresh activation
baseline. This prevents server edits made while publication readiness was
absent from falling through an upgrade gap.

### Chatbook-originated mutation

1. The authorized Personal Context service validates the local request.
2. The Personal Context repository commits the immutable canonical revision,
   local speculative manifest advance, and eligible encrypted outbox rows.
3. After commit, mutation code sends a best-effort coordinator wake.
4. The coordinator stages semantic rows through the existing cross-database
   outbox dispatcher. Receipts make stage/retry/shred idempotent.
5. The derivative local manifest is not allowed to race ahead of its semantic
   siblings.
6. Sync V2 stores the staged client envelope as non-egress ingress with
   immutable base metadata and an explicit purge generation.
7. A replayable materialization attempt moves ingress from `received` to
   `applying`. The server accepts, rejects, or creates a conflict under its
   current canonical object and purge generation.
8. On canonical acceptance, the Personal Context transaction applies the
   semantic object, advances the authoritative manifest, writes the encrypted
   ordered publication batch, and records the ingress replay receipt.
9. Sync marks the ingress `applied` only after it can verify that receipt. A
   crash after canonical commit but before this mark replays as equality and
   returns the same canonical version and batch without a second manifest
   advance.
10. Server relay stages semantic home-authority egress followed by its
    manifest in Sync V2.
11. Chatbook pulls and applies the canonical server result, then advances its
    home-manifest checkpoint.

A push item counts as accepted only after step 8 commits. `received`,
`applying`, and retryable-failed ingress produces a retryable result and leaves
the Chatbook SyncState envelope pending. Structurally invalid or unauthorized
ingress becomes a terminal rejection; a conflict becomes terminal for that
push only after the server has durably attached the canonical authority
candidate described below. The encrypted Personal Context source snapshot may
still be shredded after verified local staging because the SyncState copy is
then its durable retry owner.

A failure at steps 3 through 11 never rolls back the local edit. The deduplicated
pending count and retry state remain visible.

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

Pull-time relay also has a fixed row and wall-time budget, initially 100 source
rows or 100 milliseconds per request. Reaching the budget before a visible
lookahead produces a retryable `personal_context_relay_pending` continuation
without advancing past unavailable authority data. This protects request
latency when old duplicate or poisoned debt is present.

The Sync store, not the source outbox, owns per-device retention once an
envelope is durable.

### Pull and cursor advancement

One existing `sync_once()` call may return only one pull page. A coordinator
run performs its push phase once, then drains a bounded number of Personal
Context pull pages, initially 10. It must not emulate draining by repeatedly
rerunning the complete push phase. If more pull pages remain, it yields and
schedules one follow-up run rather than monopolizing the event loop.

For each page, Chatbook processes delivered authority envelopes in server-
cursor order. The opaque pull cursor maintains two distinct notions:

- A server scan watermark may advance across client ingress after its immutable
  ingress role is validated, even though no client receives or stores it.

The delivered/application checkpoint advances across an authority envelope
only when it is:

- Validated and applied.
- Already represented by the same canonical head.
- Retained opaquely under the negotiated unknown-version rules.
- Converted into a durable conflict whose shared candidate envelope is pinned.

Classification as ingress is permanent. Canonical acceptance never mutates an
ingress row into egress; it creates a new deterministic home-authority envelope
at a later server cursor. The scan watermark can therefore pass hidden ingress
before or between visible authority rows without losing a future canonical
publication. Neither server scan progress nor a returned raw cursor is treated
as proof that Chatbook applied authority data.

The server's Personal Context pull filter returns only `applied`
home-authority egress after activation. Client ingress and any authority row
whose canonical receipt is not verified remain invisible regardless of their
generic envelope acceptance state. Receiving a repeated authority envelope for
an already-current canonical version is an idempotent no-op that may advance
the delivered/application checkpoint.

An authentication, decryption, integrity, manifest, generation, or purge
failure stops profile-wide advancement. An authenticated but invalid canonical
object may be quarantined as a per-object condition only when the server cursor
can still be recovered without accepting bad content; otherwise advancement
stops. The delivered/application checkpoint must never advance past an
unretained authority envelope that would be required for later review or
recovery.

### Manifest rebasing

Incoming manifest comparison uses the durable home-manifest checkpoint, not the
locally speculative manifest head. After the server publishes a canonical
manifest, Chatbook rebases its local derived view while preserving every
unaccepted semantic head and outbox row.

A manifest rebase may retire acknowledged derivative manifest work. It may not
drop semantic edits, rewrite their immutable base metadata, or reinterpret a
rejected mutation as accepted.

## Scheduling and retry behavior

The coordinator is single-flight and generation-coalesced. Eligible local work
increments a persisted content-free work generation. A wake during an active
run records the newest generation; it does not start a competing run. At most
one follow-up pass is scheduled for the accumulated generation.

Application startup schedules a check after services and link state are
available, but it resumes the persisted retry budget and `retry_not_before`.
It does not turn a crash/restart loop into unlimited fresh retries. When the
previous state was idle rather than exhausted, startup begins a normal check.

Strong wake sources that may reset an exhausted ordinary retry budget are:

- Authentication restoration or an observed offline-to-online transition.
- Capability restoration or activation completion.
- A new eligible Personal Context outbox work generation while connectivity is
  not known to be offline.
- A workspace mapping change that unblocks retained objects.
- Creation or update of a conflict-resolution intent.
- **Sync now**.

Opening **My Profile** or building an agent-context snapshot is a weak activity
wake. It may start one background check only when `next_activity_check_at` has
passed, no check is running, connectivity is not known offline, and the current
retry budget is not exhausted. The initial activity interval is five minutes.
Atomically advancing a persisted activity generation and
`next_activity_check_at` means concurrent context requests schedule at most one
check. A weak activity wake never resets an exhausted retry budget.

Agent-context construction never waits for network I/O. It uses the current
immutable local snapshot immediately and merely offers the coordinator a
content-free background wake.

V1 uses a bounded jittered retry sequence around 2 seconds, 10 seconds,
30 seconds, 2 minutes, and 5 minutes. After exhaustion, automatic work waits
for a strong wake. Multiple local edits within one active or exhausted work
generation coalesce; a later online edit generation may start one new budget.
Edits made while connectivity is already known to be unavailable increase the
pending count but do not re-arm an exhausted timer sequence.

Reconnection and **Sync now** re-arm ordinary transient failures. A server
`Retry-After` or equivalent rate-limit window is persisted as
`retry_not_before` and survives restart. Manual sync does not bypass that
server instruction.

Retry-budget transitions are atomic in SyncState:

- A strong wake with a new wake/work generation increments
  `retry_budget_generation`, sets `retry_attempt_index = 0`, and moves the
  budget to `active`. Replaying the same trigger generation cannot reset it a
  second time. An active server `Retry-After` remains the earliest permitted
  attempt even after this reset.
- Each transient failure uses the delay at the current zero-based attempt
  index, persists the next index and `retry_not_before`, and stays `active`.
  After the final five-minute slot is consumed without success, it becomes
  `exhausted` and has no scheduled timer.
- A successful fully drained check moves the budget to `idle`, resets the index
  to zero, clears ordinary retry/error timing, and records the completed work
  and activity generations.
- A restart loads these exact fields. It resumes one due `active` attempt,
  waits for a future `retry_not_before`, or leaves an `exhausted` budget idle;
  startup alone does not alter the generation or index.

The coordinator does not install a permanent poll, long poll, socket, or timer
that lives for the process lifetime.

Successful completion resets the retry counter and moves the next activity
check forward. `last_success_at` is not updated until the selected Personal
Context domains reach `has_more = false` and pull-time relay reports no pending
continuation. A bounded page-drain yield is progress, not a completed check.

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

Conflicts preserve both immutable candidates. An ordinary version conflict
freezes only the affected object. A same-scope semantic-key collision freezes
both record IDs and the contested `(profile_scope_id, kind, semantic_key)` slot
so neither record can steal the key through an unrelated edit. Other objects
and key slots continue synchronizing. Context uses the last mutually
acknowledged version or omits the object when no such version exists.

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

Every Personal Context push conflict must name and deliver the current
canonical authority candidate. Before the server reports the conflict, it:

1. Reads the exact canonical head through the Personal Context service.
2. Creates or reuses a deterministic `applied` home-authority envelope for
   that version, even when the version predates the requesting device's pull
   cursor.
3. Stores its ID as `remote_envelope_id` on the generic conflict record.
4. Returns the protected authority envelope in the versioned push-conflict
   item. A replay returns the same ID and bytes.

Chatbook encrypts and pins that returned envelope in SyncState before treating
the push item as terminally conflicted or retiring its staged retry. A pull-
time conflict instead pins the incoming home-authority envelope before cursor
advancement. Thus review never depends on a future pull or on an envelope that
retention may already have removed.

User-facing actions map to the server contract as follows:

- **Keep shared profile version** uses `skip`. No replacement envelope is sent.
  After server acknowledgment, Chatbook applies the already pinned shared
  candidate through the dedicated local conflict-resolution transaction.
- **Keep local version** uses `overwrite` and includes the rebased canonical
  replacement envelope inside the resolution request.
- **Merge** uses `overwrite` and includes the user-reviewed merged canonical
  envelope inside the resolution request.
- **Keep both as separate records** uses `duplicate_rename`, only after explicit
  user choice, with a new object ID and a noncolliding semantic key.

`skip` changes no server canonical object. `overwrite`, merge, and
`duplicate_rename` route the supplied canonical envelope through the server
Personal Context service; successful resolution creates an ordered authority
publication batch ending in the authoritative manifest. The generic conflict
does not become resolved until that canonical transaction commits.

Each mutating resolution has an idempotency key and uses the same
Personalization replay-receipt boundary as ordinary ingress. The canonical
transaction records the conflict token, expected envelope IDs, result, and
publication batch before Sync marks the generic conflict resolved. A crash
between those databases replays the exact receipt and cannot apply the merge,
overwrite, or duplicate twice. `skip` is a Sync-only conflict decision and
does not create a canonical publication batch.

Chatbook does not use ordinary inbound apply to install a selected divergent
candidate. Its dedicated local resolution transaction verifies the conflict
token and expected envelope IDs, retires or redirects the conflicting local
head and staged envelope, installs the selected canonical result, releases the
exact object/key-slot freeze, and records the journal outcome atomically. For
`overwrite` or merge it remains frozen until the matching authority
publication is applied. For `duplicate_rename`, the shared candidate keeps the
contested key and the local candidate is installed under the reviewed new ID
and noncolliding key.

The merge editor uses typed fields rather than raw canonical JSON. Restrictive
values win by default for delete, `user_only`, and other privacy-reducing
controls; the user must explicitly choose a less restrictive result.

Conflict pages contain at most 20 items. Labels are **Local version** and
**Shared profile version**, not ambiguous client/server winner language.

## Restrictive privacy cleanup

A change that reduces agent visibility, replication, or retention propagates
its restrictive canonical state immediately. Every context, search, summary,
index, cache, and tool boundary consults that new head before returning data;
cleanup work cannot temporarily authorize the old value.

Each runtime atomically journals a content-free cleanup requirement bound to
the exact canonical version and purge generation before exposing the new head.
It then invalidates in-memory snapshots and removes or rewrites derived
artifacts. Completion writes a peer-local acknowledgment for that version.
Server cleanup can span separate derived stores, so its journal is replayable
and its acknowledgment is reported through the ongoing-sync contract.

The canonical mutation and authority publication do not wait for every cleanup
job, because delaying the restrictive head would preserve broader access.
Instead, **Privacy cleanup pending** remains a separate incomplete condition
until this Chatbook device and the home server have acknowledged their required
cleanup. A later transport check may update **Last checked**, but it cannot
clear or describe that privacy change as complete. Other devices create and
track their own requirement when they receive the authority publication.
Offline devices remain protected by their existing encryption and must apply
the restrictive head before serving profile context on reconnect.

Cleanup retries follow the same single-flight discipline but do not generate a
routine notification. A persistent failure that requires configuration or
manual repair becomes Attention. Cleanup acknowledgment contains no profile
value or readable semantic metadata.

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
- **Privacy cleanup pending** when a restrictive change has not received its
  required local and home-server acknowledgments.
- **Sync now**.

Attention is orthogonal to the primary state. A profile can be **Queued** with
one conflicting object while other work continues. The interface does not use
"Up to date" because another device or server edit may exist after the last
successful check.

Primary-state precedence is deterministic: an active run is **Syncing**; a
pending retry window is **Retrying**; remaining unsent semantic work is
**Queued**; otherwise an active home binding is **Linked**. Attention and
privacy-cleanup state are secondary and may coexist with any primary state.

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

One pending semantic unit is the exact `(commit_id, object_type, object_id,
version_id)` mutation. Its Personal Context source row and staged SyncState copy
count once; two genuinely pending versions count separately, and the derivative
local manifest barrier adds nothing. Conflict reviews, resolution intents,
cleanup acknowledgments, and activation are reported as separate bounded
counts rather than inflated into that semantic total.

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
The editor freezes while deletion is pending. Every Personal Context ingress,
source-publication row, authority envelope, conflict candidate, and purge
barrier carries an explicit purge generation.

For purge ordering, "accepted" means that the canonical Personalization
transaction committed, not merely that an ingress envelope became durable in
the separate Sync database. The server uses its per-profile canonical mutation
serialization and one recoverable per-profile relay lease:

1. The purge transaction advances the generation, destroys canonical and
   derived-readable content under its cleanup journal, terminalizes or fences
   every older source-publication row, and creates the new generation's
   deterministic barrier publication.
2. An older-generation ingress that was durable in Sync but had not committed
   canonically is marked `stale_generation` on replay and can never materialize.
   A canonical mutation committed before the purge is removed by the purge
   transaction.
3. Relay rechecks current generation and source-row state while holding the
   profile relay lease. It cannot stage an older-generation authority envelope
   after the barrier. A crash releases the lease without weakening the durable
   fence.
4. The barrier becomes Sync-visible only after every already-staged permissible
   older-generation authority publication precedes it and every unstaged older
   source row is terminal. Sync rejects any older-generation authority envelope
   presented after that point, regardless of server cursor.
5. Devices erase their replicas when they receive the barrier. The server
   crypto-shreds readable Personal Context Sync history and retains only the
   minimal content-free acknowledgment ledger.

A new profile cannot be created from the old link until the initiating device
has received acknowledgment. The purge request and barrier are idempotent under
restart and duplicate delivery.

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
purge-pending, privacy-cleanup-pending, ingress-retryable,
ingress-stale-generation, per-object-conflict, workspace-unmapped,
relay-pending, and relay-poisoned.

## Verification strategy

### Contract and model conformance

- The server-generated schema artifact is checked in with version and source
  commit metadata; Chatbook's vendored copy has the same checksum.
- Both applications accept valid capability, server-origin, and conflict
  fixtures and reject invalid or ambiguous variants.
- Chatbook sends the authoritative batch endpoint and server action names.
- Reserved authority identity cannot be registered or submitted by a client,
  and only applied authority envelopes pass the Personal Context egress filter.
- Pagination with ingress before and between authority rows advances the scan
  watermark while delivering every authority row exactly once and preserving
  the independent application checkpoint.
- Client ingress, ordered publication batches, activation receipts, continuity
  tokens, expected conflict candidates, and purge generations round-trip
  without changing canonical bytes.

### Repository and interruption tests

Tests use real temporary Personal Context and SyncState/Sync SQLite databases,
plus real temporary server Personalization and Sync databases. They inject a
failure and restart at every cross-database boundary:

- Chatbook canonical commit before wake.
- Personal Context outbox stage before receipt.
- Receipt before source-row shredding.
- Server Sync ingress durability before Personalization materialization.
- Server Personalization canonical/publication commit before Sync ingress
  terminalization; replay proves no second manifest advance or batch.
- Each partial semantic ordinal of a server publication batch before its
  manifest; restart preserves order and deterministic envelope IDs.
- Two after-commit/pull-time relay attempts interleave across consecutive
  batches; the shared lease and earliest-incomplete rule preserve global
  profile publication order across every restart point.
- Activation chooses only a whole-batch publication watermark. Restarts between
  baseline installation, the `covered_by_activation` CAS, encrypted source-row
  compaction, and the first post-watermark relay preserve the covered-through
  terminal proof and resume at the next sequence.
- Sync authority-envelope durability before source acknowledgment.
- Push-conflict authority-candidate creation before response and Chatbook pin;
  pull-conflict pin before cursor advancement.
- Resolution transmission, canonical resolution commit, authority
  republication, and dedicated local resolution apply.
- Restrictive head commit before cache/index cleanup and acknowledgment.
- Purge-request durability before local key destruction.
- Every ordering of old-generation ingress insertion, canonical
  materialization, authority publication, purge commit, and barrier relay.
- Every activation transition around baseline preparation, Sync installation,
  local reconcile/rebase, acknowledgment, and post-watermark publication,
  including a local edit before and during installation.
- Every persisted retry slot and the exhausted state immediately before and
  after restart.

Every case must converge without duplicate semantic mutations, lost edits,
cursor skipping, or plaintext fallback.

### Multi-device integration

A two-Chatbook-device and one-server matrix covers:

- Offline local edit, reconnect, and convergence.
- Direct server edit reaching both devices, including the originating device
  after a prior client push.
- Client ingress never reaching another device before canonical authority
  publication, and duplicate authority delivery applying as already-current.
- Concurrent record edits and same-key creations.
- Version conflicts freezing one object and key collisions freezing both
  records plus only their contested semantic-key slot.
- Safe manifest rebasing with additional pending local edits.
- Workspace mapping on one device without leaking that mapping to another.
- `device_only` exclusion and `user_only` agent exclusion.
- Retry exhaustion, restart, reconnect, rate-limit persistence, and manual
  sync.
- Throttled activity discovery in a long-lived connection, with context
  construction remaining nonblocking and exhausted budgets not storming.
- Existing-link activation while a server edit races the baseline.
- Existing-link activation with local pending edits before preparation and
  another edit during client staging; neither head nor outbox item is retired.
- Capability downgrade and restoration with both continuous and changed
  activation epochs.
- A direct server edit during version `0`, covering the journal-preserved token
  path and the invalidated-token/fresh-baseline path.
- Remove-local-copy behavior and delete-everywhere generation fencing.

### UI and accessibility

Tests mount the production `PersonalContextSettingsPanel` hierarchy and CSS at
supported terminal sizes. They cover every primary state, simultaneous
attention and privacy-cleanup state, deduplicated pending counts, long
sanitized labels, disabled-action explanations, keyboard-only conflict review,
focus return, pagination, and destructive confirmations.

### Privacy evidence

Unique canary profile values are scanned across both applications' ordinary
databases, WAL files, outboxes, conflict storage, retry state, logs,
diagnostics, recovery artifacts, migration snapshots, caches, and
application-owned backups. Only encrypted or explicitly content-free artifacts
may contain related state.

Transition tests hold an old context snapshot, search/index entry, cache, and
summary while a record becomes `user_only`, device-only, archived, or deleted.
No agent-facing boundary may return the old value after the restrictive head
commits, even before cleanup acknowledgment.

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
- Client-authored Personal Context envelopes are ingress-only; no other device
  can pull pending, failed, conflicted, stale, or accepted ingress before its
  applied home-authority publication.
- Device-only-only mutations never create any Sync outbox row.
- A crash at any documented cross-database boundary recovers idempotently.
- A replay after canonical application but before ingress terminalization
  returns the original result without another canonical version, manifest
  revision, or publication batch.
- Every authority publication batch makes all semantic envelopes durable before
  its manifest becomes pull-visible, including after a partial-batch crash.
- Consecutive publication batches become egress in monotonic per-profile order
  even when both relay entry points interleave and restart.
- Manifest sequencing cannot retire or overwrite an unaccepted semantic edit.
- Conflicts retain and deliver both encrypted candidates, freeze only the exact
  object or contested key slot, and resolve through the authoritative batched
  server contract plus a dedicated local resolution transaction.
- Pull scan watermarks may pass only immutable permanent non-egress rows;
  delivered/application checkpoints never advance past authority data that was
  neither safely handled nor durably retained.
- Existing links complete the journaled activation state machine without
  losing a server mutation racing the activation baseline or an unaccepted
  local edit made before/during installation; source compaction cannot precede
  durable baseline installation and the durable whole-batch
  `covered_by_activation` transition. Compaction preserves content-free proof
  that global publication ordering may continue at the next sequence.
- Capability downgrade preserves queued work and checkpoints, and restoration
  requires a new baseline unless the same durably issued epoch/token proves
  uninterrupted publication journaling on every version-1 exchange.
- Automatic retries stop after the bounded sequence and restart only for a
  defined strong trigger; throttled activity reads do not create request-path
  I/O or retry storms, and the exact budget generation, attempt index,
  exhausted state, and server retry window survive process restart.
- **Settings → My Profile** reports accurate queued/syncing/retrying/attention,
  privacy-cleanup, last-attempt, last-success, deduplicated pending, and
  workspace-mapping state.
- A restrictive change is excluded from agent-facing paths immediately and is
  not reported cleanup-complete until local and home-server version-bound
  acknowledgments exist.
- User-action notifications are deduplicated across restart; routine offline
  and retry transitions do not notify.
- Remove-local-copy and delete-everywhere preserve their distinct scopes, and
  an old-generation ingress or device cannot resurrect globally deleted
  content or publish behind the purge barrier.
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
