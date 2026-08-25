# ADR-040: Synchronized moodboard and Studio authority

**Status:** Proposed
**Date:** 2026-08-24
**Decision owner:** TASK-13007 requester and implementation review
**Related task:** `TASK-13007`
**Related ADRs:** `Docs/ADR/031-notes-capability-sync-domains.md`,
`Docs/ADR/034-durable-server-origin-sync-mutation-batches.md`, and
`Docs/ADR/039-canonical-notes-task-sync-and-derived-checklist-projections.md`
**Related spec:**
`Docs/superpowers/specs/2026-08-24-notes-moodboard-studio-sync-design.md`

## Decision

Synchronize Notes moodboards, explicit note placements, and accepted Studio
sidecars as three independently versioned Sync v2 domains:

- `notes.moodboard`
- `notes.moodboard_note`
- `notes.studio_document`

The existing product tables remain authoritative projection stores. A moodboard
keeps its local integer REST key and gains a portable UUIDv4 `sync_id`. An explicit
placement remains unique for one `(moodboard, note)` pair and uses a deterministic
relationship identity derived from the portable moodboard and note identities. A
Studio sidecar remains one-to-one with its Notes row and uses the note UUID as its
object identity.

Placement identity uses the existing exact relationship namespace
`notes.moodboard_note:sha256:<64-lowercase-hex>`. Its parent is the moodboard
portable UUID and its dependencies name both board and note.

Only explicit manual placements synchronize. Smart-rule matches remain derived
from synchronized Notes, organization, and conversation state on each replica.
Canonical smart rules use portable collection identities; local integer IDs are
translated at the product boundary. A smart-only match never creates a placement
envelope.

Derived matching uses a versioned backend-independent algorithm: Unicode
NFC/casefold literal matching whose compatibility ID includes the runtime's exact
Unicode data version, exact normalized sources, portable collection membership,
and inclusive UTC bounds. Updated rules use server-bound `canonical_modified_at`
metadata projected identically for every accepted `notes.note` envelope;
replica-local clocks and backend `LOWER`/`LIKE` semantics are not authoritative.

Requests never scan an owner's complete note set. Smart matches live in a
disposable local owner/dataset-scoped projection with bounded resumable rebuilds,
dirty state, atomic generation publication, and explicit completeness status. It
is neither product nor Sync authority and can be dropped and rebuilt.

The Notes row remains authoritative for title and Markdown content. The Studio
sidecar remains authoritative for structured render state. A save that changes
both is recorded as a complete ordered ADR-034 mutation group with `notes.note`
first and `notes.studio_document` second. The Studio payload binds to the exact
accepted note revision and hash instead of duplicating note title or content.

Client note-plus-Studio changes arrive as one closed Studio compound command. The
server validates its complete note intent and deterministically expands it into the
same two primitive envelopes with all-or-none append. Mutation-group fields remain
server-owned and response-only. Dataset/device/client-envelope identity locates an
existing group. A separately persisted canonical fingerprint over validated
client-controlled command fields distinguishes exact replay from changed-intent
conflict before server expansion. The plan hash verifies only the integrity of the
stored expanded plan.

Client note tombstone/restore commands are also server-expanded with the retained
Studio lifecycle step when one exists. Ordinary note upserts remain independent and
may legitimately make a Studio binding stale.

Studio note lifecycle is fail-closed by state. Before enrollment, existing Notes
behavior retains and hides the local sidecar without claiming Studio Sync. During
capture-enabled bootstrap, server REST writes append the full group and external
pushes wait. When ready, a device that lacks the Studio adapter cannot delete or
restore a note with a retained sidecar. After Studio history exists, an unhealthy
Studio coordinator blocks that lifecycle mutation rather than leaving an
unsynchronized sidecar. An ordinary `notes.note` upsert may proceed during a pure
Studio-readiness degradation only when Notes capture is healthy and the canonical
predecessor chain is clear. A pending, failed, or conflicting accepted predecessor
activates ADR-034's dataset barrier; only exact replay, repair, or explicit conflict
resolution may proceed.

Only accepted persisted Studio state synchronizes. Generation requests, prompts,
previews, failures, credentials, and raw unaccepted model responses remain
operations. Accepted provenance distinguishes server-attested execution,
client-declared execution, manual changes, and trusted legacy bootstrap without
carrying secret-bearing request values.

Canonical Studio structured payload, diagram manifest, and provenance are closed
schemas versioned by render version. Nested canonical payload contains sections
only; rendering injects title from `notes.note` and source/layout values from the
outer Studio authority. The legacy REST serializer rehydrates its established
nested `meta`/`layout` product view, and equal legacy nested request fields are
stripped at the boundary. Derived SVG and diagram compatibility aliases are rebuilt
locally, excluded from canonical hashes, and sanitized at render/export. Provider
dictionaries and arbitrary metadata are reduced to accepted product shapes before
capture. `source_note_id` must be a known live same-owner/same-dataset note when
new state is accepted; an already valid reference may remain retained if its
source is later tombstoned.

All three domains use whole-object canonical lineage, exact base comparison,
complete payload tombstones, and explicit restore intent. Complete tombstones
preserve board configuration, placement layout, and Studio sidecar state for
deterministic restore. Tombstone does not imply physical erasure from append-only
Sync history.

Conflict resolution supports overwrite and skip for all three domains.
Duplicate-rename is rejected: placement and Studio identity cannot change, and
duplicating a board without its placements would add an unapproved partial-copy
product operation.

Note lifecycle delete/restore preserves the Studio payload's previous note binding
and provenance, including legitimate stale state. Only an accepted Studio save
rebinds the sidecar to the current or planned note head.

Legacy product rows migrate first into proven owner scope and reserved
`local-unbound` dataset scope. TASK-13007 reuses schema-v60's existing
`note_task_scope_authority` relation as the sole owner-to-default-personal-dataset
authority despite its legacy name; it creates no second authority table. The row
gains one-way `task_graph_bound`, `moodboard_graph_bound`, and
`studio_graph_bound` flags. Existing rows migrate with the task flag true only
after complete task-graph consistency verification, and task callers stop treating
row presence alone as task binding. Each graph flag changes
false-to-true in the same transaction as that complete graph's verification/rekey,
so malformed unrelated state does not block the healthy unit. An empty unit still
records its binding. Conflicting authority or wrong-dataset state fails closed. The
authority row itself retains owner-only forced PostgreSQL RLS with `USING` and
`WITH CHECK`.

The additive DDL defaults task bound to true and the two new graph flags to false,
preserving old task binders that insert only owner/dataset. New binders always write
all flags explicitly. Production moodboard/Studio binding cannot activate until no
row-presence-era task server remains; otherwise an old caller could misread a
non-task-first authority row.
Local-unbound rows can support inactive compatibility behavior but can never enter
canonical capture or readiness.

PostgreSQL product and local smart-projection tables carry direct owner/dataset
scope and force RLS. Direct placement and Studio scope must agree with parent scope
through composite database constraints where possible and same-transaction
product-store validation otherwise.

Moodboard and placement readiness and writable advertisement are coupled. Studio
readiness and advertisement are independent but require `notes.note`. The three
domains remain absent from public supported and writable capabilities until their
storage, capture, bootstrap, repair, security, dependency, and live-PostgreSQL
predicates pass.

The first four child tasks deliver contracts, storage, portable smart matching,
and internal-only dormant domain machinery with no production path that can
publish canonical history. Only `TASK-13007.5` wires production enrollment/capture
and may advertise the domains. This keeps pre-activation rollback distinct from
post-history rollback.

Portable `notes.note` modification-time parsing/projection also lands dormant in
TASK-13007.2. TASK-13007.5 is the first production writer of that routing metadata;
afterward compatibility builds permanently retain new-field projection and the
old-envelope receipt-time fallback.

Enrollment is limited to the Chatbook default-personal, server-materializable
`server_trusted_v1` encryption policy. Opaque client-only policies cannot advertise
these server-inspected product domains.

Server-origin mutations append a complete canonical plan before product
materialization. Product projection is ordered, resumable, and idempotent but not a
distributed transaction. The initiating API reports success only when all steps
are applied.

The schema migration cannot be binary-reverted because older initializers reject a
newer ChaChaNotes schema. Before production history, rollback requires a
forward-compatible dormant build or restoration of a pre-migration database before
starting an older binary. After a dataset publishes canonical history, rollback
also requires retention of the capture gates, or maintenance mode plus a
pre-activation database restoration.

## Context

ADR-031 established independent versioned domains for mutable Notes capabilities
and kept the core note payload lossless. TASK-13007 is the remaining Notes parity
slice for visual moodboard organization and accepted Studio document state.

The current moodboard table has a local integer identity and soft-delete/version
fields, but no portable UUID, dataset binding, canonical lineage, or canvas state.
Its link table has only a composite pair and creation timestamp; unpin physically
deletes the row. Smart rules contain local collection IDs and computed membership.

The current Studio table is a one-to-one note sidecar with structured rendering
state but no explicit tenant/dataset scope, lifecycle, canonical lineage, or
accepted-output provenance. Studio title and Markdown already belong to the normal
Notes row. Introducing a second title/content authority would create divergence
between domains.

Moodboards and Studio both have compound operations. Board deletion may have many
placements, so cascading a tombstone group would be unbounded. Studio saves and
note delete/restore affect at most the note and one sidecar and fit ADR-034's
bounded durable group model.

The product and Sync databases do not share a transaction manager. ADR-034 already
provides the smallest durable intent and repair boundary, so this decision reuses
it rather than introducing another transaction mechanism.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Put moodboards, placements, and Studio state inside `notes.note` | They are independently mutable and would create whole-note conflicts and partial-client overwrite risk contrary to ADR-031. |
| Add Sync-owned shadow tables as the canonical product authority | This duplicates existing product authority and creates projection, backfill, and rollback failure modes without user value. |
| Use an append-only Studio revision/event product model | The current product supports one latest sidecar; immutable history is not required by TASK-13007 and would add retention and conflict policy. |
| Give Studio documents independent IDs or allow several per note | It breaks the existing one-to-one model and introduces unsupported hierarchy and lifecycle semantics. |
| Allow duplicate placements of one note on a board | It breaks current uniqueness and requires a new user-visible identity and interaction model. |
| Synchronize computed smart matches | It turns a local projection into mutable authority and risks stale materialized membership. Portable rules plus synchronized dependencies are sufficient. |
| Evaluate every smart rule directly on each request | Literal query-only boards can scan the complete note set and make counts/high pages unbounded. A disposable resumable projection preserves derived authority while bounding requests. |
| Add a moodboard/Studio-specific scope-authority table | TASK-13006 already established one immutable owner-to-default-personal-dataset authority. A second row could disagree and split one owner's Notes graph across datasets. |
| Tombstone every placement when deleting a board | A large board produces an unbounded mutation group. Retaining hidden placements preserves restore behavior with a bounded board tombstone. |
| Duplicate note title and Markdown in the Studio payload | Two canonical authorities could disagree. Ordered note-plus-Studio groups preserve the aggregate without duplication. |
| Store prompts and generation responses as provenance | They may contain secrets or sensitive transient content. Provenance needs only bounded accepted-transition facts and hashes. |
| Chunk oversized Studio payloads in this task | It adds a new multi-envelope object protocol and restore/conflict semantics. V1 fails before write and blocks readiness instead. |
| Treat product projection as atomically committed with Sync | The databases do not share a transaction manager. ADR-034 durable plans plus idempotent repair are the established boundary. |
| Permit ordinary downgrade after activation | Older binaries cannot understand future write gates and may create unsynchronized product mutations. |

## Consequences

Moodboards, manual placements, and Studio sidecars become portable without changing
their established product ownership. Integer moodboard REST routes remain
compatible while clients gain stable resource IDs and canonical revisions.

Unpin must become a soft tombstone and placement rows gain layout, lifecycle, and
scope columns. Board and note soft deletion retain placements. Studio note
delete/restore becomes a bounded note-plus-sidecar group, and active hard-delete
paths require an authorized retention-aware workflow.

Smart matching remains local and requires its input domains to be ready. Collection
filters must translate portable IDs and the currently accepted-but-ignored
collection filter must be implemented. Source-filtered boards require compatible
conversation state. Notes and related authorities also need portable normalized
query projections and a server-bound modification time so SQLite and PostgreSQL
evaluate the same rule identically. Runtimes with different Unicode data versions
cannot claim the same smart-match compatibility or writable moodboard readiness.

Studio operations must distinguish accepted persistence from generation. Provider
and model provenance is trustworthy only when server-attested; client claims stay
explicitly labeled. Existing arbitrary Studio metadata must be reduced to the
closed accepted schema or diagnosed as a readiness blocker. Client compound saves
require deterministic server expansion before Studio can activate. Legacy nested
title/source/layout values are removed only when they exactly match their external
authorities; mismatches block readiness.

Fresh and upgrade migrations become more substantial: they must prove ownership,
preserve local-unbound compatibility, canonicalize legacy rows, consolidate Studio
schema authority, enforce direct scope and RLS, and surface malformed state without
silently discarding it.

The default Sync envelope size becomes a user-visible activation constraint for
large Studio state. Active writes fail before product mutation and oversized
legacy rows block readiness until repaired or reduced.

Server-generated creates require an idempotency key once capture is active, while
inactive compatibility routes retain their prior permissive behavior. Missing
active preconditions fail before provider work, identifier allocation, append, or
product mutation.

Domain behavior can be deactivated while production history is still absent, but
the schema migration itself cannot be binary-reverted: older initializers reject a
newer ChaChaNotes schema. Pre-activation code rollback therefore retains
forward-compatible schema support or restores a pre-migration database. Rollback
is further constrained after activation by the capture gates and canonical history.
Documentation and deployment tests must make both boundaries explicit.
