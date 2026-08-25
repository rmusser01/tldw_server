# ADR-040: Synchronized moodboard and Studio authority

**Status:** Proposed  
**Date:** 2026-08-24  
**Decision owner:** TASK-13007 requester and implementation review  
**Related task:** `TASK-13007`  
**Related ADRs:** `Docs/ADR/031-notes-capability-sync-domains.md`,
`Docs/ADR/034-durable-server-origin-sync-mutation-batches.md`  
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

Only explicit manual placements synchronize. Smart-rule matches remain derived
from synchronized Notes, organization, and conversation state on each replica.
Canonical smart rules use portable collection identities; local integer IDs are
translated at the product boundary. A smart-only match never creates a placement
envelope.

The Notes row remains authoritative for title and Markdown content. The Studio
sidecar remains authoritative for structured render state. A save that changes
both is recorded as a complete ordered ADR-034 mutation group with `notes.note`
first and `notes.studio_document` second. The Studio payload binds to the exact
accepted note revision and hash instead of duplicating note title or content.

Only accepted persisted Studio state synchronizes. Generation requests, prompts,
previews, failures, credentials, and raw unaccepted model responses remain
operations. Accepted provenance distinguishes server-attested execution,
client-declared execution, manual changes, and trusted legacy bootstrap without
carrying secret-bearing request values.

All three domains use whole-object canonical lineage, exact base comparison,
complete payload tombstones, and explicit restore intent. Complete tombstones
preserve board configuration, placement layout, and Studio sidecar state for
deterministic restore. Tombstone does not imply physical erasure from append-only
Sync history.

Legacy product rows migrate first into proven owner scope and reserved
`local-unbound` dataset scope. Explicit enrollment binds the complete owner graph
to the sole default-personal dataset under a scope-authority record and dataset
fence. Local-unbound rows can support inactive compatibility behavior but can
never enter canonical capture or readiness.

PostgreSQL product tables carry direct owner/dataset scope and force RLS. Direct
placement and Studio scope must agree with parent scope through composite database
constraints where possible and same-transaction product-store validation
otherwise.

Moodboard and placement readiness and writable advertisement are coupled. Studio
readiness and advertisement are independent but require `notes.note`. The three
domains remain absent from public supported and writable capabilities until their
storage, capture, bootstrap, repair, security, dependency, and live-PostgreSQL
predicates pass.

Server-origin mutations append a complete canonical plan before product
materialization. Product projection is ordered, resumable, and idempotent but not a
distributed transaction. The initiating API reports success only when all steps
are applied.

After a dataset publishes canonical history, arbitrary rollback to a binary that
predates these capture gates is not supported with writes enabled. Rollback
requires a compatibility build retaining the gates, or maintenance mode plus a
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
conversation state.

Studio operations must distinguish accepted persistence from generation. Provider
and model provenance is trustworthy only when server-attested; client claims stay
explicitly labeled.

Fresh and upgrade migrations become more substantial: they must prove ownership,
preserve local-unbound compatibility, canonicalize legacy rows, consolidate Studio
schema authority, enforce direct scope and RLS, and surface malformed state without
silently discarding it.

The default Sync envelope size becomes a user-visible activation constraint for
large Studio state. Active writes fail before product mutation and oversized
legacy rows block readiness until repaired or reduced.

Operational rollback is simple while the domains are dormant and constrained after
activation. Documentation and deployment tests must make that boundary explicit.

