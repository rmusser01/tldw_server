# ADR-065: Portable note language and negotiated structured editing

**Status:** Accepted
**Date:** 2026-10-06
**Backfilled from:** not backfilled
**Decision owner:** Structured document editing requester, approved design 2026-09-27
**Related task:** TASK-13513
**Related spec/plan:** Chatbook `Docs/superpowers/specs/2026-09-27-structured-document-editing-design.md`, `Docs/superpowers/plans/2026-09-27-structured-document-foundations.md`, and ADR-194
**Extends:** ADR-031; its accepted version-1 payload remains unchanged.

## Decision

Persist nullable `content_language` as canonical note metadata, independent of the
language hint used for title generation. Known selector modes are `text`,
`markdown`, `yaml`, `json`, and `jsonl`. Preserve unknown nonempty strings of at
most 100 characters without normalization; this modest metadata bound follows the
existing Notes keyword boundary, rather than introducing a closed enum. Omission
on update preserves the previous value; explicit null clears it. Language-only
changes use the same optimistic note version as source changes.

Keep accepted title and content exact. Language, syntax checking, and formatting
never relax the API nonempty-note contract or add syntax-save vetoes. Existing
raw database writers continue accepting empty content.
Add the column without downgrading or erasing data on rollback. Ordinary reads,
copy-by-create, current versioned rows, and database backups preserve the metadata.
Portable archive propagation follows the separately versioned archive work.

Negotiate a new clear `notes.note` payload version and a new private Notes payload
version before publishing this field. Do not append it to ADR-031's locked v1
payload or change that payload's hash semantics. Private language stays encrypted
with note content. Mixed-version paths must report unsupported/pending, preserving
stored metadata. Until complete publication exists, active Sync rejects explicit
language writes and updates/restores of language-bearing notes before persistence;
capabilities report complete write support, not merely column availability.

Validation and explicit formatting remain local, bounded, source-preserving, and
without save/network authority. Browser/Python parser dependencies, worker choices,
and editor undo adapters require their later qualification evidence; this ADR
pre-adopts none of them. No validation API, LSP, external lint command, format on
save, or generic object reserialization is introduced.

## Context

ADR-031 deliberately fixes the initial canonical payload. A new portable field
cannot safely travel via a lossy old snapshot. Additive storage and ordinary API
support can precede negotiated publication only with an honest temporary gate.
The accepted cross-repository design requires exact source and unchanged authority.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Device-local mode only | Loses the user's portable selection. |
| Closed storage enum | Destroys round trips for future modes. |
| Extend v1 without negotiation | Old validators reject it; old writers can erase it. |
| Report support whenever the column exists | Misrepresents incomplete active-Sync writes. |
| Add canonical version-history/duplicate APIs | Existing optimistic rows and create-a-copy suffice for this foundation. |

## Consequences

A mode over 100 characters must be rejected explicitly; a later protocol change
can expand the bound if real identifiers need it. Existing null-language clients
continue to work. Active Sync temporarily limits metadata-bearing note writes
until the new negotiated contracts are complete. UI duplicate/history adapters,
portable archives, parser and worker qualification remain their owning tasks.

## Admission concurrency clarification (F2 review, 2026-10-06)

The complete-write decision linearizes at durable Sync append. Use the existing
Sync dataset transaction first, then the canonical Notes transaction/row lock;
re-read language and any supplied expected product version under that lock.
Commit the Sync append while the read-only Notes lock remains held, then release
it before existing materialization. Batch append may join the caller's existing
Sync connection, matching single-envelope insertion. Ordinary, batch and inbound
capture share this admission rule. A rejected admission stores no envelope.

A canonical change after durable append is a later write. Existing replayable
materialization failures and independently committed product steps remain intact.
This is not a general atomic commit across the two databases. Extending a product
transaction across batch materialization would change those commit/replay
semantics and is explicitly outside this clarification. Neither another unlocked
preflight nor a product-before-Sync lock order meets the contract.

PostgreSQL existing-row admission uses FOR UPDATE. An absent row requires a short
Notes table write fence followed by a fresh read so a same-ID INSERT cannot slip
through; only that absent-row path briefly serializes other Notes writes. SQLite
already supplies its database write fence. The expected product version is a
process-local admission condition and never changes v1 payloads or hashes.

## Complete publication qualification (F3, TASK-13514)

The allocated clear `notes.note` adapter/schema pair is `(2, 2)`. Its complete
snapshot contains all four v1 fields and explicit `content_language`, either null
or an exact nonempty string of at most 100 characters. The locked v1 validator and
hashes remain unchanged. New publications use one complete snapshot builder;
REST, restore, compound organization, ingestion, interop, Studio/WebClipper and
both task-projection producers retain source and language. Legacy admitted v1
materialization preserves current language; new unsafe v1 writes are refused under
the existing pre-append Notes admission lock. A v1 writer cannot downgrade a v2
head even with an otherwise current base.

Versioned pull uses the existing `SYNC_V2_PULL_TOKEN_SIGNING_SECRET`. Without it,
portable-language/writable-v2 capabilities stay unavailable. New null-language
objects and v1 heads may continue through v1; language-bearing writes and v2 heads
cannot fall back. No new secret owner or generated fallback key is introduced.
Losing the key produces actionable HTTP 503 before a versioned pull advances its
cursor. Already durable ordinary/batch capture retries retain their original
versions, hashes and materialization semantics.

A selected Notes stream containing accepted v2 data refuses an older consumer
with `notes_snapshot_upgrade_required` (HTTP 409). It does not down-project or
rehash a newer head. This check uses a bounded indexed store read after page
scanning and before cursor changes. Any later append is beyond the scanned page's
watermark and is detected on the next pull. Other domains may be pulled separately.

The actual server does not register a legacy private `notes` domain, and its API
rejects private envelope ingress. F3 retains this boundary and advertises private
transport as unavailable; it does not add a private transport. Private clients must
retain encrypted metadata and pending local intent until a peer negotiates that
contract. No language is added to clear routing metadata.

Canonical rows do not record their historical wire version. New row-derived Studio
provenance therefore uses v2. Existing storage verification may recognize the exact
v1 hash of the same owner-bound current row only when language is null and existing
revision/content/deletion checks still hold. Sidecar canonical/result hashes remain
mandatory. Known Sync heads retain strict versioned lineage. Historical migration
hashes and persisted accepted provenance are never rewritten.

### F3 review clarification

Read negotiation also uses signing readiness: without the key a modern client
can still consume safe v1 Notes through the existing unsigned path. Version 2
is absent from runtime reader capabilities/stream selection until ready. The
bounded post-scan history check still refuses inaccessible v2 data with HTTP 503
for modern readers (HTTP 409 for old readers), before cursor writes; it never
filters that history as successful delivery. Preserved language in shared
capture is conditioned on the version of the canonical row actually read even
when the caller omitted its optimistic version. This remains a process-local
admission condition; absent-row creation and wire/hash shapes are unchanged.

### Immutable publication origin scope (F4)

Existing local publication-intent payloads carry the exact originating principal and workspace, including explicit null. Drain compares both before copying source into a scoped outbox. Missing historical proof remains pending; an active profile cannot supply historical authority. This local metadata never enters wire payloads and adds no schema or transport.

### Scoped offline drafts (F5)

The existing authority-scoped browser offline queue remains the only draft storage
owner. Version-2 entries capture exact source, optional language (legacy absence
means unknown), and a draft revision. Unknown/structured modes cannot enter
Markdown conversion. A captured complete language-capable acknowledgement removes
only its matching draft; rejected and empty drafts remain local. Storage failures
fence navigation instead of reporting a saved draft. This adopts no new editor.

### Fresh default provenance (F5 review Ruling32)

Version-3 entries in the same offline queue carry `implicitDefaultLanguage` proof
only for a fresh new draft whose null editing mode has never been explicitly
selected. This known UI default may omit language on creation without requiring
language capability; cleanup still requires exact source, a valid new identity
and a version acknowledgement. Selecting even the same null mode consumes that
proof and is a draft mutation. Canonical loads, recognized metadata and ambiguous historical entries receive no
exemption. Duplication carries the coherent mode/provenance tuple through the
existing reset helper: unknown remains unknown and proven implicit default remains
eligible, while a genuinely fresh reset still starts in the known UI default.
Restoration accepts the proof only for a v3 new-draft entry with null language and
no note/base-version identity. Rekeying to a saved note consumes it. Explicit mode
changes/resets retain capability and complete language acknowledgement protection.
No new storage owner, inferred language, or downgrade of an explicit mutation.

### I1 Monaco qualification boundary (2026-10-07)

The requester authorized a removable Monaco candidate experiment after WebKit's
native textarea history failed. The installed Monaco distribution is copied into
Next's public directory and WXT's existing shared public directory by a test-only
preparation script. The candidate explicitly configures the loader to that same
origin path. Test preparation also emits same-origin static worker bootstraps
that import the installed worker assets and emit Monaco's readiness message;
this avoids the distribution's blob-worker default under extension CSP. No CDN, package
installation, source validation network request, or production editor adoption is
introduced. Markdown keeps its existing editor. Native edits use Monaco history,
never a second undo stack. BOM, mixed-newline and lone-CR source is refused before
mounting an editable model; the exact original remains in a read-only fallback.
Homogeneous LF/CRLF must additionally pass an exact model round-trip. This is an
experiment admission profile, not a new web storage normalization policy. I3 may
adopt only the independently reviewed qualified profile and loading mechanism.

### F5 source-bound creation acknowledgement (2026-10-07)

A complete implicit-default creation reply supplies the created identity, version,
and only its own associated language to a newer body-only draft. An absent language
remains unknown; rekeying consumes the fresh-default proof. An explicit mode edit
made during the request keeps its own language and stronger capability protection.
Ordinary and queued mutation cleanup require exact source, identity and version
acknowledgement even when language is unknown. This uses the existing source
acknowledgement boundary, with no exemption for arbitrary existing null-mode notes.
