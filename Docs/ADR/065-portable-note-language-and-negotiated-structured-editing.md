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
