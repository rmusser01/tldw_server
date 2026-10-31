# ADR-031: Canonical synchronized Notes capability domains

**Status:** Accepted
**Date:** 2026-08-08
**Backfilled from:** `tldw_chatbook/backlog/decisions/046-synchronized-database-notes-parity.md`
**Decision owner:** TASK-13002 requester and implementation review
**Related task:** TASK-13002
**Related spec:** `tldw_chatbook/Docs/superpowers/specs/2026-08-08-notes-server-parity-design.md`

## Decision

Treat Sync v2 as the ownership boundary for every mutable personal Notes capability, beginning with one canonical `notes.note` version-1 contract.

A `notes.note` upsert carries `title`, `content`, nullable `conversation_id`, and nullable `message_id` under `server_trusted_v1`. The server validates the field types and documented title/content limits without trimming, escaping, truncating, or otherwise rewriting accepted Markdown. Tombstones retain the existing `tombstone` operation.

Restore is not a new wire operation. It is an `upsert` with `routing_metadata.restore_intent` set to `true`, a complete base tuple that exactly references the current tombstone head, and the full canonical note payload. An ordinary upsert against a tombstone, a restore without the exact current base, or a restore against an active object is a whole-object conflict.

Client-origin pushes and normal server REST mutations pass through the same domain adapter, append-only envelope store, materializer, and object-state update. Server-origin metadata identifies the owner outside the canonical note payload. Keyword mutations remain blocked while Sync is active until their independent synchronized domains are available; a partial note payload never implies ownership of keyword state.

Later Notes capabilities use independent versioned domains for independently mutable resources. Derived indexes such as FTS and parsed wikilinks are rebuilt from canonical state rather than synchronized as competing authorities.

## Context

The server already projected `notes.note` envelopes into ChaChaNotes and captured normal REST create, update, and delete mutations. However, production composition used an accept-anything adapter, the projection accepted a legacy `body` alias and rewrote titles with whitespace stripping, and active-Sync REST restore was rejected. That combination could accept malformed payloads, produce different canonical envelopes by origin, and prevent a legitimate current-head restore while still needing to reject stale resurrection.

Chatbook parity also includes keywords, folders, manual links, attachments, tasks, activity, moodboards, and Studio documents. Folding those independently mutable resources into the base note object would create avoidable whole-note conflicts and silent partial synchronization.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Add a `restore` wire operation | It expands the protocol when existing upsert plus explicit intent and base lineage expresses the state transition safely. |
| Permit any upsert against a tombstone | A delayed ordinary edit could silently resurrect a note after a newer delete. |
| Keep REST restore outside Sync | The database projection and envelope log would diverge, so another device could not reproduce server state. |
| Keep using the generic static adapter | It cannot enforce domain payload shape, content limits, or restore intent before persistence. |
| Put keywords, folders, links, and other resources inside `notes.note` | Independently mutable state would conflict as one large object and partial clients could overwrite capabilities they do not own. |
| Normalize or sanitize Markdown during sync | Canonical bytes would differ across devices and derived wikilinks/backlinks could change meaning. Rendering is the correct escaping boundary. |

## Consequences

All accepted core note mutations have one reproducible envelope and object-state path, and a valid restore can be replayed idempotently. Clients must send complete current base metadata for updates, tombstones, and restores. Existing producers using the legacy `body` alias must migrate to `content`; invalid or oversized payloads are rejected before append.

The synchronized Notes programme must add separately owned resource domains before claiming full capability parity. Server REST endpoints must keep rejecting keyword mutations under active Sync until those domains land.

## Follow-up

- TASK-13003 through TASK-13007 add the remaining server Notes domains.
- Chatbook TASK-3701 consumes this core contract and proves an exact two-client round trip.
