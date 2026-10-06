# ADR-065: Independent Notes Knowledge provenance

**Status:** Accepted
**Date:** 2026-10-06
**Backfilled from:** not backfilled
**Decision owner:** Requester approval in the Knowledge follow-up session
**Related task:** TASK-13514
**Related spec/plan:** Docs/Design/2026-10-06-knowledge-followup-source-context.md; Docs/superpowers/plans/IMPLEMENTATION_PLAN_knowledge_provenance_20261006.md

## Decision

Preserve Knowledge source history in an owner-scoped, independently versioned `notes.provenance` v1 Sync capability, separate from editable `notes.note` Markdown.

## Context

Portable content markers retain source history only while an editor preserves them. An old client or rich-text edit can remove the marker without intentionally deleting the evidence. Core Notes v1 deliberately admits only lossless title, content and chat backlinks. Studio sidecars describe Studio documents and cannot stand in for Knowledge provenance.

The requester approved the independent sidecar proposal on 2026-10-06. ADR031 continues to govern Notes capabilities and ADR034 continues to govern durable mutation batches. Atomicity refers to complete canonical acceptance plus transactional note/provenance product projection; Sync and Notes remain separate databases with resumable projection after a crash, not a new distributed transaction protocol.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Add provenance to core note v1 | Changes its strict wire contract and couples source-history conflicts to ordinary Markdown edits. |
| Reuse Studio or graph links | Gives evidence the wrong identity and lifecycle. |
| Store only Markdown markers or an unsynchronized local column | Editors can remove markers and other clients cannot reconstruct independent history. |

## Consequences

One sidecar uses its parent note UUID and authenticated owner. Strict bounded payloads match the portable Knowledge marker. Omitted provenance preserves it; explicit replacement requires its own exact version. Note/provenance saves use existing durable Sync groups, with adjacent product projections committed together. Parent deletion tombstones both accepted heads; ordinary restore does not automatically restore evidence. Explicit evidence restore requires its retained tombstone head and an active, explicitly restored parent.

New clients prefer validated canonical provenance, surface marker disagreement and keep portable markers for old servers and exports. A retained source reference is historical evidence, never a permission grant or an assertion that a source remains live. Server materialization follows the existing server-trusted encryption policy and fails closed for unsupported private materialization. Enrollment and replay must retain independent heads, including tombstones. No live web capture/refresh policy is introduced by this decision.

## Follow-up

Implement and verify TASK-13514 against the approved design, including old/new clients, independently stale writes, owner boundaries, SQLite/PostgreSQL migrations, durable retries, deletion/restore, replay and portable export.
