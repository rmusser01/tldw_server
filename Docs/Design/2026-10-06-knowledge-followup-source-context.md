# Knowledge follow-up source context — TASK-13514

## Approved intent and bounded implementation

The requester selected the remaining Knowledge follow-ups after PR3196 merged. Preserve original evidence and enable Research to use complete available non-media source content.

For canonical Notes, fetch the exact original note through the existing owner-scoped Notes GET using the captured request scope and cancellation signal. Validate identity, active state, content and revision before ingestion. Ingest one complete note snapshot for all retrieved chunks of that note using the existing media ingestion API. Retain the original retrieved excerpts separately from the full current document and label the saved source with its original note version. A read or import failure stays visible and retryable; do not substitute an older excerpt after a failed authorized read. Successful snapshots are checkpointed and reused on retries. Workspace switches or account invalidation retire reads before upload or attachment.

Media continues to attach through its canonical ID. External web snippets continue to be explicitly labeled retrieved-excerpt snapshots. Original Knowledge answer qualifications apply to the original retrieved excerpts; adding full source context does not recertify that answer.

## Backend provenance decision requiring coordinated design

The canonical Notes API retains content and backlinks, but has no generic note metadata field. ADR031 owns the synchronized core-note payload, and existing Studio provenance belongs to Studio documents. A dedicated Knowledge provenance capability must preserve deletion, optimistic versions, restoration, Sync replay and old clients. Overloading Studio or graph-link records would create misleading product identity.

Recommended approach: independently versioned canonical note-provenance records, with owner-scoped Notes create/read integration and the existing Sync envelope/materializer machinery. Keep source references and trust qualifications separate from editable Markdown. Content markers remain a backward-compatible fallback for existing servers and historical notes. This needs a specific Notes/Sync compatibility spec before implementation.

Alternative: extend the core-note payload with optional provenance under a new version and explicitly preserve it for old-client writes. This is fewer persistence concepts but increases whole-note conflict and rollout complexity. A local-only metadata column would fail the existing synchronized-capability rule and is not a complete solution.

## ADR check

Bounded full-note snapshots require no new ADR: ADR007 keeps ResearchWorkspace canonical and ADR031 keeps Notes reads authoritative. The proposed independent provenance capability requires a new durable decision governed by ADR031. The backend design must record its contract/version and migration before implementation.

## Verification

Mounted workflow checks cover complete content beyond retrieved chunks, one upload per original note, exact identity/revision, denied/deleted/invalid source reads, and workspace retirement before ingestion. Serialization checks retain the source revision across Quick Note save/reopen. Existing handoff retry, owner, selection and tombstone coverage remains applicable. Real browser checks use disposable canonical Notes and verify the saved snapshot through the actual API.

## Proposed canonical provenance contract for approval

Implement `notes.provenance` as an independent version-1 Sync capability governed by [ADR031](../ADR/031-notes-capability-sync-domains.md). Use one owner-scoped sidecar per canonical note. The note UUID identifies its parent; independent envelope/object-state versions govern provenance changes. Editable Markdown and the existing `notes.note` v1 contract retain their current ownership.

The bounded payload contains origin, question/thread reference, source scope, trust state/reason codes, evidence origin, and source references with original identity/type/version, original retrieved excerpts and optional Research snapshot media IDs. Reuse the current client provenance validator's limits and allowed values in a matching server schema. Credentials, provider secrets and arbitrary executable metadata are rejected. A request cannot set owner identity.

| Mutation | Required behavior |
| --- | --- |
| Save a new sourced note | Persist note and sidecar through their canonical adapters in one transaction. Return both acknowledged heads; a lost response reconciles by idempotent identity. |
| Revise Markdown with an old client | Update core note only. Absence of provenance is preservation, not deletion. |
| Revise provenance | Require its exact current base independently from Markdown. Reject stale replacement without changing either accepted head. |
| Delete note | Tombstone core note and sidecar atomically. A late source import or old sidecar update cannot revive the deleted parent. |
| Restore note | Require the current core tombstone base and explicit restore intent; restore provenance only from an explicitly retained sidecar head. |
| Read / reopen | Enforce owner, active parent and normal encryption policy. Return structured provenance when the capability is available. |
| Replay / bootstrap | Declare the domain in capability discovery and replay the sidecar through the existing materializer. Rebuild projections and indexes from canonical envelopes. |

Keep content markers readable during rollout. New clients prefer validated server provenance and retain the marker on servers without the capability. Backfill only a valid marker under the note's owner and exact current version; never infer trust from the editable prose. Conflicting marker/server values preserve canonical structured provenance and expose a reconciliation result rather than silently overwriting it. An export remains self-contained by serializing validated sidecar data into the existing portable marker.

Required proof before release: SQLite and PostgreSQL migration/replay; two-client old/new writes; independent optimistic conflict; atomic note/sidecar save/delete; tombstone restore fences; unauthorized parent/source references; encryption capability behavior; export/import; lost save acknowledgment; marker fallback on an old server. Source references are pointers with retained excerpts, not permission grants. Deleting or losing access to a source must not make a previously saved excerpt appear live or currently authorized.

This is a reviewable proposal, not an implemented capability. It requires a new ADR and approval before adding storage or Sync wire contracts. Full live external web sources also need a separate capture/refresh policy, permission handling and source-version contract; the current explicit excerpt snapshots remain honest until that design exists.


## Fresh server workspace persistence

The live browser exposed a handoff path that checked only legacy migration tombstones before saving canonical Notes. Fresh server workspaces could retain a local draft without the structured provenance marker. Wait for the existing server-workspace confirmation before starting an import when the client supports server reconciliation. Pass that identity into the importer and reuse canonical Notes persistence there. Completed local handoffs keep their existing retirement behavior; confirmation must not replay attachments or revive discarded drafts. Reuse the successful source snapshot IDs and the existing canonical save/readback acknowledgment. No Notes schema or Sync contract changes are required.

## Overlapping canonical source identities

The final browser check found that snapshot ingestion can return a media ID already present among retrieved results. Keep one workspace attachment, but build its evidence from every resolved payload source with that media ID. Retain both the original Notes identity/version and the retrieved media reference regardless of processing order. A source is a snapshot when any of its retained references came from snapshot ingestion. Reuse the existing evidence validator and canonical note acknowledgment; no new source schema or merge abstraction is needed.
