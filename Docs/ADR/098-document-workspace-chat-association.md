# ADR-098: Server-owned document chat associations

**Status:** Proposed
**Date:** 2026-11-23
**Backfilled from:** not backfilled
**Decision owner:** Project owner
**Related task:** [TASK-15586](../../backlog/tasks/task-15586%20-%20Sync-Document-Workspace-conversations-across-devices.md)
**Related spec:** [Cross-device conversations in Document Workspace](../superpowers/specs/2026-11-23-document-workspace-conversation-sync-design.md)

## Decision

Keep a single owner-bound document association with each linked native Chat record. Store the canonical Media UUID and bind the link to the original conversation record. Media authorizes document access. Chat authorizes conversation access and owns the transcript, continuation, and lifecycle. Opening or admitting a turn in a linked Document Workspace chat requires both authorities and a matching association at the server operation.

## Context

Document Workspace discovers conversation tabs through an IndexedDB index on the originating browser. A chat may already be saved on the server, but another browser lacks the document index. Same-server discovery needs a durable association that survives browser changes and respects both account ownership and access to the document. [ADR-049](049-chat-history-selection-ownership.md) governs native history selection.

## Alternatives considered

| Option | Trade-off |
| --- | --- |
| Store the index with Media and reading progress | Document lookup is convenient. Chat creation, trash, and deletion would have to coordinate with a second store for the association. |
| Reuse `source` and `external_ref` on the conversation | Those fields support editable metadata and import identity. A document link requires an independently enforced owner and lifecycle contract. |
| Add a Sync v2 domain for the link | Useful for clients with independent replicas. Same-server browsers can read the association directly from the server. |

## Consequences

- One document can have many chats. Each chat has at most one document link, tied to its own native record.
- Association writes are idempotent. Losing the link response or retrying after an import preserves the existing native chat identity. Discovery requires a saved message, a live chat with a verified link, and a live document accessible to the caller.
- Automatic reconciliation of a browser link requires a matching saved Media UUID and verified chat ownership. Numeric-only legacy links require the user to confirm the document. Saving a browser-only conversation requires **Save to server** for that chat and account support for native import. Drafts and tab selection stay in the browser.
- Removing document access blocks Document Workspace discovery and retrieval. Passages already included in saved transcripts follow ordinary Chat ownership and retention. Source links require fresh Media access.
