# ADR-050: Owner-Fenced Native Chat Fork Storage

**Status:** Accepted
**Date:** 2026-09-25
**Backfilled from:** `Docs/Design/2026-09-16-chatbook-chat-parity-review-closure.md`
**Decision owner:** Requester-approved Chatbook parity design, TASK-13261
**Related task:** TASK-13261.11
**Related spec/plan:** `Docs/Design/2026-09-16-chatbook-chat-parity-review-closure.md`, `IMPLEMENTATION_PLAN_chatbook_h2_qodo_review_2026_09_23.md`

## Decision

Store each native chat fork as an owner-bound operation with an immutable receipt independent of its child chat, and fence fork admission when the destination workspace is closing or deleted.

## Context

A client can lose the response after a child commits. Repeating a copy without a durable operation identity could create a second child; storing the receipt only on the child could erase the evidence when that child is deleted. A workspace deletion racing fork preparation could otherwise admit a new child after deletion has enumerated existing chats. Fork projection must also retain only authorized, immutable source context for the selected history.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Re-run a fork request after an uncertain response | It can produce duplicate children and cannot distinguish a committed result from a precommit failure. |
| Keep the operation receipt only on the child row | Deleting the child loses the tombstone needed to prevent recreation by replay. |
| Let workspace deletion rely on its initial chat enumeration | Concurrent preparation could publish a child after that enumeration. |
| Copy source rows or asset paths directly into the child | It can transfer stale context or source-owned authority across the owner boundary. |

## Consequences

- A fork key resolves to the same operation outcome, including a burned receipt after child deletion; replay cannot create another child for that key.
- Operation, candidate, claim, reference, and quota-intent records are scoped to their direct owner, with forced row-level security on PostgreSQL.
- Workspace deletion closes admission before enumerating chats and checks for protected residual children before final deletion. Failed cascades remain closed until a safe retry.
- The current H2 increment defines projection and durable storage groundwork. Physical asset verification, publication, reclamation, and the public native-fork flow are follow-on work; an operation receipt alone does not claim those are complete.
- ADR-049 continues to govern the selected history and source-owner boundary used to prepare a fork.

## Follow-up

- Finish the staged physical asset lifecycle and public fork/recovery protocol before exposing native forking to clients.
