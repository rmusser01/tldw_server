# ADR-056: Permanent Owner-Bound Workspace Startup Receipts

**Status:** Proposed
**Date:** 2026-09-27
**Backfilled from:** not backfilled
**Decision owner:** Requester-approved Stage 2C design; implementation review pending
**Related task:** TASK-13245.8
**Related spec/plan:** `Docs/Design/2026-09-27-persona-workspace-strict-startup-refresh.md`; `Docs/superpowers/plans/2026-09-27-persona-workspace-strict-startup-implementation-plan.md`

## Decision

Strict Workspace chat startup uses an operation-specific permanent owner/key receipt containing only hashes, references and timestamps, atomically accepted with the conversation and irreversibly invalidated by actual binding changes.

## Context

Retrying after response loss must not create a second chat or choose today's changed Persona default. Legacy requests remain compatible; a dedicated closed protocol makes explicit None and versioned inheritance unambiguous. PostgreSQL shares tenant storage, while deleted chats and Workspaces must not erase retry identity or lifetime capacity.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Legacy create with ignored extra selector | Old servers can silently inherit instead of accepting explicit None. |
| Native fork receipts or generic receipt framework | Native leases, request snapshots and projection lifecycle do not match this privacy-bounded operation. |
| Expiring receipts or cascade deletion | A delayed retry could silently create/rebind a chat and recycle lifetime capacity. |
| Endpoint-only identity comparison | Change-away-and-back and concurrent Sync writes require transaction-local invalidation. |

## Consequences

- A dedicated route owns an idle outermost transaction; response projection occurs only after commit. No downgrade to legacy creation.
- Owner/key uniqueness spans Workspaces. PostgreSQL forced owner-only RLS retains orphan visibility without live-parent predicates.
- Owner serialization bounds lifetime receipt count (default 10000). All tombstones count; no automatic eviction or key recycling.
- Conversation hard deletion nulls its receipt reference permanently; Workspace deletion preserves receipts. Actual identity/scope changes invalidate in the writer transaction.
- Current access and Persona admission remain required on replay and generation. Receipts confer no access or memory authority.
- Offline writer drain, migration and compatible-binary restart are required. Backups retain private hashed tombstones; old binaries cannot safely coexist with receipt-aware writers.
- ADR050 continues governing native forks; this decision does not complete Stage 2D or client parity.

## Follow-up

Keep the strict route inactive until lifecycle, capacity, transaction and current-admission gates pass. Promote this record only with reviewed delivery evidence.
