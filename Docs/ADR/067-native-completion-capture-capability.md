# ADR-067: Owner-scoped native completion capture capability

**Status:** Proposed
**Date:** 2026-10-10
**Backfilled from:** not backfilled
**Decision owner:** Original chat remediation session; D7 accepted 2026-10-04
**Related task:** TASK13574, parent TASK13573
**Related spec/plan:** Docs/Design/2026-10-04-d7-conversation-model.md; Docs/Design/IMPLEMENTATION_PLAN_chat_native_server_settlement_13573.md

## Decision

Discover native selected-completion persistence ownership through an explicit opt-in on the existing authenticated, scoped H1 capture, retaining the original completion admission checks.

## Context

Ordinary native Full and Compact currently append both messages from the browser. The server already settles selected completions and partials, but refuses enrolled global Sync owners. A generic save_to_db or OpenAPI feature cannot distinguish those owners. Existing H1 capture and snapshot schemas are strict, so an unsolicited response field would break older clients. ADR049 continues to govern history selection; ADR050 governs native fork storage and does not grant completion authority.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Switch every native completion using global save support | Bypasses the distinction between enrolled Sync and ordinary ownership. |
| Reuse native_fork_context or add an unsolicited snapshot field | Misstates copier eligibility and breaks existing strict consumers. |
| Probe with a completion and retry browser persistence after refusal | Dispatch can have an uncertain outcome and fallback can duplicate input or reply. |
| Add a separate capability endpoint or durable registry | The authenticated H1 capture already binds account, CID, scope and selected context. |

## Consequences

The query include_native_completion=true requests native_completion_v1 on a captured send response. Its strict policy selected_history_v1 describes existing selected server persistence and original admission/final receipts; supported describes persistence ownership eligibility, not provider availability or bypass of other validation. The proof's storage_context_digest must match that exact capture. The server emits no field for unrequested, fork or unsuccessful captures, preserving their prior JSON. A missing older-server capability means the established client-managed path remains selected.

The actual completion still checks scope, owner, Sync enrollment, selected context and primary-operation requirements before admitting input. Enrollment can change after a capture, so this read never grants future mutation authority. A refused or uncertain dispatched completion cannot fall back to browser appends. Capability state stays on the original captured object; no global cache or persisted capability is added. This prerequisite changes no dispatch or settlement behavior.

## Follow-up

TASK13573 consumes the original capability and primary operation in the shared normal pipeline, handles error-before-partial receipt ordering, and prevents recovery from guessing missing saves. Sync settlement, detached runs and receipt lookup remain distinct protocol work.
