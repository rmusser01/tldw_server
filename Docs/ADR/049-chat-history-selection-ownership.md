# ADR-049: Owner-Bound Chat History Selection

**Status:** Accepted
**Date:** 2026-09-25
**Backfilled from:** `Docs/Design/2026-09-16-chatbook-chat-parity-review-closure.md`
**Decision owner:** Requester-approved Chatbook parity design, TASK-13261
**Related task:** TASK-13261.10
**Related spec/plan:** `Docs/Design/2026-09-16-chatbook-chat-parity-review-closure.md`, `IMPLEMENTATION_PLAN_chatbook_h1_qodo_review_2026_09_23.md`

## Decision

Bind a mounted chat view to an owner-validated history selection; derive continuation from that selection, and give a local copy its own conversation identity and assets before it can mutate history.

## Context

The WebUI and full-page extension previously could display an earlier turn or variant while treating the displayed rows as authoritative request history. A copied chat could retain source identifiers or asset references, allowing later edits or deletion to affect its source. The Chatbook parity design requires the selected ancestry and its owner to determine the next request and mutation authority. Legacy histories without provable ancestry remain reviewable instead of silently inferring a path from timestamps.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Send currently displayed rows as the next request's history | Presentation order and a selected variant do not prove the conversation's retained ancestry. |
| Let a local copy reuse the source conversation or asset identifiers | The child could mutate, delete, or depend on source-owned records. |
| Infer all legacy branches from timestamps | Timestamps cannot establish a unique ancestry when rows are missing, mixed, or contradictory. |

## Consequences

- History-dependent actions use the validated selected path and its current owner lease; an unavailable or changed owner cannot silently acquire mutation authority.
- Local copies receive independent conversation and asset identity. Source mutation authority is not inherited.
- Ambiguous legacy history requires explicit review before history-dependent mutation while remaining readable and exportable.
- WebUI and full-page extension share this history behavior. The compact sidepanel retains a bounded expansion flow.
- This decision does not claim the public durable native-fork protocol or cross-client synchronization is complete; those have separate delivery stages.

## Follow-up

- H2 native-fork storage and later public execution must preserve this owner and selected-history boundary.
