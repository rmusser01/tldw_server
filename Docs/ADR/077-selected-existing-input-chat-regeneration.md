# ADR-077: Settle a selected existing chat question

**Status:** Proposed
**Date:** 2026-10-08
**Backfilled from:** not backfilled
**Decision owner:** Requester-approved Regenerate repair, TASK-13260.278.18.83.45
**Related task:** TASK-13260.278.18.83.45
**Related spec/plan:** `Docs/Design/2026-10-08-selected-existing-input-chat-regeneration.md`, `Docs/Plans/IMPLEMENTATION_PLAN_uat_native_regenerate_20261008.md`

## Decision

Allow a versioned assistant message on the existing `/api/v1/chats/{chat_id}/messages` endpoint to settle against a selected, existing terminal user question. The owner validates the digest-bound selection under the message transaction, then adds only a new assistant child. The local owner follows the same selected question rule.

## Context

Regenerate currently stops when a chat has selected history. An admitted question can be retried using its admission receipt, but a loaded question can predate that protocol and has no receipt. Re-admitting or rewriting it changes its revision and ancestry and may change a legacy projection. Regenerate needs another answer to the same question, even when the chat already has answers and later turns.

## Contract

- Regenerate opts into the existing strict history capture with `required_capability: "tldw_history_existing_input_v1"`. Only that opt-in receives `acknowledged_capability` with the same value; ordinary serialized captures omit the key entirely, without omitting required nullable history fields. A generic history capture does not authorize this operation. Missing, wrong, or unbound acknowledgement refuses provider dispatch and message writes. A reopened owner probes again. A bound non-ready capture may acknowledge support without claiming a captured path; a first settlement still requires fresh owner validation. An unbound stale-projection HTTP error establishes nothing; recovery may use a separate owner-bound capability-only read that does not replace its saved selection. This is capability negotiation with the current owner, not a cross-request atomic version guarantee.
- The assistant request carries `tldw_history_existing_input_v1`: the existing `HistorySelectionV1` with purpose `send`, a terminal user node, and its exact question ID and revision in `messages`. The selected path ends **after the question**. The assistant ID and payload are stable across retries; its explicit parent, when present, equals the question ID.
- The client captures the owner-selected path, prepares the provider request from that path, and finalizes the selection using the prepared request-context digest before dispatch. It does not submit a new user message. A stale view or changed request lease refuses dispatch; a late committed answer may be followed only if the original view still owns that selection.
- First settlement checks owner, conversation, scope, interpretation, selected ancestry, question role/revision and payload within the existing owner transaction. It inserts only the assistant, tagged with the selected interpretation, and stores selection, request intent and result-state proof in the assistant's protected authority field. Existing user and answer rows, IDs, versions, authority, and suffix remain byte-for-byte unchanged.
- A request using an already occupied assistant ID validates the stored protected proof and exact payload/result state before full selection revalidation. Exact replay returns the same result after an unrelated history-fence change. A conflicting ID, changed payload/proof, or edited/deleted result fails. It never silently falls back to the unversioned append route.
- Generated text with an unknown or failed settlement remains recoverable under its original operation and assistant ID. Recovery retries settlement with the same selection and payload; it does not call the model again. Legacy histories still require explicit projection review, and the selected legacy projection stays attached to the new answer without changing the source question.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Reuse `HistoryAdmissionReferenceV1` for the old question | Old questions lack that owner-issued receipt; fabricating one would create false provenance. |
| Re-admit or edit the old question | That changes its revision and projection, duplicates a user turn, and can disturb later history. |
| Add a regeneration table or separate endpoint | The message transaction and protected authority field already provide the needed atomic write and replay proof. |

## Consequences

ADR-049 continues to govern owner-selected ancestry and explicit legacy review; ADR-050 continues to govern fork storage. This extension defines a public message request that clients must use for native Regenerate. Unsupported question content, context, or ownership produces an explicit refusal. The assistant result may be a sibling of an earlier answer, while subsequent turns remain attached to their original branch.

## Follow-up

Implement and verify the local/native paths, endpoint, owner transaction, recovery, and canonical wire contract in the related plan. Keep this record Proposed pending owner review.
