# Selected existing input Regenerate

**Task:** TASK-13260.278.18.83.45
**Decision:** [Proposed ADR-067](../ADR/067-selected-existing-input-chat-regeneration.md)
**Plan:** [Native Regenerate](../Plans/IMPLEMENTATION_PLAN_uat_native_regenerate_20261008.md)

## Behavior

Regenerate the selected saved answer by preparing the model request from the owner-captured path ending at its existing user question. Add one alternative assistant whose parent is that question. Keep the question, previous answers, later turns, IDs, versions, and protected provenance unchanged. The new answer inherits the selected parent-graph or reviewed-legacy interpretation and can be followed in the current view.

## Path

1. The mounted action identifies the selected answer and its exact parent question. Capture through the live history controller with a cursor **after** that question, while retaining the original view and lease for later follow. Opt into the existing strict capture using `required_capability: "tldw_history_existing_input_v1"` and require a matching `acknowledged_capability` on this live owner; generic H1 capture is insufficient. Reopened owners probe anew. Require a terminal user row and matching ID/revision. Decline missing/wrong/unbound acknowledgement, unsupported content, or changed owner/request settings before provider dispatch.
2. Prepare the existing provider request from the captured ancestry and original question. Finalize the selection with the prepared request-context digest. Use the current turn recovery record and one stable assistant ID; do not admit a user message or remove the original answer from storage.
3. Local settlement uses its owner transaction to validate selection, question revision and role, then writes one sibling assistant with replay proof. Native settlement sends the assistant to `/api/v1/chats/{cid}/messages` with `tldw_history_existing_input_v1` set to the complete `HistorySelectionV1` (purpose `send`, selected terminal question). The API validates and delegates under the current owner transaction; the protected proof lives on the assistant only. Sync publication, when active, contains only that assistant ID.
4. If the response is lost, retry the same assistant ID, exact payload and selection from recovery without generating again. A bound non-ready opt-in capture may prove support without inventing a captured path; an unbound HTTP error proves nothing. A stale or missing legacy projection can fail before a structured capture, so recovery may make a separate capability-only owner read with a parent-graph empty cursor while retaining its original saved selection and proof. That read cannot authorize a fresh settlement, choose a view, or make the old source valid. The owner checks committed proof before revalidating a now-stale full selection. A conflict, edited/deleted result, stale first settlement, or changed scope is a refusal. A late completion follows the new answer only when the original view still owns its selection; otherwise the saved answer remains available to follow on reload. Capability acknowledgement assumes a consistent backend for the current owner/lease; it is not an atomic version guarantee across requests.

## Qualification

Authored phase-one oracles: the mounted Playground test checks the exact provider user/assistant sequence, no duplicate user write, same-parent sibling, and submitted/stored assistant ID. Authenticated API tests check opt-in acknowledgement and ordinary wire omission, stable ID, unchanged protected question and suffix, first settlement, exact replay after retained-ancestor change, refusal of a new ID against that stale selection, conflicts, stale/foreign refusal, reviewed-legacy follow, and rollback. Service tests check old/missing/wrong acknowledgement, generic capture, owner reopening, scoped transport, and bound non-ready acknowledgement.

Planned qualification after implementation: local/native owner and recovery controls (including an unbound legacy projection error followed by a separate capability-only owner read), official SQLite/PostgreSQL fixtures, both configured clients, canonical wire types, and the live UAT workflows.
