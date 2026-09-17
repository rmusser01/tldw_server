# UAT181 / TASK13260.118 — conversation enrichment callback ownership

## Stage 1: causal diagnosis

Status: complete. Actual tagging and clustering callbacks both reproduced retained PostgreSQL checkouts.

The native retained metadata has two idle PostgreSQL transactions whose prepared SQL matches `MessageStore.count_messages_for_conversation`. The actual character message writer schedules auto-tagging after each persisted message. Its native scheduler creates a daemon `threading.Thread`; the HTTP ContextVar owner is not adopted by this thread. Auto-tagging reads the conversation, calls `count_messages_since` (falling back to the exact native COUNT), and exits below three messages. No cleanup exists around that callback.

Private official-PG probe: one expected failure for the actual native callback and one passing independent-owner control; zero skips, 3.34 seconds. Both return `insufficient_new_messages`. The unowned callback leaves its actual checkout INTRANS after the real thread joins, before pool teardown; the private owner returns it. This is separate from UAT199's update projection and UAT200's Persona SQL error. The probe does not infer a tested native replacement from retained locks.

## Stage 2: bounded implementation

Status: complete. One import and two independent operation scopes were added after causal RED.

Own only `core/Chat/conversation_enrichment.py` and new `tests/DB_Management/test_conversation_enrichment_operation_lifecycle.py`. Put `chacha_operation(independent=True)` around the entire existing auto-tagging callback body/error handler. Prove the sibling native clustering callback first, then apply the same boundary there. Keep direct and pytest-inline functions unchanged. Preserve thread construction/daemon behavior, callbacks, mutations, exception handling, and caller transaction decisions. The operation returns only its captured PostgreSQL checkout; it never auto-commits a successful callback or adopts a caller's legacy state.

Permanent controls use actual threads and actual DB functions/pool methods with observation only: threshold/missing exits, successful tagging plus spawned clustering, empty clustering, repeated cached jobs, real database failure, callback in flight after the HTTP owner closes, unrelated caller transaction commit/rollback with legacy and explicit ownership, and SQLite controls. This scheduler has no cancellation API; an already-running thread finishes under its independent owner. Do not claim cancel/join semantics not present in production.

## Stage 3: verification and handoff

Status: author verification complete; frozen independent review and parent native acceptance pending.

Required PostgreSQL fixtures with zero skips, focused plus existing enrichment/organization controls, scoped Ruff baseline/current and Bandit, frozen source/test hashes, and independent review. Parent controls runtime restart/native acceptance. No shared DB method, generic read flag, transaction plumbing, WorkerSDK, provider, or broad non-HTTP adoption changes.
