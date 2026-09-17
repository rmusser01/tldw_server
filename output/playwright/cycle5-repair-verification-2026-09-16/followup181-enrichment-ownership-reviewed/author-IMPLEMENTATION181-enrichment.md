# UAT181 / TASK13260.118 — enrichment callback ownership

## Result and scope

Frozen for independent review. Production changes only `core/Chat/conversation_enrichment.py`: one import and an independent `chacha_operation` inside each existing auto-tagging/clustering runner's try block. The original catch/logging behavior also catches cleanup failures. Direct functions, pytest-inline branch, thread construction, daemon scheduling, mutations, arguments, and results remain unchanged. No source change to ChaCha, the HTTP owner, generic query flags, WorkerSDK, provider configuration, or other workers.

Production SHA256: `c41d787260468f006bfcb144aa2980c37426920d15b8ada10299c37abf527bc7`.

New permanent test SHA256: `b1ffed90c09bee35da57a1ed62e96578189157dfce396f615bac173444a9a65e`.

`owned-manifest.json`, `owned.patch`, `review-snapshot/`, and `ast-scope-check.json` freeze the review unit. The AST check unwraps exactly the two new contexts/removes their import and verifies complete equality to the old module. That baseline SHA21189cc2… exactly matches the parent's 3272-file native source receipt for API12336.

## Causal evidence

The retained native transactions PID90388/90756 began at 04:35:18.091773/.000310 UTC and remained idle in transaction at the later metadata capture. Both prepared statements exactly match `count_messages_for_conversation`: a messages/conversations COUNT for a conversation, excluding deleted rows. Sanitized receipt `native-sanitized-count-transactions.json` includes times, PIDs, statement template, and source hashes, without credentials or user contents.

`post_message_to_conversation` schedules auto-tagging after every persisted message. Native scheduling starts a daemon thread outside the HTTP ContextVar lifetime. Auto-tagging reads the conversation, calls `count_messages_since(None)` → the exact COUNT, and exits if fewer than three new messages. No old cleanup surrounded that thread. Two completed native message creations are consistent with the two retained transactions. A later native owner-switch/bootstrap failure was reported by the parent; this author did not run a replacement/restart test.

The official-PG private probe runs actual scheduler/thread/auto-tag/read functions and observes actual pool get/return. Current callback: **one expected failure** with INTRANS checkout after thread join. Private independent-owner control: **one pass**, zero skips, 3.34s. Both return `insufficient_new_messages`. Receipts are captured before fixture pool teardown. This path does not call the failing UAT199 update SELECT or the UAT200 Persona query, preserving their separate attribution.

Permanent pre-edit RED: **11 PostgreSQL failures, 11 SQLite controls passed, zero skipped, 24.99s**. Failure assertions concern retained callback loans or extra loans beyond an unrelated caller's exact lease. Empty clustering independently leaves INTRANS; successful tagging and spawned clustering leave two IDLE loans; a real failed SELECT leaves INERROR. Thus the sibling clustering scope has separate causal proof. Original RED logs and source snapshots remain retained.

## Final verification

```sh
source .venv/bin/activate &&
TLDW_UAT_EVIDENCE_LABEL=uat181-enrichment-frozen-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs \
  tldw_Server_API/tests/DB_Management/test_conversation_enrichment_operation_lifecycle.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_chat_conversation_enrichment.py \
  tldw_Server_API/tests/Notes/test_notes_organization_sync_surfaces.py::test_auto_tag_preflights_sync_before_conversation_update \
  tldw_Server_API/tests/Notes/test_notes_organization_sync_surfaces.py::test_auto_tag_background_worker_uses_active_sync_authority \
  -q --tb=short
```

**32 passed, zero skipped, 5 warnings, 31.33s**, exit0. This is 25 new lifecycle controls and 7 existing enrichment/organization controls. The mandatory PostgreSQL runner uses official disposable fixtures; no native database, provider, browser, runtime, or configuration was touched.

The new tests cover real threads, missing/threshold exits, empty clustering, successful tagging plus child clustering and committed results, repeated cached callbacks, real SQL error cleanup, an in-flight callback finishing after the HTTP owner closes, unrelated legacy/operation caller transaction commit/rollback, successful callback discarding an unfinished write, inline caller rollback, SQLite semantics, and at-most-once checkout returns. Real SQL and actual pool calls remain active; only scheduling branch selection, callback observations, deterministic query gate, and explicit fault injection are controlled.

Root review identified a test-observation race: publishing a pool return before removing the old raw-connection ledger entry could erase a newly borrowed generation. The fixture now retires the exact observed lease before invoking the real return, records failed returns separately, and records success afterward. It never locks across pool checkout. Production was unchanged by this correction. The prior 32-pass run remains history; the final 32-pass run above uses the corrected frozen ledger.

Fresh Ruff: zero baseline/current diagnostics. Fresh production Bandit: zero findings/errors. Test Bandit: zero findings/errors with only test-assertion B101 excluded. Scoped whitespace check passes. Tests formatted before the final run. Static JSON receipts are in this packet.

## Limits and handoff

This is a bounded non-HTTP callback adoption. It does not certify all application background work or arbitrary raw-driver escapes. These scheduler functions offer no cancellation API: a started daemon thread continues under its independent owner; the test proves it can outlive the HTTP scope without early return and releases on completion, not that threads can be forcibly cancelled. No implicit success commit was introduced.

Native acceptance remains parent-owned after independent review/integration/restart. Do not mark all UAT181 accepted from these fixture results, erase the earlier native failures, or claim the unrelated UAT199/200 defects are repaired by callback ownership.
