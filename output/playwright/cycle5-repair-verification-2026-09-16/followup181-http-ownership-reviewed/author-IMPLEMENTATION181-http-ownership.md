# UAT181 / TASK13260.118 — HTTP and maintenance checkout ownership

## Result and scope

The four-file implementation is frozen for independent review. It gives each complete HTTP operation its own lazy PostgreSQL checkout state and returns only that operation’s checkouts. Health checks, initialization/publication, and the default-character executor have independent maintenance scopes. SQLite and legacy callers outside an explicit operation retain their existing connection ownership.

`owned-implementation-manifest.json`, `owned-implementation.patch`, and `review-snapshot/` contain the exact seven-file release (four production, three tests). The original dependency-wired RED remains untouched in `red-source/`, `red-test.py`, and the original `owned-manifest.json`; the implementation uses a separate manifest name deliberately.

Final frozen combined verification passes **164 tests,0 skips,117.46s** across12 files (`uat181-http-frozen-combined-green`). The independent reviewer has passed all4 targeted review counterexamples; its full77-case run is in progress. Every release hash still matches its snapshot. No native acceptance, worker adoption, global lifecycle guarantee, restart or commit is claimed.

## Production behavior

| File | Change |
| --- | --- |
| `chacha/operation_scope.py` | Explicit operation owner, per-DB captured connection/backend/depth, active command/transaction uses, external borrowing, full pure-ASGI HTTP middleware. |
| `ChaChaNotes_DB.py` | Resolve PostgreSQL state through the active owner; retain legacy thread-local fallback. Guard whole DB commands, direct connection/cursor methods, and explicit transaction lifetime. No SQL changes. |
| `API_Deps/ChaCha_Notes_DB_Deps.py` | Independent scopes for health, actual initialization, and default-character executor work. Existing async accessor/cache/admission signatures remain. |
| `main.py` | Register the owner middleware once on the common normal/minimal path. |

The owner closes immediately to new work. A command or explicit transaction already in progress can finish under its captured state. Its checkout stays out of the pool until its last use ends, including backend exceptions. Completion never commits an undecided write merely because HTTP succeeded. Existing explicit transaction exit still decides commit/rollback.

Task and thread identity distinguish an already-entered use from unrelated inherited work. A detached task that merely inherited context cannot gain its parent’s active token after closure. It must enter an independent operation. Joined work uses the same request state. Independent transactions that need separate decisions must use independent scopes.

Direct wrappers retain their original state identity. A stale connection cannot execute, commit, roll back, or create a fresh cursor under no owner or another owner, even when the pool physically reuses the same raw connection. Materialized cursor result rows remain ordinary detached data.

External connections require explicit `ExternalConnection` bindings. Their outer transaction is borrowed, never adopted from thread-local state or inferred from INTRANS. Cleanup never returns or settles the borrowed raw connection. A borrowed connection closed by its caller fails clearly instead of being replaced. An incomplete binding fails before allocation.

Two different locks have distinct jobs: allocation serialization may wait for the pool; short state bookkeeping tracks active uses/retirement. The finalizer does not acquire the allocation lock or wait for a worker. Closed-owner lookups release the owner lock before checking state use, avoiding inverse owner/state lock ordering.

## Causal proof retained

| Evidence | Outcome and meaning |
| --- | --- |
| `RED181-http-lifetime.md`, original `red-source/` | Required PG:9 expected failures/14 controls/0 skips. Real due→Buddy→Notes HTTP dependency chain leaves source-review/Buddy relations locked and blocks a real second constructor while original connections remain open. |
| `uat181-http-second-green.redacted.log` | Original23 controls pass after request/maintenance ownership; explicit/raw/nested/caller decisions and live replacement preserved. |
| `RED181-inflight-worker.md`, `inflight-red-source/` | Actual paused backend execution: both response completion and cancellation returned the checkout before query completion,2 failures. This required active-use retirement rather than a closed-context check alone. |
| `uat181-http-borrowed-closed-red.redacted.log` | Closed borrowed checkout was wrongly replaceable; permanent actual HTTP control now rejects it without return/replacement. |
| `transaction-entry-retained-red.log` |1 failure/1 control: a retained transaction object could keep a use active after wrapper construction failed. Complete entry cleanup now releases it explicitly, without relying on garbage collection. The earlier unretained fixture passed due to prompt garbage collection and is retained as a weaker probe. |
| `incomplete-binding-red.log` |2 failures before rejecting missing connection/backend bindings. |
| Reviewer `escaped-cursor-red.log`; author `uat181-http-retained-cursor-red.redacted.log` |2 failures each: stale connection could create/relabel a cursor. Existing wrapper-use guard now also covers cursor creation. Permanent PG control deliberately reacquires the original physical checkout from the real pool under a later owner. |
| Reviewer `lock-order-red.log` |1 failure using actual owner/state/acquisition methods, controlled locks and no database I/O: inverse locks deadlock. All threads joined via bounded timeout instrumentation. |
| Reviewer `checkout-wait-red.log` |1 failure using actual async owner→get_connection→open connection, with only pool I/O gated: finalization waited for acquisition. Separate allocation lock fixes this. |
| `reviewer-concurrency-green.log` |Both permanent lock-order and blocked-acquisition controls pass after the final synchronization correction. |

Reviewer REDs are in `.tmp/uat181-http-independent-20260917/`. All official-PG redacted logs are in `.tmp/fresh-uat-recovery-20260916/`. Original RED fixtures and receipts are not overwritten by final implementation snapshots.

## Permanent coverage

- Real sync due/Buddy and async Notes/keywords/collections handlers, real dependency/cache/runtime/health/default path, empty/populated schedules and actual replacement constructor.
- Request completion, response error, cancellation, joined child/sync handoff, detached closed-owner rejection and explicit child owner.
- Implicit write, raw BEGIN, ChaCha explicit/nested transaction and backend explicit transaction decisions; no HTTP-success auto-commit.
- Independent simultaneous requests and two account-specific cached DB handles. User identity/rate-limit fixtures are isolated test inputs; this does not claim live JWT or shared-default-character tenant acceptance.
- Explicit external borrowing and preserved legacy INERROR/INTRANS; closed/missing borrowed connections.
- Full streamed body and background task completion/error.
- In-flight DB, direct connection and direct cursor execution, backend failure, completion/cancellation; in-flight explicit transaction commit/rollback/error.
- Retained wrapper and new-cursor rejection across owner changes, physical pool reuse, repeated finalization, primary-error precedence and last-use cleanup failure.
- Unrelated inherited task/thread cannot borrow an active parent’s token after owner close; allocation wait and lock-order controls.
- Real initializer context/default executor; actual normal and minimal main module registration without running server startup.
- SQLite explicit transaction control, preserving the repository’s autocommit behavior outside an explicit BEGIN.

## Adjacent health contract correction

The existing `test_postgres_failed_health_probe_reports_unhealthy_then_cleans_transaction` expected health to adopt and roll back a failed legacy connection. That conflicts with the approved independent maintenance contract. The old source is preserved under `implementation-baseline/`; the original adjacent run retains1 failure/90 passes.

The renamed control now proves a healthy independent probe leaves the caller INERROR until the caller explicitly rolls back. A new control injects a real division-by-zero query at the probe’s own backend SELECT seam, observes its exact raw checkout become IDLE before pool shutdown, checks False, then proves a normal retry. It does not mock the health result or fabricate an application response. The other existing health controls and assertions remain; one existing parameter signature was formatter-normalized.

The adjacent suite with this approved correction passes92 tests/0 skips in17.53s on the preceding source, before the final two synchronization guards. The final164-case combined run includes the same adjacent tests on frozen source and passes.

## Verification commands

Run from repository root; use the approved local-network escalation for the official fixture runner. It provisions isolated test databases on the owned cluster. No AuthNZ `test_db_pool`, manual provisioning, live database mutation, provider calls or live services are involved.

Focused independent77 cases:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat181-http-independent-final node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_chacha_postgres_http_operation_lifecycle.py tldw_Server_API/tests/DB_Management/test_chacha_operation_scope.py tldw_Server_API/tests/Chat/test_chacha_notes_db_deps_postgres_health.py -q --tb=short
```

Frozen combined suite uses those three files plus:

- DB_Management/test_chacha_postgres_transactions.py
- DB_Management/test_chacha_connection_typing.py
- DB_Management/test_content_backend_cache.py
- DB_Management/unit/test_postgres_transaction_manager.py
- DB_Management/unit/test_postgres_pool_fallback.py
- Chat/test_chacha_notes_db_deps_sqlite_policy.py
- Chat/test_chacha_db_deps_error_mapping.py
- Chat/test_chacha_runtime_contract.py
- API_Deps/test_chacha_notes_db_deps_error_mapping.py

Exact argv/receipts are retained under runner label `uat181-http-frozen-combined-green`. Earlier expanded64 and focused73/76 GREEN receipts precede the final review corrections and are not substituted for final verification.

Static validation: all seven files compile. Ruff logical-path baseline9/current9 findings,0 added or removed; new module/tests have no findings. Production Bandit0 findings/0 parsing errors; test scan also has0 findings/0 parsing errors, excluding only normal pytest assertion B101. The initial private-path baseline count10 was an artifact of losing the repository’s exact-filename lint override; replay with --stdin-filename actual logical paths correctly gives9/9. Four small new/test files pass Ruff formatting; legacy large production files were not globally reformatted. Added patch lines have no trailing whitespace.

## Attribution and limits

The implementation baseline already contains root’s separate UAT192 two named-column fixes. Baseline ChaCha SHA is `5a7a43c34aaa15d0e328c145f96f5919b0c6c14b6a4147da389a410488d2ea3e`. The owned implementation diff is against that baseline, so192 is preserved and excluded from181 attribution. All1,936 SQL-looking AST string constants remain exactly equal; no SQL, migration, RLS, global read defaults, generic backend semantics or account policy change is included.

Legacy calls outside a managed operation still use legacy thread-local ownership. StudyPack and other non-HTTP adoption remains a separate stage; the existing accessor audit is in `nonhttp-accessor-audit.json`. WebSocket lifetime is not implicitly adopted. Raw driver access deliberately extracted through private `_connection`, `_get_thread_connection`, or driver attribute forwarding remains caller responsibility and is outside the safe wrapper-use guarantee. Direct backend operations that borrow their own connection retain the backend’s own transaction handling.

A caller that closes a DB checkout explicitly during an operation ends that captured state; stale wrappers cannot resurrect it. New independent work can establish its own operation. HTTP cleanup is not a global rollback or pool sweep. No caller-owned connection or current migration DDL is skipped to make replacement pass.

No production, task, tracker, config, browser, model or runtime changes beyond the approved four source files were made. Root owns final integration, native acceptance and the non-HTTP follow-on decision.
