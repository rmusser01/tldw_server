# UAT181 / TASK13260.118 — independent HTTP ownership review

## Verdict — clear on the frozen release

No unresolved source findings remain. All three review findings below were
corrected by the author with permanent controls and independently rechecked.
The final required-PG command passed **77 tests, 0 skips, 22 warnings in 113.77s**.
It includes real PostgreSQL HTTP/maintenance, owner unit, SQLite and main
registration controls. All four independent counterexample cases pass in 0.58s.

All seven source/test hashes match before and after verification and match the
retained `final-source/` snapshot. Author manifest SHA256:
`43ec41f1855ad239bc89f43443ce7858b1e03c856068967bdd03e29f7c07dfbd`.
Final ChaCha SHA256:
`8e853e529624be5d9f25e6a06dea80255f06c8068eada3def3bd1ecf04f851f8`;
owner SHA256:
`996814cc1aaa0e68b4c09e7639c68f111ce97b5c54c72d7a8f4eaf42524c096d`.

Scope: four product paths (operation_scope.py, ChaChaNotes_DB.py ownership
plumbing, ChaCha_Notes_DB_Deps.py maintenance scopes, main.py registration),
two new test files and one approved health-test contract change. No production,
runtime, browser, task, tracker or git changes were made by this reviewer.

## Causal findings — resolved in final source

Final corrections verified: (1) `cursor()` uses the existing wrapper guard;
(2) `state_for()` releases owner lock before checking state use;
(3) connection allocation uses a separate allocation lock, leaving ownership
metadata/finalization locks short. The last accepted active use still returns
the checkout exactly once. Permanent controls and the independent probes pass.
The original causal observations follow for review traceability.

1. **Escaped `connection.cursor()` loses ownership.** In the first provisional
   ChaCha source, lines598–605 create a new BackendCursorWrapper without guarding
   the connection wrapper. The new cursor captures the *current* operation,
   rather than first validating the original one. A retained connection can
   therefore execute after pool return outside any owner (state=None) or when a
   later owner has reused the same raw checkout. The existing test retained a
   pre-created cursor and checked direct execute; it missed late cursor creation.
   `test_escaped_cursor_probe.py` uses actual wrappers/owner and fake backend I/O:
   **2 expected failures**,0.46s. All four provisional production hashes still
   matched after that run. Minimal correction: existing wrapper guard on
   `cursor()` plus permanent late-cursor controls.
2. **Closed-owner lookup inverts locks.** `ChaChaOperation.state_for()` originally
   holds owner._lock while calling state.permits_current_use() (state.lock).
   Actual `_get_thread_connection()` holds state.lock while calling
   `_get_connection_for_state()` → `_get_pinned_backend()` → state_for()
   (owner._lock). A late inherited caller and an already-active acquisition can
   deadlock. `test_owner_lock_order_probe.py` retains those actual methods and
   substitutes only a normal reentrant owner lock with a bounded1s acquisition
   wait, avoiding an indefinitely hung test. **1 expected failure**,1.47s;
   active lookup hit the bounded lock timeout, and all threads joined. Minimal
   correction: snapshot state/closed under owner lock, then check state use after
   releasing owner lock. Source013 separately confirmed the lock order.
3. **Finalizer waits for pool acquisition.** The same state.lock covers
   `_open_new_connection()` and pool.get_connection(). A blocked worker checkout
   therefore makes synchronous owner.close() wait for that allocation on the
   ASGI event loop, despite the deferred-return contract. The query-inflight
   controls pause after allocation has released this lock and do not expose it.
   `test_checkout_wait_finalization_probe.py` uses the actual async owner and
   actual DB getter/open helper with a controlled fake pool: **1 expected
   failure**,0.80s. Finalization could not finish before the controller released
   the pending allocation; the checkout was ultimately returned exactly once.
   Minimal proposed correction: serialize allocation separately from short
   ownership metadata locks, preserving active-use deferral and no duplicate
   checkout installation. Source013 agrees; parent owns authorization.

Private probes are independent counterexamples, not substitutes for permanent
official-PG controls. They make no database/network/model calls. The second and
third probes ran while source013 was preparing the acknowledged local fixes;
the initial and post-finding snapshots retain those stages. They are not called
a single immutable full-suite run. The final suite and private GREEN replay both
used the exact final frozen release above.

## Remaining reviewed contracts / limits

- A pure ASGI wrapper spans downstream response iteration and background work;
  its registration is unconditional before the normal/minimal router branch.
  Request ID and drain wrappers remain outside it. HTTP-only behavior avoids
  imposing a whole-WebSocket or lifespan lease.
- Captured state does not adopt a legacy thread connection. Explicit external
  bindings retain caller ownership and start at a non-owning transaction depth.
  Closed/incomplete bindings are rejected, and operation cleanup never commits.
- Active-use identity combines OS thread and asyncio task, so a new inherited
  child cannot borrow another task's active token after closure. Joined nested
  work and an already-entered explicit transaction can finish; final pool return
  must wait for the last accepted use. New work after closure must be rejected.
- Maintenance owns separate scopes for creation/publication, health and default
  character executor work. The approved health-test change preserves an outer
  failed caller transaction rather than letting health roll it back; a separate
  failing-owned-probe control checks actual health cleanup.
- Unit registration tests import main with TEST_MODE enabled in normal and minimal
  router modes. They verify actual registration, not full production startup.
  Native replacement acceptance remains parent-owned.
- Raw driver/private `_connection` access remains explicitly caller-owned and is
  outside the wrapper active-use guarantee. The separately frozen Study Pack
  probe proves an IDLE checkout retained outside an operation; it does not prove
  an INTRANS restart blocker. Broader non-HTTP references remain candidates.

## Independent verification

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat181-http-independent-final node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/DB_Management/test_chacha_postgres_http_operation_lifecycle.py tldw_Server_API/tests/DB_Management/test_chacha_operation_scope.py tldw_Server_API/tests/Chat/test_chacha_notes_db_deps_postgres_health.py -q --tb=short
```

This uses official `pg_database_config` with PostgreSQL required, not AuthNZ
test_db_pool or a custom-created database. `independent77-final.redacted.log`
and `independent77-command.json` retain the result and command. Actual routers,
cached accessor, runtime admission, health/default maintenance, SQL and replacement
constructor are retained; identity/rate-limit inputs are isolated fixture values.

```sh
source .venv/bin/activate
python -m pytest .tmp/uat181-http-independent-20260917/test_escaped_cursor_probe.py .tmp/uat181-http-independent-20260917/test_owner_lock_order_probe.py .tmp/uat181-http-independent-20260917/test_checkout_wait_finalization_probe.py -q --tb=short
```

Fresh static checks (`static-comparison.json`): all seven Python files parse;
Ruff has **9 baseline / 9 current diagnostics, 0 added/removed** when baseline
bytes use their real logical filenames. Direct `.tmp` baseline paths produce10
because exact main.py BLE001 grandfathering no longer matches; this is a path
artifact, not a source improvement. All remaining findings belong to unchanged
main.py code. Bandit on four production paths reports **0 findings and 0 parse
errors**, with169 existing skipped checks. A statement-prefix AST heuristic
finds1980 SQL-like ChaCha literals in both versions, exactly equal; this is a
comparison heuristic, not proof of all SQL semantics. Diff review confirms
ownership plumbing only, with no SQL or scheduler changes.

`verification.json`, `frozen-author-manifest.json`, `final-source-after.json`
and `reviewer-manifest.json` pin this review. The clear verdict covers the
reviewed HTTP/maintenance unit. Native181 acceptance remains pending with the
parent; Study Pack, general worker, WebSocket and raw-driver adoption are outside
this approval.
