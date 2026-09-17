# UAT181 / TASK13260.118 — Study worker checkout investigation

## Verdict

The existing Study Pack worker has a demonstrated PostgreSQL **checkout lifetime
defect** after a joined source read followed by generation failure. Its executor
checkout remains open and **IDLE** after the actual handler's `finally` closes on
the event-loop thread. This is not evidence of an INTRANS transaction, replacement
DDL blocker, pool exhaustion, or a native Study Pack failure.

A private-only outer operation owner returns the same source checkout. The actual
Study Suggestions missing-session read/error/finally path returns all checkouts
without that wrapper. The other non-HTTP accessor references remain candidates,
not demonstrated failures.

## Actual causal chain

Paths below are relative to `/Users/macbook-dev/Documents/GitHub/tldw_server2`.
Line numbers for unchanged worker/service files are stable; the mutable ChaCha
prototype is pinned by `source-before.json` and the final source snapshot.

1. `app/services/study_pack_jobs_worker.py:139` (under `tldw_Server_API`) gives
   `handle_study_pack_job` directly to `WorkerSDK.run`. SDK line955 awaits that
   handler. There is no per-job ChaCha owner at either boundary.
2. Worker lines35–45 call the real cached `get_chacha_db_for_user_id`; accessor
   `api/v1/API_Deps/ChaCha_Notes_DB_Deps.py:790` performs the real cache/runtime
   lookup and schedules independent default-character maintenance. That
   maintenance and cache-health work both return their checkouts in this probe.
3. Worker lines69–80 instantiate the actual generation service and await it.
   `core/StudyPacks/generation_service.py:222` runs the actual source resolver in
   `asyncio.to_thread`; `source_resolver.py:191` calls the actual cached DB's
   `get_note_by_id`. Its `chacha/note_store.py:540` read returns fixture evidence
   that reaches the actual generation prompt.
4. Outside an operation, ChaCha `_connection_state()` falls back to
   `threading.local`; `_get_connection_for_state()` caches the source-thread
   checkout. The note read leaves it IDLE, but does not return it to the pool.
5. After the controlled transport failure, worker lines87–92 execute the real
   cleanup helper. Its lines25–32 call `close_connection()` on the event-loop
   thread. Legacy `close_connection()` sees only that thread's cached connection;
   it cannot return the other executor's loan.
6. A private wrapper spanning the same handler with
   `chacha_operation(independent=True)` shares captured state through the real
   thread handoff. The same existing cleanup returns the captured loan on the
   event-loop thread; no global sweep or implicit commit is used.

The representative same-thread control is
`study_suggestions_jobs_worker.py:72` → actual
`StudySuggestions/snapshot_service.py:163` session rollup → missing-session
ConflictError → worker lines83–94 cleanup. This body stays on the event-loop
thread and returns its checkout. It does not establish every successful
suggestion path or every anchor type.

## Probe fidelity and evidence

`test_study_worker_lifetime_probe.py` uses the official `pg_database_config` with
required PostgreSQL enabled. It seeds only fixture data and installs that real
CharactersRAGDB into an isolated real LRU cache/runtime. Both accessors delegate
the actual function and assert that it returns the exact seeded instance.
Real pool get/return instrumentation delegates originals; receipts record object
identities, executing threads and driver status without observer SQL. The final
receipts are captured after the real job finally and after maintenance tasks
finish, **before the event loop/executor shuts down**.

The unused Media factory returns an inert handle. Provider selection is a fixed
fixture value, and only `_call_generation_model` raises the controlled failure,
after verifying actual fixture note content in the built prompt. No model,
external API, real Jobs queue, browser, native profile or app runtime is used.
No product files were changed. Fixture teardown alone closes its own DB/pool.

The first run, retained under `first-run/`, had unchanged source hashes and
1 expected failure / 2 passes / 0 skips in4.98s. Its receipt was taken after
`asyncio.run` shutdown; it is superseded by the immediate live-loop receipt.
The second run repeated the same result in5.56s with stronger instrumentation,
but source013's approved active-use owner changes landed during its before/after
window. That run is retained separately and is **not** a frozen-source result.

Final frozen-source result and receipt identities are recorded in
`verification.json`; source provenance is in `source-before.json`,
`source-after.json`, and `source-snapshot/`. Exact runner command:

**Final: 1 expected failure, 2 passes, 0 skips, 3 warnings, 5.86s.** All eleven
source hashes matched before/after and match the retained snapshots. The owner
prototype includes source013's active-use/deferred-return extension; this run
only tests completed/joined source work, not that extension's in-flight behavior.

| Case | Cached DB identity | Body thread | Source checkout | Outstanding after finally |
| --- | --- | --- | --- | --- |
| Existing Study Pack | 4625565328 | 6153269248 | 4625579344 | One, open, IDLE |
| Private Study Pack owner control | 4625487568 | 6153269248 | 4625484752 | None |
| Existing Suggestions error control | 4623248528 | 8346099072 | 4625467408 | None |

Each accessor returned the exact cached identity listed. The event loop and all
handler closes ran on thread8346099072. Identities are process-local observation
tokens, not account identifiers. `pack-existing.json` shows the source loan has
no matching return; `pack-owned.json` shows its source loan returned from the
event-loop thread. Ruff reports0 findings; Python AST parses; Bandit reports0
findings/errors with B101 excluded only for test assertions. No private runtime
credential file was copied into this packet.

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat181-study-worker-frozen node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs .tmp/uat181-study-worker-lifetime-20260917/test_study_worker_lifetime_probe.py -q --tb=short
```

Expected outcome while production remains unchanged: the existing Study Pack
cleanup assertion fails; private owner and existing Suggestions controls pass.
This is a diagnostic RED with positive controls, not a completed repair suite.

## Smallest bounded adoption proposal — production remains held

1. Once the HTTP/maintenance owner API is reviewed, put an explicit independent
   owner around **one `handle_study_pack_job` acquisition-through-finally
   lifetime**, preserving the existing inner cleanup, result, exception and
   transaction decisions. Do not wrap the long-lived WorkerSDK run/queue, change
   pool defaults, or globally close cached DB instances. No source-selection,
   generation, scheduler or Jobs schema changes are warranted by this probe.
2. Promote this actual accessor/service/thread-handoff regression into the
   permanent Study Pack suite. Add successful real persistence with fake model
   output, repeated jobs on the same cached instance, source-read failure and
   cancellation while source work is in progress. The owner API's newly approved
   active-use/deferred-return controls are a prerequisite for the last case;
   this probe does not validate cancellation while a database call is still
   executing. Cancelling an await is not proof that `to_thread` has stopped.
3. Preserve caller ownership explicitly: hold an unrelated outer pending write
   and prove the independent job neither commits, rolls back nor returns its
   connection on success/failure; verify the caller can decide afterward. An
   injected shared DB **object** is not permission to adopt its prior thread
   connection. For an explicitly borrowed external connection, use/test the
   existing `ExternalConnection` binding contract (or leave that caller outside
   this job-owned adapter); never infer ownership from an active transaction.
   Keep the public `StudyPackGenerationService` usable with caller-owned DBs.
4. Keep SQLite rollback/regeneration controls, missing/invalid owner behavior,
   failed Media acquisition cleanup, and untouched same-thread Suggestions
   controls. The existing worker tests replace `_get_databases_for_user` and
   replace the source resolver or snapshot builder; their passing state does not
   cover this real accessor/thread lifetime. The pre-handler SDK cancellation
   controls also do not cover an already-running source thread.

Only Study Pack adoption is supported by this failure. The broader 30-reference
inventory remains an investigation list. No general non-HTTP migration or
whole-application lifecycle guarantee follows from these three cases.
