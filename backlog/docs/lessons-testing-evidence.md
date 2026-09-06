# Testing Evidence Lessons

## Nested Collections transactions do not share a PostgreSQL connection

**Incident (TASK-13153, 2026-09-05):** An output-history identity wrapper opened
a transaction, then called output creation without passing its connection.
SQLite tests passed because its pool reused the thread-local handle. PostgreSQL
creation committed independently, releasing the deletion fence before identity
capture. A forced invalid identity left the output committed; a second connection
could acquire the revision-clock lock with `FOR UPDATE NOWAIT` during the gap.

**Evidence and rule:** Both PostgreSQL regressions failed before creation reused
the wrapper's explicit connection, then passed. Collections pins the backend,
not a transaction connection. Pass the connection through composed mutations;
verify rollback and lock ownership on each backend, not only successful results.

## Closing a borrowed SQLite connection must invalidate its pool entry

**Incident (TASK-13153, 2026-09-05):** The new Media v26-to-v27 migration passed,
but constructing a second adapter for the same file failed with a closed-database
error. Bootstrap directly closed its borrowed connection before running the
migrator; the shared thread-local pool still returned that closed handle.

**Evidence and rule:** Clearing the connection through the pool's existing
`clear_thread_local_connection()` method made the real on-disk upgrade/reopen
regression pass. Verify repeated production adapter construction after migration,
not just the migrated file through a separate raw connection. Pool ownership must
be updated whenever bootstrap closes a borrowed handle.

## Delayed retry reports must not clear an operator-only block

**Incident (TASK-13153, 2026-09-05):** Recovery records its failure after releasing
filesystem exclusion. Review reproduced a delayed worker's busy-lock result
overwriting another worker's newly persisted identity block, changing the retry
time from the operator-only sentinel to sixty seconds later. The subsequent file
checks remained safe, but automatic processing resumed without operator action.

**Evidence and rule:** Two real-database/file regressions failed when a busy or
unavailable result arrived after the identity block. A conditional failure UPDATE
now leaves the entire blocked row unchanged. Test delayed failure reporting as
well as simultaneous successful transitions; releasing an external lock does not
order the later database diagnostic writes.

## A fresh read alone does not settle an uncertain commit

**Incident (TASK-13153, 2026-09-05):** Publication tests covered a commit that
completed before its acknowledgement was lost, but review reproduced a later
interleaving: the fresh outcome read saw `prepared`, the commit then completed,
and conditional abort returned false. The service incorrectly reported a
definite conflict even though the output and journal were committed.

**Evidence and rule:** Two real-database/file regressions failed on that false
conflict. The outcome read now acquires the existing revision fence on its new
connection, and a failed conditional abort triggers another outcome read.
Committed state wins; an unreadable outcome remains unconfirmed. Test a commit
that completes after the first outcome read, not only before it, and inspect
conditional-transition results before reporting rollback or rejection.

## Worker cancellation must wait for writable file descriptors to close

**Incident (TASK-13153, 2026-09-05):** A real-file test paused an offloaded output
write and called `asyncio.Task.cancel()`. AnyIO's default non-abandoning worker
wait protected its own cancellation scopes but direct asyncio cancellation still
returned before the worker closed the writable descriptor. Inspecting the
installed backend confirmed the cancelled await did not stop that worker.

**Evidence and rule:** Shielding and draining the worker task before returning
cancellation, then conditionally aborting under the verified storage lock, passed
both direct asyncio and AnyIO task-group cancellation regressions. Test both
cancellation paths when file authority depends on worker lifetime; an async task
being cancelled is not evidence that its thread has stopped writing.

## PostgreSQL optional-null predicates need an explicit parameter type

**Incident (TASK-13153, 2026-09-05):** New shared output guards passed 134 SQLite
tests but failed on PostgreSQL before the first output insert. The predicate
`? IS NULL OR token <> ?` used separate parameters for the null test and comparison;
psycopg v3 could not infer the first parameter's type. A read-only `SELECT` probe
reproduced `IndeterminateDatatype` / SQLSTATE `42P18` without any application tables.

**Evidence and rule:** `CAST(? AS TEXT) IS NULL` made the probe and isolated real
PostgreSQL guard test pass. Give nullable optional token/namespace parameters a
type at their null-test occurrence; a later separate placeholder does not supply
that type. Do not treat sanitized backend errors as evidence of an environment
failure or rely on SQLite to validate PostgreSQL parameter typing.

## PostgreSQL placeholder heuristics can mistake LIKE parameters for JSONB operators

**Incident (TASK-13153, 2026-09-05):** A shared-file guard using `LIKE ? ESCAPE '^'`
passed SQLite but failed real PostgreSQL execution. Inspecting the prepared
statement showed the shared converter had preserved that question mark as a JSONB
operator while converting the other parameters. The surrounding LIKE/ESCAPE words
triggered its expression heuristic.

**Evidence and rule:** Parenthesizing the parameter as `LIKE (?) ESCAPE '^'`
avoids the ambiguous shape without changing the matching policy or broadening this
task into SQL-parser work. Check the prepared statement when a parameterized query
works on SQLite but fails through the PostgreSQL adapter; do not assume SQL accepted
by the database is necessarily passed through unchanged by its adapter.

## Conservative path comparisons must not discard cleanup authority

**Incident (TASK-13153, 2026-09-04):** Review of guarded Reading deletion found
that lowercasing owned paths discarded one cleanup intent for distinct `A.md`
and `a.md` files on case-sensitive storage. Treating a same-named, structurally
owned output on another volume as a shared reference also lost the first
volume's cleanup authority. A second review found the case-variant loss across
different owners on the same volume.

**Evidence and rule:** Failing real-database regressions led to retaining exact
spellings within one aggregate, honoring known namespaces in reference checks
through cleanup retirement, and rejecting cross-owner case ambiguity without
mutation. Case-insensitive comparison can protect against accidental deletion;
it cannot prove file identity when deciding to discard a durable cleanup record.
Unknown generic-output namespaces remain conservative, not guessed.

## Refresh lease time after acquiring the database fence

**Incident (TASK-13153, 2026-09-04):** Reading staging validation compared its lease
against a caller timestamp captured before acquiring the database clock lock.
A writer delayed at that lock could use an obsolete time to accept an expired
reservation. A regression supplied an earlier prelock time to reproduce the gap.

**Evidence and rule:** Refreshing wall time inside the locked shared validator
closed the gap for both staging writes and adoption. Time-based authorization
must be evaluated after waits at its mutation boundary, not only before them.

## Namespace validation must stay bound to the directory used for I/O

**Incident (TASK-13153, 2026-09-04):** Initial Reading staging/cleanup verified a
volume marker and locked its file, but later reopened the output-root pathname.
Review regressions renamed that root and installed a replacement between DB
validation and file I/O. Cleanup could delete a same-named replacement-volume file
while holding the original volume's lock; writing likewise targeted the new root.

**Evidence and rule:** Descriptor-relative open/stat/unlink plus fsync on the held
directory descriptor made both replacement-root regressions pass on SQLite and
PostgreSQL. A verified pathname is not durable authority over later path resolution.

## File reservation checks must account for filesystem case aliases

**Incident (TASK-13153, 2026-09-04):** Case-sensitive SQL comparisons allowed an
uppercase generic output path to bypass a lowercase Reading reservation, and
cleanup missed uppercase shared references on default macOS storage. Re-review
found the same issue in protected lock/marker filename exclusions.

**Evidence and rule:** Failing alias regressions led to conservative lowercase
comparisons for allowed ASCII names in reservations, shared references and
protected-file exclusions. Keep these comparisons aligned; a database's text
equality does not necessarily match the output filesystem's filename identity.

## Do not signal a multiprocessing Event after terminating its waiter

**Incident (TASK-13153, 2026-09-04):** The Reading storage-lock crash test
successfully terminated its child and reacquired the OS file lock, but its own
`finally: release.set()` hung. A short pytest timeout captured the parent blocked
in `multiprocessing.Condition.notify()` waiting for the killed waiter to wake.

**Evidence and rule:** Removing event signaling after termination made the same
real-process test complete. Crash-test teardown must not depend on the terminated
process's shared synchronization state; join/terminate owned children directly.
OS lock recovery and test-event recovery are different contracts.

## Retry success must re-establish durability, not just find complete bytes

**Incident (TASK-13153, 2026-09-04):** Reading namespace provisioning wrote a valid
marker before syncing it. Injected `fsync` failure made the first attempt fail,
but retry returned the existing marker without syncing, falsely reporting success
even while the storage fault persisted.

**Evidence and rule:** A real-directory regression reproduced both attempts;
syncing the existing marker and directory on explicit provisioning retries made
both fail until sync recovered, then preserved the original namespace. Complete
readable bytes alone do not prove an interrupted provisioning attempt was durable.

## SQLite non-INTEGER primary keys need explicit NOT NULL

**Incident (TASK-13153, 2026-09-04):** Review of Reading output ownership found
`output_id BIGINT PRIMARY KEY` accepted NULL on SQLite, bypassing the composite
foreign-key reference check. The trusted registration path always read a real
output first, so happy-path and foreign-user tests did not expose the schema gap.

**Evidence and rule:** A direct-SQL NULL insertion test failed before explicit
`NOT NULL` was added, then passed. Declare non-null identity constraints explicitly
for cross-backend schemas; PostgreSQL primary-key behavior is not SQLite evidence.

## Matching numeric IDs do not establish cross-domain ownership

**Incident (TASK-13153, 2026-09-04):** Media update hooks passed Media IDs directly
to Reading highlight SQL keyed by content-item IDs. The old unit tests asserted
those hook calls and the highlight API test used a nonexistent literal parent.
Real Media/Collections adapters with colliding IDs reproduced unrelated capture
highlights becoming stale during Media edits, sync and rollback.

**Evidence and rule:** Removing the cross-domain hooks and validating surviving
Reading parents made the collision and ownership regressions pass on SQLite and
PostgreSQL. Exercise separate identity domains with deliberately colliding IDs;
mock call assertions cannot prove ownership. A rollback test must create an older
version and verify the rollback succeeded, not treat an error result as preservation.

## Memoized schema setup does not initialize per-adapter runtime flags

**Incident (TASK-13153, 2026-09-04):** Making Reading item, tag and FTS writes
transactional exposed an existing import regression: a later Collections adapter
assumed ordinary FTS deletion support after schema memoization skipped setup.
SQLite rejected its DELETE against the contentless FTS table. A single-adapter
test missed this; the reproducer needed repeated construction to hit the memo.

**Evidence and rule:** Running search-capability detection for every adapter,
including disabling FTS on PostgreSQL, made the later-adapter update/search test
and the existing import-preserves-fields test pass. Cache shared DDL work, not
per-instance state initialization. Include repeated production-factory construction
when testing adapters with memoized bootstrap.

## PostgreSQL bootstrap cannot identify duplicate columns from sanitized errors

**Incident (TASK-13153, 2026-09-04):** The first real PostgreSQL revision test
failed in the existing Collections bootstrap, before reaching the new revision
migration. Bootstrap inspected existing columns only on SQLite and unconditionally
attempted to add `output_templates.metadata_json` on PostgreSQL. The backend's
sanitized `DatabaseError` no longer carried duplicate-column text, so the legacy
message-based no-op classifier rethrew it.

**Evidence and rule:** Switching bootstrap to its existing cross-backend
`_table_columns()` helper let the PostgreSQL schema-reinitialization test pass.
Inspect schema state before backfills; do not rely on parsing redacted exception
messages. Exercise a real backend bootstrap, not only mocked SQL execution.

## Generic dict lint rules do not apply to `sqlite3.Row`

**Incident (TASK-13144, 2026-08-30):** Ruff's `SIM118` suggestion replaced
`for key in row.keys()` with `for key in row` in `PersonalizationDB`. The new
Personal Context tests stayed green, but the combined existing Personalization
regression run produced eight `IndexError` failures because iterating a
`sqlite3.Row` yields values rather than column names.

**Evidence and rule:** Restoring `.keys()` with a narrow lint exemption returned
the same 53-test combined run to green. Do not apply dict-iteration
simplifications to `sqlite3.Row`; when a shared database wrapper changes, pair
new-feature tests with its existing consumer suite.

## Lifecycle fences must be proven inside the write transaction

**Incident (TASK-13145, 2026-08-30):** The Personal Context service checked the
purge-pending state before proposal and runtime writes, and its sequential tests
passed. Independent review showed a purge could commit after that check but
before either standalone repository transaction, allowing the later write to
recreate encrypted state beyond the purge barrier.

**Evidence and rule:** Passing the expected manifest version into each write and
rechecking both the manifest head and surviving scope state under the same
`BEGIN IMMEDIATE` transaction closed the race; targeted purge-race regressions
then passed. For destructive lifecycle barriers, a service precheck is UX only:
the storage transaction must enforce the same fence.

## Exercise canonical factories, not only service fakes

**Incident (TASK-13148, 2026-08-30):** Personal Context bootstrap unit tests
passed with service fakes, but the first authenticated endpoint test using the
real factory failed with `personal_context_snapshot_unavailable`. The canonical
profile clock emitted arbitrary microseconds while canonical serialization
permits millisecond precision only, so a fresh profile could not be snapshotted.

**Evidence and rule:** Normalizing the service clock at the canonical mutation
boundary made the real authenticated bootstrap, stale-completion, receipt, and
post-link push flow pass. When a feature composes storage, canonical models, and
HTTP dependencies, include at least one test through the production factory;
service fakes cannot prove cross-layer serialization contracts.

## After-commit callbacks may precede outer projection terminalization

**Incident (TASK-13161, 2026-09-03):** The real authenticated factory test pushed
a new Personal Context record successfully, but the canonical after-commit relay
then poisoned its authority batch. Unit relay tests were green. The callback ran
after the canonical database commit but before the enclosing Sync materializer
marked the identical client-ingress envelope applied; the envelope also relied on
projected object state because its wire row omitted `object_revision`.

**Evidence and rule:** A production-factory push followed by a real pull reproduced
the failure. Restricting self-head confirmation to an exact authenticated
client-ingress match in either legal pending/applied state, and deriving its base
revision/hash from current projected state, made the same after-commit relay and
exact-once pull pass. For cross-store callbacks, test the actual outer transaction
ordering and use the store's projected head facts rather than assuming envelope
terminal state or optional wire revision fields are already complete.

**Follow-up (TASK-13170, 2026-09-04):** The original production-factory endpoint
later proved that deriving the outgoing base was insufficient: receipt confirmation
still compared authority lineage to the immutable envelope's absent wire revision,
so repeated polls remained permanently pending. A real two-store test that asserted
raw revision `None`, projected revision 1, and both semantic and manifest successors
at lineage 1→2 caught the gap.

**Correction after review (TASK-13170, 2026-09-04):** Making latest projected state
the missing revision's authority proof then created an unbudgeted persistent read and
permanently rejected a lagging companion after a later legitimate materialization
moved projection forward. The durable fix derives revision only from the immutable
ingress result lineage: an absent revision with no base is genesis 1; an absent
revision with a complete strict base is `base_revision + 1`. Projection may verify
that predecessor before the ingress receipt and may advance transactionally after
authenticated authority finalize, but it must not become historical receipt proof.

## Stable basetemp evidence must respect production path policy

**Incident (TASK-13172, 2026-09-04):** The first combined 14-file certification
run forced `--basetemp=/tmp/tldw-task-13172-remediation-matrix`. Hundreds of
otherwise unrelated cases failed or errored because legacy harnesses construct
`PersonalizationDB.for_path(tmp_path / "personalization.db")`, and the production
trusted-root guard correctly rejects `/private/tmp`. The same first seven gates
passed 214/214 when rerun under pytest's trusted default temporary root, and the
remaining groups passed 239/239 and 300/300.

**Evidence and rule:** Use a stable exact basetemp only for fixtures that place
their application databases under an explicitly configured trusted root. For a
mixed legacy matrix, preserve the production path guard and record conclusive
group results under pytest's trusted temp root; never weaken storage validation
to make evidence paths prettier.

## Security report formatters can fail on deliberate invalid-Unicode fixtures

**Incident (TASK-13173, 2026-09-05):** Bandit's text and JSON report writers both
raised `UnicodeEncodeError` on an existing lone-surrogate continuity-token fixture.
The scanner itself had completed; changing output formats did not fix reporting.

**Evidence and rule:** Serializing the scanner's issue fields with escaped Unicode
allowed an in-memory comparison against HEAD: no new findings, only the two
reviewed subprocess warnings removed, and 15 unchanged fixture findings. Preserve
invalid-input coverage and compare scan results; do not remove the negative test
or add broad suppressions merely to make a report formatter succeed.

## 2026-09-05 — Buddy UAT exposed transport and lifecycle gaps hidden by test doubles

TASK-13182's setup smoke test ignored `expected_version`, so it passed while a real defaults save followed by checkpoint advancement returned 409. Enforcing optimistic version checks in the test reproduced the failure. TASK-13184's direct Python WebSocket probe accepted a server handshake without a selected subprotocol, while Chromium rejected the same handshake before sending any message. Require the browser's protocol contract in the auth regression and verify the real browser boundary. During review, discarded Strict Mode loads and A→B→A persona switches also exposed stale-completion cases that a single switch/remount test did not cover.


## 2026-09-05 — Match CI runtime and await usable editor state

During PR #2884 / TASK-13176, VisualPackEditor tests passed64/64 on local Node26 but failed4/64 on Node20 with the workflow's deterministic shared-UI config. Pack metadata appeared before manifest-derived select values, and a recorded candidate-review request preceded completion of the disabled-button state. The shard also exposed a custom-state selection before its option existed. Waiting for the actual manifest values, available option, and accepted/rejected outcome preserved the assertions and passed93 tests in the six-file CI context on Node20. Reproduce both the CI runtime and package config; a selected pack ID or recorded request is not proof that downstream controls are ready. Do not replace these readiness checks with larger timeouts.

## Cache size is not proof of cache reuse

**Incident (TASK-13187/TASK-13188, 2026-09-04):** Manual llama.cpp snapshot
tests initially modeled a slot token count as `n_past`. Review against pinned
upstream source found that slot JSON uses `n_prompt_tokens` only after a task;
fresh idle slots omit it. The live-harness investigation also found that
completion `tokens_cached` describes the final slot size, not tokens reused.

**Evidence and rule:** Source-derived fresh/busy/malformed-slot regressions
passed after correcting the parser. At upstream commit
`4d9176092d00586775af140581bb0b558ddc4389`, `server-common.cpp:67–71`
serializes reused tokens as `timings.cache_n` and newly processed tokens as
`timings.prompt_n`. The harness now requires those counters and a separate cold
process control. No binary/model was available, so the live test remains skipped
and the production build allowlist remains empty. Pin wire fixtures to their
source, label source-derived versus live evidence, and never infer reuse from
file existence, HTTP success, similar output, or final cache size.

## Snapshot reuse evidence is configuration-specific

**Incident (TASK-13188, 2026-09-04):** The supplied Gemma sliding-window model
saved/restored 2770 tokens on llama.cpp b10816 but reused zero after restart.
Native diagnostics reported missing cache coverage; a one-variable `--swa-full`
test changed reuse to 2770 tokens with only 10 processed. The managed
runner/store/coordinator subsequently reproduced 2770/10 versus a cold control
of 0/2780. At context 16384 the native SWA allocation grew 300 to 3200 MiB.

**Evidence and rule:** Treat cache mode as compatibility identity and explicit
operator configuration, with memory/restart guidance. Do not infer another
architecture's support from a model-family label or one successful configuration.
Publish warm/cold measurements before reuse assertions so failures retain useful
evidence. Managed runtime evidence does not substitute for live client acceptance.

## Activation fixtures must distinguish authorization from publication state

**Incident (TASK-13162, 2026-09-05):** Adding canonical activation checks exposed
older relay fixtures that granted access by writing only Sync metadata. Simply
activating those fixtures would cover the pending source rows their recovery
tests were meant to exercise.

**Evidence and rule:** Production certification now performs real baseline
installation and acknowledgement before creating relay debt. Narrow authority
and budget tests use an explicit typed authorization double while preserving
their real source rows and original assertions. Keep those concerns separate;
never add a production metadata-only fallback to satisfy a test harness. Compare
broader failures against unchanged dev before attributing them to the new guard.

**Follow-up (TASK-13162):** Updating those proof fixtures exposed a deeper failure:
guarded repair verified the canonical ingress receipt, but terminalization accepted
only pending/applied rows, so a previously failed row could never unblock bootstrap.
The original recovery assertion caught it. A focused regression failed on both
SQLite and PostgreSQL before the one-line transition fix; all 22 receipt/state
cases and 48 bootstrap tests then passed. Keep the recovery assertion through
fixture repairs, and test retryable failure states as well as first application.
## 2026-09-05 — Verify session handoff under StrictMode and superseded attempts

TASK-13180 real browser UAT resumed the exact Buddy session with HTTP 200, but
full Live never requested its detail or opened a WebSocket. Effect cleanup set
`mountedRef` false while StrictMode setup replay never restored it; the ordinary
route test passed. A StrictMode route regression reproduced the missing socket,
and restoring the mounted flag during effect setup made both paths pass.
Independent review also found that an older connection's catch/finally could clear
a newer attempt's loading state. Resolve/reject regressions now keep the newer
request pending and prove another Connect click cannot issue a duplicate request.
Test effect replay and overlapping completions, including failure/finally paths,
when mounted/attempt refs guard asynchronous connection state.

## Check final task-file newlines after Backlog serialization

PR2902's pre-commit job rejected two task records after Backlog MCP updates
removed their terminal newline. `git diff --check` had passed, because that check
does not require a final newline. Normalize final newlines after the last task
edit, then include those bytes in pre-commit verification; repeated MCP writes
can otherwise reintroduce the same formatting-only CI failure.

## First-run name checks must include tombstones

**Incident (TASK-13196, 2026-09-05):** pixel-migu seed replay tests preserved
its own deletion receipt, but independent review found that a preexisting deleted
user card with the same name had no receipt. The ordinary name lookup excluded
that tombstone while SQLite's unique name constraint retained it, so canonical
DB initialization failed on every retry. A real factory regression reproduced
the conflict; an explicit include-deleted lookup preserved the user's tombstone
and restored startup. Test both deletion after seeding and deleted user content
that predates the seed.
