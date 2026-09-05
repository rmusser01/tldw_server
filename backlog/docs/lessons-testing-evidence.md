# Testing Evidence Lessons

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
