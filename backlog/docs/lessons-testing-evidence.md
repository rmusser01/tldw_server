# Testing Evidence Lessons

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
