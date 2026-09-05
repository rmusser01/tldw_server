# Reading Atomic Hard Delete Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute inline; delegation is not required.

**Goal:** Delete only the confirmed Reading revision, with transactional ownership cleanup and durable file disposal.

**Architecture:** CollectionsDatabase owns the revision fence and relational mutations. Reading services stage/adopt artifacts and a bounded internal worker drains durable cleanup intents. Existing HTTP routes expose the guard and truthful cleanup state.

**Tech Stack:** Python, FastAPI/Pydantic, existing SQLite/PostgreSQL backends, pytest.

**Spec:** `Docs/superpowers/specs/2026-09-04-reading-atomic-hard-delete-design.md`

## Global Constraints

- ADR required: yes.
- ADR path: `backlog/decisions/003-reading-atomic-hard-delete.md`.
- Reason: persisted revisions, destructive API preconditions, artifact ownership and durable cleanup.
- All Reading item DTOs expose `revision: int > 0`.
- No filesystem or network work inside mutation transactions.
- Reuse `_read_snapshot()` and the SQLite backend's existing `BEGIN IMMEDIATE` helper.
- Storage lock precedes database lock; never wait for a filesystem lock inside a database transaction.
- Capability and hard-delete readiness fail closed; unavailable hard deletion returns a non-mutating 503.
- Stop old writers before upgrade. Migrate before advertising the capability.
- No external Media/Note deletion, new dependencies, UI changes or generic job platform.
- Worktree: `/private/tmp/task-13153-reading-hard-delete`; base `59bbdd1bc990a86ee63d641a97be51c2bf6a81ed`.
- Run only focused tests. PostgreSQL skips are not passing backend evidence.
- Activate the existing Server `.venv` before Python commands; do not install a new environment implicitly.

## File responsibilities

- Modify `tldw_Server_API/app/core/DB_Management/Collections_DB.py`: migrations, revision clock, writer fence, coherent reads, ownership and conditional deletion. Preserve existing generic consumers.
- Modify `tldw_Server_API/app/core/Collections/reading_service.py`: guarded delete, stage/adopt capture completion.
- Modify `tldw_Server_API/app/api/v1/schemas/reading_schemas.py`: revision and deletion response.
- Modify `tldw_Server_API/app/api/v1/endpoints/reading.py`: preconditions and error mapping.
- Inspect/update `tldw_Server_API/app/api/v1/endpoints/{items,reading_highlights,outputs}.py` and `tldw_Server_API/app/services/{outputs_service,outputs_purge_scheduler}.py`: route all Reading-owned writes through the fence.
- Create `tldw_Server_API/app/services/reading_artifact_cleanup_service.py`: bounded retry lifecycle and staged-file reconciliation.
- Create `scripts/reading_reconcile_artifacts.py`: offline dry-run/apply ownership and volume reconciliation; no deletion.
- Create `tldw_Server_API/tests/Collections/test_reading_artifact_reconciliation.py`: manual/older archive recovery, unchanged-manifest checks and authorization boundaries.
- Modify `tldw_Server_API/app/services/startup_cleanup_workers.py`: start/stop cleanup with existing service lifecycle.
- Modify `tldw_Server_API/app/api/v1/endpoints/config_info.py`: final capability activation only.
- Modify `Docs/API-related/Reading_List_API.md` and `Docs/Published/API-related/Reading_List_API.md`: API, upgrade and pending-cleanup semantics.
- Create `tldw_Server_API/tests/Collections/test_reading_atomic_delete.py`, `test_reading_revision_mutations.py`, and `test_reading_artifact_cleanup.py`: focused storage and lifecycle contracts.
- Extend existing `test_reading_api.py`, `test_collections_postgres_integration.py`, `test_reading_endpoint_error_logs.py` and appropriate docs-info tests located by symbol search.

## Stage 1: Persist revisions and fence every Reading mutation

**Status:** In Progress — revision schema/clock, item/tag, note-link, highlight and output database-update writers implemented; artifact lifecycle/purge integration and DTO coverage remain.
**Goal:** Every returned token describes one coherent aggregate version.
**Success Criteria:** Migration, no-op, child-write, reuse and snapshot cases pass on both backends.
**Tests:** New revision module plus existing service/highlight/note-link/import suites.

**Interfaces:** Preserve existing public mutation signatures; add `revision: int` to `ContentItemRow`. Internal transaction helpers consume an explicit backend connection. All allocation and parent writes use that same connection.

- [ ] Inventory writers before editing, including SQL outside CollectionsDatabase:

```bash
rg -n 'content_items|content_item_tags|content_item_note_links|reading_highlights|reading_archive' tldw_Server_API/app
rg -n 'def transaction|BEGIN IMMEDIATE|isolation_level' tldw_Server_API/app/core/DB_Management/backends
```

Record each mutation's fence in the task notes. Include imports, reanchor, bulk, direct output deletion/retention and generic content deletion. A Reading path must not retain an unguarded alternate deletion entry point.

Initial implementation progress (2026-09-04): the positive persisted column and
durable transactional counter are present, including overflow rejection and
idempotent migration. `_next_reading_revision(connection)` is not yet connected to
production item mutations or exposed through DTOs; the capability remains absent.
The writer inventory places item/tag/note-link/highlight SQL in Collections_DB.py;
output association/purge paths still need the full ownership audit. Next is explicit
connection propagation and normalized no-op detection across those writer methods.
The real PostgreSQL run exposed an existing bootstrap defect before the new
migration: SQLite-only column inspection caused duplicate-column ALTERs on PG.
Bootstrap now uses the existing `_table_columns()` helper on both backends.
Run PostgreSQL tests with `TLDW_TEST_NO_DOCKER=1` against the existing test service;
the inherited fixture otherwise attempts replacement of a named test container.
Verification for this foundation: 11 SQLite cases pass (new revision tests plus
note-link and FTS regressions); 11 real PostgreSQL cases pass (new revision tests
plus existing Collections PostgreSQL integration). The new tests pass Ruff/Black;
changed production lines pass Black. Scoped Bandit reports zero findings. The
production module retains the same nine Ruff findings as the base revision, with
no findings in the added lines. This is not completion evidence for Stage 1's
remaining mutation guards or any hard-delete behavior.
Independent foundation review found a PostgreSQL search-path mismatch. A real PG
regression reproduced it; the fix consistently uses the backend's existing public
schema contract for revision introspection, migration and allocation. Scoped
re-review found it addressed with no new issues. This does not add general
multi-schema Collections support.
The final PG-focused run has one intentional skip: the SQLite parameter of the
PostgreSQL-only search-path test. No PostgreSQL case was skipped as unavailable.

Item-writer checkpoint (2026-09-04): `upsert_content_item()` and
`update_content_item()` now acquire the clock fence before reading and use one
explicit connection for item fields, tags, FTS and revision allocation. A compound
Reading edit advances once; normalized Reading no-ops preserve both revision and
timestamp. Generic Watchlist refresh timestamps retain their previous semantics.
ID, URL, Media-ID and list readers return the persisted token; aggregate reads
share a snapshot with their tags. The existing transaction/snapshot primitives are
reused, with no new dependencies. The database-local clock deliberately serializes
these writers across users; finer-grained allocation is outside this contract.

Rollback and synchronized two-connection tests cover insertion, compound updates,
identical upserts and reads spanning another writer's commit. Strict transactional
FTS failures exposed stale per-instance flags after memoized schema initialization;
each new adapter now detects its actual search capabilities. The existing import
regression and the new later-adapter test reproduce and cover that correction.
Independent review and scoped compatibility re-review found no outstanding issues.
Child note-link/highlight writers, output associations/purges, alternate deletion
entry points and DTO exposure remain unguarded/unimplemented in this partial
checkpoint. The capability remains absent; no atomic hard-delete behavior is claimed.

Checkpoint verification (Server virtual environment, 2026-09-04):

- `python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py -k 'not postgres' -q --tb=short`: 21 passed, 23 deselected.
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py tldw_Server_API/tests/Collections/test_collections_postgres_integration.py -k postgres -q --tb=short`: 24 passed, 1 intentional SQLite-parameter skip, 21 deselected. All selected PostgreSQL cases ran against the real test service.
- `python -m pytest tldw_Server_API/tests/Collections/test_reading_service.py tldw_Server_API/tests/Collections/test_reading_note_links_db.py tldw_Server_API/tests/Collections/test_reading_import_export.py tldw_Server_API/tests/Collections/test_content_items_fts_contentless.py -q --tb=short`: 42 passed.
- New tests pass Ruff and Black; changed production ranges pass Black;
  compilation and `git diff --check` pass. Production Ruff reports the same
  nine baseline findings as the preceding foundation commit. Scoped Bandit
  reports zero findings and no analysis errors. No full-suite run was performed.

Note-link checkpoint (2026-09-04): both association writers acquire the shared
clock fence before reading the parent and use one explicit connection for child
membership and the parent revision/timestamp. Reading membership changes advance
once; duplicate links and absent unlinks preserve the token and timestamp.
Non-Reading associations retain their prior item timestamps and revisions. Missing
or inaccessible parents cannot receive new links; unlink still returns false for
an absent association. No external Note row is edited or deleted. Independent
review found no actionable issues in this slice.

Note-link verification (Server virtual environment):

- Red run of the new SQLite note-link cases: five expected failures for missing
  revision allocation/rollback; two compatibility characterization cases passed.
- `python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py -k 'not postgres' -q --tb=short`: 28 passed, 30 deselected.
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py -k 'note_link and postgres' -q --tb=short`: 7 passed on real PostgreSQL, 51 deselected, no skips.
- `python -m pytest tldw_Server_API/tests/Collections/test_reading_note_links_db.py tldw_Server_API/tests/Collections/test_reading_api.py -k note_link -q --tb=short`: 6 passed, 14 deselected.
- Test Ruff/Black, changed production-range Black, compilation and diff checks
  pass. Scoped Bandit has zero findings/errors. Production Ruff retains its nine
  previously recorded baseline findings. No full-suite run was performed.

The next highlight slice must address the existing ID-domain mismatch before
enabling hard deletion: Media repository/runtime hooks pass Media IDs directly to
`mark_highlights_stale_if_content_changed()`, whose SQL interprets them as content
item IDs. The legacy highlight CRUD test also creates an orphan using a literal
99999 parent. Neither is proof of a valid Reading parent/ownership contract.
Highlight writers, reanchor/stale hooks, output ownership/purge and alternate
delete paths remain unfinished. This checkpoint does not advertise the capability.

Highlight checkpoint (2026-09-04): CRUD now requires an owned, surviving Reading
parent, locks the revision clock before reading, and commits child changes and
one parent token/timestamp together. Equivalent patches and missing deletes are
no-ops. The create endpoint maps invalid parents to 404. Its former literal-parent
test now creates an actual Reading capture. Reanchor reads parent/highlights in
one snapshot, computes matching outside the writer lock, rejects an obsolete
content hash or changed parent revision, then commits all material child patches
with one revision advance. The save result refreshes its revision/timestamp after
reanchoring while retaining its original creation/content-change flags.

The preceding ID-domain concern is resolved: four Media write/rollback/overwrite
hooks and their now-unused bulk stale setter were removed. Media and Reading
identities are independent, and ADR-003 explicitly excludes external Media edits
from capture mutation. Real colliding-ID regressions exercise all four Media
operations on SQLite/PostgreSQL while verifying both the Media content change and
unchanged capture/highlight data. Both Reading API documents now describe the
surviving-parent and capture-specific highlight behavior. No new ADR is required;
this implements the existing ADR-003 ownership boundary.

Independent review found one stale save-result issue; its regression failed before
the refresh fix. Scoped re-review found it addressed and no other actionable issues.
Outputs/ownership/purge, alternate item deletion paths, complete DTO snapshots and
the cleanup/readiness/guarded-delete contract remain unfinished. Capability stays
absent; this is not Stage 1 or task completion.

Highlight verification (Server virtual environment):

- New SQLite highlight tests first produced 11 expected failures; the invalid-parent
  endpoint test first returned 500 rather than 404. Real Media collision tests
  reproduced stale highlights in edit, sync and rollback paths before hook removal.
- `python -m pytest tldw_Server_API/tests/Collections/test_reading_service.py tldw_Server_API/tests/Collections/test_reading_import_export.py tldw_Server_API/tests/Collections/test_reading_revision_mutations.py -k 'not postgres' -q --tb=short`: 82 passed, 46 deselected.
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py -k '(highlight or reanchor) and postgres and not external_media' -q --tb=short`: 11 passed, 77 deselected; real PostgreSQL, no skips.
- Same module with `-k 'external_media and postgres'`: 4 passed, 84 deselected; real PostgreSQL, no skips.
- `python -m pytest tldw_Server_API/tests/Collections/test_reading_highlights_api.py tldw_Server_API/tests/Collections/test_reading_highlights_reanchor.py tldw_Server_API/tests/Collections/test_companion_reading_activity_bridge.py -q --tb=short`: 6 passed.
- `python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_media_item_update_ops.py tldw_Server_API/tests/DB_Management/test_media_db_synced_document_update_ops.py tldw_Server_API/tests/DB_Management/test_media_db_document_version_rollback_ops.py -q --tb=short`: 20 passed.
- Scoped Bandit across all seven touched production modules reports zero findings
  and no analysis errors. Ruff findings in touched files are the same 14 baseline
  findings, with none in added lines. No full-suite run was performed.
- The strengthened late-reanchor race uses a second-thread connection during
  quote matching; final focused reruns pass on both SQLite and PostgreSQL (one
  case each). Changed-line Black, compilation of all 13 touched Python files,
  and `git diff --check` pass.

Output-ownership foundation checkpoint (2026-09-04): the idempotent schema now
includes `reading_output_ownership`, keyed by a non-null output ID with same-user
composite foreign keys to the item/output and an explicit opaque storage namespace.
References restrict deletion rather than cascading away ownership evidence. The
trusted database registration primitive validates an owned surviving Reading
parent, live Reading archive output, positive expected revision and nonempty
namespace; insert and parent token advancement share the writer fence/transaction.
Identical ownership replay is a no-op (even with its original token), while changed
parent/volume claims conflict. Editable output metadata cannot assign or transfer
ownership. No namespace is guessed and no automatic legacy backfill is performed.

This is deliberately NOT production adoption: the primitive has no service/HTTP
caller, and its later staging/reconciliation caller must prove artifact provenance
and mounted-volume authority. Existing generic output mutation/purge paths are not
revision-aware yet; no archives are registered through these paths at this stage.
The audit found file-first deletion in `outputs.py` (single delete, purge, rename
and generated-output cleanup) and `outputs_purge_scheduler.py`, plus standalone
output UPDATE/DELETE SQL in `outputs_service.py` and CollectionsDatabase. These
must be integrated with durable cleanup before production ownership registration
is enabled. Same-user shared-file references and pending-path reservations remain
part of that integration. Do not treat the new foreign keys alone as file safety.

Review exposed SQLite's nullable non-INTEGER primary-key behavior; a direct-SQL
NULL regression failed before explicit NOT NULL fixed it. The existing fresh-table
bootstrap test also caught the foundation adding revision with ALTER on new stores;
both fresh CREATE definitions now contain revision, with legacy migration retained.
Scoped re-review found both corrections addressed with no new actionable issues.
ADR required: existing ADR-003 applies; no new ownership decision is introduced.
Stage 1, cleanup/readiness, guarded deletion and the overall task remain incomplete;
capability stays absent.

Ownership foundation verification (Server virtual environment, 2026-09-04):

- `python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py tldw_Server_API/tests/Collections/test_output_artifact_idempotency.py tldw_Server_API/tests/Collections/test_collections_schema_bootstrap.py -k 'not postgres' -q --tb=short`: 64 passed, 59 deselected.
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py -k '(output_ownership or schema or migration or search_path) and postgres' -q --tb=short`: 18 passed, 1 intentional SQLite parameter skip in the PostgreSQL-only search-path test, 97 deselected. All selected PostgreSQL cases executed on the existing service.
- New ownership cases failed before the schema/registration implementation; the
  NULL constraint regression failed before NOT NULL was added. Concurrent replay,
  rollback, immutable association, stale revision, wrong user, invalid origin/type,
  migration re-entry and direct-SQL reference checks are covered.
- New tests pass Ruff/Black, touched production ranges pass Black, compilation and
  diff checks pass. Scoped Bandit reports zero findings/errors; the production
  module retains the same nine previously recorded Ruff findings. No full suite.

Output database-update checkpoint (2026-09-04): metadata/chatbook-link, Media-link,
rename, format and retention writers now share one CollectionsDatabase update
boundary. The outputs service delegates instead of issuing standalone UPDATE SQL.
Path validation finishes before the clock lock; output/structural-owner reads,
updates and parent revision advancement use the same explicit connection. Material
compound edits advance once, normalized JSON/no-op replays preserve the token, and
JSON booleans remain distinct from numbers. The Media link can still be cleared.
Unowned outputs cannot infer Reading ownership from editable metadata. Active
same-user validation now prevents the old Media-link writer from changing a
soft-deleted row before raising `output_not_found`.

This checkpoint only fences database metadata, not file operations. File-first
rename/transcode/deletion, purge routing, shared paths, namespace validation and
staging/adoption still require the planned durable storage lifecycle. Production
ownership registration remains unwired; capability stays absent. Existing ADR-003
applies without a new architectural decision. Stage 1 and TASK-13153 remain In Progress.
Independent scoped review found no outstanding correctness/security findings; its
retention-only coverage suggestion is included.

Output update verification (Server virtual environment):

- Initial SQLite red run reproduced missing parent revision/rollback, JSON no-op
  and deleted-row protection; four unrelated-output assertions were corrected to
  compare persisted snapshots rather than transient `is_new/content_changed` flags.
- New output cases: 19 passed on SQLite. Added retention-only persistence/no-op and
  explicit-connection/clock-order/path-validation tests: 2 passed on SQLite.
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py tldw_Server_API/tests/Services/test_outputs_service.py tldw_Server_API/tests/Collections/test_items_and_outputs_api.py tldw_Server_API/tests/Collections/test_output_artifact_idempotency.py -k 'not postgres' -q --tb=short`: 125 passed, 78 deselected (before the two extra edge cases were added).
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py -k 'output and postgres' -q --tb=short`: 33 passed, 121 deselected, no skips, against the existing real PostgreSQL service. The added `-k '(retention_only or explicit_connection_fence) and postgres'` run passed both cases (156 deselected, no skips).
- Test Ruff/Black, changed production-range Black, compilation and diff checks
  pass. Scoped Bandit reports zero findings/errors. The two production files retain
  their same ten baseline Ruff findings (nine DB, one service import order). No full suite.

- [x] Add a real database fixture using `tmp_path` and the existing `CollectionsDatabase.from_backend` pattern. Use the following record constructor in the new test module:

```python
def make_reading(db):
    return db.upsert_content_item(
        origin="reading", url="https://example.org/a",
        canonical_url="https://example.org/a", domain="example.org",
        title="Original", summary="Body", content_hash="a",
        word_count=1, published_at=None, tags=["news"],
    )

def test_noop_preserves_revision(db):
    item = make_reading(db)
    assert item.revision > 0
    same = db.update_content_item(item.id, title="Original")
    assert same.revision == item.revision
    changed = db.update_content_item(item.id, title="Changed")
    assert changed.revision > item.revision
```

- [x] Run the new revision test and confirm failure is the absent revision behavior, not fixture setup.

```bash
python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py -q
```

- [ ] Add idempotent migration DDL: positive revision column, one durable clock row, structural Reading-output association, and file-intent rows keyed by user/path. Initialize the clock above existing tokens under the migration lock. Use backend-specific integer/locking syntax. Allocate using this transaction sequence, with explicit `connection=conn`:

```sql
UPDATE reading_revision_clock SET value = value + 1 WHERE id = 1;
SELECT value FROM reading_revision_clock WHERE id = 1;
UPDATE content_items SET revision = ? WHERE id = ? AND user_id = ? AND origin = 'reading';
```

Check overflow before allocation. Acquire SQLite's write lock before reads and PostgreSQL's clock row lock before the parent. Compare normalized stored values before allocating; update all child rows and parent revision together. Snapshot aggregate reads consistently across statements.
Use existing `_read_snapshot()` for multi-statement aggregate reads and propagate
its connection rather than opening independent reads. Preserve the backend's
existing `BEGIN IMMEDIATE` implementation; add tests for connection propagation
instead of introducing parallel locking helpers.

- [ ] Parameterize material/no-op tests for fields, tags, highlights, links and owned output changes; test fresh/legacy migration twice, restart, ID reuse and rollback. Add real two-connection tests with synchronization barriers (not sleeps) for child writes and coherent reads.
- [ ] Run new tests plus existing `test_reading_service.py`, `test_reading_highlights_api.py`, `test_reading_highlights_reanchor.py`, `test_reading_note_links_db.py`, `test_reading_import_export.py` and PostgreSQL coverage. Review then commit the exact changed paths with `feat(reading): persist guarded aggregate revisions (TASK-13153)`.

## Stage 2: Atomic relational deletion and ownership

**Status:** In Progress — internal guarded deletion and owned-file intents; reconciliation and production routing remain separate checkpoints.
**Goal:** Matching deletion removes exactly owned records or rolls back completely.
**Success Criteria:** Stale/missing/ownership cases mutate nothing; external records survive.
**Tests:** Atomic delete module, FTS regression and real PostgreSQL races.

**Interfaces:** Add `CollectionsDatabase.hard_delete_reading_item(item_id: int, *, expected_revision: int) -> bool`, returning whether artifact cleanup remains pending. Raise `KeyError` for absent/inaccessible/non-Reading items, `ReadingRevisionConflict` for stale tokens, and `ReadingArtifactOwnershipConflict` for unresolved ownership; define both exception classes in Collections_DB.py.

Current internal slice (existing ADR-003; no new ADR required): first cover exact
deletion, rejected preconditions, structurally owned outputs, ambiguous legacy
references, surviving shared files, staging cancellation and rollback phases on
both backends. Reuse `reading_artifact_paths` for pending disposal; persist intents
before removing output rows, with no file I/O or storage-lock acquisition in the
transaction. Clear collection-entry capture links, preserving their independent
source/Media records and containers. Propagate the transaction through strict FTS
deletion. Add deterministic mutation/delete commit-order checks. Keep public routes,
generic delete/purge routing, legacy reconciliation and readiness unactivated;
this checkpoint alone does not establish the complete contract.

Internal atomic-delete checkpoint (2026-09-04): the new DB primitive validates
positive revision, authority and Reading origin under the clock fence, rejects
unproven legacy ownership, reserves unshared owned files before output deletion,
removes item-local joins/highlights/FTS, clears independent collection-entry links,
deletes the exact parent revision and cancels outstanding staging atomically.
The clock remains spent after deletion. No filesystem calls or storage locks are
introduced inside this transaction. The existing cleanup worker consumes the
committed intents after restart; external Media and Notes are preserved.

Review found and regressions reproduced cleanup-authority loss from collapsing
case variants and treating separate known volumes as one shared path. Intents
now retain exact spellings; reference checks honor structural namespace through
cleanup retirement. Across different owners on the same/unknown volume, distinct
case spellings are ambiguous and reject deletion without mutation. Exact-path
shared references remain preserved. A final independent re-review found no
further issues in this internal slice. Existing ADR-003 governs these safeguards.

Checkpoint verification (existing Server virtual environment):

- Initial red run: 24 SQLite cases failed for the absent method; later case-variant,
  distinct-volume and cross-owner-alias regressions failed before their fixes.
- Initial green: 26 SQLite and 26 real PostgreSQL deletion cases passed.
- Final combined SQLite/POSIX regression: atomic deletion, archive adoption,
  artifact cleanup/storage, revision mutations and contentless FTS — 171 passed,
  149 deselected. No full suite.
- Supplemental real SQLite FTS SQL-failure check exercises the helper's exception
  handling itself (not only an exception injected after the helper returns);
  1 passed, 68 deselected.
- Final PostgreSQL atomic-delete/adoption/cleanup regression: 69 passed, 69
  deselected, no skips on the existing real service with Docker startup disabled.
  Together with the SQLite/POSIX run and supplemental FTS SQL-error test, this
  verifies 241 distinct targeted cases (earlier reruns overlap these cases).
- New test Ruff/Black, changed DB-range formatting, compilation and diff checks
  pass. DB Ruff retains the same nine baseline findings; scoped Bandit reports
  zero findings and zero errors.
- External preservation uses real Media records on the tested backend and a
  separate real SQLite Notes database, matching its independent storage boundary.
  An initial test mistakenly merged Notes and Media schema tables; corrected the
  fixture rather than changing production schemas. An unsupported test-only
  update argument was corrected to the existing upsert API.

Remaining Stage 2/3 work is not complete: explicit legacy reconciliation, generic
delete/purge and collection-link writer routing, archive production callers and
cleanup startup/readiness integration. No public endpoint or capability enabled.

- [x] Add failing tests for exact-match deletion, stale token, wrong user, non-Reading row, absent row, external preservation and ambiguous ownership. Core assertion:

```python
def test_stale_delete_keeps_newer_item(db):
    item = make_reading(db)
    changed = db.update_content_item(item.id, title="Newer")
    with pytest.raises(ReadingRevisionConflict):
        db.hard_delete_reading_item(item.id, expected_revision=item.revision)
    assert db.get_content_item(item.id).revision == changed.revision
```

- [x] Run `python -m pytest tldw_Server_API/tests/Collections/test_reading_atomic_delete.py -q`; verify the expected missing-method failure.
- [x] Under the Stage 1 fence validate owner/origin/revision and proven artifact ownership. Remove item-owned joins/highlights/FTS and outputs, reserve unlink intents for unshared files, and delete with the exact predicate:

```sql
DELETE FROM content_items
WHERE id = ? AND user_id = ? AND origin = 'reading' AND revision = ?;
```

Require rowcount one; any exception rolls back children, clock and intents. Do not call current best-effort purge helpers inside this transaction. Never delete external Media, Notes, shared tag definitions or containers.

- [ ] Inject failures after each relational phase. Assert every preexisting row remains and no intent escapes rollback. Test both mutation/delete orderings through two independent SQLite connections and PostgreSQL sessions; test blocked generic delete/child paths too.
- [ ] Implement dry-run-first legacy reconciliation with the following explicit interface:

```text
reading_reconcile_artifacts.py --user-id USER --item-id ITEM --storage-namespace NAMESPACE --manifest PATH
reading_reconcile_artifacts.py --apply PATH
```

Dry-run writes a mode-0600 manifest without creating associations. Each candidate
has item/output IDs, expected item revision, candidate-record fingerprint,
namespace and `confirmed=false`; the operator reviews local evidence and explicitly
sets confirmed entries. No title/content/path is logged. Apply validates all entries
under the existing database fence before writing any: require the same owner,
Reading origin, archive output type, matching fingerprint/revision, no conflicting
association and a previously verified namespace. Insert structural associations
and bump the parent once per compound change. It never deletes/moves records or
files. Already identical applied associations are no-ops; mismatches roll back all.

- [ ] Test legacy manual archives lacking parent references, earlier save archives,
dry-run non-mutation, unconfirmed entries, wrong user, unknown volume, edited
metadata after dry-run, mixed valid/invalid manifest rollback, and repeated apply.
Run `python -m pytest tldw_Server_API/tests/Collections/test_reading_artifact_reconciliation.py -q` through red/green before closing Stage 2.
- [ ] Run atomic-delete tests and `test_content_items_fts_contentless.py`, then relevant PostgreSQL cases. Review and commit exact paths with `feat(reading): atomically delete confirmed captures (TASK-13153)`.

## Stage 3: Durable artifact staging, adoption and cleanup

### Generic file-operation boundary investigation (2026-09-05)

After `2207a84fc1`, three isolated real-SQLite/direct-endpoint probes reproduced
the explicitly deferred filesystem gaps. Each used temporary user databases and
files, not a user's existing data; no production code changed during investigation.

| Interleaving / alias | Observed response | File outcome |
| --- | --- | --- |
| Register ownership after managed-only dispatch returns unowned, before rename | 409 | Original owned path already moved |
| Rename an unowned output referencing an owned source path | 200 | Managed source path moved |
| Rename an unowned output onto an owned destination path | 200 | Managed destination bytes overwritten |

Probe results: `/private/tmp/task-13153-file-fence-probes.log` (`PROBE` lines).
The existing 146-case immutability checkpoint does not cover these interleavings
and is not evidence that generic file operations are safe for rollout.

Root cause: PATCH performs path normalization and filesystem rename/conversion
before the final DB update. DB rollback cannot reverse a file operation, and an
unowned dispatch result does not reserve either path. The existing POSIX lock is
only used by Reading staging/adoption/cleanup; DB-only legacy ownership
registration and generic output file writers do not share that storage boundary.
Ownership/row mutations are centralized in Collections_DB.py, but coordinating
their short transactions with external file I/O needs a durable operation boundary.

Proposed next decision, **not yet approved or implemented**: preserve unmanaged
rename/conversion by extending durable path reservations plus the existing storage
lock to generic file mutations, with guarded ownership/attachment and crash
recovery. This requires an ADR/spec amendment and bounded schema/lifecycle design.
Do not insert another precheck, hold a DB mutation transaction across file I/O,
or silently disable/change previously approved unmanaged behavior. A narrower
behavioral restriction would instead require the user's explicit policy change.
Capability remains absent; TASK-13153 remains In Progress.

Subsequent user decision (2026-09-05): the reservation approach is approved.
The detailed amendment is now written in
`Docs/superpowers/specs/2026-09-05-reading-output-file-reservations-design.md`;
its independent whole-spec re-review passed with no remaining serious issues.
User review of the written spec is pending. No production code has changed. The
first bounded review identified uncertain commit acknowledgement, abort witness
cleanup ordering and missing source-identity evidence. The spec now requires
durable phase rereads, destination-before-witness abort cleanup and recorded
source fingerprints checked again before disposal. Implementation planning waits
for the reviewed written spec's user approval under the brainstorming workflow.

### Managed archive immutability checkpoint (approved 2026-09-05)

ADR required: yes. ADR path: `backlog/decisions/003-reading-atomic-hard-delete.md`.
Reason: the approved managed-file policy amends the existing ownership lifecycle;
no new ADR, schema or replacement lifecycle is needed for this checkpoint.

Files: `Collections_DB.py` owns transactional immutable-field enforcement and
managed metadata dispatch; `endpoints/outputs.py` dispatches before path
normalization or file access. Extend `test_reading_revision_mutations.py` and add
`tests/Collections/test_reading_output_updates.py` using real backend/HTTP fixtures.
Update both Reading API documents and this task's record after verification.

- [x] Add failing DB tests for compound path/format rejection, metadata-only
  updates, normalized no-ops, rollback and wrong-user/deleted outputs. Retain
  existing unowned rename/conversion tests; adjust older owned mutation cases
  to metadata-only operations to reflect the approved policy.
- [x] Add real-file HTTP tests: managed title + retention update leaves filename
  and bytes untouched and advances once; changed format + title/retention returns
  409 without any row/file changes; identical format is allowed; missing and
  foreign rows return 404. Inject rollback and check sanitized diagnostics.
  Verify unowned archive and generic rename/conversion still work.
- [x] Run those tests on SQLite and record the expected failures before coding.
- [x] Under the revision clock, reject actual owned `storage_path`/`format`
  changes with `ReadingArchiveFileImmutable`. Add managed-only update dispatch
  returning None for unowned rows; pass the same explicit connection to the
  existing update boundary. PATCH calls it before resolving any filesystem path
  and maps immutable requests to 409 `reading_archive_file_immutable`.
- [x] Run focused SQLite and real PostgreSQL tests with Docker disabled, plus
  existing output API regressions. Run scoped lint/format/Bandit, independent
  review and commit the checkpoint; keep TASK-13153 In Progress.

Scope limit: dispatch returning unowned is not a filesystem lease. A concurrent
ownership registration after dispatch and generic writes through managed aliases
remain required follow-up work, as do generated-file persistence/cleanup. Do not
claim complete managed-file safety or enable `hasReadingOptimisticDeletesV1` from
this checkpoint. No filesystem I/O may be moved into a DB mutation transaction.

Checkpoint evidence: the initial SQLite red run reproduced nine policy
failures (and two teardown errors propagating the same intentional file-access
assertions through TestClient). The first green run passed 52 cases. The expanded
SQLite/API run passed 92 cases, including the two additional single-connection and
late DB-ownership regressions. Plan and implementation reviews found no actionable
issues within this scope. Real PostgreSQL passed all 54 selected cases with no
skips (509.44s): **146 distinct targeted passes** total. No Docker provisioning or
full sweep was run. New/revised focused tests pass Ruff/Black; changed production
and legacy-test ranges pass formatting, compilation and diff checks. Scoped
Bandit reports zero findings and zero errors. The 14 existing Ruff findings
(9 DB, 1 endpoint, 4 legacy API test) match HEAD and remain outside this change.

Reproduce using the Server virtual environment and `TLDW_TEST_NO_DOCKER=1`:

```bash
python -m pytest tldw_Server_API/tests/Collections/test_reading_output_updates.py tldw_Server_API/tests/Collections/test_reading_revision_mutations.py tldw_Server_API/tests/Collections/test_items_and_outputs_api.py -k 'not postgres and (output or ownership)' --timeout=90 -q --tb=short
python -m pytest tldw_Server_API/tests/Collections/test_reading_output_updates.py tldw_Server_API/tests/Collections/test_reading_revision_mutations.py -k 'postgres and (output or ownership)' --timeout=30 -q --tb=short
```

Logs: `/private/tmp/task-13153-immutable-{red,green,sqlite-api,pg}.log` and
`/private/tmp/task-13153-immutable-bandit.json`. Existing ADR-003, the spec and both
API documentation copies record the approved policy. TASK-13153 remains In Progress;
the optimistic-delete capability remains absent. Next: fence generic filesystem
mutations against late ownership and managed source/target aliases; then resume
the remaining production archive/reconciliation/cleanup/readiness integration.

Automatic-purge follow-up (2026-09-05; existing ADR-003): Watchlist reads and
generation routes call `purge_expired_outputs()` without file-deletion permission.
Its default must skip managed Reading outputs under the same ownership fence,
including ownership registered after the scan. Preserve automatic unowned output
retention/grace cleanup and quota accounting. An explicit trusted managed-file
opt-in retains the DB primitive's disposal capability. Add failing default/explicit,
renewal, late-ownership and actual Watchlist-read regressions on both backends;
then implement, run targeted regressions/static checks, review and checkpoint.
This applies the approved policy; no new ADR or capability activation.

Automatic-purge checkpoint (2026-09-05): the database maintenance helper now
defaults to preserving managed archives, with an explicit trusted file-cleanup
opt-in. It reuses candidate discovery and the existing transactional deletion
primitive, so late ownership registration and retention renewal are rechecked
under the clock. Watchlist callers need no individual prechecks. Unowned expiry
and aged-soft-delete removal, actual counts and audiobook quota remain intact.
Both API documentation copies now state that automatic Watchlist maintenance does
not authorize managed archive deletion. Independent read-only review found no issues.

Verification (existing Server venv):

- Red: seven selected SQLite cases failed before the permission-default change;
  the separate late-ownership regression also failed before implementation.
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_reading_output_deletion.py -k 'not postgres' --timeout=30 -q --tb=short`: 26 passed, 26 deselected.
- Same module with `-k 'postgres and (purge or watchlist)' --timeout=30 -x`: 8 passed,
  44 deselected on real PostgreSQL, no skips and no Docker provisioning.
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_api.py -k 'items_and_outputs_flow or outputs_pagination_excludes_mixed_origin_rows' --timeout=90 -q --tb=short`: 2 passed, 35 deselected. The first attempt timed out during full-app fixture startup with a 30-second limit, before requests. The unchanged tests passed with a 90-second limit.
- Total: 36 distinct targeted passes; no full suite. Test Ruff/Black, changed DB
  formatting, compilation and diff checks pass. Scoped Bandit: zero findings and
  errors. DB Ruff retains its same nine baseline findings.

Rename/transcode/generated-output file operations, production archive routing,
reconciliation, cleanup startup/readiness and remaining aggregate writers/DTOs
are still pending. Capability remains absent and TASK-13153 stays In Progress.

**Status:** In Progress — local POSIX exclusion, staging/cleanup intents and guarded internal adoption implemented; owned-output disposal and production wiring remain.
**Goal:** No lost cleanup after crashes and no unlink of shared/reused paths.
**Success Criteria:** Retry/restart and writer/cleanup races pass; pending work remains observable.
**Tests:** New artifact module, existing Reading archive/API tests and output-service regressions.

Output-deletion DB slice (existing ADR-003, no new ADR): reuse the internal file
intent preparation for single-output disposal. Under the existing clock fence,
soft/hard deletion of an owned archive clears only a matching parent archive
reference and advances the parent once; soft deletion retains structural ownership,
hard deletion records unshared disposal before removing ownership/output rows.
Normalized replay is a no-op. Service bulk deletion and DB retention purge delegate
to this boundary, with retention eligibility rechecked under the clock. Keep
filesystem quota measurement outside transactions and preserve generic unowned
output behavior. Cover real-backend rollback/noops, shared siblings, stale item
tokens and retention-renewal races. Public file-first handlers and scheduler are
not activated for managed archives by this slice; their file-option semantics and
storage/readiness routing remain to be integrated before production adoption.

Output-deletion DB checkpoint (2026-09-04): `delete_output_artifact()` now fences
its current output and any Reading owner, preserves ownership on soft deletion,
queues durable unshared-file disposal on hard deletion, clears only the matching
parent archive reference with strict FTS refresh, and advances the surviving
parent once. Replays do not advance it. The extracted disposal helper excludes
only the exact output IDs being removed, so siblings sharing the same path keep
the file until the last reference is removed. Existing namespace and case-alias
safeguards remain. No filesystem work was moved into a DB transaction.

The bulk service delegates unique IDs through this boundary and reports actual
removals; mismatched user scope is rejected. Database retention purge selects
candidates, then rechecks the same backend-specific expiry/grace predicate inside
each deletion transaction. Renewal after selection survives. These are per-output
commits, not a new all-or-nothing multi-output transaction.

Verification and review:

- Red: 14 expected SQLite failures for missing parent revision/lifecycle behavior,
  ownership-FK failures in old hard-delete/purge SQL, and missing bulk scope guard;
  the existing foreign-output non-mutation case already passed.
- New module initial green: 20 SQLite cases and 20 real PostgreSQL cases passed,
  with no PostgreSQL skips and `TLDW_TEST_NO_DOCKER=1` against the existing service.
- Combined SQLite regression (new output deletion, atomic item deletion, adoption,
  cleanup, revision mutations, contentless FTS, existing output service/API and
  idempotency suites): 218 passed, 169 deselected. No full suite.
- Supplemental soft-delete grace checks on both backends: 4 passed, 40 deselected,
  no skips. Total distinct targeted evidence for this checkpoint: 242 cases;
  initial green and quota comparison reruns overlap that total.
- Four real SQLite quota comparisons showed unchanged bulk SQL and per-output
  delegation agree for metadata sizes, filesystem sizes and prior soft deletion.
  After the safety review blocked the initial broad helper-removal attempt, this
  evidence supported the narrower delegation edit. Final backend suites repeat
  those quota checks through the accepted delegation; existing unused legacy
  quota-helper definitions were left unchanged in this slice.
- New tests Ruff/Black, changed production-range Black, compilation and diff
  checks pass. Scoped Bandit has zero findings/errors. Production Ruff retains
  ten baseline findings (nine DB and one service import-order finding).
- Independent read-only review found no actionable issues in this slice.

Unfinished: public file-first delete/purge/transcode handlers and scheduler,
managed file-option semantics, bounded runtime cleanup/readiness, legacy
reconciliation and remaining aggregate writer routing. No capability activation.

Approved API/scheduler disposal slice (2026-09-05; ADR-003 amended):

1. Add failing real-backend tests for nonmutating metadata-only managed rejection,
   soft-delete compatibility, explicit deferred disposal, false-option purge skips,
   failed-transaction file preservation and locked retention renewal/custom grace.
2. Preserve the existing bool DB deletion interface for trusted internal callers;
   expose the committed deletion's path/managed classification to API services so
   they never rely on a pre-lock ownership/path snapshot for file disposal.
3. Enforce explicit managed file permission under the clock. Share the backend
   purge predicate for candidate discovery and in-transaction recheck.
4. Route single API deletion, API purge and scheduler through that boundary.
   Managed files remain for durable cleanup; unrelated files use confined,
   best-effort post-commit unlink. Report actual removals and update TTS history
   only for removed records. No new retained-file schema or dependency.
5. Run targeted SQLite/PostgreSQL and existing API/service regressions, scoped
   formatter/lint/Bandit checks, independent review, and commit a checkpoint.
   Other file mutators and full readiness remain later work; capability stays absent.

API/scheduler disposal implementation (2026-09-05): a committed `DeletedOutput`
snapshot carries the path, managed ownership and surviving-reference protection.
The existing bool interface remains available to trusted internal callers. Public
deletion supplies explicit file permission, enforced under the clock; metadata-only
managed hard deletion returns 409 and false-option purges skip it. Explicit managed
removal leaves durable cleanup to the namespace-aware worker. Generic file deletion
is confined and best effort after the DB commit, never before rollback can occur.
The API and scheduler use the same backend retention predicate for discovery and
locked recheck, honoring custom grace and include-retention selection. Counts reflect
actual removals; scheduler history updates follow only committed deletions. Existing
audiobook accounting is retained, with size measurement before mutation locks.

Review reproduced an unowned shared output acquiring Reading ownership after the
delete commit but before unlink. All surviving same-user file references now block
direct unlink, with conservative basename comparison for legacy absolute paths and
escaped LIKE literals. Another failing regression caught a resolver leaking a
rejected symlink filename through nested logging; resolver failure messages are now
static. Scoped re-review found no outstanding actionable issues. A later real PG
run exposed the query converter treating `LIKE ? ESCAPE` as JSONB syntax; prepared
statement inspection isolated it and `LIKE (?) ESCAPE` fixed the targeted PG cases.
The incident is recorded in the testing-evidence lessons, without changing the
shared SQL parser in this task.

Other file-mutating routes, cleanup startup/readiness, legacy reconciliation,
production archive adoption, aggregate/collection writer routing and DTO exposure
remain unfinished. This does not solve the pre-existing general unowned-file
attachment lifecycle or activate `hasReadingOptimisticDeletesV1`. No new retained-file
schema, dependency, PR or merge is included. ADR-003 carries the approved policy;
TASK-13153 remains In Progress.

API/scheduler checkpoint verification (existing Server virtual environment):

- Initial new-route red run: 13 expected SQLite failures and 2 compatibility passes.
  Shared-alias, both review findings and the legacy absolute alias each had failing
  regressions before their corrections. Existing log tests were updated to the new
  DB/service boundary rather than preserving fake raw-SQL behavior.
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_reading_output_disposal_routes.py tldw_Server_API/tests/Collections/test_reading_output_deletion.py tldw_Server_API/tests/Services/test_outputs_service.py tldw_Server_API/tests/Collections/test_items_and_outputs_api.py tldw_Server_API/tests/Services/test_outputs_purge_scheduler_truthiness.py -k 'not postgres' --timeout=30 -q --tb=short`: 100 passed, 42 deselected.
- After the final parenthesized SQL correction, reran all new-route SQLite cases:
  20 passed, 20 deselected (overlaps the preceding 100).
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_reading_output_disposal_routes.py tldw_Server_API/tests/Collections/test_reading_output_deletion.py -k postgres --timeout=30 -x -q --tb=short`: 42 passed, 42 deselected on the existing real PostgreSQL service; no skips or Docker provisioning. An earlier failing broad PG run was interrupted to isolate the query-conversion defect; only this final run is completion evidence.
- Total distinct targeted cases: 142. No full suite ran. Existing warnings remain;
  new/rewritten tests pass Ruff, changed ranges pass Black, compilation and diff
  checks pass. Scoped Bandit has zero findings/errors. Production Ruff retains the
  same 12 baseline findings (9 DB, 1 service, 1 scheduler, 1 endpoint); the existing
  API test module retains its same 4 baseline findings. No new lint findings.
- Read-only review and re-review completed, with both findings addressed.

**Interfaces:** Create `drain_reading_artifact_cleanup(db: CollectionsDatabase, *, storage_namespace_id: str, limit: int = 100) -> int` in the cleanup service; returns completed intent count. Reservations have a namespace, unique token, lease deadline and `staged|owned|pending` lifecycle. Adoption requires matching token, unexpired lease, surviving item and original revision. Worker claim and output registration serialize on the database fence. Namespace identity is provisioned and verified against a volume marker, not inferred from the database or hostname.

Storage prerequisite slice: implement explicit, idempotent namespace provisioning
and a fail-closed, nonblocking POSIX OS-lock context in the cleanup service before
adding unlink/adoption. Runtime access never creates a missing volume marker or
lock file; provisioning never rotates an existing identity or replaces a missing
lock beside an existing marker. Hold an open directory descriptor, reject symlink
or nonregular marker/lock files, verify the path still names the locked directory
and inode, and sanitize failures. Unsupported platforms fail closed, not through
stale-file locking. Test independent processes, busy/release, process-exit recovery,
missing/mismatched markers, distinct roots and replaced lock files. This does not
establish readiness for untested shared/network filesystems or enable capability.
ADR required: existing ADR-003 applies; no new lifecycle decision.

Storage prerequisite checkpoint (2026-09-04): the new cleanup-service module has
explicit namespace provisioning and runtime `reading_storage_lock()`. Provisioning
requires an existing output directory, persists a random opaque marker with file
and directory fsync, preserves identity on repeat and rejects a missing lock beside
an existing marker. Runtime never creates marker/lock/root state. It verifies
private regular single-link marker/lock files and directory/lock identity after
nonblocking POSIX flock acquisition. Missing/mismatched state fails unavailable;
contention fails busy for a later retry. No stale-lock-file fallback, dependency,
database I/O or diagnostic payload logging is added.

The real subprocess tests prove local exclusion and lock release after both normal
exit and explicit child termination while retaining the original lock inode.
The first crash test hung in its own event signaling after termination; a timed
stack trace isolated that test teardown defect and it was removed. Independent
review reproduced a provisioning retry bypassing a prior fsync failure. A failing
regression preceded the fix; existing-marker provisioning now syncs both marker and
directory before succeeding. Re-review found no outstanding actionable findings.
Both incidents are recorded in `backlog/docs/lessons-testing-evidence.md`.

Verification: `python -m pytest tldw_Server_API/tests/Collections/test_reading_artifact_storage.py --timeout=20 -q --tb=short`
passes 23 cases (no skips) on local macOS/POSIX with the Server virtual environment.
New files pass Ruff, Black and compilation; `git diff --check` passes. Scoped Bandit
reports zero findings/errors. No database code changed, so no new PostgreSQL evidence
is claimed; no full suite ran. Tests do not certify Windows or shared/network-volume
locking. Unsupported platform locking fails closed and capability remains absent.
This module has no production caller: staging reservations, adoption, unlink intents,
bounded retry drain, purge routing and readiness remain to be implemented. The
storage prerequisite was brought forward because relational deletion must not begin
file disposal before exclusion and namespace verification are available.

Next slice: persist unadopted staging reservations and pending cleanup in one
namespace/user/path-keyed table, with unique tokens, captured parent revision,
lease deadline and bounded retry metadata. Reserve before file creation; recheck
the token/lease/parent under the clock after taking the storage lock. Cleanup
transitions expired staging to pending while holding that same storage lock,
checks output references, unlinks outside DB transactions and retires only after
directory sync. Generic output insert/path-change writers must reject reserved
paths under the clock. Keep blocked collisions/invalid paths observable and do
not guess ownership. Test delayed writers after retirement, restart/retry,
shared-reference preservation and namespace isolation on both databases. Adoption,
owned-output/hard-delete intent creation and production routing remain later work.

Unadopted-artifact checkpoint (2026-09-04): `reading_artifact_paths` now persists
token, user, namespace, path, captured parent/revision, lease and staged/pending
state, plus sanitized bounded retry metadata. It deliberately survives parent
deletion and does not advance the capture revision before adoption. Reservation,
write validation, pending transition and retirement use the existing clock fence.
Exclusive creation rechecks the original token/revision after taking the storage
lock; expired/missing reservations cannot open a file. Cleanup holds that lock
through selection, descriptor-relative unlink, directory fsync and final DB
retirement, with no filesystem work inside a mutation transaction. Missing files
complete only on a verified namespace. Shared output references, including soft
deleted rows, are preserved. Invalid paths and file collisions remain blocked;
ordinary I/O failures retain a capped backoff and sanitized category.

Generic output insertion now uses one clock-first transaction (including
idempotency lookup), and insertion/path changes cannot attach to reservations.
Because generic outputs do not carry namespace authority yet, their path guard
conservatively covers all namespaces for that user. ASCII filename comparisons
are case-insensitive even on case-sensitive volumes to protect macOS aliases.
Existing API/idempotency regression suites passed with this guard.

Review found and reproduced root-replacement and case-alias gaps. File open,
stat/unlink and directory sync now use the held validated directory descriptor,
never a reopened root pathname. Protected marker/lock names also reject uppercase
aliases. Four failing initial review regressions and two protected-file alias
regressions preceded the fixes. Final scoped re-review found no outstanding issues.
The plan's illustrative Path.open/Path.unlink operations are therefore implemented
as descriptor-relative os.open/os.unlink, not literal pathname reopening.

Verification (Server venv; only targeted suites):

- Original missing-lifecycle red run: 8 SQLite failures before implementation.
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py tldw_Server_API/tests/Collections/test_reading_artifact_storage.py -k 'not postgres' --timeout=30 -q --tb=short`: 43 passed, 20 deselected.
- New cleanup module `-k postgres --timeout=30`: 17 passed, 17 deselected; added `-k 'postgres and (blocks_invalid_paths or bounded)'`: 5 passed, 35 deselected. This covers all 20 PostgreSQL lifecycle cases, with two rerun overlaps, no skips and no Docker startup.
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_reading_revision_mutations.py tldw_Server_API/tests/Collections/test_output_artifact_idempotency.py tldw_Server_API/tests/Collections/test_collections_schema_bootstrap.py tldw_Server_API/tests/Collections/test_items_and_outputs_api.py -k 'not postgres' -q --tb=short`: 123 passed, 80 deselected.
- New/touched service and tests pass Ruff/Black; changed DB ranges pass Black;
  compilation and diff checks pass. Scoped Bandit: zero findings/errors. DB Ruff
  retains its nine baseline findings. No full-suite or deployment-readiness claim.

Production adoption must recheck the token/lease/revision under the storage lock
and atomically replace staging with structural output ownership; it must not
bypass the new generic guard casually. This slice does not implement adoption,
legacy reconciliation, owned-output/hard-delete intent creation, purge routing or
startup-worker readiness. The service helpers have no production callers and the
capability remains absent. Existing ADR-003 applies; TASK-13153 remains In Progress.

Guarded-adoption slice: add a trusted DB adoption primitive which rechecks the
reservation's token, namespace, staged state, lease and original Reading revision
under the clock. In one transaction create the archive output, insert structural
ownership, merge the parent's archive reference, refresh its FTS entry, advance
once and remove staging. Owned lifecycle state is represented by the existing
output/ownership rows rather than a duplicate staged-table row. Add a combined
write-and-adopt service operation holding the same verified storage lock throughout
write/fsync and adoption or failed-adoption scheduling. Preserve failed/expired
staging for cleanup; never recreate the parent. Test rollback at each mutation
phase, stale/expired/cancelled completion, lock exclusion through commit and
unchanged external links on SQLite/PostgreSQL. Production archive endpoints and
purge remain unwired until owned-output disposal/readiness is complete. Existing
ADR-003 applies; no new architecture decision.

Guarded-adoption checkpoint (2026-09-04): a trusted database primitive rechecks
staging token/user/namespace/state/lease and the original parent revision under
the shared clock. It atomically inserts the archive output and structural
ownership, merges the parent's archive reference, refreshes FTS, advances the
parent once and removes staging. Existing fields, metadata, tags and external
Media/Note associations are preserved. The owned state lives in the existing
output/ownership tables, not a duplicate lifecycle record. The primitive itself
performs no filesystem work; it requires the trusted caller's completed write.

The combined write-and-adopt service holds the same verified storage lock and
directory descriptor across exclusive file creation, file/directory sync, DB
adoption and failed-adoption scheduling. Failure queues the private staged file;
if scheduling fails, its durable staged lease remains a recovery path. No parent
upsert or automatic newer-revision retry occurs. Repeating a successful consumed
token returns missing staging, without creating another archive or revision.
Production callers, owned-output disposal, reconciliation/purge integration and
readiness are still absent; capability remains absent and TASK-13153 In Progress.

The shared staged validator also refreshes wall time inside the DB fence, because
a caller's timestamp can predate waiting for that lock. A failing stale-prelock-time
regression preceded this correction. Independent scoped review found no actionable
issues in the adoption slice. Existing ADR-003 applies without a new decision.

Guarded-adoption verification (Server virtual environment):

- Initial missing-adoption red run: 13 SQLite failures. Stale-prelock lease regression also failed before the shared validator correction.
- `TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Collections/test_reading_artifact_adoption.py tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py tldw_Server_API/tests/Collections/test_reading_artifact_storage.py tldw_Server_API/tests/Collections/test_reading_revision_mutations.py tldw_Server_API/tests/Collections/test_content_items_fts_contentless.py -k 'not postgres' --timeout=30 -q --tb=short`: 137 passed, 115 deselected.
- New adoption module `-k postgres --timeout=30 -q --tb=short`: 15 passed, 15 deselected on the existing real PostgreSQL service, no skips and no Docker startup.
- After strengthening the FTS rollback injection to perform the real index update before raising, reran the adoption module on SQLite: 15 passed, 15 deselected; PostgreSQL rollback cases: 6 passed, 24 deselected, no skips. These reruns overlap the 152 distinct targeted cases above.
- Service/new tests pass Ruff/Black; changed DB ranges pass Black; compilation and diff checks pass. Scoped Bandit reports zero findings/errors. DB Ruff retains nine baseline findings. No full suite or production-readiness claim.

- [ ] Write a failing unlink-retry test: create a real archive in `tmp_path`, delete its parent, force `Path.unlink` to raise `PermissionError`, and assert the intent persists. Reopen the database and retry with unlink restored; assert file and intent are absent. Also test already-missing files as successful cleanup.
- [ ] Run `python -m pytest tldw_Server_API/tests/Collections/test_reading_artifact_cleanup.py -q` and verify missing lifecycle behavior.
- [ ] Persist the staging reservation before exclusive file creation. Use random filenames, existing path confinement and adoption under the revision fence:

```python
filename = f"reading_archive_{uuid.uuid4().hex}.md"
with destination.open("x", encoding="utf-8") as stream:
    stream.write(body)
```

Import `uuid` from stdlib; `destination` must come from existing validated user-output resolution, and `body` from existing bounded rendering. The file-open snippet runs only while holding the stable per-user storage lock, after rechecking its reservation under a short database transaction. Hold the storage lock until write/close and adoption or failed-adoption scheduling finish. Cleanup takes the same lock, rechecks state, unlinks outside the DB transaction, then records completion. Never delete/replace the lock file. Reject stale completion rather than upserting the item. Lease/token checks alone are insufficient.

- [ ] Add a deterministic barrier test for this exact sequence:

```text
writer reserves path, pauses before taking storage lock
cleanup takes storage lock, expires reservation, finds file absent, retires intent
writer resumes, takes storage lock, sees retired reservation, aborts before open
assert file absent and no owned output was created
```

Also pause the writer after open: cleanup must not unlink or declare success until
the lock releases. Kill a subprocess holding the lock and prove recovery without
deleting the lock file. Use standard-library OS locking appropriate to supported
platforms; unsupported filesystems/platforms fail readiness, never silently use
lease-only behavior.

- [ ] Persist namespace on reservations and intents; bind legacy records only via
verified upgrade/reconciliation. Test two distinct output roots using the same
PostgreSQL database: worker B must leave namespace A's intents unchanged. Remove
the volume marker to simulate an unmounted/replaced volume; cleanup stays pending
and new hard deletion returns unavailable. Restore the correct marker and prove
recovery. Shared-volume workers must pass the real interprocess lock test.

- [ ] Implement bounded worker claims and retries: missing file succeeds; permission/IO errors persist a sanitized category and capped exponential backoff; invalid paths stay blocked. Never include raw exception text or content in logs. No filesystem work while the original item transaction is open. Keep a path reservation while unlink runs so output registration cannot reuse it.
Missing-file success requires a validated volume marker and the storage lock; a
missing mount is not evidence of deletion. Worker claims do not substitute for that
lock. Namespace validation errors are sanitized and leave durable work pending.
- [ ] Start/stop the worker through existing cleanup lifecycle, independent of `OUTPUTS_PURGE_ENABLED`. Extend generic output registration/deletion only where necessary to honor managed-path reservations and owned-parent revisions.
- [ ] Add shared-reference, duplicate-worker, staging-crash, expired-lease, stale-adoption, invalid-path and filename-reuse race tests. Drain twice to prove idempotence. Run related output/Reading regressions on both database backends. Review and commit exact paths with `feat(reading): recover artifact cleanup after deletion (TASK-13153)`.

## Stage 4: HTTP contract, documentation and release gate

**Status:** Not Started
**Goal:** Clients see the complete, truthful guarded-delete capability.
**Success Criteria:** Real-factory HTTP tests and both backend suites pass; public docs match behavior.
**Tests:** Existing Reading API/error-log tests plus capability coverage.

**Interfaces:** `ReadingService.delete_item(item_id: int, *, expected_revision: int) -> bool` delegates to Stage 2. `ReadingItem.revision` is required and positive. HTTP success adds `artifact_cleanup: Literal["complete", "pending"]` while preserving existing success fields.

- [ ] Extend authenticated API tests to cover the spec response table and revision in list/detail/save/update DTOs. Assert pending success never reports failure or asks clients to repeat DELETE. Include at least one canonical factory/real SQLite HTTP flow.
- [ ] Run `python -m pytest tldw_Server_API/tests/Collections/test_reading_api.py tldw_Server_API/tests/Collections/test_reading_endpoint_error_logs.py -q`; confirm new contract tests fail before endpoint edits.
- [ ] Parse an optional positive query revision; require it only for hard deletion. Map exceptions explicitly, preserve archive semantics and existing authentication. Update DTO conversion and the service call:

```python
if hard and expected_revision is None:
    raise HTTPException(status_code=428, detail="reading_revision_required")
```

Return 409 for the two documented conflicts, 404 for inaccessible/missing rows,
503 `reading_delete_unavailable` when target-store readiness is unavailable, and
generic sanitized server errors for unexpected storage failures. Never retry with
a newer token automatically. Check readiness before any hard-delete mutation;
archive does not depend on hard-delete readiness.

- [ ] Update both Reading API docs with preconditions, errors, no-op semantics, pending cleanup, legacy ownership conflicts and stopped-writer upgrade requirement. Add the exact capability only after the complete storage/worker contract is active; test absent/false readiness states.
- [ ] Derive readiness from route enablement, schema/contract version, validated
namespace/lock support and registered healthy cleanup lifecycle. Never put this
flag in `_SHIPPED_CAPABILITIES` as true. Missing state, exceptions, failed startup
or stopped cleanup produce false; docs-info reads established state without I/O
or migrations. Where user stores differ, use a conservative deployment-wide claim
and independently recheck the authenticated target store at DELETE.
- [ ] Test startup before readiness, route disabled, schema mismatch, cleanup
startup failure, cleanup termination, namespace unavailable, capability derivation
exception, and successful recovery. For each unavailable case, assert docs-info
is false and a direct DELETE with a previously valid revision returns 503 without
changing rows or files. Keep recovery of preexisting intents operational when
new hard deletion is disabled. Document 503 and the reconciliation command in both
API documents and upgrade instructions.
- [ ] Run all touched new and existing focused suites, real PostgreSQL tests, formatter/linter checks and scoped Bandit using the Server virtual environment. Record commands, counts and backend availability. Do not claim success for skipped PostgreSQL tests or unavailable tools.
- [ ] Self-review AC1–AC6 against evidence, inspect `git diff --check`, and link ADR/spec/plan in task notes. Commit exact paths with `feat(reading): expose optimistic hard-delete contract (TASK-13153)` only after verification.
- [ ] Keep task In Progress until acceptance criteria and repository DoD are actually satisfied. Before any later merge, follow the Server policy requiring the requester’s own human-written Change summary; this plan does not authorize bypassing it.

## Plan review record

AC1 maps to Stage 1 and DTO coverage in Stage 4; AC2/AC3 to Stages 2–3;
AC4 to the real-backend and rollback checks throughout; AC5 to Stage 4;
AC6 to ADR-003. Additional reviewed safeguards are ID reuse, coherent snapshots,
staging recovery, ambiguous legacy ownership and reserved-path races.
Review amendments cover late file creation after intent retirement, storage-volume
identity, explicit legacy reconciliation, fail-closed capability/endpoint readiness,
and reuse of existing snapshot/transaction primitives. These add test requirements;
they are not claims that the proposed mechanisms have passed runtime verification.
The original review record describes proposed checks; dated checkpoint sections
above record implementation and verification evidence as execution progresses.
