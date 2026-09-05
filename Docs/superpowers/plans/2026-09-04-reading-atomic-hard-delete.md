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

**Status:** In Progress — revision schema/clock, item/tag, note-link and highlight writers implemented; output ownership integration and DTO coverage remains.
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

**Status:** Not Started
**Goal:** Matching deletion removes exactly owned records or rolls back completely.
**Success Criteria:** Stale/missing/ownership cases mutate nothing; external records survive.
**Tests:** Atomic delete module, FTS regression and real PostgreSQL races.

**Interfaces:** Add `CollectionsDatabase.hard_delete_reading_item(item_id: int, *, expected_revision: int) -> bool`, returning whether artifact cleanup remains pending. Raise `KeyError` for absent/inaccessible/non-Reading items, `ReadingRevisionConflict` for stale tokens, and `ReadingArtifactOwnershipConflict` for unresolved ownership; define both exception classes in Collections_DB.py.

- [ ] Add failing tests for exact-match deletion, stale token, wrong user, non-Reading row, absent row, external preservation and ambiguous ownership. Core assertion:

```python
def test_stale_delete_keeps_newer_item(db):
    item = make_reading(db)
    changed = db.update_content_item(item.id, title="Newer")
    with pytest.raises(ReadingRevisionConflict):
        db.hard_delete_reading_item(item.id, expected_revision=item.revision)
    assert db.get_content_item(item.id).revision == changed.revision
```

- [ ] Run `python -m pytest tldw_Server_API/tests/Collections/test_reading_atomic_delete.py -q`; verify the expected missing-method failure.
- [ ] Under the Stage 1 fence validate owner/origin/revision and proven artifact ownership. Remove item-owned joins/highlights/FTS and outputs, reserve unlink intents for unshared files, and delete with the exact predicate:

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

**Status:** Not Started
**Goal:** No lost cleanup after crashes and no unlink of shared/reused paths.
**Success Criteria:** Retry/restart and writer/cleanup races pass; pending work remains observable.
**Tests:** New artifact module, existing Reading archive/API tests and output-service regressions.

**Interfaces:** Create `drain_reading_artifact_cleanup(db: CollectionsDatabase, *, storage_namespace_id: str, limit: int = 100) -> int` in the cleanup service; returns completed intent count. Reservations have a namespace, unique token, lease deadline and `staged|owned|pending` lifecycle. Adoption requires matching token, unexpired lease, surviving item and original revision. Worker claim and output registration serialize on the database fence. Namespace identity is provisioned and verified against a volume marker, not inferred from the database or hostname.

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
No implementation or test results are claimed by this document.
