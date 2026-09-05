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

**Status:** In Progress — revision schema/clock foundation implemented; writer integration remains.
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

- [ ] Add a real database fixture using `tmp_path` and the existing `CollectionsDatabase.from_backend` pattern. Use the following record constructor in the new test module:

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

- [ ] Run the new revision test and confirm failure is the absent revision behavior, not fixture setup.

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
