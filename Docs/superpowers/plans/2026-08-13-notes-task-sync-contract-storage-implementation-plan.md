# Notes Task Sync Contract And Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish strict dormant `notes.task`/`notes.task_activity` contracts, schema-v60 collision-safe task storage, forced PostgreSQL tenancy, separate canonical task revisions, and resumable readiness state without advertising either domain.

**Architecture:** ChaChaNotes remains the product authority. Schema v60 transactionally rebuilds the existing task graph into six owner/dataset-scoped graph tables plus one private owner-to-dataset authority relation, initially using a private local-unbound sentinel for legacy REST rows; explicit enrollment later records the immutable authority and rekeys the graph under the dataset fence. A shared contract module owns parsing and hashing, while public supported/writable capability lists remain unchanged.

**Tech Stack:** Python 3.11, Pydantic, SQLite, PostgreSQL, FastAPI Sync models, pytest, Ruff, Bandit.

**Design:** `Docs/superpowers/specs/2026-08-13-notes-task-activity-sync-design.md`

**Backlog task:** `TASK-13006.1`

**ADR required:** no
**ADR path:** `Docs/ADR/039-canonical-notes-task-sync-and-derived-checklist-projections.md`
**Reason:** ADR-039 already approves this slice's product schema, tenant identity, RLS, canonical revision ownership, and readiness contracts.

---

## File map

**Create**

- `tldw_Server_API/app/core/Sync/v2/notes_task_contract.py` — sole task/activity v1 parser, canonical JSON, legacy-event conversion, and hashes.
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_contract.py` — exact task/activity/legacy vectors and dormant-capability assertions.
- `tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py` — SQLite fresh/upgrade/rollback/collision/rekey storage contracts.
- `tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_postgres_tenancy.py` — live PostgreSQL two-owner RLS/catalog/plan proof.

**Modify**

- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` — schema v60 migration, exact current-catalog verification, store wiring.
- `tldw_Server_API/app/core/DB_Management/chacha/task_store.py` — owner/dataset predicates, canonical revision/hash, local-unbound binding/rekey helpers.
- `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py` — six owner/dataset graph policies plus one owner-only scope-authority policy.
- `tldw_Server_API/app/core/DB_Management/Sync_DB.py` — independent dormant readiness records and atomic state transitions.
- `tldw_Server_API/app/core/Sync/v2/models.py` — known-but-dormant domain types and discoverable private schemas; do not add to supported lists.
- `tldw_Server_API/app/core/Sync/v2/store.py` — readiness facade methods.
- `tldw_Server_API/app/core/Sync/v2/profile.py` — sanitized internal readiness status only; no writable advertisement.
- `tldw_Server_API/app/core/Notes_Tasks/service.py` — trusted indexed scope-authority lookup and scope forwarding.
- `tldw_Server_API/app/core/Notes_Tasks/reconciler.py` — trusted scope forwarding for legacy reconciliation.
- `tldw_Server_API/app/api/v1/endpoints/notes_tasks.py` — authenticated owner scope forwarding without wire changes.
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py` — authenticated MCP owner scope forwarding without tool-schema changes.
- `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py` — scoped CRUD and revision separation.
- `tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py` — exact policy-source contract.
- `tldw_Server_API/tests/Sync/test_sync_v2_models.py` — dormant domain/capability boundary.
- `tldw_Server_API/tests/Sync/test_sync_v2_store.py` — readiness state transitions and rollback.
- `tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py` — REST compatibility through the scoped store.
- `tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py` — MCP compatibility through the scoped store.
- `tldw_Server_API/tests/DB_Management/test_chacha_migration_v48_tasks.py` — v48 compatibility fixture removes v60-only tables when synthesizing old state.

### Task 0: Start the child and attach this plan

- [ ] **Step 1: Verify Backlog state and ADR link**

```bash
backlog task edit 13006.1 -s "In Progress"
backlog task 13006.1 --plain
```

Expected: TASK-13006.1 is In Progress and links ADR-039 plus this plan before any
production change.

### Task 1: Lock the exact v1 task/activity contract

- [ ] **Step 1: Write task payload RED tests**

Add exact valid vectors plus one rejection per boundary: lowercase UUIDv4, all
required nullable fields, title/description bounds, status/completed invariant,
real due dates, estimate syntax, recurrence combinations, owner-only assignee,
NFKC/casefold tags, reserved custom keys, JSON size/depth, extra fields, and
canonical hash stability.

```python
def test_task_v1_rejects_open_with_completed_at() -> None:
    payload = valid_task_payload(status="open", completed_at="2026-08-13T10:00:00Z")
    with pytest.raises(NotesTaskContractError, match="completion"):
        parse_notes_task_v1(payload, owner_user_id=OWNER_ID)


def test_task_hash_ignores_projection_row_version() -> None:
    parsed = parse_notes_task_v1(valid_task_payload(), owner_user_id=OWNER_ID)
    assert task_object_hash(parsed, revision=7, deleted=False) == EXPECTED_VECTOR
```

- [ ] **Step 2: Write activity and legacy conversion RED tests**

Cover exact event enums/old-new shapes, provenance binding inputs, revision-1 create,
revision-2 tombstone, `created_at_client` equality, changed stable-ID fingerprints,
idempotency-key hashing/removal, every legacy event mapping, and fail-closed unknown
legacy data.

```python
def test_legacy_status_changed_maps_to_completed() -> None:
    converted = convert_legacy_task_event(
        legacy_event(event_type="status_changed", old={"status": "open"}, new={"status": "done"})
    )
    assert converted.event_type == "completed"
    assert converted.old_value == {"status": "open"}
    assert converted.new_value == {"status": "done"}
```

- [ ] **Step 3: Run the focused RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_contract.py
```

Expected: collection/import failure because `notes_task_contract.py` is absent.

- [ ] **Step 4: Implement strict task parsing and task hashes**

Use frozen Pydantic models and one canonical JSON serializer. Do not duplicate
validation in adapters or stores.

```python
class NotesTaskV1Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: UUID4
    note_id: UUID4
    title: str
    description: str | None
    status: Literal["open", "done"]
    completed_at: str | None
    priority: Literal["low", "medium", "high"] | None
    due_date: str | None
    estimate: str | None
    recurrence: NotesTaskRecurrenceV1 | None
    assignee_id: str | None
    tags: tuple[str, ...]
    custom: dict[str, JsonValue]


class NotesTaskActivityTombstoneV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    note_id: UUID4
    task_id: UUID4 | None
    deleted_at: str
    delete_reason: Literal["user_request", "correction", "policy"]
```

Implement `parse_notes_task_v1(...)`, `parse_notes_task_tombstone_v1(...)`, and
`notes_task_object_hash(...)`. Return typed immutable values; hash adapter version,
revision, lifecycle, identity, and exact payload through canonical UTF-8 JSON. Never
include projection cache, REST row version, server cursor, or read state.

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_contract.py \
  -k 'task_v1 and not activity and not legacy'
```

- [ ] **Step 5: Implement strict activity parsing and legacy conversion**

Implement `parse_notes_task_activity_v1(...)`,
`parse_notes_task_activity_tombstone_v1(...)`,
`notes_task_activity_object_hash(...)`, and `convert_legacy_task_event(...)`. Use the
spec's exact old/new schemas for all five legacy source families, remove only a
validated raw idempotency key, and bind tombstone `deleted_at` to normalized
`SyncEnvelopeCreate.created_at_client`.

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_contract.py \
  -k 'activity or legacy or deleted_at or idempotency'
```

- [ ] **Step 6: Add known-but-dormant domain types**

Extend `SyncDomain` and private schema discovery so internal tests/bootstrap can
name `notes.task` and `notes.task_activity`. Keep both absent from
`SYNC_V2_SUPPORTED_DOMAINS`, `SYNC_V2_SUPPORTED_OPERATIONS`, public version maps,
factory advertisement checks, and selected-dataset writable maps.

- [ ] **Step 7: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  -k 'notes_task or notes_task_activity or supported_domain'
git add tldw_Server_API/app/core/Sync/v2/notes_task_contract.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py
git commit -m "feat(sync): define dormant Notes task contracts"
```

### Task 2: Migrate the SQLite task graph to schema v60

- [ ] **Step 1: Write migration RED tests**

Assert current version 60, fresh/59→60 parity, exact six graph schemas plus the
private scope-authority relation, composite
owner/dataset identities and FKs, `ON UPDATE CASCADE`, task canonical revision/hash,
event lifecycle columns, drift table, bounded indexes, integer storage classes,
post-DDL rollback, collision failure, and version-last behavior.

```python
def test_v59_to_v60_preserves_rows_as_local_unbound(tmp_path: Path) -> None:
    db = v59_database_with_task_graph(tmp_path)
    upgraded = CharactersRAGDB(db_path=db)
    task = upgraded.get_task_scoped(OWNER_ID, LOCAL_UNBOUND, TASK_ID)
    assert task["id"] == TASK_ID
    assert task["canonical_revision"] == task["version"]
    assert task["canonical_hash"].startswith("sha256:")
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py
```

Expected: current schema is 59 and v60 structures are absent.

- [ ] **Step 3: Add fixed SQLite v60 DDL and conversion helpers**

Define the six exact graph replacement table/index statements, the private
scope-authority relation, and pure row converters in `ChaChaNotes_DB.py`. Add unit
tests for canonical column/check/index SQL, source-row
conversion, local-unbound scope, and malformed-metadata diagnostics before invoking
the initializer.

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py \
  -k 'sqlite_ddl or converts_source or source_diagnostic'
```

- [ ] **Step 4: Implement transactional SQLite v60 migration**

Implement `_migrate_from_v59_to_v60_sqlite(conn)` and
`_verify_note_task_schema_sqlite(conn)`, then add the exact 59→60 initializer branch.
Under the existing initializer lock/transaction:

1. require exact v59 authority and required source tables;
2. reject any v60 target-table collision;
3. create six replacement graph tables plus the empty scope-authority relation with exact checks;
4. map each source row to its proven owner and private local-unbound sentinel;
5. compute canonical task hashes with the shared contract;
6. source-count/hash verify every table;
7. swap tables and recreate bounded indexes; and
8. compare-and-set schema version 59→60 last.

Do not consult Sync DB from this migration. Do not rewrite malformed metadata into
custom data; preserve product rows and mark readiness-blocking source diagnostics.
Add injected failures after create, copy, index creation, and verification; every
case must leave schema version 59 and the original tables intact.

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py \
  -k 'sqlite_upgrade or sqlite_rollback or sqlite_current_catalog or sqlite_concurrent'
```

- [ ] **Step 5: Add exact owner/dataset task-store methods and compatibility scope**

All canonical task/event/projection/read/reconciliation methods take owner and
dataset scope before IDs. Preserve existing REST/MCP behavior through one trusted
service compatibility adapter: authenticated callers pass owner explicitly and the
service resolves either the enrolled dataset or the private local-unbound sentinel.
Do not make endpoints or MCP tools accept a client-selected dataset. Update the REST,
MCP, and reconciler call sites in this step and prove their public schemas unchanged.
Add `bind_local_task_graph_to_dataset()` which verifies the complete source set,
rejects target collisions, updates the sentinel scope, and records the immutable
owner authority in one product transaction through cascading composite FKs. Empty
graphs still record the target; same-target replay is idempotent, different-target
rebind fails closed, and every shared task-store write must match the authority.
Normal compatibility resolution is one indexed owner lookup and performs no table
locks, catalog verification, RLS toggles, or graph scans.

```python
def get_task(
    self, *, owner_user_id: str, dataset_id: str, task_id: str,
    include_deleted: bool = False, conn: TaskConnection | None = None,
) -> dict[str, Any] | None:
    ...
```

- [ ] **Step 6: Run scoped-store compatibility RED/GREEN**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py \
  tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py \
  -k 'task or checklist or reconcile'
```

Expected: legacy REST and MCP behavior is unchanged, cross-owner direct store access
fails, and no caller can select the local-unbound sentinel.

- [ ] **Step 7: Prove canonical revision separation**

Add tests where `set_task_projection()` and `mark_task_unlinked()` retain their REST
row-version compatibility but do not change `canonical_revision/hash`; canonical
task mutation advances both once.

- [ ] **Step 8: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py \
  tldw_Server_API/tests/DB_Management/test_chacha_migration_v48_tasks.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py \
  tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/task_store.py \
  tldw_Server_API/app/core/Notes_Tasks/service.py \
  tldw_Server_API/app/core/Notes_Tasks/reconciler.py \
  tldw_Server_API/app/api/v1/endpoints/notes_tasks.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py \
  tldw_Server_API/tests/DB_Management/test_chacha_migration_v48_tasks.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py \
  tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py
git commit -m "feat(notes): add scoped task schema v60"
```

### Task 3: Enforce and verify PostgreSQL tenancy

- [ ] **Step 1: Write server-free catalog and live RLS RED tests**

Cover schema-version lock order, fixed relation locks, exact columns/defaults/types,
validated PK/FKs/checks, index definitions/predicates, table/schema owner, ENABLE +
FORCE RLS, sole exact policies, rollback/no-version-bump, two-owner same-ID
coexistence, cross-owner get/list/update/read-state denial, and index-backed pages.

- [ ] **Step 2: Run RED with PostgreSQL required**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_postgres_tenancy.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py \
  -k 'note_task or task_event or task_projection'
```

Expected: fail, never skip; v60 PostgreSQL DDL/RLS is absent.

- [ ] **Step 3: Add PostgreSQL v60 fixed DDL and server-free catalog verifier**

Add bounded, fixed catalog queries and exact expected column/default/type/constraint/
index/policy sets. First turn the server-free catalog drift matrix GREEN; include
extra-policy, OR-true check, weakened partial predicate, wrong FK scope, and PG18
NOT-NULL catalog rows.

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py \
  -k 'postgres_catalog or postgres_sql or rls_policy'
```

- [ ] **Step 4: Implement PostgreSQL v60 migration ordering**

Implement `_migrate_from_v59_to_v60_postgres(conn)` and
`_verify_note_task_schema_postgres(conn)`, then add the exact 59→60/current-60
initializer branches. Follow the reviewed v59 ordering: schema-version row lock first, fixed relation
locks, catalog-owner/RLS verification, exact temporary FORCE handling only when
needed, validate before DDL, transactional rebuild/copy/swap, restore FORCE, install
the six graph policies plus the owner-only scope-authority policy, exact verify,
then version bump last. Current-v60
startup locks and verifies the full catalog; it never repairs drift silently.

- [ ] **Step 5: Add all graph and scope-authority forced-RLS policies**

Use owner/dataset predicates plus owned-note/task/event EXISTS checks in both USING
and WITH CHECK for the six graph relations. Use an owner-only predicate for the
private scope-authority relation. Enumerate all policies and reject extra permissive policies.

- [ ] **Step 6: Run live GREEN and commit**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_postgres_tenancy.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_postgres_tenancy.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
git commit -m "feat(notes): enforce task graph PostgreSQL tenancy"
```

### Task 4: Add dormant readiness state and source diagnostics

- [ ] **Step 1: Write readiness RED tests**

Test independent `notes_task_v1` and `notes_task_activity_v1` rows through
`not_enrolled`, `enrolling`, `bootstrapping`, `verifying`, `ready`, and `blocked`;
monotonic cursor/count/fingerprint; atomic coupled-capture flag; rollback; malformed
state; local-unbound rejection; and sanitized diagnostics. Assert capabilities omit
both domains in every state, including forged ready metadata.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  -k 'notes_task_readiness or notes_task_activity_readiness or dormant_task_domain'
```

- [ ] **Step 3: Implement one bounded transition helper**

Store readiness in dataset metadata through one transaction-held resolver. Validate
state transitions, cursor monotonicity, fixed reason codes, count/hash shapes, and
atomic `task_activity_capture_enabled` changes. Facade methods delegate; they do not
mutate raw metadata independently.

- [ ] **Step 4: Expose sanitized internal status without capability support**

Profile/bootstrap diagnostics may report state, counts, hashed cursor, and reason
code to an authorized owner. They must not return task text, event values, raw
Markdown, actor data, or local-unbound identifiers.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  -k 'notes_task or notes_task_activity or capability'
git add tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/profile.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py
git commit -m "feat(sync): persist dormant task readiness"
```

### Task 5: Run the PR gate and close TASK-13006.1

- [ ] **Step 1: Run the required test matrix**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_postgres_tenancy.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_service.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
```

Expected: all selected tests pass and live PostgreSQL tests report no skip.

- [ ] **Step 2: Run static/security checks**

```bash
PRODUCTION_PATHS=(
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
  tldw_Server_API/app/core/DB_Management/chacha/task_store.py
  tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py
  tldw_Server_API/app/core/DB_Management/Sync_DB.py
  tldw_Server_API/app/core/Sync/v2/notes_task_contract.py
  tldw_Server_API/app/core/Sync/v2/models.py
  tldw_Server_API/app/core/Sync/v2/store.py
  tldw_Server_API/app/core/Sync/v2/profile.py
  tldw_Server_API/app/core/Notes_Tasks/service.py
  tldw_Server_API/app/core/Notes_Tasks/reconciler.py
  tldw_Server_API/app/api/v1/endpoints/notes_tasks.py
  tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py
)
../../.venv/bin/ruff check --no-cache "${PRODUCTION_PATHS[@]}"
../../.venv/bin/bandit -q "${PRODUCTION_PATHS[@]}"
PYTHONPYCACHEPREFIX=/tmp/task13006-pycache ../../.venv/bin/python -m py_compile "${PRODUCTION_PATHS[@]}"
git diff --check
```

If a legacy whole-file Ruff baseline is encountered, record the exact pre-existing
codes and run the established file-specific ignore only for that file; Bandit and
`py_compile` still cover every production path above.

- [ ] **Step 3: Verify capability omission and self-review**

Inspect the full diff for accidental additions to public supported/writable maps,
unscoped ID predicates, raw diagnostics, schema repair, or later-PR adapter logic.

- [ ] **Step 4: Update Backlog and commit closeout docs**

Check every TASK-13006.1 AC/DoD item, record commands/results and the live PostgreSQL
server identity, add concise implementation notes, set Done, then commit only the
task file and any directly relevant documentation.
