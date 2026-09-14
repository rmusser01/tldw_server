# Dormant Notes Task Sync Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete strict `notes.task` v1 adapter, product materializer, resumable task bootstrap, and dormant server-origin capture primitives without exposing either task domain.

**Architecture:** The adapter owns exact lineage/conflict decisions; the materializer writes owner/dataset-scoped ChaChaNotes rows and canonical revision/hash under the shared dataset fence. Bootstrap uses trusted source verification and stable task IDs. Factory wiring exists for internal bootstrap/tests only; public support, enrollment, and writable capabilities remain absent until TASK-13006.4.

**Tech Stack:** Python 3.11, Pydantic contracts from TASK-13006.1, Sync v2 adapters/materializers, SQLite, PostgreSQL, pytest, Ruff, Bandit.

**Design:** `Docs/superpowers/specs/2026-08-13-notes-task-activity-sync-design.md`

**Backlog task:** `TASK-13006.2`

**ADR required:** no
**ADR path:** `Docs/ADR/039-canonical-notes-task-sync-and-derived-checklist-projections.md`
**Reason:** This plan directly implements ADR-039's already-approved task-domain lifecycle without changing its boundary.

---

## File map

**Create**

- `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_task.py` — task lineage, authorization context, restore, and stable conflicts.
- `tldw_Server_API/app/core/Sync/v2/materializers/notes_task.py` — idempotent scoped product projection and postcondition verification.
- `tldw_Server_API/app/core/Sync/v2/notes_task_bootstrap.py` — resumable trusted task capture/count/fingerprint verification.
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_adapter.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_materializer.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_bootstrap.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_capture.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_postgres_contract.py`

**Modify**

- `tldw_Server_API/app/core/Sync/v2/domain_adapters/__init__.py`
- `tldw_Server_API/app/core/Sync/v2/materializers/__init__.py`
- `tldw_Server_API/app/core/Sync/v2/adapters.py` — context callbacks for authorized note/task heads.
- `tldw_Server_API/app/core/Sync/v2/factory.py` — internal adapter/materializer/bootstrap wiring with public-support guard unchanged.
- `tldw_Server_API/app/core/Sync/v2/service.py` — bounded context reads and dormant bootstrap acceptance.
- `tldw_Server_API/app/core/Sync/v2/profile.py` — invoke task bootstrap only through private readiness path.
- `tldw_Server_API/app/core/DB_Management/chacha/task_store.py` — exact canonical task CAS/postcondition helpers.
- `tldw_Server_API/app/core/Notes_Tasks/service.py` — optional, uninjected canonical task capture seam.
- `tldw_Server_API/tests/Sync/test_sync_v2_factory.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- `tldw_Server_API/tests/Notes_Tasks/unit/test_service.py`

### Task 0: Start the child and attach this plan

- [ ] **Step 1: Move TASK-13006.2 In Progress**

```bash
backlog task edit 13006.2 -s "In Progress" --plan $'1. Implement strict notes.task lineage.\n2. Materialize canonical tasks with split-commit repair.\n3. Bootstrap legacy tasks in bounded resumable pages.\n4. Add dormant capture without public capability.\n5. Run SQLite/live-PostgreSQL/static gates.\nDetailed plan: Docs/superpowers/plans/2026-08-13-notes-task-sync-lifecycle-implementation-plan.md\nADR required: no; ADR-039 applies.'
backlog task 13006.2 --plain
```

Expected: TASK-13006.1 is Done, TASK-13006.2 is In Progress, and the approved plan
and ADR are linked before production work.

### Task 1: Implement strict task adapter lineage

- [ ] **Step 1: Write adapter RED tests**

Cover create-empty-head, exact-base update, completion/reopen, metadata and
recurrence-state changes, tombstone, explicit restore, ordinary upsert against
tombstone, stale base, changed task/note identity, deleted/missing/foreign note,
malformed routing metadata, and exact replay.

```python
def test_task_restore_requires_exact_tombstone_base() -> None:
    result = adapter.evaluate(
        task_envelope(operation="upsert", restore_intent=True, base_hash=WRONG_HASH),
        dataset,
        context_with_task_head(tombstone_head()),
    )
    assert result.status == "conflict"
    assert result.error_code == "notes_task_restore_base_mismatch"
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_adapter.py
```

- [ ] **Step 3: Implement `NotesTaskDomainAdapter`**

Parse only through `notes_task_contract.py`. Require adapter version 1 and
`upsert|tombstone`. Bind object ID to payload task ID and parent ID to note ID.
Reuse `tldw_Server_API/app/core/Sync/v2/domain_adapters/_lineage.py` base comparison
helpers; do not add a parallel lineage algorithm.

```python
class NotesTaskDomainAdapter:
    domain = "notes.task"

    def evaluate(
        self, envelope: SyncEnvelopeCreate, dataset: SyncDataset, context: AdapterContext
    ) -> AdapterResult:
        payload = parse_notes_task_envelope(envelope, owner_user_id=context.owner_user_id)
        note = context.get_authorized_note(payload.note_id)
        head = context.get_head("notes.task", str(payload.task_id))
        return evaluate_task_lineage(envelope, payload, note=note, head=head)
```

Return stable sanitized conflict/error codes; never include title, metadata, note
existence, or another owner's identity.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_adapter.py
git add tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_task.py \
  tldw_Server_API/app/core/Sync/v2/domain_adapters/__init__.py \
  tldw_Server_API/app/core/Sync/v2/adapters.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_adapter.py
git commit -m "feat(sync): add strict Notes task adapter"
```

### Task 2: Project task envelopes idempotently

- [ ] **Step 1: Write materializer RED tests**

Test create/update/complete/reopen/tombstone/restore; canonical vs REST revisions;
same-cursor replay; lower/divergent cursor; exact product postcondition repair;
owner/dataset/note predicates; task ID collision across datasets; transaction
split after product commit but before Sync apply-status commit; idempotent replay
repairs that split without duplicating product state; and no activity side effect.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_materializer.py
```

- [ ] **Step 3: Add focused task-store CAS helpers and turn store tests GREEN**

Implement `apply_sync_task_create`, `apply_sync_task_upsert`,
`apply_sync_task_tombstone`, `apply_sync_task_restore`, and
`verify_sync_task_postcondition`. Each receives the existing product transaction,
owner, dataset, exact base/head, canonical revision/hash, and payload. It never
records an event; TASK-13006.4 coordinates the task/activity group.

- [ ] **Step 4: Implement product commit then Sync-status repair**

Follow `notes_link.py` materializer transaction/error mapping. Acquire no process
mutex; use the service's dataset materialization transaction and bound product
connection. The product DB and Sync DB do not share a transaction: commit the product
row, then mark Sync apply status. On replay after a split, verify the exact product
postcondition and advance Sync status without rewriting or duplicating the task.

- [ ] **Step 5: Run the injected split-commit matrix**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_materializer.py \
  -k 'split_commit or postcondition or replay or rollback_product_failure'
```

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_materializer.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py \
  -k 'sync_task or canonical_revision or projection_revision'
git add tldw_Server_API/app/core/DB_Management/chacha/task_store.py \
  tldw_Server_API/app/core/Sync/v2/materializers/notes_task.py \
  tldw_Server_API/app/core/Sync/v2/materializers/__init__.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_materializer.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py
git commit -m "feat(notes): materialize canonical task envelopes"
```

### Task 3: Bootstrap existing tasks resumably

- [ ] **Step 1: Write bootstrap RED tests**

Cover rejection of local-unbound/unbound input, an already-bound authorized dataset,
stable IDs, keyset pages, trusted bootstrap routing, legacy metadata validation,
concurrent capture supplied by a test harness,
resume after each injected failure, source drift, count/fingerprint mismatch,
blocked diagnostics, exact replay, and final task-ready while activity remains not
ready. Public capability maps must still omit both domains.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_bootstrap.py
```

- [ ] **Step 3: Add bounded source paging and stable-ID helpers**

Add an owner/dataset-scoped product page ordered by stable task ID with a hard limit
of 500. Add pure helpers for bootstrap envelope ID, trusted routing metadata, and
aggregate fingerprint. Turn page-boundary and stable-replay vectors GREEN before
service wiring.

Implementation targets: `TaskStore.page_tasks_for_sync_bootstrap(...)`,
`_task_bootstrap_envelope_id(...)`, `_task_bootstrap_routing(...)`, and
`_task_bootstrap_fingerprint(...)`.

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_bootstrap.py \
  -k 'page or stable_id or fingerprint or routing'
```

- [ ] **Step 4: Implement one bounded bootstrap page**

Use a maximum page of 500 and require that the product graph is already bound to the
authorized real dataset. Reject the local-unbound sentinel; this dormant child neither
enrolls nor rekeys production data. TASK-13006.4 owns product rekey and capture-before-
scan orchestration. For each task, build one stable trusted bootstrap envelope whose
ID derives from bootstrap ID + task ID + canonical hash. Capture with the existing
batch primitive and a source verifier; never directly insert accepted envelopes.

```python
class NotesTaskBootstrapper:
    PAGE_LIMIT = 500

    def bootstrap(self, *, service: SyncV2Service, dataset: SyncDataset) -> SyncDataset:
        ...
```

Each call source-verifies, appends at most one page through the existing batch
primitive, and persists the next key plus running count/fingerprint. An injected
split after Sync append resumes by exact replay; it never deletes accepted history.

- [ ] **Step 5: Reconcile final count and fingerprint**

Compare owner/dataset product count and aggregate fingerprint with applied
`notes.task` bootstrap envelopes under the dataset fence. Set only the task readiness
row to ready; coupled writable readiness remains false. Source drift or mismatch is
fail-closed and records only bounded row IDs/reason codes.

- [ ] **Step 6: Wire private bootstrap/factory paths**

Register the strict adapter/materializer internally and validate their presence in
the factory, but keep public supported-domain constants unchanged. Profile may call
the bootstrapper only through the private task readiness operation; the public
domain request validator still rejects `notes.task`.

- [ ] **Step 7: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_factory.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py
git add tldw_Server_API/app/core/Sync/v2/notes_task_bootstrap.py \
  tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/app/core/Sync/v2/profile.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_factory.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py
git commit -m "feat(sync): bootstrap dormant Notes tasks"
```

### Task 4: Add dormant server-origin task capture primitives

- [ ] **Step 1: Write capture RED tests**

Inject a capture callback into `NotesTaskService` and prove create/update/status/
delete/restore inputs produce exact canonical task steps with stable IDs and bases.
Prove no callback preserves legacy behavior, a callback failure rolls back the
product transaction where the seam is used, and no endpoint/factory injects it yet.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_capture.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_service.py \
  -k 'task_capture or legacy_without_sync'
```

- [ ] **Step 3: Add the smallest capture seam**

Define one protocol/callback that accepts the canonical before/after task mutation,
actor context, operation, and stable idempotency key and returns a product mutation
result. Do not build activity/note groups here; TASK-13006.4 owns expansion. Keep the
default `None`, so shipped REST/MCP behavior is unchanged.

- [ ] **Step 4: Add PostgreSQL source/plan proof**

In the server-free PostgreSQL contract tests, assert every materializer/store query
binds owner+dataset before task ID, task rows are locked before CAS, and task pages
use the v60 index. In the live test, execute create/update/delete/restore under a
non-superuser, non-BYPASSRLS role for two owners.

- [ ] **Step 5: Run GREEN and commit**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_postgres_contract.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_service.py
git add tldw_Server_API/app/core/Notes_Tasks/service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_postgres_contract.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_service.py
git commit -m "feat(notes): add dormant task capture seam"
```

### Task 5: Run the PR gate and close TASK-13006.2

- [ ] **Step 1: Run focused and regression tests**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_adapter.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_postgres_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_factory.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_service.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py
```

Expected: all pass; live PostgreSQL tests do not skip.

- [ ] **Step 2: Run static/security checks**

```bash
PRODUCTION_PATHS=(
  tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_task.py
  tldw_Server_API/app/core/Sync/v2/domain_adapters/__init__.py
  tldw_Server_API/app/core/Sync/v2/materializers/notes_task.py
  tldw_Server_API/app/core/Sync/v2/materializers/__init__.py
  tldw_Server_API/app/core/Sync/v2/notes_task_bootstrap.py
  tldw_Server_API/app/core/Sync/v2/adapters.py
  tldw_Server_API/app/core/Sync/v2/factory.py
  tldw_Server_API/app/core/Sync/v2/service.py
  tldw_Server_API/app/core/Sync/v2/profile.py
  tldw_Server_API/app/core/DB_Management/chacha/task_store.py
  tldw_Server_API/app/core/Notes_Tasks/service.py
)
../../.venv/bin/ruff check --no-cache "${PRODUCTION_PATHS[@]}"
../../.venv/bin/bandit -q "${PRODUCTION_PATHS[@]}"
PYTHONPYCACHEPREFIX=/tmp/task13006-pycache ../../.venv/bin/python -m py_compile "${PRODUCTION_PATHS[@]}"
git diff --check
```

- [ ] **Step 3: Self-review the dormant boundary**

Prove public models/endpoints/capabilities/enrollment still reject or omit both
domains. Reject any change that records activity directly or rewrites projections;
those belong to later child tasks.

- [ ] **Step 4: Complete Backlog notes and status**

Check TASK-13006.2 AC/DoD, record exact commands/live PG evidence, document any plan
deviation, add implementation notes, and set Done.
