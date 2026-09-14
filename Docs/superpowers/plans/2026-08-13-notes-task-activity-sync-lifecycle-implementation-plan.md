# Dormant Notes Task Activity Sync Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement immutable `notes.task_activity` v1 evaluation, product materialization, exact legacy-event bootstrap, and dormant task-transition capture without advertising or enrolling either task domain.

**Architecture:** Activity is an immutable audit stream, not a mutable task projection. The adapter accepts create revision 1 and one-way tombstone revision 2 only; corrections are new stable event IDs. Bootstrap transforms the five legacy event families exactly and pages by `(created_at, id)`. Capture derives stable activity IDs from the accepted task transition, but remains private until TASK-13006.4 can atomically append the full task/activity/note plan.

**Tech Stack:** Python 3.11, Pydantic contracts from TASK-13006.1, Sync v2 adapters/materializers, SQLite, PostgreSQL, pytest, Ruff, Bandit.

**Design:** `Docs/superpowers/specs/2026-08-13-notes-task-activity-sync-design.md`

**Backlog task:** `TASK-13006.3`

**ADR required:** no
**ADR path:** `Docs/ADR/039-canonical-notes-task-sync-and-derived-checklist-projections.md`
**Reason:** This plan directly implements ADR-039's approved immutable activity lifecycle and dormant activation boundary.

---

## File map

**Create**

- `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_task_activity.py` — immutable lifecycle and stable conflict decisions.
- `tldw_Server_API/app/core/Sync/v2/materializers/notes_task_activity.py` — idempotent owner/dataset-scoped event projection.
- `tldw_Server_API/app/core/Sync/v2/notes_task_activity_bootstrap.py` — exact legacy conversion and resumable capture.
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_adapter.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_materializer.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_bootstrap.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_capture.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_postgres_contract.py`

**Modify**

- `tldw_Server_API/app/core/Sync/v2/domain_adapters/__init__.py`
- `tldw_Server_API/app/core/Sync/v2/materializers/__init__.py`
- `tldw_Server_API/app/core/Sync/v2/adapters.py` — bounded activity-head and note/task authorization context.
- `tldw_Server_API/app/core/Sync/v2/factory.py` — private adapter/materializer/bootstrap wiring only.
- `tldw_Server_API/app/core/Sync/v2/service.py` — private bootstrap and capture repair seam.
- `tldw_Server_API/app/core/Sync/v2/profile.py` — private readiness invocation without capability exposure.
- `tldw_Server_API/app/core/DB_Management/chacha/task_store.py` — exact event identity, cursor ordering, and postcondition helpers.
- `tldw_Server_API/app/core/Notes_Tasks/service.py` — optional dormant transition-capture callback.
- `tldw_Server_API/tests/Sync/test_sync_v2_factory.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- `tldw_Server_API/tests/Notes_Tasks/unit/test_service.py`

### Task 0: Start the child and attach this plan

- [ ] **Step 1: Move TASK-13006.3 In Progress**

```bash
backlog task edit 13006.3 -s "In Progress" --plan $'1. Enforce immutable notes.task_activity lineage.\n2. Materialize activity by Sync cursor and stable ID.\n3. Bootstrap exact legacy events resumably.\n4. Add complete dormant transition capture and repair.\n5. Run SQLite/live-PostgreSQL/static gates.\nDetailed plan: Docs/superpowers/plans/2026-08-13-notes-task-activity-sync-lifecycle-implementation-plan.md\nADR required: no; ADR-039 applies.'
backlog task 13006.3 --plain
```

Expected: TASK-13006.2 is Done, TASK-13006.3 is In Progress, and its plan/ADR are
linked before production work.

### Task 1: Enforce immutable activity lineage

- [ ] **Step 1: Write adapter RED tests**

Cover create revision 1, exact replay, changed stable-ID reuse, task-less note event,
same-note task event, missing/foreign note or task, canonical old/new values,
server-bound provenance, exact tombstone revision 2, repeated tombstone, attempted
update/restore, and correction as a new event ID. Under untrusted client origin,
reject lifecycle events (`created`, `updated`, `completed`, `reopened`, `deleted`,
`restored`, projection events) and accept only `corrected` with an existing authorized
same-owner/dataset/note/task `corrects_activity_id`. Trusted coordinator/bootstrap
origins may create the closed lifecycle event set.

```python
def test_activity_rejects_changed_stable_id_reuse() -> None:
    result = adapter.evaluate(
        activity_envelope(event_id=EVENT_ID, new_value={"status": "done"}),
        dataset,
        context_with_activity_head(activity_head(event_id=EVENT_ID)),
    )
    assert result.status == "conflict"
    assert result.error_code == "notes_task_activity_identity_reused"
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_adapter.py
```

Expected: collection or behavior failures because the activity adapter is absent.

- [ ] **Step 3: Implement one strict adapter**

Parse only through `notes_task_contract.py`. Bind object ID to activity ID and parent
ID to note ID. Require the optional task to be in the same owner/dataset/note. Accept
only `upsert` at revision 1 against an empty head and `tombstone` at revision 2 against
the exact live revision. Use the authenticated adapter context origin: direct clients
may create only `corrected`, and its target activity must resolve in the identical
scope; trusted coordinator/bootstrap origins may create lifecycle events. Actor and
source provenance remain server-bound. Do not create an activity-update or restore
path.

```python
if head is None:
    return accept_create(envelope, required_revision=1)
if exact_replay(head, envelope):
    return accept_replay(head)
if envelope.operation == "tombstone" and exact_live_base(head, envelope):
    return accept_tombstone(envelope, required_revision=2)
return conflict("notes_task_activity_immutable")
```

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_adapter.py
git add tldw_Server_API/app/core/Sync/v2/domain_adapters \
  tldw_Server_API/app/core/Sync/v2/adapters.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_adapter.py
git commit -m "feat(sync): add immutable task activity adapter"
```

### Task 2: Materialize activity with exact ordering and scope

- [ ] **Step 1: Write materializer RED tests**

Cover revision-1 insert, exact replay, changed replay rejection, revision-2 tombstone,
postcondition verification, owner/dataset/note/task scope, cursor ordering by
`(sync_server_cursor, activity_id)`, and rollback on an injected product-write
failure. Creation time is used only by the legacy bootstrap source scan.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_materializer.py
```

- [ ] **Step 3: Add exact product pages and lifecycle CAS helpers**

Add connection-aware create/tombstone/get/page methods. Every predicate includes
`owner_user_id` and `dataset_id`; task-bearing rows join the task and note scope. Use
keyset pagination by `(sync_server_cursor, activity_id)` with the TASK-13006.1
indexes and cap requested pages at 1,000.
Implementation targets: `TaskStore.create_sync_task_activity(...)`,
`TaskStore.tombstone_sync_task_activity(...)`,
`TaskStore.get_sync_task_activity(...)`, and
`TaskStore.page_sync_task_activity(...)`.

- [ ] **Step 4: Implement the materializer**

Use the accepted Sync transaction and product connection. Insert the canonical event
only once, verify its immutable semantic row, and tombstone by exact lifecycle CAS.
Do not project read/dismiss state into Sync. Product and Sync commits remain split:
after a product commit/Sync-status failure, exact replay verifies the immutable row
and advances apply status without inserting a second event.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_materializer.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py
git add tldw_Server_API/app/core/Sync/v2/materializers \
  tldw_Server_API/app/core/DB_Management/chacha/task_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_materializer.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py
git commit -m "feat(sync): materialize immutable task activity"
```

### Task 3: Bootstrap legacy events exactly and resumably

- [ ] **Step 1: Write legacy transform RED vectors**

Pin the spec's exact mappings for `created`, `updated`, `status_changed`, `unlinked`,
and `deleted`, including old/new normalization, source/provenance sentinel, optional
task identity, and the rule that read/dismiss rows never become activity envelopes.

- [ ] **Step 2: Write bootstrap RED tests**

Cover deterministic `(created_at, id)` pages, equal timestamps, page boundary replay,
stable envelope/activity IDs, count/fingerprint verification, split-commit repair,
source drift, malformed legacy rows, empty source, and cross-owner isolation.

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_bootstrap.py
```

- [ ] **Step 4: Add the exact legacy converter and source page**

Reuse the contract converter from TASK-13006.1 and add an owner/dataset-scoped source
page ordered by `(created_at, id)`, hard-capped at 1,000. Turn every five-family
mapping/rejection vector and equal-timestamp page boundary GREEN before appending.
Implementation targets: `TaskStore.page_legacy_events_for_sync_bootstrap(...)`,
`legacy_task_event_to_activity(...)`, and `_activity_bootstrap_envelope_id(...)`.

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_bootstrap.py \
  -k 'legacy_mapping or rejects_legacy or source_page or equal_timestamp'
```

- [ ] **Step 5: Implement one resumable bootstrap page**

Read only through keyset pages of at most 1,000. Transform through the strict contract
module, derive IDs from the immutable legacy event identity, use the source row's
canonical timestamp, and persist transition progress/count/fingerprint through the
shared readiness records. Fail closed on a changed source fingerprint.

```python
for row in task_store.page_legacy_events(after=cursor, limit=page_size):
    payload = legacy_task_event_to_activity(row)
    batch.append(build_bootstrap_activity(payload, provenance=LEGACY_SENTINEL))
```

- [ ] **Step 6: Reconcile count/fingerprint and readiness**

After the last page, compare source count and aggregate fingerprint with applied
activity envelopes. Persist only activity readiness. An injected split after Sync
append resumes by exact replay; source drift fails closed without deleting history.

- [ ] **Step 7: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py
git add tldw_Server_API/app/core/Sync/v2/notes_task_activity_bootstrap.py \
  tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/app/core/Sync/v2/profile.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py
git commit -m "feat(sync): bootstrap legacy task activity"
```

### Task 4: Add dormant transition capture and repair

- [ ] **Step 1: Write capture RED tests**

Cover create, title/description update, recurrence/metadata-only update, completion,
reopen, projection unlink, delete, and restore→`restored`. Prove every portable task
mutation derives the exact canonical event/value shape and stable activity ID;
retries do not duplicate, a failed second write is repairable, and public
capabilities/enrollment still omit both task domains.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_capture.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_service.py
```

- [ ] **Step 3: Implement a private capture primitive**

Expose a disabled-by-default callback from the task service. Given an accepted task
transition, return the strict activity payload plus stable object/envelope IDs. Do not
append separately from the task; TASK-13006.4 will make the full mutation plan atomic.

```python
capture = task_activity_capture
if capture is not None:
    capture.record_dormant_transition(before=before, after=after, operation=operation)
```

- [ ] **Step 4: Prove PostgreSQL ordering and RLS**

Add server-free SQL/catalog tests and live PostgreSQL tests for owner isolation,
same-note task references, cursor-plus-ID paging, exact replay, and rollback. Run with
PostgreSQL required; a skip is a failure for this child PR.

- [ ] **Step 5: Run GREEN and commit**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_postgres_contract.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_service.py
git add tldw_Server_API/app/core/Notes_Tasks/service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_postgres_contract.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_service.py
git commit -m "feat(sync): add dormant task activity capture"
```

### Task 5: Run the PR gate and close TASK-13006.3

- [ ] **Step 1: Run focused and regression tests**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_adapter.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_capture.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_postgres_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_adapter.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Notes_Tasks
```

Expected: all pass; live PostgreSQL tests do not skip.

- [ ] **Step 2: Run static/security checks**

```bash
PRODUCTION_PATHS=(
  tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_task_activity.py
  tldw_Server_API/app/core/Sync/v2/domain_adapters/__init__.py
  tldw_Server_API/app/core/Sync/v2/materializers/notes_task_activity.py
  tldw_Server_API/app/core/Sync/v2/materializers/__init__.py
  tldw_Server_API/app/core/Sync/v2/notes_task_activity_bootstrap.py
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

Confirm no public request schema, endpoint, supported/writable capability, enrollment,
or automatic product mutation advertises either domain. Confirm read/dismiss state is
absent and corrections always create new activity identities.

- [ ] **Step 4: Complete Backlog notes and status**

Check TASK-13006.3 AC/DoD, record exact commands and live PostgreSQL evidence, document
plan deviations, add implementation notes, and set the task Done.
