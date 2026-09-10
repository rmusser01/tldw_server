# Notes Task Checklist Convergence and Sync Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Atomically couple task, activity, and optional note mutations; converge managed Markdown checklists through durable anchors and explicit drift; then expose `notes.task` and `notes.task_activity` together only when enrollment and both bootstraps are ready.

**Architecture:** A single coordinator builds the complete deterministic mutation plan before any append; server-origin plans use the existing ADR-034 batch and authenticated client task pushes are expanded through the same planner before append. Managed Markdown markers carry stable task identity plus the last-common canonical revision/hash; immutable Sync envelopes and group metadata are durable projection authority, while product cache/drift rows are rebuildable. The two domains share one activation predicate and are never advertised independently. The 1,000-envelope mutation-group ceiling permits at most 499 managed task transitions in one note reconciliation (`2N + 1`).

**Tech Stack:** Python 3.11, Textual server API, Sync v2 mutation groups, ChaChaNotes SQLite/PostgreSQL, Pydantic, pytest, Ruff, Bandit.

**Design:** `Docs/superpowers/specs/2026-08-13-notes-task-activity-sync-design.md`

**Backlog task:** `TASK-13006.4`

**ADR required:** no
**ADR path:** `Docs/ADR/039-canonical-notes-task-sync-and-derived-checklist-projections.md`
**Reason:** This plan activates the coupled boundary and convergence algorithm already approved by ADR-039; it introduces no additional architecture decision.

---

## File map

**Create**

- `tldw_Server_API/app/core/Sync/v2/notes_task_coordinator.py` — deterministic compound planning and one-batch append.
- `tldw_Server_API/app/core/Notes_Tasks/projection_markers.py` — pure hidden-marker parse/render and canonical line identity.
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_coordinator.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_projection.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activation.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_task_end_to_end.py`

**Modify**

- `tldw_Server_API/app/core/Notes_Tasks/markdown_parser.py` — recognize managed checklist markers without changing ordinary Markdown parsing.
- `tldw_Server_API/app/core/Notes_Tasks/reconciler.py` — exact three-way classification and explicit drift handling.
- `tldw_Server_API/app/core/Notes_Tasks/service.py` — route REST/MCP task mutations through the coordinator.
- `tldw_Server_API/app/core/Notes_Tasks/models.py` — bounded projection/drift response types only where public APIs require them.
- `tldw_Server_API/app/core/DB_Management/chacha/task_store.py` — rebuildable projection cache, drift, product rekey, and retention-blocker methods; never durable anchor authority.
- `tldw_Server_API/app/core/DB_Management/Sync_DB.py` — historical immutable task-envelope lookup and mutation-group anchor metadata.
- `tldw_Server_API/app/core/Sync/v2/store.py` — owner/dataset-scoped historical envelope and group-anchor facade.
- `tldw_Server_API/app/core/Sync/v2/service.py` — coordinator entry, coupled enrollment/readiness, retention and repair.
- `tldw_Server_API/app/core/Sync/v2/factory.py` — coupled supported/writable activation.
- `tldw_Server_API/app/core/Sync/v2/models.py` — domain constants and bounded internal plan types.
- `tldw_Server_API/app/core/Sync/v2/profile.py` — coupled bootstrap/enrollment response.
- `tldw_Server_API/app/core/Sync/v2/replay.py` — repair full mutation groups without partial projection.
- `tldw_Server_API/app/core/Sync/v2/restore.py` — linked task/projection restore coordination.
- `tldw_Server_API/app/api/v1/endpoints/notes_tasks.py` — expose stable drift/conflict outcomes.
- `tldw_Server_API/app/api/v1/endpoints/sync.py` — coupled enrollment/capability behavior.
- `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py` — public capability schemas for the two domains.
- `Docs/API/Sync_V2_M1.md` — document versions, enrollment, grouping, markers, and failure semantics.
- `tldw_Server_API/tests/Notes_Tasks/unit/test_markdown_parser.py`
- `tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py`
- `tldw_Server_API/tests/Notes_Tasks/unit/test_service.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_retention.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py`

### Task 0: Start the child and attach this plan

- [ ] **Step 1: Move TASK-13006.4 In Progress**

```bash
backlog task edit 13006.4 -s "In Progress" --plan $'1. Add immutable-envelope projection anchors and rebuildable caches.\n2. Expand server/client task mutations into deterministic groups.\n3. Implement exact three-way convergence and drift claims.\n4. Fence retention/delete/restore/relink.\n5. Rekey product state and activate both domains together.\n6. Run end-to-end SQLite/live-PostgreSQL/static gates.\nDetailed plan: Docs/superpowers/plans/2026-08-13-notes-task-checklist-convergence-activation-implementation-plan.md\nADR required: no; ADR-039 applies.'
backlog task 13006.4 --plain
```

Expected: TASK-13006.3 is Done, TASK-13006.4 is In Progress, and the plan/ADR are
linked before production work.

### Task 1: Parse managed markers and persist projection authority

- [ ] **Step 1: Write marker RED tests**

Cover render/parse round trips, stable task UUID, canonical revision/hash, NFKC/control
characters, duplicate markers, forged markers, ordinary checklist lines, reordered
lines, and malformed hidden comments. Keep the marker syntax private to the helper.

```python
def test_managed_marker_round_trip() -> None:
    marker = render_task_marker(TASK_ID, revision=7, object_hash=HASH)
    assert parse_task_marker(marker) == TaskMarker(TASK_ID, 7, HASH)
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notes_Tasks/unit/test_markdown_parser.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_projection.py
```

- [ ] **Step 3: Implement one pure marker module**

Use standard-library parsing. Return a typed result or a stable validation error; do
not embed DB or Sync calls. The Markdown parser may call this helper but ordinary
unmarked checklist behavior must remain unchanged.

- [ ] **Step 4: Persist authoritative anchors in Sync group metadata**

Extend the existing mutation-group metadata with the historical task envelope ID,
canonical task revision/hash, note envelope ID/hash, and projection version. Add an
owner/dataset-scoped Sync-store lookup that resolves that immutable historical task
envelope exactly; a marker that cannot resolve its named envelope is drift.
Implementation targets: `SyncDatabase.get_historical_task_envelope(...)`,
`SyncV2Store.get_historical_task_envelope(...)`, and
`_validate_task_projection_group_metadata(...)`.

- [ ] **Step 5: Add rebuildable product cache and drift methods**

Add owner/dataset/note/task-scoped create/get/CAS/page methods to `task_store.py`.
These rows cache the authoritative Sync group/envelope references and are rebuildable;
they do not become last-common authority. Drift stores privacy-safe reason codes and
lifecycle, not raw note/task content. Rebuild missing caches only from a valid marker,
its exact Sync group metadata, and the immutable historical task envelope named by
that metadata—never from current task heads or checklist text.

- [ ] **Step 6: Run cache-loss/retention GREEN and commit**

Before committing, prove cache deletion rebuilds from the marker's immutable envelope,
marker removal creates drift without losing the historical reference, task/note
tombstones retain the named anchor envelope, and open drift keeps every referenced
group member ineligible for compaction.

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Notes_Tasks/unit/test_markdown_parser.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_projection.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py
git add tldw_Server_API/app/core/Notes_Tasks/projection_markers.py \
  tldw_Server_API/app/core/Notes_Tasks/markdown_parser.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/DB_Management/chacha/task_store.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_markdown_parser.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_projection.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py
git commit -m "feat(notes): add managed task projection authority"
```

### Task 2: Build one deterministic compound mutation coordinator

- [ ] **Step 1: Write coordinator RED tests**

Cover task-only metadata changes (`task + activity`), projection-affecting changes
(`task + activity + note`), note-origin checklist changes, REST/MCP origins, Sync
client task pushes, stable IDs under retry, exact parent/base tuples, full preflight
before append, injected second/third-step failure, acceptance of exactly 499 managed
task changes (`2N + 1 = 999`), and pre-append rejection of 500 (`1,001`).

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_coordinator.py
```

- [ ] **Step 3: Implement pure deterministic plan construction**

Build all envelopes in memory before calling `server_origin_batch`. Derive activity,
note, envelope, and idempotency IDs from the mutation identity and canonical payloads.
Reject a plan over 1,000 steps before any append; note reconciliation therefore caps
managed task transitions at 499.

Implementation targets: `NotesTaskCoordinator.plan_task_mutation(...)`,
`NotesTaskCoordinator.plan_note_reconciliation(...)`, `_task_activity_id(...)`, and
`_validate_task_mutation_plan(...)`.

```python
plan = [task_envelope, activity_envelope]
if note_projection_changed:
    plan.append(note_envelope)
validate_mutation_group(plan, max_steps=1000)
return server_origin_batch.append_atomic(plan)
```

- [ ] **Step 4: Append server-origin plans and repair split commits**

Replace the dormant callback in task service with one injected coordinator. Keep one
mutation authority: append the complete compound Sync batch, then materialize product
state. Sync and product commits are separate; injected splits replay the same accepted
group and verify product postconditions. Repair never appends only the missing
activity or note step and never promises cross-database rollback.

- [ ] **Step 5: Expand authenticated client task pushes before append**

In the Sync service, after device/dataset/domain/version authorization and strict task
validation but before any append, expand an incoming `notes.task` envelope into the
same compound plan. Derive activity/group IDs from the authenticated task envelope ID
and canonical payload; add the optional note projection from the exact base/head.
Append all steps atomically in Sync. Exact client replay must resolve the identical
group; changed envelope-ID reuse conflicts before activation.
Implementation targets: `SyncV2Service._expand_task_client_push(...)` and the existing
batch append transaction; do not add a second append engine.

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_coordinator.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  -k 'client_push or client_replay or client_group or changed_identity'
```

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_coordinator.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_service.py
git add tldw_Server_API/app/core/Sync/v2/notes_task_coordinator.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Notes_Tasks/service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_coordinator.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_service.py
git commit -m "feat(sync): coordinate task activity and note mutations"
```

### Task 3: Implement explicit three-way checklist convergence

- [ ] **Step 1: Write the reconciliation matrix as RED tests**

For each managed task compare `(anchor, current task, current note line)` and cover:
unchanged, task-only, note-only, equal concurrent result, incompatible concurrent
change, completion/reopen, description edit, reorder, unlink, missing marker, duplicate
marker, cache loss, explicit REST task vs unmarked checklist, and unrelated note edits.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_projection.py
```

- [ ] **Step 3: Implement the minimal three-way classifier**

Return one of `no_change`, `task_to_note`, `note_to_task`, `same_result`, or `drift`.
Only a marker plus matching anchor grants managed authority. Unmarked checklists stay
ordinary Markdown. On incompatible edits, preserve both product states and write one
privacy-safe drift row; never silently overwrite an explicit task.

- [ ] **Step 4: Route reconciliation through the coordinator**

Translate non-drift decisions into deterministic compound plans. Advance anchors only
after the full mutation group materializes and postconditions pass. A crash between
append and projection must replay the same plan and converge without duplicate events.

- [ ] **Step 5: Implement exact drift claims and outcomes**

Require owner/dataset, drift ID, expected drift lifecycle revision, current task head,
current note head, and authoritative anchor group. Test `keep_task`,
`accept_markdown`, `unlink`, and `dismiss`; stale or changed claims conflict without
mutation. Resolution appends its complete deterministic group first, then CAS-updates
the rebuildable drift row. Dismissal changes only drift lifecycle and never task/note
content.

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_projection.py \
  -k 'resolve_drift or stale_claim or dismiss_drift or keep_task or accept_markdown or unlink'
```

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_projection.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_coordinator.py
git add tldw_Server_API/app/core/Notes_Tasks/reconciler.py \
  tldw_Server_API/app/core/Sync/v2/notes_task_coordinator.py \
  tldw_Server_API/tests/Notes_Tasks/unit/test_reconciler.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_projection.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_coordinator.py
git commit -m "feat(notes): converge managed task checklists"
```

### Task 4: Fence retention, delete, restore, and relink

- [ ] **Step 1: Write lifecycle/retention RED tests**

Cover linked task tombstone retaining required task/activity/note envelopes, open drift
blocking compaction, resolved drift releasing blockers, note delete, task unlink,
explicit restore/relink, note restore with missing task, repeated restore, cache loss,
and stale candidate revalidation under the dataset materialization fence.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_retention.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  -k 'notes_task or task_activity or checklist_projection'
```

- [ ] **Step 3: Add exact blocker and resolution reads**

Use owner/dataset-scoped, capped, index-backed pages. Revalidate task bindings, current
heads, immutable Sync group/envelope anchor references, open drift, devices, exact
version acks, and restore windows on the same guarded connection before retention CAS.
Product projection caches are never retention authority. Do not introduce physical GC
or a recurrence scheduler.

- [ ] **Step 4: Implement delete/restore/relink coordination**

Produce complete deterministic plans, require exact tombstone bases for restore, and
recreate/advance projection anchors only after all referenced product state is valid.
Unlink preserves task identity; relink requires authorized destination note scope.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_retention.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_projection.py
git add tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/replay.py \
  tldw_Server_API/app/core/Sync/v2/restore.py \
  tldw_Server_API/app/core/DB_Management/chacha/task_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_retention.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py
git commit -m "feat(sync): fence task projection lifecycle"
```

### Task 5: Activate coupled enrollment and capabilities

- [ ] **Step 1: Write activation RED tests**

Cover default omission, gate off, unbound/local sentinel dataset, one bootstrap ready,
both bootstraps ready, source diagnostic failure, explicit enrollment rekey, product-
commit/Sync-readiness split and idempotent resume, active re-enrollment, device version
negotiation, selected-dataset authorization, and the invariant that support/writability
always includes both domains or neither. Inject concurrent REST and MCP mutations
before each source scan, between task/activity scans, and after each scan; every
transition must appear exactly once after resume. Add direct-client authorization
vectors: lifecycle activity creates are rejected, a `corrected` event with an exact
authorized same-scope target is accepted, and missing/foreign/cross-task targets fail
without revealing existence.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py
```

- [ ] **Step 3: Implement one coupled readiness predicate**

Require rollout gate, dataset authorization/enrollment, task readiness, activity
readiness, projection repair readiness, and supported adapter version 1 for both
domains. Reuse it from adapter enforcement, service capabilities, profile/bootstrap,
and public endpoint responses.

- [ ] **Step 4: Rekey local-unbound rows atomically at enrollment**

Under the dataset fence, validate collision-free ownership, rekey all six task-side
product tables (including rebuildable cache/drift rows) from the private sentinel to
the enrolled dataset, verify exact counts/fingerprints, and commit that product
transaction. Sync readiness and capture activation are a later Sync-DB transition
under the same fence: if it fails, remain non-writable and resume idempotently from
the verified product state. Never claim cross-database rollback or rewrite historical
Sync envelopes.

- [ ] **Step 5: Enable coupled capture, run both bootstraps, then become ready**

With external task-domain Sync writes still disabled, commit one Sync readiness
transition that enables both REST/MCP task and activity capture together. Only then
run the task source scan and activity source scan resumably. Capture closes the race
for mutations concurrent with either scan; exact event/envelope IDs deduplicate source
rows and captured transitions. After both count/fingerprint verifications pass, commit
one coupled readiness transition. Until that last commit, supported/writable maps omit
both domains. Add split-resume tests after product rekey, capture enable, every task
page, every activity page, and final readiness.

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activation.py \
  -k 'capture_before_scan or concurrent_rest or concurrent_mcp or split_resume or coupled_ready'
```

- [ ] **Step 6: Wire public schemas/endpoints and docs**

Advertise `notes.task: [1]` and `notes.task_activity: [1]` together. Add stable,
sanitized activation/drift errors and document payload versions, group limits,
enrollment, marker authority, conflict behavior, client activity limited to exact
same-scope `corrected` events, and no recurrence scheduler. Keep the service preflight
that rejects direct client lifecycle events even if adapter/factory wiring changes.

- [ ] **Step 7: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py
git add tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/app/core/Sync/v2/profile.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/api/v1/endpoints/sync.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  Docs/API/Sync_V2_M1.md \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py
git commit -m "feat(sync): activate coupled task domains"
```

### Task 6: Prove end-to-end convergence and close TASK-13006

- [ ] **Step 1: Run end-to-end SQLite and required PostgreSQL tests**

Cover two devices plus server-origin REST/MCP, create/edit/complete/reopen, recurrence
metadata propagation without scheduling, note-origin checklist edits, concurrent equal
and incompatible edits, deletion, restoration, relink, pagination, pull/ack, crash at
each mutation-group boundary, split-commit repair, and retention revalidation.

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_end_to_end.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_coordinator.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_projection.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activation.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_postgres_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_postgres_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py \
  tldw_Server_API/tests/Sync/test_sync_v2_retention.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/Notes_Tasks
```

Expected: all pass; live PostgreSQL tests do not skip.

- [ ] **Step 2: Run static/security checks**

```bash
PRODUCTION_PATHS=(
  tldw_Server_API/app/core/Sync/v2/notes_task_coordinator.py
  tldw_Server_API/app/core/Sync/v2/service.py
  tldw_Server_API/app/core/Sync/v2/store.py
  tldw_Server_API/app/core/Sync/v2/factory.py
  tldw_Server_API/app/core/Sync/v2/models.py
  tldw_Server_API/app/core/Sync/v2/profile.py
  tldw_Server_API/app/core/Sync/v2/replay.py
  tldw_Server_API/app/core/Sync/v2/restore.py
  tldw_Server_API/app/core/DB_Management/Sync_DB.py
  tldw_Server_API/app/core/DB_Management/chacha/task_store.py
  tldw_Server_API/app/core/Notes_Tasks/projection_markers.py
  tldw_Server_API/app/core/Notes_Tasks/markdown_parser.py
  tldw_Server_API/app/core/Notes_Tasks/reconciler.py
  tldw_Server_API/app/core/Notes_Tasks/service.py
  tldw_Server_API/app/core/Notes_Tasks/models.py
  tldw_Server_API/app/api/v1/endpoints/notes_tasks.py
  tldw_Server_API/app/api/v1/endpoints/sync.py
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
)
../../.venv/bin/ruff check --no-cache "${PRODUCTION_PATHS[@]}"
../../.venv/bin/bandit -q "${PRODUCTION_PATHS[@]}"
PYTHONPYCACHEPREFIX=/tmp/task13006-pycache ../../.venv/bin/python -m py_compile "${PRODUCTION_PATHS[@]}"
git diff --check
```

- [ ] **Step 3: Self-review scope and security**

Confirm every public mutation uses the coordinator, neither domain can activate alone,
all task-side predicates bind owner and dataset, no raw drift content leaks, group/page
caps are enforced before work, and no scheduler, physical GC, or unrelated Notes API
change entered the diff.

- [ ] **Step 4: Complete Backlog and final documentation**

Check TASK-13006.4 and parent TASK-13006 AC/DoD, record exact SQLite/live PostgreSQL and
static evidence, add concise implementation notes and any real lesson learned, set both
tasks Done, and commit the closeout documentation.
