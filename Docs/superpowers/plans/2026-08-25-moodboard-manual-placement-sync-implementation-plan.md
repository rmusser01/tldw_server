# Dormant Moodboard and Manual Placement Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement complete but dormant `notes.moodboard` and `notes.moodboard_note` lifecycle, bootstrap, repair, and compatibility APIs while synchronizing only explicit placements.

**Architecture:** Two strict adapters own whole-object lineage and one scoped product store materializes portable board UUIDs into local integer rows. Board and placement bootstrap use independent bounded keysets but one coupled readiness/capture boundary. Existing hybrid REST ordering remains unchanged; a separate opaque keyset endpoint exposes canonical placement order. Factory wiring is internal-only and cannot enroll, capture production mutations, or advertise either domain until TASK-13007.5.

**Tech Stack:** Python 3.11, Pydantic contracts from TASK-13007.1, derived projection from TASK-13007.2, Sync v2 adapters/materializers, SQLite, PostgreSQL, FastAPI, pytest, Ruff, Bandit, OpenAPI exporter, openapi-typescript.

**Design:** `Docs/superpowers/specs/2026-08-24-notes-moodboard-studio-sync-design.md`

**Backlog task:** `TASK-13007.3`

**ADR required:** no
**ADR path:** `Docs/ADR/040-synchronized-moodboards-and-studio-authority.md`
**Reason:** ADR-040 already approves the two-domain identity, lifecycle, parent/dependency, conflict, bootstrap, API compatibility, and dormant rollout contracts.

---

## File map

**Create**

- `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_moodboard.py` — strict board v1 lineage, restore, dependency, and overwrite/skip evaluation.
- `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_moodboard_note.py` — deterministic placement identity, parent/dependency, retained-child, and lineage evaluation.
- `tldw_Server_API/app/core/Sync/v2/materializers/notes_moodboard.py` — idempotent owner/dataset-scoped board projection and postcondition checks.
- `tldw_Server_API/app/core/Sync/v2/materializers/notes_moodboard_note.py` — idempotent soft placement projection and restore.
- `tldw_Server_API/app/core/Sync/v2/notes_moodboard_bootstrap.py` — bounded board bootstrap and source verification.
- `tldw_Server_API/app/core/Sync/v2/notes_moodboard_note_bootstrap.py` — bounded placement bootstrap and source verification.
- `tldw_Server_API/app/core/Sync/v2/notes_moodboard_coordinator.py` — internal-only singleton planners, capture harness, repair dispatch, and coupled state checks.
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_adapter.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_adapter.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_materializer.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_materializer.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_bootstrap.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_bootstrap.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_coordinator.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_postgres_contract.py`

**Modify**

- `tldw_Server_API/app/core/DB_Management/chacha/moodboard_sync_store.py` — scoped CAS, portable/local identity lookup, placement pages, bootstrap verification, and repair postconditions.
- `tldw_Server_API/app/core/DB_Management/Sync_DB.py` — private bootstrap transitions and coupled capture-disabled guards.
- `tldw_Server_API/app/core/Sync/v2/store.py` — bootstrap/readiness facade methods.
- `tldw_Server_API/app/core/Sync/v2/factory.py` — strict adapter/materializer/bootstrap wiring for internal harnesses only.
- `tldw_Server_API/app/core/Sync/v2/service.py` — generic push/pull/repair/conflict compatibility for private domains and explicit `duplicate_rename` rejection.
- `tldw_Server_API/app/core/Sync/v2/replay.py` — repair dispatch for dormant domains.
- `tldw_Server_API/app/core/Sync/v2/restore.py` — preview retained board/placement state without fabricating group membership.
- `tldw_Server_API/app/core/Sync/v2/materializers/__init__.py`
- `tldw_Server_API/app/api/v1/schemas/notes_moodboards.py` — canvas, portable/canonical fields, placement input/response, restore, cursor, and concurrency models.
- `tldw_Server_API/app/api/v1/endpoints/notes.py` — compatible canvas/pin/unpin additions plus restore, layout patch, and placement keyset list routes; production Sync capture remains unwired.
- `tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py` — scoped soft placement/product lifecycle.
- `tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_schemas.py` — exact bounds and cursor models.
- `tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py` — legacy and additive API behavior.
- `tldw_Server_API/tests/Sync/test_sync_v2_factory.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- `tldw_Server_API/tests/Services/test_openapi_contracts.py`
- `Docs/Notes/Moodboards.md`
- `Docs/API/sync-v2.md` — describe dormant internal schemas without claiming support.
- `apps/tldw-frontend/lib/api/openapi.fingerprint.json`

**Generate for verification; do not commit**

- `apps/tldw-frontend/lib/api/generated/openapi.json`
- `apps/tldw-frontend/lib/api/generated/schema.d.ts`

### Task 0: Start the child after the portable projection lands

- [ ] **Step 1: Verify dependency and attach this plan**

```bash
backlog task 13007.2 --plain
backlog task edit 13007.3 -s "In Progress" \
  --doc Docs/superpowers/plans/2026-08-25-moodboard-manual-placement-sync-implementation-plan.md \
  --plan $'1. Implement strict board and placement adapters.\n2. Materialize both domains idempotently with retained-child lifecycle.\n3. Bootstrap and repair both domains through bounded internal-only coordinators.\n4. Add compatible canvas/placement REST contracts and canonical keyset listing.\n5. Prove SQLite/live PostgreSQL behavior, capability exclusion, and close the child.\n\nADR required: no\nADR path: Docs/ADR/040-synchronized-moodboards-and-studio-authority.md\nReason: ADR-040 already governs this child.'
```

Expected: TASK-13007.2 is Done; TASK-13007.3 becomes In Progress before production edits.

### Task 1: Implement strict board and placement adapters

- [ ] **Step 1: Write board lifecycle RED tests**

Cover create with empty head, UUIDv4 identity, exact retry, changed retry, no-op, update, stale base, tombstone with complete payload, explicit restore, restore-live conflict, update-deleted conflict, wrong parent, owner/dataset mismatch, unsupported adapter/operation, overwrite/skip, and `duplicate_rename` rejection.

- [ ] **Step 2: Write placement lifecycle RED tests**

Cover exact namespaced ID recomputation, `parent_id=moodboard_id`, dependencies on board and note, create/live-parent requirement, update/move, equal order tie behavior, tombstone/unpin, restore/repin, retained row under deleted board/note, edit denial until parents restore, cross-scope injection, integer/display bounds, exact retry, stale base, overwrite/skip, and rename rejection.

```python
def test_placement_rejects_identity_not_derived_from_members() -> None:
    outcome = evaluate(placement_envelope(object_id=UNRELATED_ID), context=live_parents())
    assert outcome.conflict_type == "notes_moodboard_note_identity_mismatch"


def test_board_delete_does_not_emit_placement_steps() -> None:
    plan = coordinator.plan_board_tombstone(board_head(), idempotency_key="delete-1")
    assert [step.domain for step in plan.steps] == ["notes.moodboard"]
```

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_adapter.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_adapter.py
```

Expected: imports fail because the adapter modules do not exist.

- [ ] **Step 4: Implement board adapter with shared lineage helpers**

Parse through `notes_moodboard_studio_contract.py`, require exact current base for every non-create mutation, compute the complete normalized object hash, and return reviewable conflicts without product reads. Semantic no-op returns the current head.

- [ ] **Step 5: Implement placement adapter with projected-head dependencies**

Recompute the relationship ID and validate both dependencies against the immutable applied-head snapshot. Allow retained tombstones beneath deleted parents but deny new/edit operations. Do not treat a smart match as a placement dependency or product row.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_adapter.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_adapter.py
git add tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_moodboard.py \
  tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_moodboard_note.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_adapter.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_adapter.py
git commit -m "feat(sync): evaluate moodboard placement lifecycle"
```

### Task 2: Materialize both domains idempotently

- [ ] **Step 1: Write materializer/store RED tests**

Cover portable UUID to local integer lookup, owner/dataset session scope, create/update/no-op, product `version` versus canonical revision, complete tombstone, restore, placement soft unpin, board/note retained-child hiding, postcondition replay after product/status split, same-cursor idempotency, divergent same-cursor denial, missing/wrong parent, cross-dataset collision, and safe error messages.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_materializer.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py
```

- [ ] **Step 3: Add scoped CAS and postcondition helpers**

`moodboard_sync_store.py` must expose explicit owner/dataset methods such as:

```python
def upsert_moodboard_from_sync(
    self,
    *,
    owner_user_id: str,
    dataset_id: str,
    sync_id: str,
    payload: Mapping[str, object],
    canonical_revision: int,
    canonical_hash: str,
    server_cursor: int,
) -> Mapping[str, object]:
    """Apply one accepted board head or return its exact postcondition."""
    raise NotImplementedError
```

All placement writes resolve the scoped live/tombstoned board and note inside the same product transaction. Every retry reloads and verifies exact product postconditions before deciding to write.

- [ ] **Step 4: Implement two focused materializers**

Keep materializers thin: parse the already-accepted envelope, set product scope, call the scoped store, convert only expected CAS/dependency failures into stable conflict results, and let retryable DB failures surface for repair.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_materializer.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py
git add tldw_Server_API/app/core/DB_Management/chacha/moodboard_sync_store.py \
  tldw_Server_API/app/core/Sync/v2/materializers/notes_moodboard.py \
  tldw_Server_API/app/core/Sync/v2/materializers/notes_moodboard_note.py \
  tldw_Server_API/app/core/Sync/v2/materializers/__init__.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_materializer.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py
git commit -m "feat(sync): materialize moodboards and placements"
```

### Task 3: Bootstrap, verify, repair, and wire internal-only coordinators

- [ ] **Step 1: Write bounded bootstrap RED tests**

Board keyset is `(sync_id)` and placement keyset is `(moodboard_sync_id,note_id)`. Cover empty graph, page limits, exact privacy-safe fingerprints, trusted source verifier, interrupted page, product commit/status split, resume, stale source correction, final full verification, malformed/oversized blockers, unknown legacy collection, deleted board/note retention, already-deleted placement rows, and proof that historical hard-deleted links are not fabricated.

- [ ] **Step 2: Write coordinator/factory/repair RED tests**

Cover deterministic singleton plans, required UUIDv4 allocation bound to idempotency replay, internal harness capture, exact replay rematerialization, conflict repair, pull/ack, skip/overwrite, rename rejection, partial component wiring failure, and explicit proof that normal REST calls create no envelopes in this child.

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_coordinator.py \
  tldw_Server_API/tests/Sync/test_sync_v2_factory.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py
```

- [ ] **Step 4: Implement separate bootstrappers with coupled completion**

Each bootstrap persists its own cursor/count/fingerprint, but the coordinator cannot report the pair ready unless both final source verifications and product postconditions pass. Local-unbound graph binding and rekey happen through the v61 authority transaction before source scan.

- [ ] **Step 5: Add internal-only factory and repair wiring**

Register strict adapters, materializers, bootstrappers, and repair dispatch so isolated tests can exercise the lifecycle. The public settings/capability path must still reject the domains, and `resolve_notes_moodboard_coordinator()` must return inactive for every production REST call.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_coordinator.py \
  tldw_Server_API/tests/Sync/test_sync_v2_factory.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py
git add tldw_Server_API/app/core/Sync/v2/notes_moodboard_bootstrap.py \
  tldw_Server_API/app/core/Sync/v2/notes_moodboard_note_bootstrap.py \
  tldw_Server_API/app/core/Sync/v2/notes_moodboard_coordinator.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/core/Sync/v2/replay.py \
  tldw_Server_API/app/core/Sync/v2/restore.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_coordinator.py \
  tldw_Server_API/tests/Sync/test_sync_v2_factory.py
git commit -m "feat(sync): bootstrap dormant moodboard domains"
```

### Task 4: Add canvas, placement, restore, and canonical keyset APIs

- [ ] **Step 1: Write schema and endpoint RED tests**

Cover optional canvas on create/update, `sync_id` and canonical lineage in responses, bodyless legacy pin, optional placement body, patch layout, optimistic version/base headers, soft unpin, explicit repin/restore, board restore, freeform/masonry behavior, manual metadata in hybrid results, legacy hybrid order, and exact stable 404/409/422 mappings.

For `GET /moodboards/{id}/placements`, cover default 50/max 200, `state=live|tombstoned|all`, opaque authenticated cursor, `(order_index,object_id)` order, equal-order tie, owner/dataset scope, tamper rejection, and no offset fallback.

```python
def test_placement_list_tie_breaks_by_object_id(client) -> None:
    response = client.get(f"/api/v1/notes/moodboards/{BOARD_ID}/placements")
    keys = [(row["order_index"], row["object_id"]) for row in response.json()["items"]]
    assert keys == sorted(keys)
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_schemas.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  -k 'moodboard or placement'
```

- [ ] **Step 3: Implement compatible product/API behavior only**

Translate local collection IDs at the REST boundary, store canonical canvas/placement fields, and preserve legacy response aliases. A bodyless pin uses approved default layout values. Unpin becomes a soft tombstone; repin restores the same deterministic identity. These endpoints still call product storage directly in this child and must not invoke the dormant coordinator.

- [ ] **Step 4: Implement signed opaque keyset cursor**

Reuse the repository cursor-signing utility. Bind cursor claims to authenticated owner, resolved dataset, moodboard portable identity, state filter, and last `(order_index,object_id)`. Reject malformed/tampered/cross-filter cursors without revealing row existence.

- [ ] **Step 5: Update docs, fingerprint, and generated client proof**

Update `Docs/Notes/Moodboards.md` with canvas fields, placement lifecycle, hybrid versus canonical ordering, cursor/state filters, concurrency, and the dormant Sync disclaimer. Update `Docs/API/sync-v2.md` only to document known internal schemas; do not list them as supported.

```bash
PYTHON=../../.venv/bin/python bun --cwd apps/tldw-frontend run generate:api-types
PYTHONPATH=. ../../.venv/bin/python Helper_Scripts/export_openapi_schema.py \
  --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

Inspect the generated `schema.d.ts`, commit only the reviewed fingerprint, and verify the semantic delta is limited to approved moodboard/placement routes and schemas.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_schemas.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  -k 'moodboard or placement'
git add tldw_Server_API/app/api/v1/schemas/notes_moodboards.py \
  tldw_Server_API/app/api/v1/endpoints/notes.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/moodboard_sync_store.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_schemas.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  Docs/Notes/Moodboards.md Docs/API/sync-v2.md \
  apps/tldw-frontend/lib/api/openapi.fingerprint.json
git commit -m "feat(notes): add moodboard placement contracts"
```

### Task 5: Prove tenancy, pagination, dormancy, and close TASK-13007.3

- [ ] **Step 1: Run SQLite and required-live PostgreSQL matrices**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_adapter.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_adapter.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_note_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_coordinator.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_postgres_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_factory.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_schemas.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py
```

Expected: live PostgreSQL does not skip; RLS, keyset plans, bootstrap pages, crash repair, conflicts, manual-only Sync, and legacy hybrid order all pass.

- [ ] **Step 2: Run static/security/OpenAPI/diff gates**

```bash
PRODUCTION_PATHS=(
  tldw_Server_API/app/core/DB_Management/chacha/moodboard_sync_store.py
  tldw_Server_API/app/core/DB_Management/Sync_DB.py
  tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_moodboard.py
  tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_moodboard_note.py
  tldw_Server_API/app/core/Sync/v2/materializers/notes_moodboard.py
  tldw_Server_API/app/core/Sync/v2/materializers/notes_moodboard_note.py
  tldw_Server_API/app/core/Sync/v2/materializers/__init__.py
  tldw_Server_API/app/core/Sync/v2/notes_moodboard_bootstrap.py
  tldw_Server_API/app/core/Sync/v2/notes_moodboard_note_bootstrap.py
  tldw_Server_API/app/core/Sync/v2/notes_moodboard_coordinator.py
  tldw_Server_API/app/core/Sync/v2/store.py
  tldw_Server_API/app/core/Sync/v2/factory.py
  tldw_Server_API/app/core/Sync/v2/service.py
  tldw_Server_API/app/core/Sync/v2/replay.py
  tldw_Server_API/app/core/Sync/v2/restore.py
  tldw_Server_API/app/api/v1/schemas/notes_moodboards.py
  tldw_Server_API/app/api/v1/endpoints/notes.py
)
../../.venv/bin/ruff check --no-cache "${PRODUCTION_PATHS[@]}"
../../.venv/bin/bandit -q "${PRODUCTION_PATHS[@]}"
PYTHONPYCACHEPREFIX=/tmp/task13007-3-pycache ../../.venv/bin/python -m py_compile "${PRODUCTION_PATHS[@]}"
PYTHONPATH=. ../../.venv/bin/python Helper_Scripts/export_openapi_schema.py \
  --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
git diff --check
```

- [ ] **Step 3: Self-review domain boundaries**

Confirm no smart-only match becomes a placement, board delete emits no unbounded child plan, placement uniqueness remains `(moodboard,note)`, public hybrid and canonical placement order remain distinct, REST product mutations still create no Sync envelopes, and both domains remain absent from public supported/writable maps.

- [ ] **Step 4: Finalize the child**

Record exact evidence, check TASK-13007.3 AC/DoD, add implementation notes/touched files and any real lesson, set the child Done, and commit the closeout. Do not activate either domain in this PR.
