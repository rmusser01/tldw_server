# Dormant Accepted Studio Document Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement complete but dormant `notes.studio_document` storage, lifecycle, bootstrap, repair, provenance, and compound-save planning while preserving the established Studio REST representation.

**Architecture:** One strict adapter synchronizes sections-only canonical sidecars bound to exact `notes.note` heads; title/Markdown remain note authority. A compatibility serializer injects legacy nested title/source/layout and rebuilds disposable diagram caches at the REST boundary. Internal planners synthesize deterministic singleton or note-first/Studio-second groups and persist a separate private client-intent fingerprint before append, but production Studio and Notes endpoints remain on their existing product path until TASK-13007.5.

**Tech Stack:** Python 3.11, Pydantic contracts from TASK-13007.1, Sync v2 mutation groups, SQLite, PostgreSQL forced RLS, FastAPI, pytest, Ruff, Bandit, OpenAPI exporter, openapi-typescript.

**Design:** `Docs/superpowers/specs/2026-08-24-notes-moodboard-studio-sync-design.md`

**Backlog task:** `TASK-13007.4`

**ADR required:** no
**ADR path:** `Docs/ADR/040-synchronized-moodboards-and-studio-authority.md`
**Reason:** ADR-040 already defines Studio authority, provenance, source binding, compound commands, lifecycle, compatibility, cache, conflict, and dormant rollout policy.

---

## File map

**Create**

- `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_studio_document.py` — strict sidecar identity, note/source binding, lineage, provenance, restore, and resolution evaluation.
- `tldw_Server_API/app/core/Sync/v2/materializers/notes_studio_document.py` — idempotent scoped Studio projection and compatibility-cache invalidation.
- `tldw_Server_API/app/core/Sync/v2/notes_studio_bootstrap.py` — bounded sidecar bootstrap, legacy canonicalization, source verification, and repair.
- `tldw_Server_API/app/core/Sync/v2/notes_studio_coordinator.py` — internal-only manual/derive/regenerate/diagram/lifecycle planners and accepted-state capture harness.
- `tldw_Server_API/app/core/Notes/studio_compat.py` — canonical-to-legacy response rehydration, equal nested-input stripping, and sanitized cache reconstruction.
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_adapter.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_materializer.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_bootstrap.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_coordinator.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_client_compound.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_postgres_contract.py`
- `tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_compat.py`
- `Docs/Notes/Studio.md` — REST/canonical authority, accepted-persistence, provenance, limits, and stale-state guide.

**Modify**

- `tldw_Server_API/app/core/DB_Management/Sync_DB.py` — private Studio bootstrap transitions plus `sync_studio_compound_intents` storage/catalog for incoming intent fingerprints.
- `tldw_Server_API/app/core/DB_Management/chacha/note_store.py` — scoped Studio CAS, tombstone/restore, bootstrap pages, source checks, and postcondition reads.
- `tldw_Server_API/app/core/Sync/v2/store.py` — bootstrap and private intent-record facade methods.
- `tldw_Server_API/app/core/Sync/v2/factory.py` — internal-only strict adapter/materializer/bootstrap wiring.
- `tldw_Server_API/app/core/Sync/v2/service.py` — private Studio compound expansion, overlap rejection, replay lookup, and generic repair/push behavior.
- `tldw_Server_API/app/core/Sync/v2/replay.py` — Studio repair dispatch.
- `tldw_Server_API/app/core/Sync/v2/restore.py` — retained Studio preview semantics.
- `tldw_Server_API/app/core/Sync/v2/materializers/__init__.py`
- `tldw_Server_API/app/core/Notes/studio_markdown.py` — sections-only canonical helpers and title/layout injection inputs.
- `tldw_Server_API/app/core/Notes/studio_service.py` — accepted-state reduction, deterministic planners, authoritative outer fields, cache-free manifests, and product-only compatibility path.
- `tldw_Server_API/app/api/v1/schemas/notes_studio.py` — strict accepted-write/precondition models and compatible response fields.
- `tldw_Server_API/app/api/v1/endpoints/notes.py` — manual Studio save contract and optional concurrency/idempotency inputs; production Sync capture remains unwired.
- `tldw_Server_API/tests/ChaChaNotesDB/test_note_studio_db.py`
- `tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_service.py`
- `tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_factory.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- `tldw_Server_API/tests/Sync/test_sync_v2_store.py`
- `tldw_Server_API/tests/Services/test_openapi_contracts.py`
- `Docs/API/sync-v2.md` — document the private known domain without advertising support.
- `apps/tldw-frontend/lib/api/openapi.fingerprint.json`

**Generate for verification; do not commit**

- `apps/tldw-frontend/lib/api/generated/openapi.json`
- `apps/tldw-frontend/lib/api/generated/schema.d.ts`

### Task 0: Start the child after TASK-13007.1 lands

- [ ] **Step 1: Verify dependency and attach this plan**

```bash
backlog task 13007.1 --plain
backlog task edit 13007.4 -s "In Progress" \
  --doc Docs/superpowers/plans/2026-08-25-accepted-studio-document-sync-implementation-plan.md \
  --plan $'1. Implement strict Studio identity, lineage, source, and provenance evaluation.\n2. Materialize sections-only sidecars idempotently and preserve REST compatibility.\n3. Bootstrap legacy sidecars with diagnostics and repair.\n4. Build internal accepted-state and client compound planners with separate intent fingerprints.\n5. Update Studio API/OpenAPI/docs without production Sync capture.\n6. Prove SQLite/live PostgreSQL behavior and close the child.\n\nADR required: no\nADR path: Docs/ADR/040-synchronized-moodboards-and-studio-authority.md\nReason: ADR-040 already governs this child.'
```

Expected: TASK-13007.1 is Done; TASK-13007.4 becomes In Progress. It may proceed independently of TASK-13007.2/.3.

### Task 1: Implement strict Studio adapter semantics

- [ ] **Step 1: Write identity/lineage RED tests**

Cover `object_id=payload.note_id=parent_id`, create/update/no-op, exact retry, changed retry, stale base, whole-payload tombstone, explicit restore, ordinary upsert against tombstone, standalone Studio tombstone rejection, overwrite/skip, `duplicate_rename` rejection, and cross-owner/dataset failures.

- [ ] **Step 2: Write note/source/provenance RED tests**

Cover exact note revision/hash binding, planned note head overlay, sidecar-only current head, stale-note conflict, same-scope live source note, retained reference after source tombstone, new derivation from tombstoned source rejection, unknown/cross-scope non-enumerating error, excerpt membership after CRLF normalization, server/client/bootstrap attestation, provider/model pairing, server-stamped `accepted_at`, restore preserving provenance, and transient/secret key rejection.

```python
def test_studio_identity_must_equal_note_and_parent() -> None:
    outcome = evaluate(studio_envelope(object_id=OTHER_NOTE), context=note_head())
    assert outcome.conflict_type == "notes_studio_identity_mismatch"


def test_client_claimed_server_attestation_is_rewritten() -> None:
    accepted = evaluate(client_studio_envelope(attestation="server"), context=note_head())
    assert accepted.payload["accepted_provenance"]["attestation"] == "client_declared"
```

- [ ] **Step 3: Write hash/cache/sanitization RED tests**

Cover exact sections-only result hash, object hash, normalized excerpt/companion hashes, canonical diagram context/order, canonical JSON render hash, cache/alias exclusion, derived cache rebuild, diagram/Mermaid escaping at render/export boundaries, payload limit preflight, and recursive unknown/depth/count rejection.

- [ ] **Step 4: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_adapter.py
```

Expected: import failure because the adapter module is absent.

- [ ] **Step 5: Implement the strict adapter**

Use the TASK-13007.1 contract parser and applied-head snapshot only. Server execution context—not client payload—supplies provider/model, server attestation, acceptance time, and authenticated device binding. Tombstone/restore preserve the accepted payload and prior provenance. Return stable reviewable conflicts without reading product tables directly.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_adapter.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_contract.py
git add tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_studio_document.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_adapter.py
git commit -m "feat(sync): evaluate accepted Studio documents"
```

### Task 2: Materialize canonical sidecars and preserve REST compatibility

- [ ] **Step 1: Write store/materializer RED tests**

Cover owner/dataset-scoped create/update/no-op, note-parent/source verification in one transaction, canonical/product revision separation, tombstone/restore, legitimate stale binding retention, same-cursor replay, divergent same-cursor denial, product-commit/status split repair, payload-size preflight before write, and derived-cache invalidation.

- [ ] **Step 2: Write compatibility RED tests**

Cover REST fetch/save/render/regenerate/diagram/export rehydrating:

- `payload_json.meta.title` from current/planned note;
- `meta.source_note_id` from the outer sidecar;
- `layout.template_type`, `handwriting_mode`, and `render_version` from outer fields;
- diagram aliases only when derivable; and
- sanitized locally rebuilt `cached_svg` excluded from canonical storage/hashes.

Equal nested legacy input is stripped; mismatched nested title/source/layout/diagram alias returns `422` or a bootstrap blocker rather than silently choosing an authority.

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_materializer.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_studio_db.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_compat.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_service.py
```

- [ ] **Step 4: Add scoped Studio CAS/postcondition methods**

`note_store.py` receives explicit owner/dataset methods for canonical projection and bootstrap. All writes validate the scoped note and optional source note in the same transaction and persist only sections-only payload, outer authority fields, canonical lineage, lifecycle, and accepted provenance.

- [ ] **Step 5: Implement thin materializer and compatibility serializer**

`NotesStudioDocumentMaterializer` calls the scoped store and maps only expected lineage/dependency collisions. `studio_compat.py` is the sole REST compatibility boundary; render/export receive outer values explicitly and never depend on canonical nested `meta` or `layout`.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_materializer.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_studio_db.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_compat.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_service.py
git add tldw_Server_API/app/core/DB_Management/chacha/note_store.py \
  tldw_Server_API/app/core/Sync/v2/materializers/notes_studio_document.py \
  tldw_Server_API/app/core/Sync/v2/materializers/__init__.py \
  tldw_Server_API/app/core/Notes/studio_compat.py \
  tldw_Server_API/app/core/Notes/studio_markdown.py \
  tldw_Server_API/app/core/Notes/studio_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_materializer.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_studio_db.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_compat.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_service.py
git commit -m "feat(notes): materialize canonical Studio sidecars"
```

### Task 3: Bootstrap legacy sidecars and repair split state

- [ ] **Step 1: Write bootstrap RED tests**

Use `(note_id)` keysets. Cover empty source, bounded page, exact source count/cursor/fingerprint, interrupted page, resume, final verification, product/Sync split repair, already-deleted parent bootstrapping tombstoned, legitimate stale note binding, live same-scope source verification, retained tombstoned source, malformed JSON, unknown fields, unsupported render version, oversized object, invalid timestamp/reference, nested title/source/layout mismatch, diagram alias mismatch, and cache discard/rebuild.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_bootstrap.py
```

- [ ] **Step 3: Implement bounded, source-verified bootstrap**

Bind/rekey only the Studio graph through the v61 authority transaction. Canonicalize valid legacy rows under `trusted_bootstrap_v1` using normalized product modification time. Store privacy-safe diagnostic code/hash/count for invalid rows; never drop, guess, or rewrite mismatched authority. Capture each trusted source row with a verifier, then recheck the complete bounded aggregate before readiness can advance.

- [ ] **Step 4: Wire internal factory/repair/restore paths**

Register the Studio adapter, materializer, bootstrapper, and repair dispatch for isolated internal use. Restore preview returns the latest non-superseded projected sidecar state but never invents standalone tombstones or incomplete mutation-group metadata. Public support/writable maps remain unchanged.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_factory.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py
git add tldw_Server_API/app/core/Sync/v2/notes_studio_bootstrap.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/app/core/Sync/v2/replay.py \
  tldw_Server_API/app/core/Sync/v2/restore.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_factory.py
git commit -m "feat(sync): bootstrap dormant Studio documents"
```

### Task 4: Plan accepted state and client compound commands deterministically

- [ ] **Step 1: Write accepted-transition RED tests**

Cover manual save, derive, current deterministic regenerate, and diagram save. Assert only successfully persisted accepted state yields a plan; prompt, provider request, raw output, failed result, preview, title suggestion, credential/token fields, and returned-only generation yield none. Derive/diagram server plans stamp actual provider/model; manual/current regenerate use null provider/model.

- [ ] **Step 2: Write compound/replay RED tests**

Cover sidecar-only singleton; note-first/Studio-second plan; planned note revision/hash binding; all-or-none append; server-owned group fields; overlap with separate note envelope rejection; stable lookup by dataset/device/client envelope; canonical client-controlled intent fingerprint; exact replay; changed intent conflict; separately verified stored plan hash; tampered plan; server timestamp/provenance variance; and crash/resume.

```python
def test_changed_client_intent_conflicts_before_expansion() -> None:
    first = push_studio_compound(client_id="e1", note_content="A")
    changed = push_studio_compound(client_id="e1", note_content="B")
    assert first.accepted
    assert changed.error_code == "notes_studio_client_intent_conflict"
```

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_coordinator.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_client_compound.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  -k 'studio or compound_intent'
```

- [ ] **Step 4: Add the private intent record**

Add exact SQLite/PostgreSQL `sync_studio_compound_intents` storage with a scoped unique lookup, canonical incoming fingerprint, group ID, stored plan hash, created/updated timestamps, and immutable replay semantics. It contains hashes and identifiers only—never note/Studio plaintext. Startup verifies exact columns/indexes; drift fails closed.

- [ ] **Step 5: Implement one internal planner and client expansion path**

```python
@dataclass(frozen=True, slots=True)
class NotesStudioMutationPlan:
    steps: tuple[ServerOriginMutationStep, ...]
    client_intent_fingerprint: str


def plan_accepted_studio_transition(
    *, kind, note_head, studio_head, accepted_state, note_change=None, execution=None
) -> NotesStudioMutationPlan:
    """Build the deterministic accepted-state mutation plan."""
    raise NotImplementedError
```

Validate the complete overlay before append. Persist/compare incoming intent before server timestamps or provenance stamping. Compute and verify the expanded plan hash separately. Use existing ADR-034 append/materialize machinery; do not add another transaction protocol.

- [ ] **Step 6: Keep production callers unwired**

The coordinator is callable from internal tests only. `studio_service.py`, REST Notes delete/restore, and public client push reject or bypass the dormant domain exactly as before. No production operation can publish Studio history in this child.

- [ ] **Step 7: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_coordinator.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_client_compound.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  -k 'studio or compound_intent'
git add tldw_Server_API/app/core/Sync/v2/notes_studio_coordinator.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_coordinator.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_client_compound.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py
git commit -m "feat(sync): plan accepted Studio mutations"
```

### Task 5: Update Studio REST, OpenAPI, generated types, and docs

- [ ] **Step 1: Write API RED tests**

Cover manual accepted save, fetch, derive, regenerate, diagram, render/export compatibility, optional expected Studio/note revisions, inactive legacy omission behavior, exact nested compatibility stripping, mismatch `422`, canonical payload limit behavior only when internal capture-enabled harness is used, stable 404/409/413/422 mappings, and proof failures/previews/title suggestions create no canonical state.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_service.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  -k 'studio'
```

- [ ] **Step 3: Implement compatible REST contracts**

Add/complete a manual sidecar-save route using `NoteStudioDocumentUpsertRequest` with optional inactive-mode preconditions. Reduce generated provider dictionaries to the closed canonical schema before product persistence. Pass outer title/source/layout into render/regenerate/diagram/export explicitly and rehydrate legacy responses through `studio_compat.py`.

- [ ] **Step 4: Document and regenerate contracts**

Create `Docs/Notes/Studio.md` documenting canonical versus REST shapes, accepted-persistence boundary, provenance attestation, source/excerpt checks, stale binding, cache ownership, size limits, concurrency, and the dormant Sync disclaimer. Update `Docs/API/sync-v2.md` without advertising the domain.

```bash
PYTHON=../../.venv/bin/python bun --cwd apps/tldw-frontend run generate:api-types
PYTHONPATH=. ../../.venv/bin/python Helper_Scripts/export_openapi_schema.py \
  --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

Review the generated Studio types and complete OpenAPI semantic diff; commit only the approved fingerprint.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_service.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_compat.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  -k 'studio'
git add tldw_Server_API/app/api/v1/schemas/notes_studio.py \
  tldw_Server_API/app/api/v1/endpoints/notes.py \
  tldw_Server_API/app/core/Notes/studio_service.py \
  tldw_Server_API/app/core/Notes/studio_markdown.py \
  tldw_Server_API/app/core/Notes/studio_compat.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_service.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py \
  Docs/Notes/Studio.md Docs/API/sync-v2.md \
  apps/tldw-frontend/lib/api/openapi.fingerprint.json
git commit -m "feat(notes): harden accepted Studio contracts"
```

### Task 6: Prove tenancy, exclusion, repair, and close TASK-13007.4

- [ ] **Step 1: Run SQLite and required-live PostgreSQL matrices**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_adapter.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_materializer.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_coordinator.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_client_compound.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_studio_postgres_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_factory.py \
  tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py \
  tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_studio_db.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_compat.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_studio_service.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py \
  tldw_Server_API/tests/Services/test_openapi_contracts.py
```

Expected: PostgreSQL does not skip; RLS/source authorization, bootstrap diagnostics, canonical hashes, compound replay, cache exclusion, sanitization, and API compatibility all pass.

- [ ] **Step 2: Run static/security/OpenAPI/diff gates**

```bash
PRODUCTION_PATHS=(
  tldw_Server_API/app/core/DB_Management/Sync_DB.py
  tldw_Server_API/app/core/DB_Management/chacha/note_store.py
  tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_studio_document.py
  tldw_Server_API/app/core/Sync/v2/materializers/notes_studio_document.py
  tldw_Server_API/app/core/Sync/v2/materializers/__init__.py
  tldw_Server_API/app/core/Sync/v2/notes_studio_bootstrap.py
  tldw_Server_API/app/core/Sync/v2/notes_studio_coordinator.py
  tldw_Server_API/app/core/Sync/v2/store.py
  tldw_Server_API/app/core/Sync/v2/factory.py
  tldw_Server_API/app/core/Sync/v2/service.py
  tldw_Server_API/app/core/Sync/v2/replay.py
  tldw_Server_API/app/core/Sync/v2/restore.py
  tldw_Server_API/app/core/Notes/studio_compat.py
  tldw_Server_API/app/core/Notes/studio_markdown.py
  tldw_Server_API/app/core/Notes/studio_service.py
  tldw_Server_API/app/api/v1/schemas/notes_studio.py
  tldw_Server_API/app/api/v1/endpoints/notes.py
)
../../.venv/bin/ruff check --no-cache "${PRODUCTION_PATHS[@]}"
../../.venv/bin/bandit -q "${PRODUCTION_PATHS[@]}"
PYTHONPYCACHEPREFIX=/tmp/task13007-4-pycache ../../.venv/bin/python -m py_compile "${PRODUCTION_PATHS[@]}"
PYTHONPATH=. ../../.venv/bin/python Helper_Scripts/export_openapi_schema.py \
  --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
git diff --check
```

- [ ] **Step 3: Self-review accepted-state and dormancy boundaries**

Confirm note title/Markdown never enter Studio canonical payload, caches never enter hashes, prompts/credentials/raw/failed/unaccepted output never persist, client provenance cannot claim server attestation, compound replay compares incoming intent rather than stamped plan bytes, production REST/push creates no Studio envelopes, and Studio remains absent from public capabilities.

- [ ] **Step 4: Finalize the child**

Record exact evidence, check TASK-13007.4 AC/DoD, add concise implementation notes/touched files and any real lesson, set the child Done, and commit closeout. Do not activate Studio in this PR.
