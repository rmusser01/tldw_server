# Notes Moodboard and Studio Contract and Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish strict dormant contracts and schema-v61 tenant-scoped product storage for moodboards, manual placements, Studio sidecars, and per-graph scope authority without exposing any new public Sync capability.

**Architecture:** ChaChaNotes remains product authority. Schema v61 transactionally upgrades the legacy moodboard, placement, Studio, and scope-authority relations; legacy product rows remain owner-proven in `local-unbound` scope until an explicit graph binder rekeys them. One frozen contract module owns canonical JSON, identifiers, hashes, limits, and legacy diagnostics, while one readiness module validates private metadata and all public supported/writable maps continue to omit the three domains.

**Tech Stack:** Python 3.11, Pydantic v2, SQLite, PostgreSQL forced RLS, FastAPI Sync v2 models, pytest, Ruff, Bandit.

**Design:** `Docs/superpowers/specs/2026-08-24-notes-moodboard-studio-sync-design.md`

**Backlog task:** `TASK-13007.1`

**ADR required:** no
**ADR path:** `Docs/ADR/040-synchronized-moodboards-and-studio-authority.md`
**Reason:** ADR-040 already approves the storage, identity, scope-authority, canonicalization, readiness, and dormant-rollout boundaries implemented by this child.

---

## File map

**Create**

- `tldw_Server_API/app/core/Sync/v2/notes_moodboard_studio_contract.py` — sole v1 parser, canonical serializer, hashes, placement identity, bounds, and legacy diagnostic codes for all three domains.
- `tldw_Server_API/app/core/Sync/v2/notes_moodboard_studio_readiness.py` — pure parsing/redaction for three private readiness records and coupled moodboard readiness.
- `tldw_Server_API/app/core/DB_Management/chacha/moodboard_sync_store.py` — scoped storage/binding primitives used by later adapters, bootstrap, and repair without moving legacy REST methods.
- `tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_contract.py` — exact cross-runtime vectors, field bounds, identity, lineage, and dormant capability tests.
- `tldw_Server_API/tests/ChaChaNotesDB/test_notes_moodboard_studio_sync_migration_v61.py` — SQLite fresh/60→61/rollback/catalog/legacy conversion proof.
- `tldw_Server_API/tests/ChaChaNotesDB/test_notes_moodboard_studio_sync_postgres_tenancy.py` — required live PostgreSQL RLS, catalog, race, and plan proof.

**Modify**

- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` — schema v61 DDL, resumable PostgreSQL migration progress, bounded conversion, catalog verification, runtime Studio helper demotion, and moodboard sync-store composition.
- `tldw_Server_API/app/core/DB_Management/chacha/task_store.py` — flag-aware sole authority lookup and atomic task-graph binding.
- `tldw_Server_API/app/core/DB_Management/chacha/note_store.py` — scoped Studio row reads/writes and canonical lineage columns while retaining REST compatibility.
- `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py` — exact owner/dataset product policies and owner-only authority policy.
- `tldw_Server_API/app/core/DB_Management/Sync_DB.py` — private readiness transitions for moodboard, placement, and Studio.
- `tldw_Server_API/app/core/Sync/v2/models.py` — known-but-dormant domain literals, operations, and private schemas only.
- `tldw_Server_API/app/core/Sync/v2/store.py` — readiness facade methods.
- `tldw_Server_API/app/core/Sync/v2/profile.py` — authorized, privacy-safe internal readiness diagnostics.
- `tldw_Server_API/app/core/Notes_Tasks/service.py` — compatibility scope resolution only when `task_graph_bound=true`.
- `tldw_Server_API/app/core/Notes_Tasks/reconciler.py` — the same flag-aware task scope forwarding.
- `tldw_Server_API/app/api/v1/endpoints/notes_tasks.py` — preserve existing task API behavior through flag-aware scope resolution.
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py` — preserve task MCP behavior through flag-aware scope resolution.
- `tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py` — upgrade fixture and v61 authority amendment coverage.
- `tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_postgres_tenancy.py` — old-binder insert and graph-flag race coverage.
- `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py` — row-present/task-unbound compatibility tests.
- `tldw_Server_API/tests/ChaChaNotesDB/test_note_studio_db.py` — legacy Studio CRUD compatibility against the v61 table.
- `tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py` — legacy moodboard CRUD compatibility against scoped v61 rows.
- `tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py` — exact new policy-source contract.
- `tldw_Server_API/tests/Sync/test_sync_v2_models.py` — known/private versus advertised domain boundary.
- `tldw_Server_API/tests/Sync/test_sync_v2_store.py` — readiness transition and rollback coverage.

### Task 0: Start the child and attach the reviewed plan

- [ ] **Step 1: Confirm approved inputs and task state**

```bash
backlog task edit 13007.1 -s "In Progress"
backlog task 13007.1 --plain
git status --short
```

Expected: TASK-13007.1 is In Progress, links ADR-040 and the approved design, and the worktree contains only the approved planning checkpoint before production edits.

- [ ] **Step 2: Attach this plan to the task**

```bash
backlog task edit 13007.1 \
  --doc Docs/superpowers/plans/2026-08-25-notes-moodboard-studio-contract-storage-implementation-plan.md \
  --plan $'1. Lock strict dormant v1 contracts and capability exclusion.\n2. Upgrade SQLite product storage and per-graph scope authority to schema v61.\n3. Enforce exact PostgreSQL catalog and forced-RLS tenancy.\n4. Add private readiness and demote the runtime Studio schema helper.\n5. Prove REST compatibility and run the required PR gate.\n\nADR required: no\nADR path: Docs/ADR/040-synchronized-moodboards-and-studio-authority.md\nReason: ADR-040 already governs this child.'
```

### Task 1: Lock canonical v1 contracts and dormant catalog entries

- [ ] **Step 1: Write canonical JSON and identity RED tests**

Cover sorted ASCII keys, compact UTF-8 JSON, `ensure_ascii=false`, NaN/float rejection, JS-safe integers, timestamp normalization, lowercase UUIDv4, extension-key syntax, depth/count/byte limits, exact object-hash framing, and the exact placement identity vector.

```python
def test_placement_id_is_exact_namespaced_digest() -> None:
    payload = valid_placement_payload(
        moodboard_id="253fbb6d-8bc9-4e7f-bce0-56ac1fd46227",
        note_id="28467075-bde3-4478-883c-125a5672873c",
    )
    assert placement_object_id(payload) == EXPECTED_NAMESPACED_SHA256


def test_canonical_json_rejects_float_even_when_finite() -> None:
    with pytest.raises(SyncContractError, match="integer"):
        canonical_json_bytes({"x": 1.0})
```

- [ ] **Step 2: Write closed payload RED tests**

Add one valid vector and one rejection per approved bound/cross-field rule for moodboard smart rules/canvas, placements/display, Studio sections, diagram manifest, and provenance. Explicitly test sections-only Studio state, `cached_svg` rejection, provider/model pairing, excerpt/source pairing, source-graph equality, render/result hashes, and tombstones retaining complete payloads.

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_contract.py
```

Expected: import failure because the contract module does not exist.

- [ ] **Step 4: Implement one frozen contract module**

Use strict frozen Pydantic models and one canonical serializer. Keep server-bound owner/dataset, acceptance timestamps, and attestation out of client-selectable payload fields.

```python
class NotesMoodboardNoteV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    moodboard_id: UUID4
    note_id: UUID4
    x: Annotated[int, Field(ge=JS_SAFE_MIN, le=JS_SAFE_MAX)]
    y: Annotated[int, Field(ge=JS_SAFE_MIN, le=JS_SAFE_MAX)]
    width: Annotated[int, Field(ge=1, le=1_000_000)]
    height: Annotated[int, Field(ge=1, le=1_000_000)]
    order_index: Annotated[int, Field(ge=JS_SAFE_MIN, le=JS_SAFE_MAX)]
    display: BoundedExtensionMap


class NotesStudioDocumentV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    note_id: UUID4
    source_note_id: UUID4 | None
    payload_json: StudioSectionsV1
    template_type: Literal["lined", "grid", "cornell"]
    handwriting_mode: Literal["off", "accented"]
    excerpt_snapshot: str | None
    excerpt_hash: Sha256Digest | None
    diagram_manifest_json: StudioDiagramManifestV1 | None
    companion_content_hash: Sha256Digest
    render_version: Literal[1]
    note_revision: Annotated[int, Field(ge=1)]
    note_hash: Sha256Digest
    accepted_provenance: StudioAcceptedProvenanceV1
```

Expose `parse_*_v1`, `canonical_json_bytes`, `placement_object_id`, `*_object_hash`, `studio_result_hash`, `diagram_render_hash`, and bounded privacy-safe legacy diagnostic helpers. Do not duplicate these rules in DB or API layers.

- [ ] **Step 5: Add known-but-dormant domains**

Extend `SyncDomain`, `SYNC_V2_KNOWN_DOMAINS`, `SYNC_V2_INTERNAL_OPERATIONS`, and `_sync_v2_internal_domain_schemas()` for:

```python
NOTES_MOODBOARD_STUDIO_DOMAINS = (
    "notes.moodboard",
    "notes.moodboard_note",
    "notes.studio_document",
)
```

Keep them absent from `SYNC_V2_SUPPORTED_DOMAINS`, `SYNC_V2_SUPPORTED_OPERATIONS`, server-supported version output, dataset writable maps, and factory public validation.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py
git add tldw_Server_API/app/core/Sync/v2/notes_moodboard_studio_contract.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py
git commit -m "feat(sync): define dormant moodboard Studio contracts"
```

### Task 2: Upgrade SQLite product storage and sole scope authority to v61

- [ ] **Step 1: Write fresh/upgrade/rollback RED tests**

Assert current schema 61, fresh/60→61 catalog parity, exact moodboard/placement/Studio columns, scoped unique keys, parent consistency, Boolean storage, indexes, local-unbound conversion, canonical lineage, diagnostics, version-last behavior, and rollback at every create/copy/index/verify checkpoint.

Add exact authority tests for:

- defaults `task_graph_bound=true`, moodboard/Studio false;
- existing authority rows verified before migrating task true;
- row presence with task false resolving to local-unbound;
- explicit all-flag inserts by new binders;
- immutable same-dataset replay and wrong-dataset rejection;
- empty graph binding; and
- interleaved first enrollment where one dataset wins without partial rekey.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_notes_moodboard_studio_sync_migration_v61.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py
```

Expected: schema remains 60 and graph flags/scoped product columns are absent.

- [ ] **Step 3: Implement fixed SQLite v61 migration**

Add `_migrate_from_v60_to_v61_sqlite(conn)` using create/copy/verify/swap/index/version-last ordering. Generate UUIDv4 moodboard IDs once during conversion, derive placement identity in Python, retain legacy rows and diagnostics, default existing canvases to masonry, retain Studio sidecars, and never fabricate historically deleted links.

The migration must reject ambiguous owner proof, duplicate portable IDs, malformed/oversized canonical state, incompatible Studio nested authority, or task-graph mismatch before blessing readiness.

- [ ] **Step 4: Implement flag-aware authority and scoped stores**

Update task callers so the query is explicit:

```sql
SELECT dataset_id
FROM note_task_scope_authority
WHERE owner_user_id = ? AND task_graph_bound = 1
```

Add moodboard and Studio graph bind helpers that lock the authority row, insert all flags explicitly when absent, verify/rekey only their complete graph, flip one flag in the same transaction, and reject a different immutable target. `moodboard_sync_store.py` owns scoped sync/bootstrap reads; legacy REST methods retain current signatures and resolve to local-unbound before activation.

- [ ] **Step 5: Demote the runtime Studio schema helper**

Change `_ensure_note_studio_schema_*` from competing `CREATE TABLE IF NOT EXISTS` authority to exact current-schema verification/delegation. Current v61 startup fails closed on drift instead of silently creating a legacy table.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_notes_moodboard_studio_sync_migration_v61.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_studio_db.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/moodboard_sync_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/task_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/note_store.py \
  tldw_Server_API/app/core/Notes_Tasks/service.py \
  tldw_Server_API/app/core/Notes_Tasks/reconciler.py \
  tldw_Server_API/app/api/v1/endpoints/notes_tasks.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_notes_moodboard_studio_sync_migration_v61.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_chacha_task_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_studio_db.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py
git commit -m "feat(notes): add scoped moodboard Studio storage"
```

### Task 3: Run a bounded resumable PostgreSQL v61 migration and enforce exact RLS

- [ ] **Step 1: Write server-free catalog and required-live RED tests**

Cover fixed lock order, explicit `lock_timeout`/`statement_timeout`, columns/defaults/types/nullability, PK/FK/check/index definitions, table/schema owner, ENABLE+FORCE RLS, exact `USING` and `WITH CHECK`, extra-policy rejection, two-owner/same-ID coexistence, wrong-dataset denial, relationship injection, old-binder inserts, first-enrollment races, and indexed keyset plans using a non-table-owner role.

Seed a large legacy fixture and inject failure after every schema, bounded-copy page, index, constraint-validation, RLS, aggregate-verification, and version step. Assert each retry resumes from durable privacy-safe migration progress, never recopies a completed page incompatibly, and never exposes a partially blessed v61 catalog.

- [ ] **Step 2: Run RED with PostgreSQL required**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_notes_moodboard_studio_sync_postgres_tenancy.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
```

Expected: fail, never skip; v61 PostgreSQL schema, resumable progress protocol, and policies do not exist.

- [ ] **Step 3: Implement the bounded resumable PostgreSQL v61 algorithm**

Acquire the schema-version row lock first and fixed relation locks only for each short metadata transition. Set bounded lock and statement timeouts on every transaction. Create additive target structures plus a private `chacha_schema_migration_progress` record keyed by migration/phase; it stores only keyset cursor, count, aggregate fingerprint, status, and update time.

Copy/rekey legacy rows in deterministic `(owner_user_id, legacy_primary_key)` pages under explicit row and wall-clock budgets, committing progress with each page. Resume from the last verified cursor after interruption. After source count/fingerprint and target postconditions match, build/verify indexes, add or validate constraints, install forced RLS, run exact catalog verification, mark progress complete, and bump schema version last. Current-v61 startup verifies rather than repairs drift. Every phase is idempotent so a timeout or crash is a safe retry, not an instruction to restart an unbounded copy.

- [ ] **Step 4: Install exact policies and parent-scope enforcement**

Force RLS on `moodboards`, `moodboard_notes`, and `note_studio_documents`. Retain owner-only forced RLS on `note_task_scope_authority`. Use composite constraints where the current parent catalog permits them and same-transaction store validation otherwise; RLS alone never proves matching parent dataset.

- [ ] **Step 5: Run live GREEN and commit**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/ChaChaNotesDB/test_notes_moodboard_studio_sync_migration_v61.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_notes_moodboard_studio_sync_postgres_tenancy.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_postgres_tenancy.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_notes_moodboard_studio_sync_postgres_tenancy.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_postgres_tenancy.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
git commit -m "feat(notes): enforce moodboard Studio tenancy"
```

Expected: the required-live large upgrade executes multiple pages, survives every injected boundary failure, resumes from stored progress, observes the configured timeouts, and finishes with exact v61 catalog/RLS/version state.

### Task 4: Persist private readiness without capability exposure

- [ ] **Step 1: Write readiness RED tests**

Test exact independent records through `not_enrolled -> enrolling -> bootstrapping -> verifying -> ready` and `blocked`, monotonic keyset cursor/count/fingerprint, fixed reason codes, malformed metadata, local-unbound rejection, capture-disabled invariant, coupled board/placement state, independent Studio state, transaction rollback, and redacted diagnostics. Assert all public capability surfaces omit all three domains even for forged ready metadata.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  -k 'moodboard or studio_document or dormant'
```

- [ ] **Step 3: Implement pure parsing and one transition boundary**

`notes_moodboard_studio_readiness.py` validates exact metadata and returns structured parse errors without raising into profile responses. `SyncDatabase` owns transaction-held compare-and-set transitions; `SyncV2Store` only delegates. Capture flags cannot become true in this child.

- [ ] **Step 4: Expose privacy-safe internal status only**

Authorized diagnostics may expose state, stable code, bounded count, resume phase, and hashed cursor/fingerprint. Never expose names, note text, Studio content, excerpts, provider request data, or local-unbound identifiers.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  -k 'moodboard or studio_document or dormant'
git add tldw_Server_API/app/core/Sync/v2/notes_moodboard_studio_readiness.py \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/profile.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py
git commit -m "feat(sync): persist dormant moodboard Studio readiness"
```

### Task 5: Prove compatibility and close TASK-13007.1

- [ ] **Step 1: Run the full required child matrix**

```bash
TLDW_TEST_POSTGRES_REQUIRED=1 PYTHONPATH=. ../../.venv/bin/python -m pytest -q --tb=short \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_moodboard_studio_contract.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_notes_moodboard_studio_sync_migration_v61.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_notes_moodboard_studio_sync_postgres_tenancy.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_migration_v60.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_task_sync_postgres_tenancy.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_studio_db.py \
  tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py \
  tldw_Server_API/tests/Notes_NEW/integration/test_notes_studio_api.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
```

Expected: all pass; PostgreSQL tests do not skip; existing REST shapes and behavior remain compatible.

- [ ] **Step 2: Run static, security, and diff gates**

```bash
PRODUCTION_PATHS=(
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
  tldw_Server_API/app/core/DB_Management/chacha/moodboard_sync_store.py
  tldw_Server_API/app/core/DB_Management/chacha/task_store.py
  tldw_Server_API/app/core/DB_Management/chacha/note_store.py
  tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py
  tldw_Server_API/app/core/Notes_Tasks/service.py
  tldw_Server_API/app/core/Notes_Tasks/reconciler.py
  tldw_Server_API/app/api/v1/endpoints/notes_tasks.py
  tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py
  tldw_Server_API/app/core/DB_Management/Sync_DB.py
  tldw_Server_API/app/core/Sync/v2/notes_moodboard_studio_contract.py
  tldw_Server_API/app/core/Sync/v2/notes_moodboard_studio_readiness.py
  tldw_Server_API/app/core/Sync/v2/models.py
  tldw_Server_API/app/core/Sync/v2/store.py
  tldw_Server_API/app/core/Sync/v2/profile.py
)
../../.venv/bin/ruff check --no-cache "${PRODUCTION_PATHS[@]}"
../../.venv/bin/bandit -q "${PRODUCTION_PATHS[@]}"
PYTHONPYCACHEPREFIX=/tmp/task13007-1-pycache ../../.venv/bin/python -m py_compile "${PRODUCTION_PATHS[@]}"
git diff --check
```

- [ ] **Step 3: Self-review dormant and authority invariants**

Confirm there is no production enrollment/capture path, no public capability entry, no second scope-authority table, no row-presence task binding, no unscoped PostgreSQL query, no raw diagnostic content, and no destructive downgrade migration.

- [ ] **Step 4: Finalize the Backlog task and commit evidence**

Check every TASK-13007.1 AC/DoD item only after evidence exists, add concise implementation notes with touched files and exact test/Bandit results, record any genuine lesson learned, set the child Done, and commit the closeout documentation. Do not start TASK-13007.2 or TASK-13007.4 until this commit is review-clean.
