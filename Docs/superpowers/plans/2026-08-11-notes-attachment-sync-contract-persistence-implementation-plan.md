# Notes Attachment Sync Contract And Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the adapter-v2 contract, version negotiation, schema-v59 Notes registry, immutable Sync revision bindings, version-aware cursors/acknowledgments, and PostgreSQL tenancy boundary without enabling Notes attachment mutations.

**Architecture:** ChaChaNotes owns current attachment metadata in a focused store, Sync owns immutable envelope/binding history and blob identity, and the feature remains gated off. Additive Sync tables dual-write version-1 cursor/ack state during rollback; v2 blob acknowledgments use blob ID. SQLite and PostgreSQL migrations are fail-closed and fresh/upgrade equivalent.

**Tech Stack:** Python 3.11, Pydantic, SQLite, PostgreSQL, FastAPI capability schemas, pytest, Ruff, Bandit.

**Design:** `Docs/superpowers/specs/2026-08-11-notes-attachment-sync-and-blob-lifecycle-design.md`

**Backlog task:** `TASK-13005.1`

**ADR required:** yes
**ADR path:** `Docs/ADR/038-canonical-notes-attachment-registry-and-blob-lifecycle.md`
**Reason:** This slice creates durable product/Sync schemas, adapter-version negotiation, ownership, RLS, and rollback contracts.

---

### Task 1: Lock the adapter-v2 and capability contract

**Files:**
- Create: `tldw_Server_API/app/core/Sync/v2/attachment_refs_v2.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/adapters.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`

- [ ] **Step 1: Write failing contract tests**

Cover exact lowercase UUIDv4/digest validation, positive sizes, extra-field rejection,
restore intent only in routing metadata, derived availability omission, immutable
creation fields, canonical object hash, adapter-v1 write rejection, v1/v2 object-ID
collision, version-map caps (100 domains/eight versions), and omission-as-v1. Add
exact acceptance vectors proving `created_at == normalize(created_at_client)`,
`created_by == authenticated_device_id`, server-origin identity follows the trusted
capture contract, legacy provenance is accepted only from a verified bootstrap step,
and `last_modified` equals the canonical mutation timestamp. Exact replay must not
enrich or rewrite any of those fields after submission.

```python
def test_attachment_ref_v2_rejects_client_availability() -> None:
    payload = _v2_payload(availability="available")
    with pytest.raises(AttachmentRefV2ValidationError):
        validate_attachment_ref_v2(payload)
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py \
  -k 'adapter_version or attachment_ref_v2'
```

Expected: failures for missing v2 model/parser/capability fields.

- [ ] **Step 3: Implement the minimal strict parser and models**

Keep one canonical payload model. Compute `object_hash` only from semantic fields;
exclude availability, resolved blob ID, storage status, and retention release.

```python
class AttachmentRefV2Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    attachment_id: UUID4
    parent_domain: Literal["notes.note"]
    parent_object_id: UUID4
    file_name: str
    original_file_name: str
    content_type: str
    size_bytes: int = Field(ge=1)
    blob_hash: str
    created_at: str
    last_modified: str
    created_by: str
```

- [ ] **Step 4: Advertise server-supported and dataset-writable versions separately**

Add the bounded device `supported_adapter_versions` map. Do not advertise v2 as
writable until `notes_attachment_v2.state == "ready"` and the dedicated rollout gate
is on. Existing devices with no map remain version 1.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py -k 'capabilit or adapter_version or attachment_ref'
git add tldw_Server_API/app/core/Sync/v2/attachment_refs_v2.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/app/core/Sync/v2/adapters.py \
  tldw_Server_API/app/core/Sync/v2/factory.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py
git commit -m "feat(sync): define attachment ref v2 contract"
```

### Task 2: Add schema-v59 registry and owner-scoped product store

**Files:**
- Create: `tldw_Server_API/app/core/DB_Management/chacha/note_attachment_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/__init__.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_migration_v59.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_store.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_postgres_tenancy.py`
- Test: `tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py`

- [ ] **Step 1: Write migration/store RED tests**

Assert SQLite v58→v59 and PostgreSQL SQL contracts, fresh/upgrade parity, transaction
rollback, two concurrent initializers serializing on the schema-version authority,
exact checks/indexes, same-owner note FK behavior, duplicate live-name conflict,
tombstone/restore CAS, cross-owner denial, and required live-PostgreSQL two-owner
isolation (skip only when DSN is unavailable). Assert bounded list/detail query counts
and PostgreSQL index-backed owner/dataset/note/name lookup plans.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_migration_v59.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_postgres_tenancy.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
```

Expected: schema/store/policy are absent.

- [ ] **Step 3: Implement v59 and the focused store**

Create `note_attachments` exactly as ADR-038 specifies. Use `client_id`, dataset ID,
stable attachment/note UUIDs, positive version/size, lowercase digest, lifecycle
coherence, and partial unique live filename. Expose only owner/dataset-scoped methods:
`get`, `list_page`, `create`, `compare_and_set`, `tombstone`, and `restore`.

- [ ] **Step 4: Implement verified PostgreSQL migration/RLS ordering**

Follow the v58 migration pattern: schema-version lock, relation locks, schema-owner
and FORCE-RLS catalog verification, exact temporary NO FORCE set, validation before
DDL/version, exact FORCE restoration, then full ChaCha RLS reinstall. `USING` and
`WITH CHECK` must verify both registry owner and referenced Notes owner.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_migration_v59.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_postgres_tenancy.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
git add tldw_Server_API/app/core/DB_Management/chacha/note_attachment_store.py \
  tldw_Server_API/app/core/DB_Management/chacha/__init__.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_migration_v59.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_store.py \
  tldw_Server_API/tests/ChaChaNotesDB/test_note_attachment_postgres_tenancy.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
git commit -m "feat(notes): add canonical attachment registry"
```

### Task 3: Add immutable Sync bindings and safe storage namespaces

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/blob_store.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_store.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_notes_postgres_contract.py`

- [ ] **Step 1: Write binding/namespace RED tests**

Cover immutable `(dataset,attachment,revision)` identity, pending→resolved CAS,
retention release monotonicity, exact digest/size match, v2 dataset namespace
confinement, and locked verify/copy/reverify/CAS relocation of a legacy global key
without unlinking it. Add SQLite query-count and PostgreSQL query-plan assertions for
binding lookup, unresolved-binding paging, and namespace resolution. Add paired
acceptance vectors with bytes already present vs absent: immutable
`availability_at_acceptance` must record the observed state, later exact digest/size
resolution may fill only the binding blob ID, and payload, object hash, request
fingerprint, envelope identity, and idempotent replay result must remain identical.

- [ ] **Step 2: Run RED, implement, and run GREEN**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_postgres_contract.py \
  -k 'attachment_binding or storage_namespace or legacy_blob_relocation'
```

Add `sync_attachment_revision_bindings` and
`sync_dataset_storage_namespaces`. Never select a v2 blob through the legacy
`attachment_id` provenance column. New storage paths contain only the server-issued
namespace and validated lowercase digest.

- [ ] **Step 3: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/blob_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_notes_postgres_contract.py
git commit -m "feat(sync): persist attachment revision bindings"
```

### Task 4: Migrate version-aware cursors and acknowledgments

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_store.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_models.py`

- [ ] **Step 1: Write RED migration/cursor/ack tests**

Test fresh/upgrade parity, seeding as v1, transactional v1 dual-write, max
reconciliation after simulated old-binary writes, version-set upgrade replay, opaque
token limits/signature/set mismatch, v1 numeric cursors, delivered-watermark ack
validation, blob-ID acknowledgments, and non-overwriting replacement evidence.

- [ ] **Step 2: Run RED and implement minimal side tables/readers**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  -k 'adapter_cursor or version_ack or blob_id_ack or pull_token'
```

Use new tables rather than rebuilding the legacy PK in place. Version-1 reads take
the verified max, writes dual-write in one transaction, v2 writes only new tables.
Reject a version-set removal unless the device is revoked/re-registered.

- [ ] **Step 3: Run GREEN, static checks, and commit**

```bash
PYTHONPATH=. ../../.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_attachment_refs.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py
../../.venv/bin/ruff check --no-cache \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/attachment_refs_v2.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py
../../.venv/bin/bandit -q \
  tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/Sync/v2/attachment_refs_v2.py \
  tldw_Server_API/app/core/Sync/v2/blob_store.py
git diff --check
git add tldw_Server_API/app/core/DB_Management/Sync_DB.py \
  tldw_Server_API/app/core/Sync/v2/models.py \
  tldw_Server_API/app/core/Sync/v2/store.py \
  tldw_Server_API/app/core/Sync/v2/service.py \
  tldw_Server_API/app/api/v1/schemas/sync_v2_models.py \
  tldw_Server_API/tests/Sync/test_sync_v2_store.py \
  tldw_Server_API/tests/Sync/test_sync_v2_service.py \
  tldw_Server_API/tests/Sync/test_sync_v2_models.py
git commit -m "feat(sync): version attachment cursors and acknowledgments"
```

### Task 5: Slice verification and disabled rollout proof

- [ ] Run the live PostgreSQL tenancy test when configured; record an honest skip otherwise.
- [ ] Prove `SYNC_V2_ENABLE_NOTES_ATTACHMENT_SYNC` defaults off and no Notes REST mutation path uses v2 yet.
- [ ] Run touched Ruff/formatter, Bandit, `py_compile`, and `git diff --check` with exact paths.
- [ ] Update `TASK-13005.1` implementation notes, check its AC/DoD, and mark only
  that child Done.
