# Notes Organization Sync v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Notes keywords, keyword collections, folders, and their user-visible memberships durable Sync v2 state on SQLite and PostgreSQL without breaking integer REST identifiers, legacy devices, or existing datasets.

**Architecture:** Implement the six-domain contract approved in the design and ADR-032 as a vertical extension of the existing strict `notes.note` path. Stable product-store identities and deterministic relationship IDs feed strict domain adapters; accepted compound mutations are atomically appended to the Sync store as ordered groups and then materialized, resumably, into a per-user ChaChaNotes database. Existing datasets enter an explicit initializing state while current organization state is captured, and active-Sync REST routes use one small Notes organization coordinator rather than maintaining a parallel write path.

**Tech Stack:** Python 3.11+, FastAPI/Pydantic, SQLite and PostgreSQL database adapters, Sync v2 envelope store and materializers, pytest/pytest-asyncio, Bandit and Ruff.

## Global Constraints

- Execute tasks in order and one at a time. Do not parallelize tests or implementation work.
- Apply test-driven development for every behavior change: write the named focused test, run it and observe the specified failure, implement the smallest production change, then rerun the same test.
- Treat `Docs/superpowers/specs/2026-08-08-notes-organization-sync-design.md`, `Docs/ADR/032-durable-server-origin-sync-mutation-batches.md`, and `Docs/ADR/033-canonical-folder-link-suppression-preserves-source-provenance.md` as normative.
- Preserve existing integer REST and storage IDs. Only canonical Sync identities use resource `sync_id` values or deterministic relationship hashes.
- Never publish part of a mutation group to the Sync log. Product materialization may be a durable applied prefix, but it must resume in `mutation_step` order and must not skip a failed or conflicted step.
- The six organization domains are one enrollment/readiness group. Partial enrollment and writes while `initializing` or `failed` must fail closed.
- Do not add the organization domains to the generic media compatibility adapter.
- Do not synchronize flashcards, source IDs, folder keys, file paths, source content, credentials, FTS rows, display counts, or other derived data.
- Do not convert existing soft deletes into cascades. Dormant relationships and hierarchy pointers survive resource tombstones unless an explicit unlink or merge tombstones them.
- Keep route handlers thin. The portable coordinator owns plans and Sync capture; materializers and the ChaCha persistence seam own product projection.
- Use parameterized SQL and the repository's database transaction abstractions. Validate every payload and path component at the boundary.
- Run only the focused command listed at each red-green checkpoint. Run the consolidated task suite once in Task 10.
- Each implementation commit must leave the focused tests for that task green and pass `git diff --check`.

---

### Task 1: Add the public six-domain contract and deterministic identity helpers

**Files:**

- Create: `tldw_Server_API/app/core/Sync/v2/notes_organization.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Test: `tldw_Server_API/tests/Sync/test_sync_v2_models.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_identity.py`

- [ ] **Step 1: Write failing capability and schema tests**

Add assertions that `SyncDomain`, `sync_v2_domain_schemas()`, and the capability payload expose exactly these organization domains at schema version 1 with `upsert` and `tombstone`:

```python
NOTES_ORGANIZATION_DOMAINS = (
    "notes.keyword",
    "notes.keyword_link",
    "notes.keyword_collection",
    "notes.keyword_collection_link",
    "notes.folder",
    "notes.folder_link",
)

@pytest.mark.parametrize("domain", NOTES_ORGANIZATION_DOMAINS)
def test_notes_organization_schema_is_server_trusted_v1(domain: str) -> None:
    schema = sync_v2_domain_schemas()[domain]
    assert schema["schema_version"] == 1
    assert schema["encryption_policy"] == "server_trusted_v1"
    assert {"upsert", "tombstone"}.issubset(schema)
```

Run:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_models.py -k notes_organization
```

Expected: FAIL because the six domains and their schemas are absent.

- [ ] **Step 2: Implement the domain group and strict public payload schemas**

In `models.py`, extend `SyncDomain`, add the immutable `NOTES_ORGANIZATION_DOMAINS` tuple, and add six schemas to `sync_v2_domain_schemas()`. Keep this group separate from `M1_SYNC_DOMAINS` so existing datasets are not silently enrolled.

In `notes_organization.py`, define strict Pydantic models with `extra="forbid"`:

```python
class KeywordUpsertPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keyword: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]

class KeywordLinkPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject_type: Literal["note", "conversation"]
    subject_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    keyword_sync_id: str

class KeywordCollectionUpsertPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
    parent_sync_id: str | None = None

class KeywordCollectionLinkPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_sync_id: str
    keyword_sync_id: str

class FolderUpsertPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    parent_sync_id: str | None = None

class FolderLinkPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note_id: str
    folder_sync_id: str
```

Resource tombstones accept `{}` only. Link tombstones validate the same identity payload as link upserts. Add `parse_notes_organization_payload(domain, operation, payload)` returning a normalized `dict[str, object]` or raising the existing Sync validation error.

- [ ] **Step 3: Write failing deterministic-ID vectors and invalid-ID tests**

Test all three required vectors verbatim:

```python
assert organization_link_id(
    "notes.keyword_link", ["note", "note-123", "kw-456"]
) == "notes.keyword_link:sha256:10f9eab3be80b6e439ce1bcf8fae952527bde7d7e026d0e227f0a87ada963be0"
assert organization_link_id(
    "notes.keyword_collection_link", ["collection-123", "kw-456"]
) == "notes.keyword_collection_link:sha256:e9427c2d8bc4cfa8586130bc1fcc54cf432ca6dbb3df77bab3e65033b6148199"
assert organization_link_id(
    "notes.folder_link", ["note-123", "folder-456"]
) == "notes.folder_link:sha256:9076b60d9d8476f852736928ef3661cb06d9ba55696dd4504657c753f414b670"
```

Also reject a wrong member count, unsupported relationship domain, uppercase digest, truncated digest, and a payload whose members do not reproduce `object_id`.

Run:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_identity.py
```

Expected: FAIL because the identity helpers do not exist.

- [ ] **Step 4: Implement canonical relationship IDs and UUID validation**

Add these public interfaces in `notes_organization.py`:

```python
def new_organization_sync_id() -> str:
    """Return a canonical lowercase UUIDv4 string."""

def validate_resource_sync_id(value: str) -> str:
    """Return a canonical UUIDv4 string or raise a Sync validation error."""

def organization_link_id(domain: SyncDomain, members: Sequence[str]) -> str:
    """Hash canonical UTF-8 JSON with sorted keys and compact separators."""

def validate_organization_object_id(
    domain: SyncDomain,
    object_id: str,
    payload: Mapping[str, object],
) -> None:
    """Validate a resource UUID or recompute and compare a relationship ID."""
```

The canonical byte string must come from:

```python
json.dumps(
    {"domain": domain, "members": list(members), "schema_version": 1},
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
).encode("utf-8")
```

- [ ] **Step 5: Verify and commit Task 1**

Run, in order:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_models.py -k notes_organization
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_identity.py
git diff --check
git add tldw_Server_API/app/core/Sync/v2/models.py tldw_Server_API/app/core/Sync/v2/notes_organization.py tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_identity.py
git commit -m "feat(sync): define Notes organization domains"
```

Expected: both focused test commands pass.

---

### Task 2: Add ChaCha schema v55 stable identities and a focused projection seam

**Files:**

- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/keyword_store.py`
- Create: `tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py`
- Create: `Docs/ADR/033-canonical-folder-link-suppression-preserves-source-provenance.md`
- Modify: `Docs/ADR/README.md`
- Modify: `Docs/superpowers/specs/2026-08-08-notes-organization-sync-design.md`
- Modify: `tldw_Server_API/app/api/v1/schemas/notes_schemas.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_note_folders_postgres.py`
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v55.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_postgres_contract.py`

- [ ] **Step 1: Write failing v54-to-v55 migration tests**

Build a schema-v54 fixture containing active and soft-deleted keyword, collection, and folder rows. After initialization, assert:

- schema version is 55;
- every row has a non-null canonical UUIDv4 `sync_id`;
- IDs are unique across rows within each table;
- rerunning initialization preserves every generated ID; and
- the SQLite migration map contains a `54 -> 55` step.

Run:

```bash
pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v55.py
```

Expected: FAIL because schema v55 and `sync_id` columns do not exist.

- [ ] **Step 2: Implement backend-equivalent v55 migration**

Set `_CURRENT_SCHEMA_VERSION = 55`, add `_migrate_from_v54_to_v55`, register it in `_sqlite_linear_migration_steps()`, and update `_initialize_schema_postgres()` for fresh and upgraded databases.

For SQLite, rebuild tables only where required to guarantee `NOT NULL`; preserve all IDs, soft-deletion fields, foreign keys, and indexes. For PostgreSQL, add nullable columns, backfill UUIDv4 strings in application code, verify no null/duplicate values, then set `NOT NULL` and create unique indexes. Use explicit index names:

```text
idx_keywords_sync_id_unique
idx_keyword_collections_sync_id_unique
idx_note_folders_sync_id_unique
```

Do not derive a Sync ID from integer IDs, names, timestamps, or paths.

- [ ] **Step 3: Write failing product-store identity and hierarchy tests**

Cover:

- new keyword/collection/folder rows receive an ID before insertion;
- `get_*`, list, create, restore, and mutation responses retain the same `sync_id`;
- case-insensitive keyword and collection uniqueness remains unchanged;
- folder descendant paths recalculate transactionally after rename/move;
- self-parenting, ancestor cycles, missing/deleted parents, and paths over 500 characters fail without partial changes; and
- soft deletion keeps parent pointers and relationship rows intact.
- source-backed folder links remain locally preserved but are excluded from effective reads and snapshots after a canonical folder-link tombstone; a canonical upsert restores visibility.
- pre-existing cycles are rejected during both ancestor and descendant traversal without looping or partial mutation.

Run:

```bash
pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py -k "sync_id or hierarchy or soft_delete"
```

Expected: FAIL on missing returned IDs and missing Sync-oriented hierarchy operations.

- [ ] **Step 4: Implement the portable ChaCha organization projection seam**

Create backend-neutral dataclasses and methods rather than embedding SQL in Sync materializers:

```python
@dataclass(frozen=True)
class OrganizationResource:
    domain: SyncDomain
    sync_id: str
    local_id: int
    name: str
    parent_sync_id: str | None
    deleted: bool
    version: int

@dataclass(frozen=True)
class OrganizationRelationship:
    domain: SyncDomain
    object_id: str
    payload: Mapping[str, object]

@dataclass(frozen=True)
class OrganizationSnapshot:
    resources: tuple[OrganizationResource, ...]
    relationships: tuple[OrganizationRelationship, ...]

class NotesOrganizationSyncStore:
    def get_resource(
        self, domain: SyncDomain, sync_id: str
    ) -> OrganizationResource | None:
        """Return one active or deleted resource by canonical identity."""

    def snapshot(self) -> OrganizationSnapshot:
        """Return a transactionally consistent organization snapshot."""

    def apply_resource(
        self,
        *,
        domain: SyncDomain,
        object_id: str,
        operation: SyncOperation,
        payload: Mapping[str, object],
    ) -> OrganizationResource:
        """Apply one resource envelope in a ChaCha transaction."""

    def apply_relationship(
        self,
        *,
        domain: SyncDomain,
        object_id: str,
        operation: SyncOperation,
        payload: Mapping[str, object],
        routing_metadata: Mapping[str, object],
    ) -> None:
        """Apply one relationship envelope in a ChaCha transaction."""
```

The class receives one initialized per-user `CharactersRAGDB`. Each public apply method uses one ChaCha transaction. Folder subtree path updates happen inside that transaction. `routing_metadata.bootstrap_capture` and origin provenance deltas are not interpreted here until Tasks 6 and 9.

Implement ADR-033 in fresh SQLite/PostgreSQL schemas and the v54-to-v55 migration.
`note_folder_sync_suppressions` has one unique `(note_id, folder_id)` pair. Folder
relationship reads and snapshots use `(manual UNION source) MINUS suppression`.
Canonical folder-link upsert removes suppression and ensures the manual row;
tombstone removes the manual row and inserts suppression. Never delete source
membership or source-key rows from the canonical apply path.

Extend keyword and folder CRUD responses and Pydantic response schemas additively with `sync_id`. Keep integer `id` fields unchanged.

- [ ] **Step 5: Verify PostgreSQL DDL and projection parity**

Add mock-cursor contract assertions for parameter style, `chacha_keywords` table mapping, unique indexes, `NOT NULL`, and transaction boundaries. Extend the live PostgreSQL folder test behind its existing environment skip.

Run:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_postgres_contract.py
```

Expected after implementation: PASS without requiring a live PostgreSQL server; the existing optional integration remains skip-safe.

- [ ] **Step 6: Verify and commit Task 2**

Run, in order:

```bash
pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v55.py
pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py -k "sync_id or hierarchy or soft_delete"
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_postgres_contract.py
git diff --check
git add Docs/ADR/033-canonical-folder-link-suppression-preserves-source-provenance.md Docs/ADR/README.md Docs/superpowers/specs/2026-08-08-notes-organization-sync-design.md tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha/keyword_store.py tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py tldw_Server_API/app/api/v1/schemas/notes_schemas.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py tldw_Server_API/tests/ChaChaNotesDB/test_note_folders_postgres.py tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v55.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_postgres_contract.py
git commit -m "feat(notes): add stable organization identities"
```

---

### Task 3: Add atomic mutation-group storage to the Sync database

**Files:**

- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_models.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_store.py`

- [ ] **Step 1: Write failing envelope metadata and all-or-none append tests**

Add `mutation_group_id`, `mutation_step`, `mutation_step_count`, and `mutation_plan_hash` round-trip tests. Verify the four fields are either all absent for legacy single envelopes or all present and internally valid:

```python
assert envelope.mutation_step == 0
assert envelope.mutation_step_count == 3
assert envelope.mutation_plan_hash == expected_sha256
```

Store tests must prove:

- `insert_envelopes_atomic()` inserts a complete ordered group;
- duplicate `(dataset_id, mutation_group_id, mutation_step)` returns the identical existing group only when every envelope fingerprint matches;
- a mismatched replay is a stable idempotency conflict;
- an injected failure at step 2 leaves zero rows; and
- a single legacy envelope still works.

Run:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_store.py -k mutation_group
```

Expected: FAIL because group columns and atomic batch insertion are absent.

- [ ] **Step 2: Extend envelope models and canonical fingerprints**

Add optional group fields to `SyncEnvelopeCreate` and required-consistent optional fields to `SyncEnvelope`. Validate group IDs as non-empty opaque strings, steps as zero-based, counts as positive, `step < count`, and plan hashes as lowercase SHA-256 hex.

Include group metadata in envelope fingerprints, serialization, row conversion, and idempotency comparison. A legacy envelope with all four values absent must retain its existing fingerprint behavior.

- [ ] **Step 3: Implement the migration-safe Sync schema extension**

Extend fresh-table DDL and `_ensure_envelope_m1_columns()` with the four nullable fields. Add indexes through `_ensure_envelope_m1_indexes()`:

```text
UNIQUE(dataset_id, mutation_group_id, mutation_step)
INDEX(dataset_id, mutation_group_id, mutation_step)
```

Use backend-compatible partial uniqueness where required so legacy null group IDs do not collide.

- [ ] **Step 4: Implement atomic store interfaces**

Add:

```python
def insert_envelopes_atomic(
    self, envelopes: Sequence[SyncEnvelopeCreate]
) -> list[SyncEnvelope]:
    """Insert one complete validated group or return its exact stored replay."""

def list_mutation_group(
    self, dataset_id: str, mutation_group_id: str
) -> list[SyncEnvelope]:
    """Return a complete mutation group ordered by zero-based step."""
```

Validate before opening the transaction that all envelopes share dataset, group ID, step count, and plan hash, and that steps are exactly `range(step_count)`. Allocate server sequences and insert every row inside one Sync DB transaction. On duplicate group lookup, compare the complete ordered plan before returning the stored group.

- [ ] **Step 5: Verify and commit Task 3**

Run, in order:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_models.py -k mutation_group
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_store.py -k mutation_group
git diff --check
git add tldw_Server_API/app/core/Sync/v2/models.py tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/app/core/Sync/v2/store.py tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py
git commit -m "feat(sync): persist atomic mutation groups"
```

---

### Task 4: Evaluate, append, and resume compound server-origin groups

**Files:**

- Modify: `tldw_Server_API/app/core/Sync/v2/adapters.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/domain_adapters/_lineage.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Create: `tldw_Server_API/app/core/Sync/v2/server_origin_batch.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_server_origin_batch.py`

- [ ] **Step 1: Write failing virtual-head and group-resume tests**

Prove that later planned steps evaluate against earlier accepted steps before insertion. Cover one object updated twice in a group, a parent created before a child, a relationship created after both resources, and a dependency ordered after its consumer being rejected.

Group capture tests must prove:

- evaluation failure inserts no envelopes and does not touch ChaChaNotes;
- append failure inserts no envelopes and does not touch ChaChaNotes;
- materializer failure at step 2 keeps steps 0 and 1 applied, leaves 2 failed and later steps pending, then resumes from step 2;
- a conflict blocks its step and every later step;
- replaying the same idempotency key returns the same group; and
- a different plan under the key returns a stable conflict.

Run:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_server_origin_batch.py
```

Expected: FAIL because compound capture does not exist.

- [ ] **Step 2: Extend adapter context with planned-head overlays**

Use explicit structural types instead of storing service objects in adapters:

```python
SyncHead = SyncEnvelope | SyncEnvelopeCreate
SyncHeadLookup = Callable[[SyncDomain, str], SyncHead | None]
SyncDomainHeadLoader = Callable[[SyncDomain], Sequence[SyncHead]]

@dataclass(frozen=True)
class SyncAdapterContext:
    prior_envelopes: Sequence[SyncHead] = ()
    get_head: SyncHeadLookup | None = None
    list_heads: SyncDomainHeadLoader | None = None
```

Update lineage helpers to accept stored or planned heads. The batch evaluator maintains a `(domain, object_id) -> SyncHead` overlay and passes it to each step. Existing single-envelope evaluation continues to pass stored history only.

- [ ] **Step 3: Define canonical batch planning interfaces**

Add:

```python
@dataclass(frozen=True)
class ServerOriginMutationStep:
    domain: SyncDomain
    operation: SyncOperation
    object_id: str
    payload: Mapping[str, object]
    parent_id: str | None = None
    routing_metadata: Mapping[str, object] = field(default_factory=dict)
    stable_key: str | None = None

@dataclass(frozen=True)
class ServerOriginBatchResult:
    dataset: SyncDataset
    envelopes: tuple[SyncEnvelope, ...]
    fully_applied: bool

def capture_server_origin_mutation_batch(
    *,
    service: SyncV2Service,
    user_id: str,
    steps: Sequence[ServerOriginMutationStep],
    source: str,
    idempotency_key: str,
) -> ServerOriginBatchResult:
    """Preflight, atomically append, and ordered-materialize one complete plan."""

def resume_server_origin_mutation_group(
    *, service: SyncV2Service, dataset_id: str, mutation_group_id: str
) -> ServerOriginBatchResult:
    """Resume at the first non-applied step without skipping a blocked step."""
```

Hash a canonical ordered representation of every step to obtain `mutation_plan_hash`. Derive a stable group ID from dataset, trusted source, and idempotency key. Generate envelope IDs deterministically from group ID and step so a retry cannot fork history.

- [ ] **Step 4: Implement ordered persistence and materialization**

The coordinator sequence is fixed:

1. Load the personal dataset and verify every requested domain is enrolled and ready inside the Sync transaction boundary.
2. Evaluate every step with the planned-head overlay.
3. Append the complete group using `insert_envelopes_atomic()`.
4. Read apply state in step order.
5. Skip only already-applied prefixes whose recorded fingerprint matches.
6. Apply the first non-applied step; stop immediately on failed/conflicted status.
7. Continue until all steps are applied.

Never roll back an already committed product projection in another database, and never label the two databases as atomically committed.

- [ ] **Step 5: Verify existing single capture remains compatible**

Run:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py
```

Expected after implementation: PASS; the existing single-object API may delegate to a one-step group only if its public return type and idempotency behavior remain unchanged.

- [ ] **Step 6: Verify and commit Task 4**

Run, in order:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py -k planned
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_server_origin_batch.py
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py
git diff --check
git add tldw_Server_API/app/core/Sync/v2/adapters.py tldw_Server_API/app/core/Sync/v2/domain_adapters/_lineage.py tldw_Server_API/app/core/Sync/v2/service.py tldw_Server_API/app/core/Sync/v2/server_origin_batch.py tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py tldw_Server_API/tests/Sync/test_sync_v2_server_origin_batch.py
git commit -m "feat(sync): coordinate resumable mutation groups"
```

---

### Task 5: Implement strict organization domain adapters and conflict policy

**Files:**

- Create: `tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_organization.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/domain_adapters/__init__.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_adapters.py`

- [ ] **Step 1: Write failing adapter contract tests**

Parameterize all six domains for:

- strict payload parsing and unknown-field rejection;
- canonical object-ID validation;
- optimistic `base_version`/`base_hash` lineage;
- exact tombstone restore lineage;
- idempotent replay;
- update-vs-delete and delete-vs-update reviewable conflicts;
- missing, tombstoned, or foreign-owner dependencies;
- case-insensitive keyword/collection uniqueness;
- parent self-reference, ancestor cycles, and pre-existing cycles;
- duplicate relationship upserts/tombstones; and
- folder derived-path uniqueness and length.

Assert stable conflict codes, including:

```text
notes_organization_domain_not_ready
notes_organization_identity_mismatch
notes_organization_dependency_missing
notes_organization_dependency_deleted
notes_organization_ownership_mismatch
notes_organization_name_conflict
notes_organization_hierarchy_cycle
notes_organization_path_conflict
notes_organization_base_conflict
```

Run:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_adapters.py
```

Expected: FAIL because strict adapters are absent.

- [ ] **Step 2: Add current-head query support**

Expose bounded store operations used by adapter callbacks:

```python
def get_current_head(
    self, dataset_id: str, domain: SyncDomain, object_id: str
) -> SyncEnvelope | None:
    """Return the canonical current head for one owner-scoped object."""

def list_current_heads(
    self, dataset_id: str, domain: SyncDomain, *, limit: int, offset: int
) -> list[SyncEnvelope]:
    """Return a bounded page of canonical current heads for one domain."""
```

Use current object-state/head indexes; do not scan the full envelope history. The service supplies owner-scoped callbacks and overlays planned batch heads before stored heads.

- [ ] **Step 3: Implement one adapter parameterized by domain**

Create:

```python
@dataclass(slots=True)
class NotesOrganizationDomainAdapter:
    domain: SyncDomain
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Return accepted, rejected, or conflict after strict domain validation."""
```

Reuse the strict Notes lineage functions for base and tombstone behavior. Add domain-specific dependency extraction, hierarchy walking, and uniqueness comparison over current canonical heads. Relationships require their referenced resource heads to be active and owned by the same dataset; keyword-note and folder-note relationships also require active `notes.note` heads, while conversation relationships require active `chat.conversation` heads.

Permit `routing_metadata.bootstrap_capture is True` only when the caller is a trusted server-origin path, the organization group is `initializing`, and Task 7's snapshot verifier confirms the dormant local relationship. Ordinary client envelopes cannot set this bypass.

- [ ] **Step 4: Verify and commit Task 5**

Run, in order:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_adapters.py
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_domain_adapters.py
git diff --check
git add tldw_Server_API/app/core/Sync/v2/domain_adapters/notes_organization.py tldw_Server_API/app/core/Sync/v2/domain_adapters/__init__.py tldw_Server_API/app/core/Sync/v2/store.py tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_adapters.py
git commit -m "feat(sync): validate Notes organization changes"
```

---

### Task 6: Materialize all six domains and register production factories

**Files:**

- Create: `tldw_Server_API/app/core/Sync/v2/materializers/notes_organization.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/materializers/__init__.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_materializer.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_factory.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_postgres_contract.py`

- [ ] **Step 1: Write failing resource and relationship materializer tests**

Cover every domain for upsert, tombstone, exact restore, idempotent reapply, stale apply, and injected database failure. Specifically prove:

- remote resource `sync_id` resolves one stable local integer row;
- parent `sync_id` becomes local `parent_id` and folder path is derived;
- resource tombstones retain relationship rows and child parent pointers;
- link tombstones remove only the explicit relationship projection;
- a relationship cannot materialize against a missing/deleted dependency;
- source provenance metadata is ignored on a remote apply;
- object apply status becomes applied only after the product transaction commits; and
- a failure records a retryable apply error without advancing later group steps.

Run:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_materializer.py
```

Expected: FAIL because the materializer is absent.

- [ ] **Step 2: Implement the domain materializer**

Create:

```python
@dataclass(slots=True)
class NotesOrganizationMaterializer:
    note_db: CharactersRAGDB
    domain: SyncDomain

    def apply(
        self, envelope: SyncEnvelope, *, store: SyncV2Store
    ) -> MaterializationResult:
        """Project one accepted organization envelope and record apply state."""
```

Mirror `NotesMaterializer`'s current-state conflict handling and apply-status ordering. Delegate all product reads/writes to `NotesOrganizationSyncStore`. Validate payload and object identity again at the materialization boundary; do not trust stored arbitrary mappings.

- [ ] **Step 3: Register strict adapters and per-user materializers**

Update `default_sync_v2_registry()` to register six `NotesOrganizationDomainAdapter` instances. Update the production service factory to bind six `NotesOrganizationMaterializer` instances to the authenticated user's ChaChaNotes database. Keep the generic media adapter list unchanged.

Test that every advertised domain has exactly one adapter and materializer and that service construction cannot advertise a domain without a materializer.

- [ ] **Step 4: Verify SQLite and PostgreSQL projection contracts**

Run, in order:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_materializer.py
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_postgres_contract.py -k materializer
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_factory.py -k notes_organization
```

Expected: all pass; PostgreSQL contract tests assert equivalent SQL and transaction semantics.

- [ ] **Step 5: Commit Task 6**

```bash
git diff --check
git add tldw_Server_API/app/core/Sync/v2/materializers/notes_organization.py tldw_Server_API/app/core/Sync/v2/materializers/__init__.py tldw_Server_API/app/core/Sync/v2/factory.py tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_materializer.py tldw_Server_API/tests/Sync/test_sync_v2_factory.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_postgres_contract.py
git commit -m "feat(sync): materialize Notes organization domains"
```

---

### Task 7: Bootstrap existing datasets and make implicit pulls device-aware

**Files:**

- Modify: `tldw_Server_API/app/core/Sync/v2/models.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/profile.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/service.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Sync_DB.py`
- Create: `tldw_Server_API/app/core/Sync/v2/notes_organization_bootstrap.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/factory.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_service.py`
- Create: `tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_bootstrap.py`

- [ ] **Step 1: Write failing enrollment-state tests**

Cover:

- requesting a strict subset of the six domains is rejected;
- requesting the full group transitions an existing personal dataset to `initializing`, not `ready`;
- organization push, server-origin capture, and organization pull fail closed while initializing or failed;
- ordinary Notes reads continue;
- a new capable personal dataset also seeds before ready; and
- the profile exposes only `initializing`, `ready`, or `failed` plus a safe count/error-code summary.

Run:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py -k notes_organization
```

Expected: FAIL because profile validation only knows the milestone-one group and datasets have no organization readiness state.

- [ ] **Step 2: Add durable group enrollment state**

Store readiness in dataset metadata under a versioned, server-owned key:

```json
{
  "notes_organization_v1": {
    "bootstrap_id": "opaque-id",
    "state": "initializing",
    "captured_count": 0,
    "expected_count": 0,
    "error_code": null
  }
}
```

Add compare-and-set store methods that update dataset domains and metadata together. Group initialization adds all six enrolled domains atomically with state `initializing`; it never exposes a partial set. Readiness checks must occur in the same Sync transaction as accepted client pushes and server-origin appends so a concurrent state transition cannot admit an unsafe write.

- [ ] **Step 3: Write failing deterministic bootstrap and resume tests**

Build mixed active/deleted resources, nested hierarchies, active relationships, and dormant relationships. Assert ordered capture:

1. keyword, collection, and folder resource upserts, with parents before children;
2. active relationships;
3. tombstones for already-deleted resources.

Inject interruption between bounded batches and prove restart reuses the bootstrap ID and stable per-object keys, does not duplicate history, verifies already-correct projections without replaying them, preserves dormant relationships via trusted `bootstrap_capture`, reconciles a changed snapshot while still initializing, and reaches ready only when final counts match.

Run:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_bootstrap.py
```

Expected: FAIL because the bootstrapper does not exist.

- [ ] **Step 4: Implement the injected bootstrapper**

Define a service seam:

```python
class SyncDatasetBootstrapper(Protocol):
    def bootstrap(
        self, *, service: SyncV2Service, user_id: str, dataset: SyncDataset
    ) -> SyncDataset:
        """Capture an enrolled dataset's local source state before readiness."""

@dataclass(slots=True)
class NotesOrganizationBootstrapper:
    note_db: CharactersRAGDB
    batch_size: int = 200

    def bootstrap(
        self, *, service: SyncV2Service, user_id: str, dataset: SyncDataset
    ) -> SyncDataset:
        """Capture, verify, and mark the complete six-domain group ready."""
```

The production factory injects the user-bound implementation. Snapshot through `NotesOrganizationSyncStore`, sort deterministically, append bounded groups through Task 4, and mark bootstrap envelopes applied only after re-reading and matching local state. The verifier, not client routing metadata, authorizes dormant relationship capture.

- [ ] **Step 5: Write and implement legacy-device implicit-pull isolation**

Add a test with two devices sharing one dataset: the upgraded device requests the six domains; the legacy device registers only core domains. An implicit legacy pull returns no organization envelopes, while an explicit unsupported-domain request is rejected.

Change service selection from:

```python
selected = dataset.domains if domains is None else domains
```

to the registered device's requested-domain intersection when `domains is None`. Use the device returned by `_require_registered_device()`; do not accept client-supplied capability data on the pull request as authority.

Run:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_service.py -k "implicit_pull and legacy_device"
```

Expected before implementation: FAIL because implicit pulls default to every enrolled dataset domain.

- [ ] **Step 6: Verify and commit Task 7**

Run, in order:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py -k notes_organization
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_bootstrap.py
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_service.py -k "implicit_pull and legacy_device"
git diff --check
git add tldw_Server_API/app/core/Sync/v2/models.py tldw_Server_API/app/core/Sync/v2/profile.py tldw_Server_API/app/core/Sync/v2/service.py tldw_Server_API/app/core/Sync/v2/store.py tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/app/core/Sync/v2/notes_organization_bootstrap.py tldw_Server_API/app/core/Sync/v2/factory.py tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_bootstrap.py
git commit -m "feat(sync): bootstrap Notes organization state"
```

---

### Task 8: Route direct organization REST mutations through a portable coordinator

**Files:**

- Create: `tldw_Server_API/app/core/Sync/v2/notes_organization_coordinator.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes.py`
- Modify: `tldw_Server_API/tests/Notes/test_notes_api_integration.py`
- Create: `tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py`

- [ ] **Step 1: Write failing direct-route capture tests**

For active ready Sync, cover:

- create, rename, and soft-delete keyword;
- create, update parent/name, and soft-delete collection;
- link and unlink collection keyword;
- link and unlink conversation keyword;
- create nested folder path, rename/move folder, and soft-delete folder;
- link and unlink note folder.

After each successful route, assert the canonical envelope object ID, normalized payload, parent/base lineage, apply status, and resulting ChaCha row. Inject preflight, append, and materialization failures and assert the documented Sync error response plus no unlogged successful product write.

Run:

```bash
pytest -q tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py -k direct
```

Expected: FAIL because routes write ChaChaNotes directly or reject keyword writes under active Sync.

- [ ] **Step 2: Implement the small portable coordinator**

Define a user-bound orchestration API:

```python
@dataclass(frozen=True)
class PlannedNotesMutation:
    steps: tuple[ServerOriginMutationStep, ...]
    load_result: Callable[[], object]

@dataclass(slots=True)
class NotesOrganizationCoordinator:
    service: SyncV2Service
    note_db: CharactersRAGDB
    user_id: str

    def capture(
        self,
        *,
        steps: Sequence[ServerOriginMutationStep],
        source: str,
        idempotency_key: str,
    ) -> ServerOriginBatchResult:
        """Capture one planned Notes mutation through durable Sync authority."""

    def plan_keyword_create(self, keyword: str) -> PlannedNotesMutation:
        """Plan creation without mutating the product database."""

    def plan_keyword_rename(
        self, keyword_id: int, keyword: str
    ) -> PlannedNotesMutation:
        """Plan a versioned keyword rename by stable identity."""

    def plan_collection_change(
        self, collection_id: int | None, name: str, parent_id: int | None
    ) -> PlannedNotesMutation:
        """Plan a collection create, rename, or hierarchy change."""

    def plan_folder_path(self, path: str) -> PlannedNotesMutation:
        """Plan missing folder segments in parent-before-child order."""

    def plan_relationship(
        self, domain: SyncDomain, members: Mapping[str, str], present: bool
    ) -> PlannedNotesMutation:
        """Plan a deterministic relationship upsert or tombstone."""
```

`PlannedNotesMutation` contains ordered Sync steps and a projection result loader; it does not mutate product state during planning. Route-specific helpers may remain inline, but all active-Sync organization mutations must delegate capture to this coordinator.

- [ ] **Step 3: Replace direct route writes when the full group is ready**

Use the existing direct database implementation only when Sync v2 is inactive for the user. When active:

- require all six domains and `ready` state;
- derive stable resource IDs from existing rows or create them during planning;
- build deterministic relationship IDs;
- capture through the atomic group path; and
- reload the product response after materialization.

Replace `_note_keywords_sync_unsupported_error` with the readiness guard. Do not unblock keyword writes for partial/initializing/failed enrollment.

- [ ] **Step 4: Preserve route compatibility**

Run the existing notes API integration tests and verify status codes, integer route IDs, response shapes, case normalization, permissions, and inactive-Sync behavior remain compatible:

```bash
pytest -q tldw_Server_API/tests/Notes/test_notes_api_integration.py -k "keyword or collection or folder"
```

Expected after implementation: PASS.

- [ ] **Step 5: Verify and commit Task 8**

Run, in order:

```bash
pytest -q tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py -k direct
pytest -q tldw_Server_API/tests/Notes/test_notes_api_integration.py -k "keyword or collection or folder"
git diff --check
git add tldw_Server_API/app/core/Sync/v2/notes_organization_coordinator.py tldw_Server_API/app/api/v1/endpoints/notes.py tldw_Server_API/tests/Notes/test_notes_api_integration.py tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py
git commit -m "feat(notes): capture organization REST mutations"
```

---

### Task 9: Make compound note writes, folder provenance, and keyword merge lossless

**Files:**

- Modify: `tldw_Server_API/app/core/Sync/v2/notes_organization_coordinator.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/keyword_store.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes.py`
- Modify: `tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py`

- [ ] **Step 1: Write failing inline note-mutation group tests**

Cover create, full update, patch, and bulk import with keywords and folders. Assert each logical note operation produces one complete ordered group:

1. `notes.note` upsert;
2. new keyword resource upserts;
3. keyword-link upserts/tombstones;
4. new folder resource upserts parent-before-child; and
5. folder-link upserts/tombstones.

Verify an injected rejection at any preflight step writes neither the note nor organization state, an append failure writes neither, and a materialization interruption resumes without duplicate note versions or relationships.

Run:

```bash
pytest -q tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py -k "inline or bulk_import"
```

Expected: FAIL because the note envelope is currently captured separately from direct keyword/folder writes.

- [ ] **Step 2: Route note create/update/patch/import through one group**

Extend the coordinator with:

```python
def plan_note_with_organization(
    self,
    *,
    note_step: ServerOriginMutationStep,
    keywords: Sequence[str] | None,
    folder_paths: Sequence[str] | None,
) -> PlannedNotesMutation:
    """Plan one note mutation and all organization deltas as one group."""
```

Read current effective relationships to emit only actual deltas. Reuse existing resources by `sync_id`; create missing keyword/folder resource steps before their link steps. Bulk import uses one group per note so one invalid item does not combine unrelated notes into a single cross-item transaction.

- [ ] **Step 3: Write failing effective-folder-union tests**

Exercise manual and source membership combinations:

- absent to manual present emits one upsert;
- absent to source present emits one upsert plus safe origin provenance metadata;
- removing one source while another/manual membership remains emits no canonical tombstone;
- removing the final effective source emits one tombstone;
- origin materialization applies provenance and canonical membership in one ChaCha transaction;
- remote materialization ignores provenance; and
- canonical tombstones preserve source rows while suppressing their effective relationship, and canonical upserts clear that suppression;
- routing metadata contains no source content, credential, or filesystem path.

Run:

```bash
pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py -k effective_sync_union
```

Expected: FAIL because effective-union delta planning and provenance routing are absent.

- [ ] **Step 4: Implement origin-only provenance handling**

Compute prospective union state before capture. Store only an allow-listed routing delta such as operation plus opaque local row identifiers needed by the origin transaction. The materializer accepts the delta only when server capture marks the envelope as locally originated; otherwise it ignores the field. Apply provenance and effective relationship projection within the same ChaCha transaction.

- [ ] **Step 5: Write failing merge-plan tests**

For keyword merge, assert the ordered group contains:

- target link upserts for every unique note/conversation/collection relationship;
- source link tombstones for every moved canonical relationship; and
- the source keyword tombstone last.

Assert target duplicates collapse idempotently, the complete plan persists before the first projection, retry resumes, and a source keyword with active or dormant `flashcard_keywords` fails before append with:

```text
notes_keyword_merge_unsynchronized_dependency
```

An ordinary keyword soft delete with flashcard links remains allowed and dormant.

Run:

```bash
pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py -k sync_merge
```

Expected: FAIL because merge currently moves all relationship tables directly and has no canonical group.

- [ ] **Step 6: Implement synchronized merge planning**

Add a read-only keyword merge planner that enumerates note, conversation, collection, and flashcard dependencies before capture. Fail closed on flashcard dependencies. Build the complete ordered mutation group, append it atomically, and let the materializer perform the same final local state as the existing direct merge path.

- [ ] **Step 7: Verify and commit Task 9**

Run, in order:

```bash
pytest -q tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py -k "inline or bulk_import or merge"
pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py -k effective_sync_union
pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py -k sync_merge
git diff --check
git add tldw_Server_API/app/core/Sync/v2/notes_organization_coordinator.py tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py tldw_Server_API/app/core/DB_Management/chacha/keyword_store.py tldw_Server_API/app/api/v1/endpoints/notes.py tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py
git commit -m "feat(notes): synchronize compound organization writes"
```

---

### Task 10: Complete restore, repair, public documentation, and release evidence

**Files:**

- Modify: `tldw_Server_API/app/core/Sync/v2/restore.py`
- Modify: `tldw_Server_API/app/core/Sync/v2/replay.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py`
- Modify: `tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py`
- Modify: `Docs/API/Sync_V2_M1.md`
- Modify: `backlog/tasks/task-13003 - Synchronize-Notes-keywords-collections-and-folders.md`
- Modify if an incident produces a reusable lesson: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Write failing restore-preview and repair tests**

Add all six domains to preview counts and ordered restore execution. Prove restore respects resource-before-link dependencies, parent-before-child hierarchy, complete mutation groups, and dormant relationships. Repair must resume failed groups at their first unapplied step and report blocked conflicts without skipping them.

Run:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py -k notes_organization
```

Expected: FAIL because restore and repair do not enumerate the six domains or group semantics.

- [ ] **Step 2: Implement restore and repair integration**

Use the same strict adapters and materializers. Do not introduce a restore-only payload shape. Restore order is resources parent-first, then relationships, then tombstones; mutation groups remain ordered units. Repair exposes safe group ID, failing step, conflict/error code, and retry result without including note content or secrets.

- [ ] **Step 3: Update the public Sync contract**

Document in `Docs/API/Sync_V2_M1.md`:

- six domain schemas and exact payloads;
- resource and relationship identity rules;
- stable hash algorithm and vectors;
- complete-group enrollment and readiness states;
- device-aware implicit pull behavior;
- durable append versus resumable product materialization;
- hierarchy, soft-delete, effective membership, merge, and conflict rules;
- bootstrap behavior and repair observability; and
- explicit non-goals, especially flashcards and provenance.

Examples must use synthetic IDs and must not contain real note content, source text, filesystem paths, credentials, or API keys.

- [ ] **Step 4: Run the focused task suite one command at a time**

Run, in order, stopping on the first failure:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_identity.py
pytest -q tldw_Server_API/tests/ChaChaNotesDB/test_notes_organization_migration_v55.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_keyword_store.py tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_store.py -k "mutation_group or envelope"
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py tldw_Server_API/tests/Sync/test_sync_v2_server_origin_batch.py
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_adapters.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_materializer.py
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_bootstrap.py
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_service.py -k "notes_organization or implicit_pull or legacy_device"
pytest -q tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py
pytest -q tldw_Server_API/tests/Notes/test_notes_api_integration.py -k "note or keyword or collection or folder"
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py -k notes_organization
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_postgres_contract.py tldw_Server_API/tests/ChaChaNotesDB/test_note_folders_postgres.py
```

Do not substitute unrelated repository-wide tests for these focused checks. Record optional PostgreSQL skips separately from failures.

- [ ] **Step 5: Run static and security checks on touched production code**

Run:

```bash
ruff check tldw_Server_API/app/core/Sync/v2 tldw_Server_API/app/core/DB_Management/Sync_DB.py tldw_Server_API/app/core/DB_Management/chacha tldw_Server_API/app/api/v1/endpoints/notes.py tldw_Server_API/app/api/v1/schemas/notes_schemas.py
bandit -q -r tldw_Server_API/app/core/Sync/v2 tldw_Server_API/app/core/DB_Management/chacha/organization_sync_store.py tldw_Server_API/app/api/v1/endpoints/notes.py
git diff --check
```

Expected: exit 0. If a repository baseline issue is outside touched code, record the exact command, output, and scope; do not claim it passed.

- [ ] **Step 6: Review against every acceptance criterion**

Create a short evidence table in the task Implementation Notes mapping AC 1-8 to tests and implementation files. Review the diff for:

- accidental partial enrollment;
- any product write before full Sync-group append;
- cross-database atomicity claims;
- client-controlled bootstrap bypasses;
- integer IDs used as canonical identity;
- remote use of origin provenance;
- silent flashcard movement;
- cascaded soft-delete cleanup;
- missing user ownership checks; and
- unsupported domains leaking to legacy devices.

- [ ] **Step 7: Complete Backlog hygiene only after evidence is green**

Update all eight acceptance criteria and Definition of Done checkboxes, add concise Implementation Notes, and record:

```text
ADR required: yes
ADR path: Docs/ADR/032-durable-server-origin-sync-mutation-batches.md
Reason: This task adds a durable cross-database mutation-group contract, domain ownership boundaries, bootstrap policy, and long-lived Sync semantics.
```

Add a lesson only if execution uncovers a concrete reusable incident. Then set TASK-13003 to Done through the Backlog CLI.

- [ ] **Step 8: Final verification and commit**

Run the two most representative fresh checks immediately before the final commit:

```bash
pytest -q tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_adapters.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_materializer.py tldw_Server_API/tests/Sync/test_sync_v2_server_origin_batch.py
pytest -q tldw_Server_API/tests/Notes/test_notes_organization_sync_api.py tldw_Server_API/tests/Sync/test_sync_v2_notes_organization_bootstrap.py
git diff --check
git status --short
```

Then commit:

```bash
git add Docs/API/Sync_V2_M1.md backlog/tasks/task-13003\ -\ Synchronize-Notes-keywords-collections-and-folders.md tldw_Server_API/app/core/Sync/v2/restore.py tldw_Server_API/app/core/Sync/v2/replay.py tldw_Server_API/tests/Sync/test_sync_v2_restore_preview.py tldw_Server_API/tests/Sync/test_sync_v2_replay_repair.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py
git commit -m "docs(sync): complete Notes organization contract"
```

If a lessons file changed, stage it explicitly before the commit.

---

## Completion evidence required before PR handoff

- The task file contains a completed AC 1-8 evidence map and concise Implementation Notes.
- All focused commands above have fresh pass/skip output recorded; failures are not described as passes.
- SQLite behavior is executed; PostgreSQL has mock-contract evidence plus the existing live integration result or an explicit environment skip.
- Ruff, Bandit, and `git diff --check` have fresh exit-zero evidence for their stated scope.
- The branch contains no uncommitted task changes and every commit is reviewable in the order above.
- The final human-authored PR summary explains the six-domain contract, durable group boundary, bootstrap behavior, device isolation, tests, and any genuine skips. Do not post a generated summary until a human has approved it.
