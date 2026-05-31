# VN Pack Portability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Jobs-backed `.tldw-vnpack` export/import for backup-grade VN asset-pack portability.

**Architecture:** Add a VN-specific portability package under `tldw_Server_API/app/core/VN_Assets/portability/` rather than extending Chatbooks. Export creates backup-grade ZIP archives from VN pack metadata plus generated-file bytes; import uses async preview, conflict planning, and a journaled commit that stages bytes until new item IDs exist.

**Tech Stack:** FastAPI, Pydantic, SQLite through `CharactersRAGDB`, core Jobs manager, AuthNZ generated-file storage, pytest/httpx, Next.js/React, Vitest, Playwright.

---

## Reference Documents

- Spec: `Docs/superpowers/specs/2026-04-25-vn-pack-portability-design.md`
- Parent feature spec: `Docs/superpowers/specs/2026-04-24-vn-asset-packs-design.md`
- Existing backend module: `tldw_Server_API/app/core/VN_Assets/`
- Existing API endpoint: `tldw_Server_API/app/api/v1/endpoints/vn_assets.py`
- Existing frontend workbench: `apps/tldw-frontend/components/vn-assets/VNAssetsWorkbench.tsx`

## File Structure

Create:

- `tldw_Server_API/app/core/VN_Assets/portability/__init__.py`
- `tldw_Server_API/app/core/VN_Assets/portability/constants.py`
  Archive schema version, filename constants, limits, job operation names, trust modes, error codes.
- `tldw_Server_API/app/core/VN_Assets/portability/models.py`
  Dataclasses or typed dict helpers for manifest sections, preview plans, conflicts, and journal records.
- `tldw_Server_API/app/core/VN_Assets/portability/fingerprints.py`
  Canonical JSON encoding, section checksums, archive SHA-256 helpers, source item fingerprints.
- `tldw_Server_API/app/core/VN_Assets/portability/archive.py`
  Safe ZIP writer/reader helpers, member path validation, size accounting, duplicate normalized path checks.
- `tldw_Server_API/app/core/VN_Assets/portability/exporter.py`
  Export assembler that reads pack/slot/item metadata and generated-file bytes into staging.
- `tldw_Server_API/app/core/VN_Assets/portability/preview.py`
  Async preview validator/planner that validates archives and produces immutable preview plans.
- `tldw_Server_API/app/core/VN_Assets/portability/importer.py`
  Journaled import commit executor for create-new and update-existing modes.
- `tldw_Server_API/app/core/VN_Assets/portability/conflicts.py`
  Conflict detection and update-existing identity matching rules.
- `tldw_Server_API/tests/VN_Assets/test_portability_db.py`
- `tldw_Server_API/tests/VN_Assets/test_portability_archive.py`
- `tldw_Server_API/tests/VN_Assets/test_portability_export.py`
- `tldw_Server_API/tests/VN_Assets/test_portability_preview.py`
- `tldw_Server_API/tests/VN_Assets/test_portability_import.py`
- `tldw_Server_API/tests/VN_Assets/test_portability_api.py`
- `apps/tldw-frontend/components/vn-assets/PortabilityPanel.tsx`
- `apps/tldw-frontend/__tests__/vn-assets/PortabilityPanel.test.tsx`

Modify:

- `tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py`
  Add portability tables and repository methods.
- `tldw_Server_API/app/core/VN_Assets/jobs.py`
  Add portability job constants, queues, payload builders, idempotency keys.
- `tldw_Server_API/app/core/VN_Assets/worker.py`
  Register/dispatch portability job handlers if this worker is the established VN job worker entry point.
- `tldw_Server_API/app/services/vn_asset_jobs_worker.py`
  Wire portability job processing if service startup owns VN jobs.
- `tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py`
  Add export, preview, conflict, commit, status, and download schemas.
- `tldw_Server_API/app/api/v1/endpoints/vn_assets.py`
  Add export, import preview, import commit, job status, and cleanup endpoints.
- `apps/tldw-frontend/types/vn-assets.ts`
  Add portability request/response types.
- `apps/tldw-frontend/lib/api/vnAssets.ts`
  Add portability API client functions.
- `apps/tldw-frontend/components/vn-assets/VNAssetsWorkbench.tsx`
  Add a portability step or panel entry.
- `apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts`
  Extend smoke coverage for portability affordances.

## Task 1: Portability Schema And Repository

**Files:**

- Modify: `tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py`
- Create: `tldw_Server_API/tests/VN_Assets/test_portability_db.py`

- [x] **Step 1: Write failing DB schema tests**

Add tests that initialize `VNAssetPacksRepository` and assert the new tables exist:

```python
def test_portability_tables_are_created(chacha_db):
    repo = VNAssetPacksRepository.initialized(chacha_db)

    table_names = {
        row["name"]
        for row in chacha_db.execute_query(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }

    assert "vn_pack_portability_jobs" in table_names
    assert "vn_pack_import_previews" in table_names
    assert "vn_pack_import_journal" in table_names
```

Also test round-trip repository methods:

```python
def test_portability_job_round_trips(chacha_db):
    repo = VNAssetPacksRepository.initialized(chacha_db)

    created = repo.create_portability_job(
        owner_user_id=7,
        job_id="job-export-1",
        operation="export",
        status="queued",
        stage="queued",
        pack_id=12,
    )

    loaded = repo.get_portability_job(created["id"], owner_user_id=7)
    assert loaded["job_id"] == "job-export-1"
    assert loaded["operation"] == "export"
```

- [x] **Step 2: Run the failing DB tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_db.py -v
```

Expected: FAIL because tables and repository methods do not exist.

- [x] **Step 3: Add schema SQL**

Extend `VN_ASSET_SCHEMA_SQL` with:

```sql
CREATE TABLE IF NOT EXISTS vn_pack_portability_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE,
    owner_user_id INTEGER NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    pack_id INTEGER REFERENCES vn_asset_packs(id),
    preview_id INTEGER,
    import_id INTEGER,
    archive_path TEXT,
    archive_sha256 TEXT,
    canonical_payload_fingerprint TEXT,
    progress_json TEXT NOT NULL DEFAULT '{}',
    warnings_json TEXT NOT NULL DEFAULT '[]',
    error_code TEXT,
    error_message TEXT,
    download_url TEXT,
    expires_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_pack_import_previews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,
    archive_path TEXT NOT NULL,
    archive_sha256 TEXT,
    canonical_payload_fingerprint TEXT,
    schema_version TEXT,
    bundle_summary_json TEXT NOT NULL DEFAULT '{}',
    validation_warnings_json TEXT NOT NULL DEFAULT '[]',
    conflicts_json TEXT NOT NULL DEFAULT '[]',
    proposed_plan_json TEXT NOT NULL DEFAULT '{}',
    quota_estimate_json TEXT NOT NULL DEFAULT '{}',
    required_choices_json TEXT NOT NULL DEFAULT '[]',
    expires_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vn_pack_import_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_user_id INTEGER NOT NULL,
    preview_id INTEGER NOT NULL REFERENCES vn_pack_import_previews(id),
    job_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    trust_mode TEXT NOT NULL,
    target_mode TEXT NOT NULL,
    target_pack_id INTEGER,
    archive_path TEXT,
    archive_sha256 TEXT,
    canonical_payload_fingerprint TEXT,
    id_maps_json TEXT NOT NULL DEFAULT '{}',
    created_records_json TEXT NOT NULL DEFAULT '{}',
    cleanup_status_json TEXT NOT NULL DEFAULT '{}',
    warnings_json TEXT NOT NULL DEFAULT '[]',
    error_code TEXT,
    error_message TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME
);
```

Add indexes for `owner_user_id`, `job_id`, `status`, `expires_at`, and `canonical_payload_fingerprint`.

- [x] **Step 4: Add repository methods**

Add focused methods:

- `create_portability_job(...)`
- `update_portability_job(job_id, fields, owner_user_id=None)`
- `get_portability_job_by_job_id(job_id, owner_user_id=None)`
- `create_import_preview(...)`
- `update_import_preview(preview_id, fields, owner_user_id=None)`
- `get_import_preview(preview_id, owner_user_id=None)`
- `create_import_journal(...)`
- `update_import_journal(import_id, fields, owner_user_id=None)`
- `get_import_journal(import_id, owner_user_id=None)`

Use `_json_or_none` or a new `_json_dump` helper consistently for JSON columns.

- [x] **Step 5: Run DB tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_db.py -v
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py tldw_Server_API/tests/VN_Assets/test_portability_db.py
git commit -m "feat(vn-assets): add portability metadata tables"
```

## Task 2: Archive Safety, Checksums, And Canonical Fingerprints

**Files:**

- Create: `tldw_Server_API/app/core/VN_Assets/portability/__init__.py`
- Create: `tldw_Server_API/app/core/VN_Assets/portability/constants.py`
- Create: `tldw_Server_API/app/core/VN_Assets/portability/fingerprints.py`
- Create: `tldw_Server_API/app/core/VN_Assets/portability/archive.py`
- Create: `tldw_Server_API/tests/VN_Assets/test_portability_archive.py`

- [x] **Step 1: Write archive safety tests**

Test unsafe member rejection:

```python
import io
import zipfile

from tldw_Server_API.app.core.VN_Assets.portability.archive import validate_archive_members


def _zip_with_members(names: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name in names:
            zf.writestr(name, b"data")
    return buffer.getvalue()


def test_validate_archive_members_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "bad.tldw-vnpack"
    archive_path.write_bytes(_zip_with_members(["manifest.json", "../escape.png"]))

    with pytest.raises(ValueError, match="unsafe_archive_member"):
        validate_archive_members(archive_path)
```

Test duplicate normalized paths, absolute paths, Windows drive letters, null bytes, unexpected top-level names, and missing required files.

- [x] **Step 2: Write canonical fingerprint tests**

```python
from tldw_Server_API.app.core.VN_Assets.portability.fingerprints import canonical_payload_fingerprint


def test_canonical_payload_fingerprint_ignores_export_metadata():
    left = {
        "manifest": {"exported_at": "2026-01-01", "export_id": "a", "pack_title": "Demo"},
        "items": [{"slot_key": "sprite.happy", "checksum": "abc"}],
    }
    right = {
        "manifest": {"exported_at": "2026-02-01", "export_id": "b", "pack_title": "Demo"},
        "items": [{"checksum": "abc", "slot_key": "sprite.happy"}],
    }

    assert canonical_payload_fingerprint(left) == canonical_payload_fingerprint(right)
```

- [x] **Step 3: Run failing tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_archive.py -v
```

Expected: FAIL because portability package does not exist.

- [x] **Step 4: Implement constants**

In `constants.py` define:

```python
VNPACK_SCHEMA_VERSION = "tldw.vnpack.v1"
VNPACK_EXTENSION = ".tldw-vnpack"
MANIFEST_PATH = "manifest.json"
CHECKSUMS_PATH = "checksums/sha256.json"
ALLOWED_TOP_LEVEL_DIRS = {"assets", "metadata", "checksums", "signatures"}
ALLOWED_TOP_LEVEL_FILES = {"manifest.json", "README.md"}
REQUIRED_MEMBERS = {
    "manifest.json",
    "metadata/pack.json",
    "metadata/slots.json",
    "metadata/items.json",
    "checksums/sha256.json",
}
TRUST_MODE_TRUSTED_RESTORE = "trusted_restore"
TRUST_MODE_UNTRUSTED_IMPORT = "untrusted_import"
ASSET_BYTES_STATUS_PRESENT = "present"
ASSET_BYTES_STATUS_MISSING = "missing"
```

- [x] **Step 5: Implement member validation and hashing**

`archive.py` responsibilities:

- normalize member names with `PurePosixPath`
- reject absolute paths, `..`, empty parts, null bytes, drive letters, backslashes, duplicate normalized names
- reject unknown top-level entries
- enforce required members
- accumulate uncompressed size and per-file size limits

`fingerprints.py` responsibilities:

- `sha256_bytes(data: bytes) -> str`
- `sha256_file(path: Path) -> str`
- `canonical_json_bytes(payload: Any) -> bytes`
- `canonical_payload_fingerprint(payload: Mapping[str, Any]) -> str`

- [x] **Step 6: Run archive tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_archive.py -v
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/VN_Assets/portability tldw_Server_API/tests/VN_Assets/test_portability_archive.py
git commit -m "feat(vn-assets): add vnpack archive validation"
```

## Task 3: Backup Export Assembler And Job

**Files:**

- Create: `tldw_Server_API/app/core/VN_Assets/portability/models.py`
- Create: `tldw_Server_API/app/core/VN_Assets/portability/exporter.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/jobs.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/worker.py`
- Modify: `tldw_Server_API/app/services/vn_asset_jobs_worker.py`
- Create: `tldw_Server_API/tests/VN_Assets/test_portability_export.py`

- [x] **Step 1: Write export assembler tests**

Use existing VN repository helpers to create a pack with one slot and two items: one with bytes and one missing bytes. Mock generated-file lookup and byte reader.

Assertions:

- ZIP contains required files.
- `manifest.json` does not contain `archive_sha256`.
- `metadata/items.json` includes both items.
- missing-byte item has `asset_bytes_status="missing"` and no `asset_path`.
- `checksums/sha256.json` includes every metadata section and present asset.
- redacted provenance omits full prompt text.

- [x] **Step 2: Run failing export tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_export.py -v
```

Expected: FAIL because exporter does not exist.

- [x] **Step 3: Implement export models**

In `models.py`, define lightweight dataclasses:

```python
@dataclass(frozen=True)
class VNPackExportOptions:
    include_character_payload: bool = False
    include_world_book_payloads: bool = False
    include_full_provenance: bool = False
    strict: bool = False
    warn_for_sharing: bool = True


@dataclass(frozen=True)
class VNPackExportResult:
    archive_path: Path
    archive_sha256: str
    canonical_payload_fingerprint: str
    file_size_bytes: int
    warnings: list[str]
```

- [x] **Step 4: Implement `VNPackExporter`**

Constructor dependencies:

- `repo: VNAssetPacksRepository`
- `owner_user_id: int`
- `generated_files_repo`
- `read_generated_file_bytes: Callable[[dict[str, Any]], bytes]`
- `staging_root: Path`

Core method:

```python
async def export_pack(
    self,
    *,
    pack_id: int,
    options: VNPackExportOptions,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> VNPackExportResult:
    ...
```

Keep this service independent of FastAPI.

- [x] **Step 5: Add job payload helpers**

In `VN_Assets/jobs.py`, add:

- `VN_PACK_EXPORT_JOB_TYPE = "vn_pack_export"`
- `build_pack_export_payload(...)`
- `pack_export_idempotency_key(...)`
- `create_pack_export_job(...)`

Use batch groups like `vn_assets:user:{user_id}:pack:{pack_id}:portability:export:{request_id}`.

- [x] **Step 6: Wire worker handler**

Find the existing VN worker dispatch in `worker.py` / `vn_asset_jobs_worker.py`. Add a handler for `vn_pack_export` that:

- marks VN portability stage `collecting_metadata`
- invokes `VNPackExporter`
- updates `vn_pack_portability_jobs` with `archive_path`, `archive_sha256`, `canonical_payload_fingerprint`, warnings, file size, and expiry
- completes or fails the Jobs record

- [x] **Step 7: Run export tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_export.py tldw_Server_API/tests/VN_Assets/test_generation_jobs.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/VN_Assets tldw_Server_API/app/services/vn_asset_jobs_worker.py tldw_Server_API/tests/VN_Assets/test_portability_export.py
git commit -m "feat(vn-assets): export vnpack bundles"
```

## Task 4: Export API And Schemas

**Files:**

- Modify: `tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_assets.py`
- Create: `tldw_Server_API/tests/VN_Assets/test_portability_api.py`

- [x] **Step 1: Write API tests for export**

Add tests that POST `/api/v1/vn-assets/packs/{pack_id}/export` and assert:

- returns `202`
- returns `job_id`
- creates a portability job row
- rejects non-owned pack
- `GET /api/v1/vn-assets/portability/exports/{job_id}` composes Jobs lifecycle and VN stage fields
- download endpoint rejects incomplete jobs

- [x] **Step 2: Run failing API tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_api.py -v
```

Expected: FAIL because schemas/routes do not exist.

- [x] **Step 3: Add schemas**

Add Pydantic models:

- `VNPackExportRequest`
- `VNPackExportResponse`
- `VNPackPortabilityJobResponse`
- `VNPackExportDownloadResponse` if a JSON metadata endpoint is needed

Use strict booleans for export options and forbid unexpected fields where practical.

- [x] **Step 4: Add endpoints**

Routes:

- `POST /packs/{pack_id}/export`
- `GET /portability/exports/{job_id}`
- `GET /portability/exports/{job_id}/download`
- `POST /portability/exports/{job_id}/cancel`

Download must:

- require job owner
- require completed status
- validate archive path is under the per-user export directory
- return `FileResponse` with a `.tldw-vnpack` filename

- [x] **Step 5: Run API tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_api.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py tldw_Server_API/app/api/v1/endpoints/vn_assets.py tldw_Server_API/tests/VN_Assets/test_portability_api.py
git commit -m "feat(vn-assets): expose vnpack export api"
```

## Task 5: Async Import Preview

**Files:**

- Create: `tldw_Server_API/app/core/VN_Assets/portability/preview.py`
- Create: `tldw_Server_API/app/core/VN_Assets/portability/conflicts.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/jobs.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/worker.py`
- Modify: `tldw_Server_API/app/services/vn_asset_jobs_worker.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_assets.py`
- Create: `tldw_Server_API/tests/VN_Assets/test_portability_preview.py`

- [x] **Step 1: Write preview validator tests**

Cover:

- traversal archive rejected
- missing required file rejected
- checksum mismatch rejected
- malformed metadata rejected
- unsupported schema rejected
- missing primary character payload produces required action `link_existing_character` or `fail_import`
- missing asset bytes produce missing-byte counts and required-slot impact
- preview computes canonical payload fingerprint deterministically

- [x] **Step 2: Run failing preview tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_preview.py -v
```

Expected: FAIL because preview planner does not exist.

- [x] **Step 3: Implement conflict matching**

In `conflicts.py`, implement:

- pack conflict signals by canonical payload fingerprint, pack title, character fingerprint
- character conflict signals by card hash, normalized name, avatar hash
- world-book conflict signals by title and canonical content hash
- slot identity: `asset_type + slot_key`
- item identity: source item fingerprint, then checksum under matched slot, then ambiguous `variant_index`

Return structured conflicts with stable IDs so the UI can render and commit can validate decisions.

- [x] **Step 4: Implement `VNPackImportPreviewer`**

Core method:

```python
async def create_preview(
    self,
    *,
    archive_path: Path,
    owner_user_id: int,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ...
```

It should validate archive members before reading payload files and never extract paths blindly.

- [x] **Step 5: Wire preview job and endpoints**

Routes:

- `POST /import/previews`
- `GET /import/previews/{preview_id}`
- `POST /import/previews/{preview_id}/cancel`
- `DELETE /import/previews/{preview_id}`

Upload endpoint stores archive under a per-user temp directory and creates the preview job.

- [x] **Step 6: Run preview tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_preview.py tldw_Server_API/tests/VN_Assets/test_portability_api.py -v
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/VN_Assets tldw_Server_API/app/services/vn_asset_jobs_worker.py tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py tldw_Server_API/app/api/v1/endpoints/vn_assets.py tldw_Server_API/tests/VN_Assets/test_portability_preview.py tldw_Server_API/tests/VN_Assets/test_portability_api.py
git commit -m "feat(vn-assets): preview vnpack imports"
```

## Task 6: Import Commit, Journal, And Create-New Restore

**Files:**

- Create: `tldw_Server_API/app/core/VN_Assets/portability/importer.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/jobs.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/worker.py`
- Modify: `tldw_Server_API/app/services/vn_asset_jobs_worker.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_assets.py`
- Create: `tldw_Server_API/tests/VN_Assets/test_portability_import.py`

- [x] **Step 1: Write import commit tests**

Use an exported test archive from Task 3 fixtures. Assert:

- commit rejects expired previews
- commit rejects mutated archive checksum
- create-new import creates a new pack
- generated-file IDs are new and owned by importing user
- `source_ref` uses new item ID
- trusted restore preserves review state for items with bytes
- untrusted import resets items with bytes to draft
- missing-byte items become hidden in both modes
- journal records ID maps and created generated-file IDs

- [x] **Step 2: Run failing import tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_import.py -v
```

Expected: FAIL because importer does not exist.

- [x] **Step 3: Implement staged import**

`VNPackImporter` stages bytes under per-user temp storage until local item IDs exist. Flow:

1. Revalidate preview and archive hash.
2. Create journal.
3. Preflight quota from preview decoded bytes.
4. Create or link character.
5. Create pack.
6. Create slots.
7. Create item rows with `generated_file_id=None`.
8. Register staged bytes with `save_and_register_vn_asset_image(..., item_id=new_item_id, pack_id=new_pack_id, check_quota=True)`.
9. Patch item rows with generated-file IDs.
10. Apply trusted/untrusted review state.
11. Mark journal completed.

Do not create active generated-file rows before new item IDs exist.

- [x] **Step 4: Implement cleanup-on-failure**

On failure after generated-file registration:

- journal records generated-file IDs
- best-effort unregister/delete runs
- failed journal exposes cleanup status
- retry can resume only from safe stages; otherwise requires cleanup first

- [x] **Step 5: Add commit endpoints**

Routes:

- `POST /import/commit`
- `GET /portability/imports/{job_id}`
- `POST /portability/imports/{job_id}/cancel`
- `POST /portability/imports/{import_id}/cleanup`

- [x] **Step 6: Run import tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_import.py tldw_Server_API/tests/VN_Assets/test_portability_api.py -v
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/VN_Assets tldw_Server_API/app/services/vn_asset_jobs_worker.py tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py tldw_Server_API/app/api/v1/endpoints/vn_assets.py tldw_Server_API/tests/VN_Assets/test_portability_import.py tldw_Server_API/tests/VN_Assets/test_portability_api.py
git commit -m "feat(vn-assets): import vnpack backups"
```

## Task 7: Update-Existing Mode

**Files:**

- Modify: `tldw_Server_API/app/core/VN_Assets/portability/conflicts.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/portability/preview.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/portability/importer.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/jobs.py`
- Modify: `tldw_Server_API/app/core/VN_Assets/worker.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/vn_assets.py`
- Modify: `tldw_Server_API/tests/VN_Assets/test_portability_preview.py`
- Modify: `tldw_Server_API/tests/VN_Assets/test_portability_import.py`

- [x] **Step 1: Write update-existing preview tests**

Assert:

- slots match by `asset_type + slot_key`
- same slot key with different labels creates a confirmation-required diff
- item matches by source fingerprint first
- item matches by checksum second
- variant-index-only match is ambiguous and requires selection
- duplicate matches block automatic update

- [x] **Step 2: Write update-existing commit tests**

Assert:

- adds missing slots/items without deleting local records
- refuses risky diffs without confirmation token
- does not replace local byte-backed item with missing-byte imported item
- does not hard-delete local files

- [x] **Step 3: Run failing update tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_preview.py::test_update_existing_identity_rules tldw_Server_API/tests/VN_Assets/test_portability_import.py::test_update_existing_non_destructive -v
```

Expected: FAIL until update mode is implemented.

- [x] **Step 4: Implement update preview diff**

Preview should produce:

- matched slots
- added slots
- matched items
- added items
- risky diffs requiring confirmation
- blocked diffs

Each diff must have a stable `diff_id`.

- [x] **Step 5: Implement non-destructive update commit**

Commit only executes actions present in the accepted preview. Do not infer new actions at commit time beyond revalidation.

- [x] **Step 6: Run update tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets/test_portability_preview.py tldw_Server_API/tests/VN_Assets/test_portability_import.py -v
```

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/VN_Assets/portability tldw_Server_API/tests/VN_Assets/test_portability_preview.py tldw_Server_API/tests/VN_Assets/test_portability_import.py
git commit -m "feat(vn-assets): support non-destructive vnpack updates"
```

## Task 8: WebUI Portability Panel

**Files:**

- Modify: `apps/tldw-frontend/types/vn-assets.ts`
- Modify: `apps/tldw-frontend/lib/api/vnAssets.ts`
- Create: `apps/tldw-frontend/components/vn-assets/PortabilityPanel.tsx`
- Modify: `apps/tldw-frontend/components/vn-assets/VNAssetsWorkbench.tsx`
- Create: `apps/tldw-frontend/__tests__/vn-assets/PortabilityPanel.test.tsx`
- Modify: `apps/tldw-frontend/__tests__/vn-assets/vnAssetsApi.test.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts`

- [x] **Step 1: Write API client tests**

Assert URL/method/body for:

- `exportVNAssetPack`
- `getVNPackExportJob`
- `createVNPackImportPreview`
- `getVNPackImportPreview`
- `commitVNPackImport`
- `getVNPackImportJob`

- [x] **Step 2: Write component tests**

Test:

- backup warning is visible
- character/world-book/full provenance toggles are explicit
- export button calls API
- upload starts preview
- required character resolution is rendered
- trust mode selector changes commit payload
- update-existing risky diff requires confirmation

- [x] **Step 3: Run failing frontend tests**

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-assets/PortabilityPanel.test.tsx __tests__/vn-assets/vnAssetsApi.test.ts
```

Expected: FAIL because UI/API client does not exist.

- [x] **Step 4: Add frontend types and API functions**

Keep TypeScript names aligned with backend schemas. Use `FormData` for preview upload.

- [x] **Step 5: Build `PortabilityPanel`**

UI requirements:

- backup-bundle label
- export option toggles
- no-encryption warning
- export job progress/download
- import upload
- preview status
- conflict/required-choice rendering
- trust mode selector
- commit progress

Keep layout consistent with existing VN workbench components; do not introduce a new design system.

- [x] **Step 6: Wire into workbench**

Add a `Portability` workflow step to `VNAssetsWorkbench.tsx` after Review.

- [x] **Step 7: Run frontend tests**

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-assets
```

Expected: PASS.

- [x] **Step 8: Run Playwright smoke**

```bash
cd apps/tldw-frontend
bunx playwright test e2e/smoke/vn-assets.spec.ts --reporter=line
```

Expected: PASS.

- [x] **Step 9: Commit**

```bash
git add apps/tldw-frontend/types/vn-assets.ts apps/tldw-frontend/lib/api/vnAssets.ts apps/tldw-frontend/components/vn-assets apps/tldw-frontend/__tests__/vn-assets apps/tldw-frontend/e2e/smoke/vn-assets.spec.ts
git commit -m "feat(vn-assets): add portability workbench"
```

## Task 9: Final Verification And Security

**Files:**

- No planned source edits unless checks fail.

- [ ] **Step 1: Run backend VN suite**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/VN_Assets -v
```

Expected: PASS.

- [ ] **Step 2: Run frontend VN tests**

```bash
cd apps/tldw-frontend
bunx vitest run __tests__/vn-assets
```

Expected: PASS.

- [ ] **Step 3: Run Playwright smoke**

```bash
cd apps/tldw-frontend
bunx playwright test e2e/smoke/vn-assets.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 4: Run Bandit on touched backend paths**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/VN_Assets \
  tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py \
  tldw_Server_API/app/api/v1/endpoints/vn_assets.py \
  tldw_Server_API/app/api/v1/schemas/vn_asset_schemas.py \
  tldw_Server_API/app/services/vn_asset_jobs_worker.py \
  -f json -o /tmp/bandit_vn_pack_portability.json
```

Expected: command exits 0 and JSON has no new findings in touched code.

- [ ] **Step 5: Run diff checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files modified before final commit.

- [ ] **Step 6: Commit verification fixes if needed**

Only if prior checks required changes:

```bash
git add <changed-files>
git commit -m "fix(vn-assets): finalize vnpack portability"
```

## Implementation Notes

- Prefer `rg` and existing VN test fixtures before adding new fixtures.
- Keep archive validation independent from FastAPI so tests can run without an app.
- Do not store `archive_sha256` inside `manifest.json`.
- Do not create active generated-file records before new local item IDs exist.
- Missing-byte imported items must be hidden in both trust modes.
- Update-existing mode must be non-destructive by default.
- Use generated-file `source_feature="vn_assets"` for imported images.
- Keep prompt snapshots redacted unless `include_full_provenance=true`.
- Preserve exact old-to-new ID maps in the import journal for cleanup and support.
