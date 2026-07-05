# Visual Identity Expression Packs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement V1 shared Visual Identity Expression Packs for character/persona chats: import and edit SillyTavern-style expression packs, bind active packs to characters or personas by default, resolve expression assets during chat, support animated GIF/WebP and capability-gated AVIF originals, and preserve a clean bridge to VN asset generation.

**Architecture:** Add a focused `Visual_Identities` backend module with its own ChaChaNotes repository adapter, service layer, archive importer, storage validator, Jobs integration, API schemas, and endpoint module under `/api/v1/visual-identities`. Keep existing Persona Visual endpoints and legacy character mood images compatible. On the frontend, add typed API methods, expression resolver utilities, a pack/draft management UI, and chat runtime integration that reuses the current character mood baseline and existing Persona Buddy image renderer patterns.

**Tech Stack:** FastAPI, Pydantic, SQLite ChaChaNotes, existing Jobs manager, Pillow, Python `zipfile`, React, TypeScript, Ant Design, existing `TldwApiClient` domain mixins, Vitest, pytest, Bandit.

**Boundary Spec:** `Docs/superpowers/specs/2026-07-01-visual-identity-expression-packs-design.md`

**Source Reference:** [TavernSprite SillyTavern expression guide](https://tavernsprite.com/blog/sillytavern-add-use-character-expressions-guide/)

**Backlog Plan Task:** `TASK-12090`

---

## Scope Check

- [ ] V1 is large enough to require staged execution, but each stage is independently reviewable and testable.
- [ ] V1 includes backend data/API/import/runtime contracts, frontend pack management, and chat portrait/stage integration.
- [ ] V1 does not implement full VN scene composition, generated VN scripts, Live2D, Rive, Lottie, SVG animation packs, server-side emotion classification, or multi-character VN scene casting.
- [ ] V1 keeps Persona Visual Packs as a compatibility surface. It does not migrate existing persona packs or remove existing persona visual routes.
- [ ] V1 stores enough source metadata for VN asset generation to target expression slots later, and it adds an import path from existing generated files so VN-generated images can become expression assets without duplicating upload logic.

## Non-Negotiable Boundaries

- [ ] Preserve existing Persona Visual API routes under the persona endpoint group.
- [ ] Preserve legacy character `extensions` mood image behavior as a fallback.
- [ ] Do not place the new route surface inside the already large `persona.py` endpoint file.
- [ ] Do not store image bytes in ChaChaNotes rows.
- [ ] Do not activate ZIP imports directly. Every ZIP import creates a draft first, then activation creates an immutable pack version.
- [ ] Do not trust ZIP filenames, content types, dimensions, frame counts, archive paths, or decompression size without validation.
- [ ] Do not allow AVIF upload when runtime capability checks cannot verify dimensions and MIME safely.
- [ ] Do not rewrite historical messages when the active pack or binding changes. Historical message metadata keeps its resolved visual identity fields.
- [ ] Do not make the selected persona override the character actor in character chat. The default displayed actor remains the selected character unless the UI is explicitly rendering the persona as the actor.

## Existing Seams To Reuse

- [ ] Backend route/dependency pattern from `tldw_Server_API/app/api/v1/endpoints/vn_assets.py`.
- [ ] Router registration through `tldw_Server_API/app/api/v1/router_groups/core.py`.
- [ ] ChaChaNotes focused repository pattern from `tldw_Server_API/app/core/DB_Management/VNAssetPacks_DB.py`.
- [ ] Persona Visual MIME and image validation lessons from `tldw_Server_API/app/core/Persona/visual_asset_constraints.py` and `tldw_Server_API/app/core/Persona/visual_service.py`.
- [ ] Persona Visual manifest and renderer concepts from `tldw_Server_API/app/core/Persona/visuals.py` and `apps/packages/ui/src/components/Common/PersonaBuddy/SpriteFrameRenderer.tsx`.
- [ ] Existing character mood baseline from `apps/packages/ui/src/utils/character-mood.ts`.
- [ ] Existing chat message portrait resolution seam in `apps/packages/ui/src/components/Common/Playground/useMessageState.ts`.
- [ ] Existing persisted message metadata path in `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`, especially `_build_stream_persist_metadata_extra`.

## V1 Public Contract

Add a new backend endpoint module:

- [ ] `tldw_Server_API/app/api/v1/endpoints/visual_identities.py`

Register it through `router_groups/core.py` with:

- [ ] `prefix=f"{API_V1_PREFIX}/visual-identities"`
- [ ] `tags=("visual-identities",)`
- [ ] `route_key="visual-identities"`

Initial route surface:

- [ ] `GET /api/v1/visual-identities/capabilities`
- [ ] `GET /api/v1/visual-identities/expression-slots`
- [ ] `GET /api/v1/visual-identities/packs`
- [ ] `POST /api/v1/visual-identities/packs`
- [ ] `GET /api/v1/visual-identities/packs/{pack_id}`
- [ ] `PATCH /api/v1/visual-identities/packs/{pack_id}`
- [ ] `DELETE /api/v1/visual-identities/packs/{pack_id}`
- [ ] `POST /api/v1/visual-identities/packs/{pack_id}/assets`
- [ ] `POST /api/v1/visual-identities/packs/{pack_id}/assets/from-generated-file`
- [ ] `GET /api/v1/visual-identities/packs/{pack_id}/assets/{asset_id}/content`
- [ ] `POST /api/v1/visual-identities/imports/zip`
- [ ] `GET /api/v1/visual-identities/drafts/{draft_id}`
- [ ] `PATCH /api/v1/visual-identities/drafts/{draft_id}/slots/{slot_key}`
- [ ] `POST /api/v1/visual-identities/drafts/{draft_id}/activate`
- [ ] `POST /api/v1/visual-identities/bindings`
- [ ] `DELETE /api/v1/visual-identities/bindings/{binding_id}`
- [ ] `GET /api/v1/visual-identities/bindings/resolve`

Resolution query shape:

```text
GET /api/v1/visual-identities/bindings/resolve?actor_kind=character&actor_id=123&expression_key=happy
```

Response contract includes:

```json
{
  "actor_kind": "character",
  "actor_id": 123,
  "pack_id": 10,
  "pack_version_id": 4,
  "expression_key": "happy",
  "requested_expression_key": "joy",
  "asset_id": 88,
  "asset_url": "/api/v1/visual-identities/packs/10/assets/88/content",
  "fallback_reason": null,
  "is_animated": true,
  "content_type": "image/webp"
}
```

## Data Model

Create a focused repository file:

- [ ] `tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py`

Tables created by `ensure_visual_identity_tables(db: CharactersRAGDB)`:

- [ ] `visual_identity_packs`
- [ ] `visual_identity_pack_drafts`
- [ ] `visual_identity_pack_versions`
- [ ] `visual_identity_assets`
- [ ] `visual_identity_bindings`
- [ ] `visual_identity_idempotency`

Required columns:

```sql
visual_identity_packs:
  id INTEGER PRIMARY KEY
  owner_user_id INTEGER NOT NULL
  title TEXT NOT NULL
  description TEXT NOT NULL DEFAULT ''
  status TEXT NOT NULL CHECK(status IN ('active', 'archived', 'deleted'))
  active_version_id INTEGER
  default_expression_key TEXT NOT NULL DEFAULT 'neutral'
  source_kind TEXT NOT NULL DEFAULT 'manual'
  source_context_json TEXT NOT NULL DEFAULT '{}'
  created_at TEXT NOT NULL
  updated_at TEXT NOT NULL
  version INTEGER NOT NULL DEFAULT 1

visual_identity_pack_drafts:
  id INTEGER PRIMARY KEY
  owner_user_id INTEGER NOT NULL
  pack_id INTEGER
  title TEXT NOT NULL
  status TEXT NOT NULL CHECK(status IN ('importing', 'ready_for_review', 'failed', 'abandoned', 'activated'))
  source_kind TEXT NOT NULL
  source_filename TEXT NOT NULL DEFAULT ''
  import_job_id TEXT
  validation_summary_json TEXT NOT NULL DEFAULT '{}'
  slot_map_json TEXT NOT NULL DEFAULT '{}'
  default_expression_key TEXT NOT NULL DEFAULT 'neutral'
  error_json TEXT NOT NULL DEFAULT '{}'
  created_at TEXT NOT NULL
  updated_at TEXT NOT NULL
  version INTEGER NOT NULL DEFAULT 1

visual_identity_pack_versions:
  id INTEGER PRIMARY KEY
  pack_id INTEGER NOT NULL
  owner_user_id INTEGER NOT NULL
  version_number INTEGER NOT NULL
  default_expression_key TEXT NOT NULL DEFAULT 'neutral'
  manifest_json TEXT NOT NULL
  created_at TEXT NOT NULL
  UNIQUE(pack_id, version_number)

visual_identity_assets:
  id INTEGER PRIMARY KEY
  owner_user_id INTEGER NOT NULL
  pack_id INTEGER
  draft_id INTEGER
  pack_version_id INTEGER
  expression_key TEXT NOT NULL
  original_expression_key TEXT NOT NULL DEFAULT ''
  display_label TEXT NOT NULL DEFAULT ''
  source_filename TEXT NOT NULL
  storage_relpath TEXT NOT NULL
  content_type TEXT NOT NULL
  bytes INTEGER NOT NULL
  sha256 TEXT NOT NULL
  width INTEGER NOT NULL
  height INTEGER NOT NULL
  is_animated INTEGER NOT NULL DEFAULT 0
  frame_count INTEGER
  duration_ms INTEGER
  preview_relpath TEXT
  deleted INTEGER NOT NULL DEFAULT 0
  created_at TEXT NOT NULL
  updated_at TEXT NOT NULL

visual_identity_bindings:
  id INTEGER PRIMARY KEY
  owner_user_id INTEGER NOT NULL
  actor_kind TEXT NOT NULL CHECK(actor_kind IN ('character', 'persona'))
  actor_id INTEGER NOT NULL
  pack_id INTEGER NOT NULL
  active_version_id INTEGER NOT NULL
  status TEXT NOT NULL CHECK(status IN ('active', 'deleted'))
  created_at TEXT NOT NULL
  updated_at TEXT NOT NULL
  version INTEGER NOT NULL DEFAULT 1

visual_identity_idempotency:
  id INTEGER PRIMARY KEY
  owner_user_id INTEGER NOT NULL
  scope TEXT NOT NULL
  resource_id TEXT NOT NULL
  idempotency_key TEXT NOT NULL
  payload_hash TEXT NOT NULL
  status TEXT NOT NULL
  response_json TEXT
  created_at TEXT NOT NULL
  updated_at TEXT NOT NULL
  UNIQUE(owner_user_id, scope, resource_id, idempotency_key)
```

Indexes:

- [ ] `idx_visual_identity_packs_owner_status` on `(owner_user_id, status)`
- [ ] `idx_visual_identity_drafts_owner_status` on `(owner_user_id, status)`
- [ ] `idx_visual_identity_assets_pack_expression` on `(pack_id, pack_version_id, expression_key, deleted)`
- [ ] `idx_visual_identity_assets_draft_expression` on `(draft_id, expression_key, deleted)`
- [ ] `idx_visual_identity_bindings_actor_active` unique partial index on `(owner_user_id, actor_kind, actor_id)` where `status = 'active'`

## Expression Slot Contract

Canonical V1 slots:

- [ ] `neutral`
- [ ] `happy`
- [ ] `excited`
- [ ] `sad`
- [ ] `angry`
- [ ] `thinking`
- [ ] `confused`
- [ ] `surprised`

Required built-in aliases:

```text
default -> neutral
normal -> neutral
calm -> neutral
joy -> happy
joyful -> happy
cheerful -> happy
hype -> excited
thrilled -> excited
upset -> sad
sorrowful -> sad
mad -> angry
annoyed -> angry
furious -> angry
thoughtful -> thinking
pondering -> thinking
unsure -> confused
puzzled -> confused
shocked -> surprised
astonished -> surprised
```

Unrecognized SillyTavern ZIP names become custom expression slots with normalized keys prefixed by `custom:` in backend metadata and displayed without the prefix in UI labels.

## Stage 0: Implementation Hygiene Gate

**Goal:** Start implementation from a clean reviewable path without disturbing unrelated worktree changes.

**Success Criteria:**

- [ ] Confirm the active Backlog task is `TASK-12090` or a child implementation task created from this plan.
- [ ] Confirm unrelated dirty files are ignored and not staged.
- [ ] Confirm implementer has read this plan and the boundary spec.
- [ ] Confirm whether implementation will run inline or through subagent-driven execution.

**Commands:**

```bash
git status --short
```

```bash
python - <<'PY'
from pathlib import Path
for path in [
    "Docs/superpowers/specs/2026-07-01-visual-identity-expression-packs-design.md",
    "Docs/superpowers/plans/2026-07-01-visual-identity-expression-packs-implementation-plan.md",
]:
    assert Path(path).exists(), path
PY
```

## Stage 1: Expression Slots And Format Capabilities

**Goal:** Add deterministic expression-key normalization and runtime image-format capability reporting.

**Files:**

- [ ] Add `tldw_Server_API/app/core/Visual_Identities/__init__.py`
- [ ] Add `tldw_Server_API/app/core/Visual_Identities/expression_slots.py`
- [ ] Add `tldw_Server_API/app/core/Visual_Identities/constraints.py`
- [ ] Add `tldw_Server_API/tests/Visual_Identities/test_expression_slots.py`
- [ ] Add `tldw_Server_API/tests/Visual_Identities/test_visual_identity_capabilities.py`

**Implementation Steps:**

- [ ] Define `CANONICAL_EXPRESSION_SLOTS`, `EXPRESSION_ALIASES`, and `CUSTOM_EXPRESSION_PREFIX = "custom:"`.
- [ ] Implement `normalize_expression_key(value: str) -> str | None`.
- [ ] Implement `normalize_expression_filename(filename: str) -> str | None` that strips extension, lowercases, replaces non-alphanumeric runs with `_`, and maps aliases.
- [ ] Implement `is_custom_expression_key(value: str) -> bool`.
- [ ] Implement `display_label_for_expression_key(value: str) -> str`.
- [ ] Define allowed static/animated MIME sets: PNG, JPEG, WebP, GIF as baseline; AVIF only when `supports_avif()` returns true.
- [ ] Implement `build_visual_identity_capabilities()` with upload max bytes, archive max bytes, max dimension, max frame count, supported MIME types, and AVIF enabled flag.

**Tests To Write First:**

```python
def test_normalize_builtin_expression_aliases() -> None:
    assert normalize_expression_key("default") == "neutral"
    assert normalize_expression_key("normal") == "neutral"
    assert normalize_expression_key("joy") == "happy"
```

```python
def test_unrecognized_filename_becomes_custom_expression() -> None:
    assert normalize_expression_filename("bashful smile.PNG") == "custom:bashful_smile"
```

```python
def test_capabilities_include_avif_only_when_runtime_supports_it(monkeypatch) -> None:
    monkeypatch.setattr("tldw_Server_API.app.core.Visual_Identities.constraints.supports_avif", lambda: False)
    capabilities = build_visual_identity_capabilities()
    assert "image/avif" not in capabilities["supported_mime_types"]
```

**Verification:**

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Visual_Identities/test_expression_slots.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_capabilities.py
```

## Stage 2: Repository And Schema

**Goal:** Add Visual Identity persistence in ChaChaNotes through a focused repository adapter.

**Files:**

- [ ] Add `tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py`
- [ ] Add `tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py`

**Implementation Steps:**

- [ ] Implement `ensure_visual_identity_tables(db: CharactersRAGDB) -> None`.
- [ ] Reject non-SQLite backends before opening a transaction, matching the VN asset repository behavior.
- [ ] Preserve outer transaction rollback behavior.
- [ ] Implement `VisualIdentityRepository.initialized(db)`.
- [ ] Implement pack methods: `create_pack`, `get_pack`, `list_packs`, `update_pack`, `archive_pack`, `mark_pack_deleted`.
- [ ] Implement draft methods: `create_draft`, `get_draft`, `update_draft_slot_map`, `set_draft_status`, `list_draft_assets`.
- [ ] Implement version methods: `create_pack_version`, `get_pack_version`, `set_active_version`.
- [ ] Implement asset methods: `create_asset`, `get_asset`, `list_assets_for_version`, `list_assets_for_draft`, `mark_asset_deleted`.
- [ ] Implement binding methods: `upsert_binding`, `delete_binding`, `get_binding_for_actor`, `resolve_active_binding`.
- [ ] Implement idempotency helpers mirroring the VN asset repository contract: claim, complete, and replay response for matching payload hashes.

**Tests To Write First:**

```python
def test_visual_identity_tables_are_created(chacha_db: CharactersRAGDB) -> None:
    ensure_visual_identity_tables(chacha_db)
    cursor = chacha_db.execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'visual_identity_%'"
    )
    table_names = {row[0] for row in cursor.fetchall()}
    assert {
        "visual_identity_packs",
        "visual_identity_pack_drafts",
        "visual_identity_pack_versions",
        "visual_identity_assets",
        "visual_identity_bindings",
        "visual_identity_idempotency",
    }.issubset(table_names)
```

```python
def test_binding_upsert_keeps_one_active_binding_per_actor(chacha_db: CharactersRAGDB) -> None:
    repo = VisualIdentityRepository.initialized(chacha_db)
    first = repo.upsert_binding(owner_user_id=1, actor_kind="character", actor_id=7, pack_id=10, active_version_id=1)
    second = repo.upsert_binding(owner_user_id=1, actor_kind="character", actor_id=7, pack_id=11, active_version_id=2)
    assert second["id"] == first["id"]
    assert repo.get_binding_for_actor(owner_user_id=1, actor_kind="character", actor_id=7)["pack_id"] == 11
```

**Verification:**

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py
```

## Stage 3: Storage, Validation, And Generated-File Import

**Goal:** Store image originals safely on disk, validate dimensions and animations, and support importing existing generated files from the VN asset pipeline.

**Files:**

- [ ] Add `tldw_Server_API/app/core/Visual_Identities/storage.py`
- [ ] Add `tldw_Server_API/tests/Visual_Identities/test_visual_identity_storage.py`
- [ ] Modify `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- [ ] Add storage directory helper `DatabasePaths.get_user_visual_identities_dir(user_id: int) -> Path`

**Implementation Steps:**

- [ ] Store files under `Databases/user_databases/{user_id}/visual_identities`.
- [ ] Use content-derived SHA-256 filenames with the original extension retained only after validation.
- [ ] Validate MIME by file headers and Pillow image loading, not by client-provided `content_type`.
- [ ] Validate max bytes, max width, max height, max frame count, and nonzero dimensions.
- [ ] Detect `is_animated`, `frame_count`, and approximate `duration_ms` for GIF/WebP/AVIF where Pillow exposes frames.
- [ ] Store animated originals unchanged.
- [ ] Create a first-frame preview only when Pillow can seek frame zero without raising.
- [ ] Return a typed `VisualIdentityStoredAsset` dataclass with relpath, MIME, bytes, SHA-256, dimensions, animation fields, and preview relpath.
- [ ] Implement `copy_generated_file_to_expression_asset(owner_user_id, pack_id, expression_key, generated_file_id, source_feature)` that reads an `AuthnzGeneratedFilesRepo` record, validates it through the same path, and records `source_kind="generated_file"` with `source_context_json` containing `generated_file_id` and optional `source_feature`.

**Tests To Write First:**

```python
def test_rejects_image_over_max_dimension(tmp_path: Path) -> None:
    image_path = tmp_path / "large.png"
    Image.new("RGBA", (4097, 32)).save(image_path)
    with pytest.raises(ValueError, match="image_dimensions_exceed_limit"):
        validate_and_store_visual_identity_asset(
            source_path=image_path,
            owner_user_id=1,
            expression_key="happy",
            storage_root=tmp_path / "store",
        )
```

```python
def test_animated_gif_original_is_stored_and_marked_animated(tmp_path: Path) -> None:
    gif_path = tmp_path / "blink.gif"
    frames = [Image.new("RGBA", (8, 8), color) for color in ("red", "blue")]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    stored = validate_and_store_visual_identity_asset(
        source_path=gif_path,
        owner_user_id=1,
        expression_key="surprised",
        storage_root=tmp_path / "store",
    )
    assert stored.content_type == "image/gif"
    assert stored.is_animated is True
    assert stored.frame_count == 2
```

```python
def test_avif_is_rejected_when_capability_is_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("tldw_Server_API.app.core.Visual_Identities.constraints.supports_avif", lambda: False)
    with pytest.raises(ValueError, match="unsupported_mime_type"):
        validate_visual_identity_mime(content_type="image/avif", content=b"avif-bytes")
```

**Verification:**

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_storage.py
```

## Stage 4: Service Layer And Resolution Priority

**Goal:** Implement pack lifecycle, binding, activation, and deterministic expression resolution above the repository/storage layers.

**Files:**

- [ ] Add `tldw_Server_API/app/core/Visual_Identities/service.py`
- [ ] Add `tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py`

**Implementation Steps:**

- [ ] Implement `VisualIdentityService(db, owner_user_id, jobs_manager=None)`.
- [ ] Implement pack creation with `status="active"` and no active version until first activation.
- [ ] Implement draft activation in one transaction: create or update pack, create version, attach draft assets to version, update active version, set draft `activated`.
- [ ] Implement binding upsert after activation when `actor_kind` and `actor_id` are provided.
- [ ] Implement `resolve_expression_asset(actor_kind, actor_id, requested_expression_key, manual_override_expression_key=None, mood_expression_key=None)`.
- [ ] Apply expression priority in this order: manual override, requested expression, mood expression, pack default, neutral/default/normal asset, legacy fallback signal, neutral placeholder signal.
- [ ] Return fallback reason values from this enum: `manual_override`, `requested`, `mood`, `pack_default`, `neutral_alias`, `legacy_character_mood`, `placeholder`.
- [ ] Validate actor ownership by confirming character/persona exists for the current user before binding.
- [ ] Preserve inactive/deleted pack assets if referenced by active versions or message metadata.

**Tests To Write First:**

```python
def test_resolve_prefers_manual_override_over_mood(chacha_db: CharactersRAGDB) -> None:
    result = service.resolve_expression_asset(
        actor_kind="character",
        actor_id=character_id,
        requested_expression_key=None,
        manual_override_expression_key="angry",
        mood_expression_key="happy",
    )
    assert result.expression_key == "angry"
    assert result.fallback_reason == "manual_override"
```

```python
def test_activation_binds_pack_to_character_by_default(chacha_db: CharactersRAGDB) -> None:
    activated = service.activate_draft(draft_id=draft["id"], actor_kind="character", actor_id=character_id)
    binding = repo.get_binding_for_actor(owner_user_id=1, actor_kind="character", actor_id=character_id)
    assert binding["pack_id"] == activated["pack_id"]
```

```python
def test_deleted_pack_does_not_resolve_for_new_messages(chacha_db: CharactersRAGDB) -> None:
    repo.mark_pack_deleted(pack_id=pack_id, owner_user_id=1)
    result = service.resolve_expression_asset(actor_kind="character", actor_id=character_id, requested_expression_key="happy")
    assert result.fallback_reason == "placeholder"
```

**Verification:**

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_service.py
```

## Stage 5: ZIP Import Drafts And Jobs

**Goal:** Import SillyTavern-style ZIP packs through Jobs into reviewable drafts.

**Files:**

- [ ] Add `tldw_Server_API/app/core/Visual_Identities/archive_import.py`
- [ ] Add `tldw_Server_API/app/core/Visual_Identities/jobs.py`
- [ ] Add `tldw_Server_API/tests/Visual_Identities/test_visual_identity_archive_import.py`
- [ ] Add `tldw_Server_API/tests/Visual_Identities/test_visual_identity_jobs.py`

**Implementation Steps:**

- [ ] Define job domain `visual_identities`.
- [ ] Define job type `visual_identity_import_zip`.
- [ ] Implement `create_visual_identity_import_zip_job(owner_user_id, draft_id, upload_path, source_filename, idempotency_key)` with owner user id, draft id, upload path, source filename, and idempotency payload hash.
- [ ] Validate archive size before opening.
- [ ] Reject encrypted ZIP entries.
- [ ] Reject absolute paths, `..`, backslashes normalized to traversal, empty filenames, path duplicates after normalization, symlinks, directories used as files, nested archives, and unsupported image entries.
- [ ] Enforce entry count limit, per-file compressed/uncompressed byte limits, total uncompressed byte limit, and decompression ratio limit.
- [ ] Map filenames into canonical/custom expression keys.
- [ ] When multiple entries map to the same expression key, keep the first deterministic normalized path and add the rest to `validation_summary_json.duplicates`.
- [ ] Store valid image assets against the draft.
- [ ] Mark draft `ready_for_review` if at least one valid asset exists.
- [ ] Mark draft `failed` if no valid assets exist or archive validation fails before any asset can be accepted.

**Archive Limits:**

```python
MAX_EXPRESSION_ZIP_BYTES = 100 * 1024 * 1024
MAX_EXPRESSION_ZIP_ENTRIES = 128
MAX_EXPRESSION_ZIP_TOTAL_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_EXPRESSION_ZIP_DECOMPRESSION_RATIO = 100
MAX_EXPRESSION_ASSET_BYTES = 25 * 1024 * 1024
```

**Tests To Write First:**

```python
def test_zip_import_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("../happy.png", b"not used")
    with pytest.raises(ValueError, match="archive_path_not_allowed"):
        inspect_visual_identity_zip(archive_path)
```

```python
def test_zip_import_maps_default_and_custom_slots(tmp_path: Path) -> None:
    archive_path = build_zip_with_images(tmp_path, ["default.png", "bashful.png"])
    draft = importer.import_zip_to_draft(archive_path, source_filename="pack.zip")
    assert draft.slot_map["neutral"].asset_id is not None
    assert draft.slot_map["custom:bashful"].asset_id is not None
```

```python
def test_import_job_marks_draft_ready_for_review(chacha_db: CharactersRAGDB, tmp_path: Path) -> None:
    archive_path = build_zip_with_images(tmp_path, ["neutral.png"])
    job_result = run_visual_identity_import_zip_job(
        db=chacha_db,
        owner_user_id=1,
        draft_id=draft_id,
        archive_path=archive_path,
        source_filename="pack.zip",
    )
    draft = repo.get_draft(job_result["draft_id"], owner_user_id=1)
    assert draft["status"] == "ready_for_review"
```

**Verification:**

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_archive_import.py \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_jobs.py
```

## Stage 6: API Schemas, Endpoints, And Router Registration

**Goal:** Expose the Visual Identity contract through authenticated FastAPI endpoints.

**Files:**

- [ ] Add `tldw_Server_API/app/api/v1/schemas/visual_identity_schemas.py`
- [ ] Add `tldw_Server_API/app/api/v1/endpoints/visual_identities.py`
- [ ] Modify `tldw_Server_API/app/api/v1/router_groups/core.py`
- [ ] Add `tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py`

**Implementation Steps:**

- [ ] Use `get_chacha_db_for_user`, `get_request_user`, and `rbac_rate_limit` from the existing API dependency modules.
- [ ] Use `JobManager()` dependency pattern from `vn_assets.py`.
- [ ] Use `AuthnzGeneratedFilesRepo` dependency pattern from `vn_assets.py` for `/assets/from-generated-file`.
- [ ] Add response models for capabilities, slots, packs, drafts, assets, bindings, and resolved expression assets.
- [ ] Add upload endpoints with idempotency keys for mutating archive/generated-file operations.
- [ ] Return 404 for missing or foreign-user packs/drafts/assets.
- [ ] Return 409 for active binding conflicts that cannot be resolved by upsert.
- [ ] Return 422 for invalid actor kind, expression key, file, archive, or unsupported MIME.
- [ ] Return `FileResponse` for asset content with the validated content type and immutable cache headers for versioned assets.
- [ ] Make `POST /drafts/{draft_id}/activate` accept optional `actor_kind` and `actor_id`; when present, bind the activated pack to that actor by default.

**Schema Names:**

- [ ] `VisualIdentityCapabilitiesResponse`
- [ ] `VisualIdentityExpressionSlotResponse`
- [ ] `VisualIdentityPackCreate`
- [ ] `VisualIdentityPackUpdate`
- [ ] `VisualIdentityPackResponse`
- [ ] `VisualIdentityAssetResponse`
- [ ] `VisualIdentityDraftResponse`
- [ ] `VisualIdentityDraftSlotUpdate`
- [ ] `VisualIdentityDraftActivateRequest`
- [ ] `VisualIdentityBindingRequest`
- [ ] `VisualIdentityBindingResponse`
- [ ] `VisualIdentityResolveResponse`
- [ ] `VisualIdentityImportZipStartResponse`

**Tests To Write First:**

```python
def test_capabilities_endpoint_reports_supported_formats(client: TestClient) -> None:
    response = client.get("/api/v1/visual-identities/capabilities", headers=auth_headers)
    assert response.status_code == 200
    assert "image/gif" in response.json()["supported_mime_types"]
```

```python
def test_activate_draft_with_character_binds_by_default(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/visual-identities/drafts/{draft_id}/activate",
        json={"actor_kind": "character", "actor_id": character_id},
        headers=auth_headers,
    )
    assert response.status_code == 200
    resolved = client.get(
        f"/api/v1/visual-identities/bindings/resolve?actor_kind=character&actor_id={character_id}&expression_key=neutral",
        headers=auth_headers,
    )
    assert resolved.json()["pack_id"] == response.json()["pack_id"]
```

```python
def test_asset_content_requires_owner(client: TestClient) -> None:
    response = client.get(f"/api/v1/visual-identities/packs/{foreign_pack_id}/assets/{asset_id}/content", headers=auth_headers)
    assert response.status_code == 404
```

**Verification:**

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Visual_Identities/test_visual_identities_api.py
```

## Stage 7: Chat Message Metadata Integration

**Goal:** Persist resolved expression metadata for assistant messages without changing old message rendering.

**Files:**

- [ ] Modify `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- [ ] Add `tldw_Server_API/tests/Character_Chat/test_visual_identity_expression_metadata.py`

**Implementation Steps:**

- [ ] Extend `_build_stream_persist_metadata_extra` with optional visual identity fields.
- [ ] Persist only scalar metadata fields in `metadata.extra`.
- [ ] Use keys: `visual_actor_kind`, `visual_actor_id`, `visual_pack_id`, `visual_pack_version_id`, `visual_expression_key`, `visual_asset_id`, `visual_fallback_reason`.
- [ ] Keep existing `mood_label`, `mood_confidence`, and `mood_topic` fields unchanged.
- [ ] Thread optional resolved visual identity metadata through non-streaming and streaming character chat persistence paths.
- [ ] Do not fail chat generation if visual identity resolution fails. Log the error and persist mood metadata as today.

**Tests To Write First:**

```python
def test_stream_persist_metadata_includes_visual_identity_fields() -> None:
    extra = _build_stream_persist_metadata_extra(
        speaker_character_id=5,
        speaker_character_name="Ari",
        turn_taking_mode="single",
        validation_degraded=False,
        persist_fingerprint="fp",
        mood_label="happy",
        mood_confidence=0.8,
        mood_topic=None,
        usage=None,
        visual_identity={
            "actor_kind": "character",
            "actor_id": 5,
            "pack_id": 10,
            "pack_version_id": 2,
            "expression_key": "happy",
            "asset_id": 99,
            "fallback_reason": "mood",
        },
    )
    assert extra["visual_expression_key"] == "happy"
    assert extra["visual_asset_id"] == 99
```

```python
def test_visual_identity_resolution_failure_does_not_block_character_reply(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.resolve_character_visual_identity",
        raising_resolver,
    )
    response = client.post(
        character_chat_message_url,
        json={"message": "Hello"},
        headers=auth_headers,
    )
    assert response.status_code == 200
```

**Verification:**

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Character_Chat/test_visual_identity_expression_metadata.py
```

## Stage 8: Frontend Types, API Client, And Resolver Utilities

**Goal:** Add typed frontend support for Visual Identity API calls and expression resolution.

**Files:**

- [x] Add `apps/packages/ui/src/types/visual-identities.ts`
- [x] Add `apps/packages/ui/src/services/tldw/domains/visual-identities.ts`
- [x] Modify `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- [x] Add `apps/packages/ui/src/utils/visual-identity-expressions.ts`
- [x] Add `apps/packages/ui/src/utils/visual-identity-emote.ts`
- [x] Add `apps/packages/ui/src/utils/__tests__/visual-identity-expressions.test.ts`
- [x] Add `apps/packages/ui/src/utils/__tests__/visual-identity-emote.test.ts`
- [x] Add `apps/packages/ui/src/services/__tests__/tldw-api-client.visual-identities.test.ts`

**Implementation Steps:**

- [x] Add TypeScript interfaces matching the backend schema names.
- [x] Add `visualIdentityMethods` domain mixin.
- [x] Add methods for capabilities, slots, pack CRUD, asset upload, generated-file import, draft read/update/activate, binding upsert/delete/resolve, and ZIP import start.
- [x] Import and merge `visualIdentityMethods` into `TldwApiClient` declaration merging and `Object.assign`.
- [x] Implement frontend expression alias normalization using the same V1 slot baseline as `character-mood.ts`.
- [x] Implement `parseVisualIdentityEmoteCommand(input: string)` that returns `{ expressionKey, rawExpression }` for `/emote happy`, `/emote anger`, and custom expression labels; return `null` for regular messages.
- [x] Keep `/emote` parsing client-side so it does not send a chat message.

**Tests To Write First:**

```ts
it("maps anger slash command to angry expression", () => {
  expect(parseVisualIdentityEmoteCommand("/emote anger")).toEqual({
    expressionKey: "angry",
    rawExpression: "anger"
  })
})
```

```ts
it("keeps regular messages untouched", () => {
  expect(parseVisualIdentityEmoteCommand("please /emote happy")).toBeNull()
})
```

```ts
it("normalizes custom expression labels", () => {
  expect(normalizeVisualIdentityExpressionKey("bashful smile")).toBe("custom:bashful_smile")
})
```

**Verification:**

```bash
bunx vitest run \
  apps/packages/ui/src/utils/__tests__/visual-identity-expressions.test.ts \
  apps/packages/ui/src/utils/__tests__/visual-identity-emote.test.ts
```

## Stage 9: Pack Management And Draft Review UI

**Goal:** Let users import, review, edit, activate, and bind expression packs from character/persona workflows.

**Files:**

- [x] Add `apps/packages/ui/src/components/Common/VisualIdentity/VisualIdentityPackPanel.tsx`
- [x] Add `apps/packages/ui/src/components/Common/VisualIdentity/VisualIdentityDraftReview.tsx`
- [x] Add `apps/packages/ui/src/components/Common/VisualIdentity/ExpressionSlotGrid.tsx`
- [x] Add `apps/packages/ui/src/components/Common/VisualIdentity/ExpressionAssetUploader.tsx`
- [x] Add `apps/packages/ui/src/components/Common/VisualIdentity/__tests__/VisualIdentityDraftReview.test.tsx`
- [x] Add `apps/packages/ui/src/components/Common/VisualIdentity/__tests__/ExpressionSlotGrid.test.tsx`
- [x] Modify `apps/packages/ui/src/components/Option/Characters/CharacterEditorForm.tsx`.
- [x] Modify `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`.

**Implementation Steps:**

- [x] Add a reusable panel that accepts `actorKind: "character" | "persona"` and `actorId: number | string`.
- [x] Load current active binding by calling `resolveVisualIdentityBinding` for `neutral`.
- [x] Show active pack, active version, default expression, supported formats, and fallback status.
- [x] Add ZIP import button that calls `startVisualIdentityZipImport`.
- [x] Poll the returned draft through terminal status-aware polling until ready, failed, cancelled, quarantined, activated, or timeout. No generic Visual Identity job status endpoint/helper exists in the current V1 surface.
- [x] Show draft slot grid with canonical slots first and custom slots after.
- [x] Allow per-slot image upload with immediate validation errors from the API.
- [ ] Allow default expression selection only from slots with valid assets. Deferred: this requires a backend draft default-expression update or activation-time override API.
- [x] Activate draft with actor binding by default, matching the user's requirement that the pack applies to the selected character/persona it is associated with.
- [x] Replace the current `CharacterEditorForm.tsx` "Mood images (coming soon)" dashed block with `VisualIdentityPackPanel` and a collapsed section titled "Legacy mood images". Do not remove stored legacy data.
- [x] Add `VisualIdentityPackPanel` to `VisualPackEditor.tsx` in a separate section titled "Expression packs" so Persona Visual Pack management remains visible and unchanged.

**UX Constraints:**

- [x] Use compact grids and toolbars rather than a marketing-style page.
- [x] Use image thumbnails for expressions when asset content URLs are available; packless draft assets show a stable unavailable-preview state.
- [x] Use icons for upload, replace, delete, activate, and refresh actions where existing icon libraries provide them.
- [x] Show animated assets as images, but do not create layout shifts when animations load.
- [x] Respect reduced-motion by pausing preview animation when the user has reduced motion enabled.

**Tests To Write First:**

```tsx
it("activates a ready draft with the current character binding", async () => {
  render(<VisualIdentityPackPanel actorKind="character" actorId={7} />)
  await user.click(screen.getByRole("button", { name: /activate/i }))
  expect(api.activateVisualIdentityDraft).toHaveBeenCalledWith(
    expect.any(Number),
    expect.objectContaining({ actor_kind: "character", actor_id: 7 })
  )
})
```

```tsx
it("shows custom expression slots after canonical slots", () => {
  render(<ExpressionSlotGrid slots={[neutralSlot, customBashfulSlot, happySlot]} />)
  expect(screen.getAllByTestId("expression-slot").map((node) => node.textContent)).toEqual([
    "Neutral",
    "Happy",
    "Bashful"
  ])
})
```

**Verification:**

```bash
bunx vitest run \
  apps/packages/ui/src/components/Common/VisualIdentity/__tests__/VisualIdentityDraftReview.test.tsx \
  apps/packages/ui/src/components/Common/VisualIdentity/__tests__/ExpressionSlotGrid.test.tsx
```

## Stage 10: Chat Runtime, Portraits, Picker, And Stage View

**Goal:** Use active expression packs in character/persona chat with manual override, `/emote`, mood fallback, and a basic single-character VN-style stage.

**Files:**

- [x] Add `apps/packages/ui/src/components/Common/VisualIdentity/VisualIdentityImage.tsx`
- [x] Add `apps/packages/ui/src/components/Common/VisualIdentity/ExpressionPicker.tsx`
- [x] Add `apps/packages/ui/src/components/Common/VisualIdentity/VisualIdentityStage.tsx`
- [x] Add `apps/packages/ui/src/hooks/useVisualIdentityResolver.ts`
- [x] Modify `apps/packages/ui/src/components/Common/Playground/message-types.ts`
- [x] Modify `apps/packages/ui/src/components/Common/Playground/useMessageState.ts`
- [x] Modify `apps/packages/ui/src/hooks/chat/useChatActions.ts`
- [x] Modify `apps/packages/ui/src/components/Common/Playground/Message.tsx`, specifically the `portraitPanel` image rendering and avatar preview rendering that use `portraitImage`.
- [x] Modify `apps/packages/ui/src/components/Option/Playground/PlaygroundChat.tsx`
- [x] Modify `apps/packages/ui/src/components/Option/Playground/PlaygroundCompareCluster.tsx`
- [x] Modify `apps/packages/ui/src/hooks/useMessageOption.tsx`
- [x] Modify `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts`
- [x] Modify `apps/packages/ui/src/store/option/types.ts`
- [x] Add `apps/packages/ui/src/components/Common/VisualIdentity/__tests__/VisualIdentityImage.test.tsx`
- [x] Add `apps/packages/ui/src/components/Common/VisualIdentity/__tests__/ExpressionPicker.test.tsx`
- [x] Add `apps/packages/ui/src/components/Common/VisualIdentity/__tests__/VisualIdentityStage.test.tsx`
- [x] Add `apps/packages/ui/src/components/Common/Playground/__tests__/visual-identity-message-state.test.tsx`
- [x] Add `apps/packages/ui/src/hooks/__tests__/useVisualIdentityResolver.test.tsx`

**Implementation Steps:**

- [x] Add optional props to `MessageCharacterProps`: `visualActorKind`, `visualActorId`, `visualPackId`, `visualPackVersionId`, `visualExpressionKey`, `visualAssetId`, `visualAssetUrl`, `visualFallbackReason`.
- [x] Teach `useMessageState` to prefer `visualAssetUrl` over legacy mood images for bot messages when the speaker matches the selected character.
- [x] Keep `resolveCharacterMoodImageUrl` as the fallback after visual identity resolution.
- [x] Add `useVisualIdentityResolver` to resolve the active actor's current expression and cache by actor, pack version, expression key, and asset id.
- [x] Add `ExpressionPicker` in the chat toolbar or existing character controls area.
- [x] On picker click, set session manual override and immediately update the stage/next assistant expression resolution.
- [x] In `useChatActions`, parse `/emote` before submit. If it is a valid emote command, update manual override and skip the network send.
- [x] Add a basic single-character `VisualIdentityStage` that displays the active character/persona centered with the current resolved expression.
- [x] Keep the stage disabled unless character/persona identity display is enabled and a visual identity binding resolves.
- [x] Respect `prefers-reduced-motion` in `VisualIdentityImage`; static images render normally, animated GIF/WebP/AVIF originals render as browser images unless reduced motion requires a still preview.

**Tests To Write First:**

```tsx
it("uses visual identity asset before legacy mood avatar", () => {
  const state = renderUseMessageState({
    isBot: true,
    visualAssetUrl: "/api/v1/visual-identities/packs/1/assets/2/content",
    moodLabel: "happy",
    characterIdentity: characterWithLegacyHappyImage,
  })
  expect(state.portraitImage).toBe("/api/v1/visual-identities/packs/1/assets/2/content")
})
```

```tsx
it("handles emote commands without sending chat", async () => {
  await submitChatInput("/emote surprised")
  expect(sendMessage).not.toHaveBeenCalled()
  expect(setManualExpressionOverride).toHaveBeenCalledWith("surprised")
})
```

```tsx
it("renders still preview when reduced motion is enabled", () => {
  mockReducedMotion(true)
  render(<VisualIdentityImage assetUrl="/animated.webp" previewUrl="/preview.png" isAnimated />)
  expect(screen.getByRole("img")).toHaveAttribute("src", "/preview.png")
})
```

**Verification:**

```bash
bunx vitest run \
  apps/packages/ui/src/components/Common/VisualIdentity/__tests__/VisualIdentityImage.test.tsx \
  apps/packages/ui/src/components/Common/VisualIdentity/__tests__/ExpressionPicker.test.tsx \
  apps/packages/ui/src/components/Common/Playground/__tests__/visual-identity-message-state.test.tsx
```

**Stage 10 Verification Run:**

- [x] `bunx vitest run src/components/Common/VisualIdentity/__tests__/VisualIdentityImage.test.tsx src/components/Common/VisualIdentity/__tests__/ExpressionPicker.test.tsx src/components/Common/VisualIdentity/__tests__/VisualIdentityStage.test.tsx src/components/Common/VisualIdentity/__tests__/VisualIdentityPackPanel.test.tsx src/components/Common/VisualIdentity/__tests__/VisualIdentityDraftReview.test.tsx src/components/Common/VisualIdentity/__tests__/ExpressionSlotGrid.test.tsx src/components/Common/Playground/__tests__/visual-identity-message-state.test.tsx src/hooks/__tests__/useVisualIdentityResolver.test.tsx src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx src/components/Option/Playground/__tests__/PlaygroundChat.server-load-state.test.tsx` passed: 10 files, 28 tests.
- [x] `bun run test:characters-harness` passed: 3 files, 104 tests. Known Ant Design shadow-root warning remains.
- [x] `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --project tsconfig.json --pretty false` still reports 85 existing baseline diagnostics outside Stage 10 files; no diagnostics matched `VisualIdentity`, `useVisualIdentityResolver`, `useChatActions`, `useMessageOption`, `PlaygroundChat`, `PlaygroundCompareCluster`, `useServerChatLoader`, `message-types`, `useMessageState`, `Common/Playground/Message`, or `store/option/types`.
- [x] `git diff --check` passed.
- [x] Bandit skipped: Stage 10 touched frontend TypeScript/React files only.

## Stage 11: VN Asset Bridge

**Goal:** Tie expression packs to the VN asset generation path without turning this V1 into full VN scene generation.

**Files:**

- [ ] Add `tldw_Server_API/app/core/Visual_Identities/vn_bridge.py`
- [ ] Add `tldw_Server_API/tests/Visual_Identities/test_visual_identity_vn_bridge.py`
- [ ] Add `apps/packages/ui/src/components/Common/VisualIdentity/__tests__/GeneratedFileImportAction.test.tsx`

**Implementation Steps:**

- [ ] Define `VisualIdentitySourceRef` with `source_feature`, `source_id`, `source_label`, and `metadata`.
- [ ] Support `source_feature="vn_assets"` and `source_feature="manual_upload"` initially.
- [ ] Implement `create_asset_from_generated_file(owner_user_id, pack_id, expression_key, generated_file_id, source_feature)` in the service using the storage helper from Stage 3.
- [ ] Record generated-file provenance in asset `source_context_json` or draft `source_context_json`.
- [ ] Add a frontend action in the pack panel for importing a generated file into a selected expression slot when a generated file id is available from VN asset UI state.
- [ ] Keep future generation prompt orchestration outside this stage. The bridge accepts generated files that already exist.

**Tests To Write First:**

```python
def test_generated_file_import_uses_same_validation_as_upload(chacha_db: CharactersRAGDB) -> None:
    with pytest.raises(ValueError, match="unsupported_mime_type"):
        service.create_asset_from_generated_file(pack_id=pack_id, expression_key="happy", generated_file_id="bad-file")
```

```python
def test_generated_file_import_records_vn_asset_provenance(chacha_db: CharactersRAGDB) -> None:
    asset = service.create_asset_from_generated_file(
        pack_id=pack_id,
        expression_key="happy",
        generated_file_id="generated-1",
        source_feature="vn_assets",
    )
    assert asset["source_context"]["source_feature"] == "vn_assets"
    assert asset["source_context"]["generated_file_id"] == "generated-1"
```

**Verification:**

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Visual_Identities/test_visual_identity_vn_bridge.py
```

```bash
bunx vitest run \
  apps/packages/ui/src/components/Common/VisualIdentity/__tests__/GeneratedFileImportAction.test.tsx
```

## Stage 12: End-To-End Regression And Documentation

**Goal:** Verify the feature as a cohesive slice and document operational limits.

**Files:**

- [ ] Add `Docs/Design/Visual_Identity_Expression_Packs.md`
- [ ] Add or update frontend locale strings for Visual Identity labels.
- [ ] Add e2e coverage where the existing frontend test harness supports authenticated WebUI flows.

**Implementation Steps:**

- [ ] Document supported formats and AVIF capability behavior.
- [ ] Document ZIP import safety limits.
- [ ] Document binding behavior for characters vs personas.
- [ ] Document fallback priority and legacy mood image compatibility.
- [ ] Document the `/emote` client command and manual picker behavior.
- [ ] Document how VN-generated files can be imported into expression slots.
- [ ] Run backend unit/integration tests for all new Visual Identity tests.
- [ ] Run frontend Vitest tests for new Visual Identity utilities/components and touched Playground behavior.
- [ ] Run Bandit on touched backend code.
- [ ] Run TypeScript or the repo's existing frontend validation command if available in the current package scripts.

**Verification:**

```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Visual_Identities \
  tldw_Server_API/tests/Character_Chat/test_visual_identity_expression_metadata.py
```

```bash
bunx vitest run \
  apps/packages/ui/src/utils/__tests__/visual-identity-expressions.test.ts \
  apps/packages/ui/src/utils/__tests__/visual-identity-emote.test.ts \
  apps/packages/ui/src/components/Common/VisualIdentity/__tests__/VisualIdentityDraftReview.test.tsx \
  apps/packages/ui/src/components/Common/VisualIdentity/__tests__/ExpressionSlotGrid.test.tsx \
  apps/packages/ui/src/components/Common/VisualIdentity/__tests__/VisualIdentityImage.test.tsx \
  apps/packages/ui/src/components/Common/VisualIdentity/__tests__/ExpressionPicker.test.tsx \
  apps/packages/ui/src/components/Common/VisualIdentity/__tests__/GeneratedFileImportAction.test.tsx \
  apps/packages/ui/src/components/Common/Playground/__tests__/visual-identity-message-state.test.tsx
```

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Visual_Identities \
  tldw_Server_API/app/core/DB_Management/VisualIdentity_DB.py \
  tldw_Server_API/app/api/v1/endpoints/visual_identities.py \
  tldw_Server_API/app/api/v1/schemas/visual_identity_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  -f json -o /tmp/bandit_visual_identity_expression_packs.json
```

## Acceptance Criteria

- [ ] A user can import a SillyTavern-style ZIP into a draft, review slots, and activate it.
- [ ] Activation with a selected character/persona binds the pack to that actor by default.
- [ ] PNG, JPEG, WebP, and GIF expression assets validate and render.
- [ ] Animated GIF/WebP originals are preserved and displayed, with still previews available for reduced motion.
- [ ] AVIF is accepted only when backend capability checks confirm support.
- [ ] `/emote happy` and `/emote anger` update the active expression without sending a chat message.
- [ ] The chat portrait prefers active visual identity assets, then legacy mood images, then static avatar.
- [ ] Message metadata stores actor kind/id, pack id, pack version id, expression key, asset id, and fallback reason for resolved assistant expressions.
- [ ] Persona Visual routes continue to pass existing tests.
- [ ] Legacy character mood image behavior continues to pass existing tests.
- [ ] VN-generated files can be imported into expression slots through the same validation path as uploads.

## Rollout Notes

- [ ] Ship the backend API and frontend management UI behind normal authenticated access, not an experimental global flag.
- [ ] Keep the runtime resolver tolerant of missing bindings so existing chats are unaffected.
- [ ] Treat missing assets in old message metadata as unavailable and fall back without broken images.
- [ ] Surface ZIP validation failures in draft/job status rather than toast-only errors.
- [ ] Prefer additive docs and schema fields; avoid renaming current mood fields.

## Final Review Checklist

- [ ] All new backend tests pass.
- [ ] All new frontend tests pass.
- [ ] Existing Persona Visual tests pass.
- [ ] Existing character mood tests pass.
- [ ] Bandit reports no new findings in touched backend code.
- [ ] `git diff --check` passes.
- [ ] Backlog task records touched files, verification, and final summary.
- [ ] Commit includes only files for this feature slice.
