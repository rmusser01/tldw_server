# Persona Visual Pack Duplicate-To-Persona Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a same-user Persona Visual workflow that duplicates one persona's visual pack to a different persona as a draft with copied manifest-referenced assets.

**Architecture:** Keep duplication inside the Persona Visual service and expose it through one synchronous API route that returns the existing `PersonaVisualPackResponse`. Promote manifest asset collection/remapping into a shared helper so import and duplicate use the same traversal. Extend the DB layer narrowly for same-user cross-persona lineage and safe status transitions without weakening normal pack creation.

**Tech Stack:** FastAPI, Pydantic, SQLite-backed `CharactersRAGDB`, `PersonaVisualService`, Next/React shared UI package, Vitest, Pytest, Bandit.

---

## Spec And Tracking

- Spec: `Docs/superpowers/specs/2026-05-09-persona-visual-duplicate-to-persona-design.md`
- Backlog: `TASK-193`
- GitHub: #1449, #1450

Key decisions from the spec:

- Duplicate only to a different same-user persona.
- Return `PersonaVisualPackResponse`; do not expose `asset_id_map`.
- Do not add idempotency keys in V1.
- Copy only manifest-referenced source assets.
- Never activate or archive source or target active packs.
- Keep this in Persona/Buddy visual-pack scope, not VN/CYOA.

Design-review refinements before implementation:

- Normalize a provided duplicate title after trimming; if it becomes empty, fall back to `Copy of {source title}` before calling the DB layer.
- Treat missing source files and checksum mismatches as stored-state conflicts at the API boundary, not ordinary user validation failures.
- Verify failed duplicate attempts do not activate anything and remove any copied files. A failed target pack may remain with status `failed`; do not expose it as a usable draft.
- Keep the duplicate target list scoped to active catalog personas in the UI, while the backend remains the authority for same-user target validation.

## File Structure

Backend:

- Create: `tldw_Server_API/app/core/Persona/visual_manifest_assets.py`
  - Owns manifest asset ID collection and remapping.
- Modify: `tldw_Server_API/app/core/Persona/visual_portability/importer.py`
  - Uses shared remapping helper instead of private helper.
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`
  - Allows explicit same-user cross-persona `parent_pack_id` validation for duplicate creation.
  - Adds a narrow status-update helper for failed-to-draft transitions.
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Exports any new persona visual DB helper names.
- Modify: `tldw_Server_API/app/core/Persona/visual_service.py`
  - Adds duplicate orchestration, preflight, safe storage path resolution, and cleanup.
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
  - Adds duplicate request schema.
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
  - Adds duplicate endpoint and error mapping.

Backend tests:

- Create: `tldw_Server_API/tests/Persona/test_persona_visual_manifest_assets.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visual_service.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`

Frontend:

- Modify: `apps/packages/ui/src/types/persona-visuals.ts`
  - Adds duplicate request and target persona types.
- Modify: `apps/packages/ui/src/services/persona-visuals.ts`
  - Adds duplicate API call and target persona listing helper.
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
  - Adds duplicate UI state/action/card and success handoff behavior.
- Modify: `apps/packages/ui/src/routes/sidepanel-persona.tsx`
  - Passes a callback that switches to the target persona's Visuals tab.

Frontend tests:

- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

Docs:

- Modify: `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`
  - Mark duplicate-to-persona as implemented once the implementation lands.

---

### Task 1: Share Manifest Asset Helpers

**Files:**
- Create: `tldw_Server_API/app/core/Persona/visual_manifest_assets.py`
- Modify: `tldw_Server_API/app/core/Persona/visual_portability/importer.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visual_manifest_assets.py`

- [x] **Step 1: Write failing tests for asset collection and remapping**

Create `tldw_Server_API/tests/Persona/test_persona_visual_manifest_assets.py`:

```python
from copy import deepcopy

from tldw_Server_API.app.core.Persona.visual_manifest_assets import (
    collect_visual_manifest_asset_ids,
    remap_visual_manifest_assets,
)


def test_collect_visual_manifest_asset_ids_reads_all_supported_references() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {
            "idle": {
                "frames": [{"asset_id": "asset-frame"}],
                "asset_ids": ["asset-sheet"],
                "preview_asset_id": "asset-preview",
            }
        },
    }

    assert collect_visual_manifest_asset_ids(manifest) == {
        "asset-frame",
        "asset-sheet",
        "asset-preview",
    }


def test_remap_visual_manifest_assets_returns_copy_with_supported_references_remapped() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {
            "idle": {
                "frames": [{"asset_id": "source-a"}, {"asset_id": "source-b"}],
                "asset_ids": ["source-a", "source-c"],
                "preview_asset_id": "source-b",
            }
        },
    }
    original = deepcopy(manifest)

    remapped = remap_visual_manifest_assets(
        manifest,
        {"source-a": "target-a", "source-b": "target-b"},
    )

    assert manifest == original
    animation = remapped["animations"]["idle"]
    assert animation["frames"] == [
        {"asset_id": "target-a"},
        {"asset_id": "target-b"},
    ]
    assert animation["asset_ids"] == ["target-a", "source-c"]
    assert animation["preview_asset_id"] == "target-b"
```

- [x] **Step 2: Run the new tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_manifest_assets.py -q
```

Expected: import failure because `visual_manifest_assets.py` does not exist.

- [x] **Step 3: Implement the shared helper**

Create `tldw_Server_API/app/core/Persona/visual_manifest_assets.py`:

```python
from __future__ import annotations

from copy import deepcopy
from typing import Any


def collect_visual_manifest_asset_ids(manifest: dict[str, Any]) -> set[str]:
    asset_ids: set[str] = set()
    animations = manifest.get("animations")
    if not isinstance(animations, dict):
        return asset_ids
    for animation in animations.values():
        if not isinstance(animation, dict):
            continue
        frames = animation.get("frames")
        if isinstance(frames, list):
            for frame in frames:
                if isinstance(frame, dict):
                    asset_id = str(frame.get("asset_id") or "").strip()
                    if asset_id:
                        asset_ids.add(asset_id)
        listed_asset_ids = animation.get("asset_ids")
        if isinstance(listed_asset_ids, list):
            for asset_id in listed_asset_ids:
                normalized = str(asset_id or "").strip()
                if normalized:
                    asset_ids.add(normalized)
        preview_asset_id = str(animation.get("preview_asset_id") or "").strip()
        if preview_asset_id:
            asset_ids.add(preview_asset_id)
    return asset_ids


def remap_visual_manifest_assets(
    manifest: dict[str, Any],
    asset_id_map: dict[str, str],
) -> dict[str, Any]:
    remapped = deepcopy(manifest)
    animations = remapped.get("animations")
    if not isinstance(animations, dict):
        return remapped
    for animation in animations.values():
        if not isinstance(animation, dict):
            continue
        frames = animation.get("frames")
        if isinstance(frames, list):
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                asset_id = str(frame.get("asset_id") or "")
                if asset_id in asset_id_map:
                    frame["asset_id"] = asset_id_map[asset_id]
        asset_ids = animation.get("asset_ids")
        if isinstance(asset_ids, list):
            animation["asset_ids"] = [
                asset_id_map.get(str(asset_id), asset_id)
                for asset_id in asset_ids
            ]
        preview_asset_id = str(animation.get("preview_asset_id") or "")
        if preview_asset_id in asset_id_map:
            animation["preview_asset_id"] = asset_id_map[preview_asset_id]
    return remapped
```

- [x] **Step 4: Update importer to use the shared helper**

In `tldw_Server_API/app/core/Persona/visual_portability/importer.py`:

```python
from tldw_Server_API.app.core.Persona.visual_manifest_assets import (
    remap_visual_manifest_assets,
)
```

Replace:

```python
remapped_manifest = _remap_visual_manifest_assets(visual_manifest, id_maps["assets"])
```

with:

```python
remapped_manifest = remap_visual_manifest_assets(visual_manifest, id_maps["assets"])
```

Then remove the private `_remap_visual_manifest_assets()` function and the now-unused `deepcopy` import from `importer.py`.

- [x] **Step 5: Run helper and import regression tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_manifest_assets.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q
```

Expected: helper tests pass and existing import/export API tests still pass.

- [x] **Step 6: Commit helper extraction**

```bash
git add tldw_Server_API/app/core/Persona/visual_manifest_assets.py \
  tldw_Server_API/app/core/Persona/visual_portability/importer.py \
  tldw_Server_API/tests/Persona/test_persona_visual_manifest_assets.py
git commit -m "Extract persona visual manifest asset helpers"
```

---

### Task 2: Add Narrow DB Support For Duplicate Lineage

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visual_service.py`

- [x] **Step 1: Write service-level failing tests for cross-persona parent creation and status transition**

Append focused expectations to the duplicate service tests in Task 3, or temporarily add this failing test to `tldw_Server_API/tests/Persona/test_persona_visual_service.py`:

```python
def test_db_allows_explicit_cross_persona_parent_for_duplicate_path(db_instance: CharactersRAGDB) -> None:
    user_id = "user-1"
    source_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Source"})
    target_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    source_pack = db_instance.create_persona_visual_pack(
        persona_id=source_persona_id,
        user_id=user_id,
        title="Source Pack",
    )

    target_pack = db_instance.create_persona_visual_pack(
        persona_id=target_persona_id,
        user_id=user_id,
        title="Duplicate",
        parent_pack_id=source_pack["id"],
        parent_persona_id=source_persona_id,
        status="failed",
        provenance="mixed",
    )
    updated = db_instance.update_persona_visual_pack_status(
        pack_id=target_pack["id"],
        persona_id=target_persona_id,
        user_id=user_id,
        status="draft",
        expected_version=target_pack["version"],
    )

    assert updated["parent_pack_id"] == source_pack["id"]
    assert updated["status"] == "draft"
```

- [x] **Step 2: Run the focused test and verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py::test_db_allows_explicit_cross_persona_parent_for_duplicate_path -q
```

Expected: failure because `parent_persona_id` and `update_persona_visual_pack_status()` do not exist.

- [x] **Step 3: Extend pack creation with explicit parent persona**

In `create_persona_visual_pack()` in `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`, add an optional keyword after `parent_pack_id`:

```python
parent_persona_id: str | None = None,
```

Change parent validation to:

```python
if parent_pack_id:
    self._require_persona_visual_pack_owner(
        conn,
        pack_id=str(parent_pack_id),
        persona_id=str(parent_persona_id or persona_id),
        user_id=user_id,
    )
```

Do not change existing callers; the default remains same-persona validation.

- [x] **Step 4: Add status-update helper**

In `persona_state_store.py`, near `update_persona_visual_pack_manifest()`, add:

```python
def update_persona_visual_pack_status(
    self,
    *,
    pack_id: str,
    persona_id: str,
    user_id: str,
    status: str,
    expected_version: int | None = None,
) -> dict[str, Any] | None:
    status_value = self._normalize_persona_visual_enum(
        status,
        allowed=self._ALLOWED_PERSONA_VISUAL_PACK_STATUSES,
        field_name="status",
    )
    if status_value == "active":
        raise InputError("Use activate_persona_visual_pack for active status transitions.")
    now = self._get_current_utc_timestamp_iso()
    bool_cast = bool if self.backend_type == BackendType.POSTGRESQL else int
    params: list[Any] = [
        status_value,
        None,
        now,
        pack_id,
        user_id,
        persona_id,
        bool_cast(False),
    ]
    where_sql = "id = ? AND user_id = ? AND persona_id = ? AND deleted = ?"
    if expected_version is not None:
        where_sql += " AND version = ?"
        params.append(int(expected_version))
    query = (
        "UPDATE persona_visual_packs "
        "SET status = ?, active_at = ?, last_modified = ?, version = version + 1 "
        f"WHERE {where_sql}"  # nosec B608
    )
    with self.transaction() as conn:
        self._require_active_persona_profile_owner(conn, persona_id=persona_id, user_id=user_id)
        self._require_persona_visual_pack_owner(
            conn,
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        prepared_query, prepared_params = self._prepare_backend_statement(query, tuple(params))
        cursor = conn.execute(prepared_query, prepared_params or ())
        if cursor.rowcount == 0 and expected_version is not None:
            raise ConflictError(
                "Persona visual pack version mismatch.",
                entity="persona_visual_packs",
                entity_id=pack_id,
            )
    return self.get_persona_visual_pack(pack_id=pack_id, persona_id=persona_id, user_id=user_id)
```

- [x] **Step 5: Export the helper**

In `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`, add `update_persona_visual_pack_status` to the persona visual method export list near `update_persona_visual_pack_manifest`.

- [x] **Step 6: Run the DB-focused test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py::test_db_allows_explicit_cross_persona_parent_for_duplicate_path -q
```

Expected: pass.

- [x] **Step 7: Commit DB support**

```bash
git add tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/Persona/test_persona_visual_service.py
git commit -m "Support persona visual duplicate lineage"
```

---

### Task 3: Implement PersonaVisualService Duplication

**Files:**
- Modify: `tldw_Server_API/app/core/Persona/visual_service.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visual_service.py`

- [ ] **Step 1: Write failing service test for successful duplicate**

Add to `tldw_Server_API/tests/Persona/test_persona_visual_service.py`:

```python
def _manifest_with_all_reference_shapes(asset_a: str, asset_b: str) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {
            "idle": {"animation_id": "idle"},
            "listening": {"animation_id": "idle"},
            "thinking": {"animation_id": "idle"},
            "speaking": {"animation_id": "idle"},
            "error": {"animation_id": "idle"},
        },
        "animations": {
            "idle": {
                "frames": [
                    {"asset_id": asset_a, "duration_ms": 100},
                    {"asset_id": asset_b, "duration_ms": 100},
                ],
                "asset_ids": [asset_a],
                "preview_asset_id": asset_b,
                "frame_rate": 2,
            }
        },
    }


def test_duplicate_pack_to_persona_copies_referenced_assets_and_remaps_manifest(
    service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    user_id = "user-1"
    source_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Source"})
    target_persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    source_pack = db_instance.create_persona_visual_pack(
        persona_id=source_persona_id,
        user_id=user_id,
        title="Source Pack",
    )
    asset_a = service.create_asset_from_upload(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=source_pack["id"],
        content=_png_bytes(width=2, height=2),
        mime_type="image/png",
        original_filename="idle-a.png",
        asset_role="frame",
    )
    asset_b = service.create_asset_from_upload(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=source_pack["id"],
        content=_png_bytes(width=3, height=3),
        mime_type="image/png",
        original_filename="idle-b.png",
        asset_role="preview",
    )
    unused = service.create_asset_from_upload(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=source_pack["id"],
        content=_png_bytes(width=4, height=4),
        mime_type="image/png",
        original_filename="unused.png",
        asset_role="generated_candidate",
    )
    updated_source = db_instance.update_persona_visual_pack_manifest(
        pack_id=source_pack["id"],
        persona_id=source_persona_id,
        user_id=user_id,
        manifest=_manifest_with_all_reference_shapes(asset_a["id"], asset_b["id"]),
        expected_version=source_pack["version"],
    )

    duplicated = service.duplicate_pack_to_persona(
        source_persona_id=source_persona_id,
        user_id=user_id,
        pack_id=updated_source["id"],
        target_persona_id=target_persona_id,
        title="Target Draft",
    )

    assert duplicated["status"] == "draft"
    assert duplicated["persona_id"] == target_persona_id
    assert duplicated["parent_pack_id"] == source_pack["id"]
    copied_assets = db_instance.list_persona_visual_assets(
        pack_id=duplicated["id"],
        persona_id=target_persona_id,
        user_id=user_id,
    )
    assert len(copied_assets) == 2
    copied_ids = {asset["id"] for asset in copied_assets}
    assert asset_a["id"] not in copied_ids
    assert asset_b["id"] not in copied_ids
    assert unused["checksum_sha256"] not in {asset["checksum_sha256"] for asset in copied_assets}
    assert all(f"persona_visuals/{target_persona_id}/{duplicated['id']}/" in asset["storage_key"] for asset in copied_assets)
    remapped_animation = duplicated["manifest"]["animations"]["idle"]
    assert {frame["asset_id"] for frame in remapped_animation["frames"]} == copied_ids
    assert set(remapped_animation["asset_ids"]).issubset(copied_ids)
    assert remapped_animation["preview_asset_id"] in copied_ids
```

- [ ] **Step 2: Write failing service tests for V1 guardrails**

Add tests for same-persona, missing manifest asset row, and missing source file:

```python
def test_duplicate_pack_rejects_same_persona_target(service: PersonaVisualService, db_instance: CharactersRAGDB) -> None:
    persona_id, pack = _create_pack(db_instance)

    with pytest.raises(PersonaVisualServiceError) as exc_info:
        service.duplicate_pack_to_persona(
            source_persona_id=persona_id,
            user_id="user-1",
            pack_id=pack["id"],
            target_persona_id=persona_id,
        )

    assert exc_info.value.code == "same_persona_target_unsupported"
```

For missing file, create an asset, update the manifest to reference it, remove `Path(asset["storage_path"])`, and assert `source_asset_missing`.

For missing row, update the manifest to reference `"does-not-exist"` and assert `invalid_manifest`.

For checksum mismatch, mutate the stored source file bytes after upload and assert `source_asset_checksum_mismatch`.

For partial failure cleanup, induce an exception after the first copied file is created, then assert copied files under the target pack storage directory were removed and no target pack was promoted to `draft` or `active`.

- [ ] **Step 3: Run service duplicate tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py -q
```

Expected: failures because `duplicate_pack_to_persona()` does not exist.

- [ ] **Step 4: Add source asset path resolution helper**

In `PersonaVisualService`, add a private helper based on the exporter pattern:

```python
def _asset_storage_path(self, *, user_id: str, storage_key: str) -> Path:
    prefix = f"{VISUAL_STORAGE_PREFIX}/"
    relative_key = storage_key[len(prefix):] if storage_key.startswith(prefix) else storage_key
    relative_path = Path(*Path(relative_key).parts)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise PersonaVisualServiceError(
            "invalid_storage_path",
            "Persona visual storage path escapes the user visual directory.",
        )
    base = DatabasePaths.get_user_persona_visuals_dir(user_id).resolve(strict=False)
    target_path = (base / relative_path).resolve(strict=False)
    if not target_path.is_relative_to(base):
        raise PersonaVisualServiceError(
            "invalid_storage_path",
            "Persona visual storage path escapes the user visual directory.",
        )
    return target_path
```

If using `Path(*Path(relative_key).parts)` proves awkward with normalized keys, import and reuse `_safe_relative_storage_path()` from the exporter only if it does not create an import cycle. Otherwise keep the helper local.

- [ ] **Step 5: Add duplicate orchestration**

In `PersonaVisualService`, implement:

```python
def duplicate_pack_to_persona(
    self,
    *,
    source_persona_id: str,
    user_id: str,
    pack_id: str,
    target_persona_id: str,
    title: str | None = None,
) -> dict[str, Any]:
    if str(source_persona_id) == str(target_persona_id):
        raise PersonaVisualServiceError(
            "same_persona_target_unsupported",
            "Persona visual packs can only be duplicated to a different persona in V1.",
        )

    source_pack = self._db.get_persona_visual_pack(
        pack_id=pack_id,
        persona_id=source_persona_id,
        user_id=user_id,
    )
    if not source_pack:
        raise PersonaVisualServiceError("pack_not_found", "Persona visual pack not found for user.")

    target_persona = self._db.get_persona_profile(
        persona_id=target_persona_id,
        user_id=user_id,
    )
    if not target_persona:
        raise PersonaVisualServiceError(
            "target_persona_not_found",
            "Target persona not found for user.",
            details={"target_persona_id": target_persona_id},
        )

    source_manifest = source_pack.get("manifest") if isinstance(source_pack.get("manifest"), dict) else {}
    source_assets = self._db.list_persona_visual_assets(
        pack_id=pack_id,
        persona_id=source_persona_id,
        user_id=user_id,
    )
    source_assets_by_id = {str(asset["id"]): asset for asset in source_assets}
    referenced_asset_ids = collect_visual_manifest_asset_ids(source_manifest)
    missing_asset_ids = sorted(referenced_asset_ids - set(source_assets_by_id))
    if missing_asset_ids:
        raise PersonaVisualServiceError(
            "invalid_manifest",
            "Persona visual manifest references assets that are not in the source pack.",
            details={"asset_ids": missing_asset_ids},
        )

    # Preflight bytes and checksums before creating target records.
    preflight_assets = []
    for asset_id in sorted(referenced_asset_ids):
        asset = source_assets_by_id[asset_id]
        source_path = self._asset_storage_path(
            user_id=user_id,
            storage_key=str(asset.get("storage_key") or ""),
        )
        if not source_path.is_file():
            raise PersonaVisualServiceError(
                "source_asset_missing",
                "Persona visual source asset file is missing.",
                details={"asset_id": asset_id},
            )
        content = source_path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        if checksum != str(asset.get("checksum_sha256") or ""):
            raise PersonaVisualServiceError(
                "source_asset_checksum_mismatch",
                "Persona visual source asset checksum does not match metadata.",
                details={"asset_id": asset_id},
            )
        preflight_assets.append((asset, content))

    title_value = str(title or "").strip()
    if not title_value:
        title_value = f"Copy of {source_pack['title']}"

    target_pack = self._db.create_persona_visual_pack(
        persona_id=target_persona_id,
        user_id=user_id,
        title=title_value,
        renderer_type=str(source_pack.get("renderer_type") or "sprite_frames"),
        status="failed",
        parent_pack_id=pack_id,
        parent_persona_id=source_persona_id,
        provenance="mixed",
        manifest={
            "manifest_version": 1,
            "renderer_type": str(source_pack.get("renderer_type") or "sprite_frames"),
            "states": {},
            "animations": {},
        },
    )
    copied_file_paths: list[Path] = []
    try:
        asset_id_map: dict[str, str] = {}
        copied_assets: list[dict[str, Any]] = []
        for source_asset, content in preflight_assets:
            copied = self.create_asset_from_upload(
                persona_id=target_persona_id,
                user_id=user_id,
                pack_id=str(target_pack["id"]),
                content=content,
                mime_type=str(source_asset.get("mime_type") or "application/octet-stream"),
                original_filename=source_asset.get("original_filename"),
                asset_role=str(source_asset.get("asset_role") or "frame"),
                provenance="mixed",
            )
            if copied.get("storage_path"):
                copied_file_paths.append(Path(str(copied["storage_path"])))
            asset_id_map[str(source_asset["id"])] = str(copied["id"])
            copied_assets.append(copied)

        remapped_manifest = remap_visual_manifest_assets(source_manifest, asset_id_map)
        validation = validate_visual_manifest(
            remapped_manifest,
            available_asset_ids={str(asset["id"]) for asset in copied_assets},
            available_asset_dimensions={
                str(asset["id"]): (int(asset["width"]), int(asset["height"]))
                for asset in copied_assets
                if asset.get("width") is not None and asset.get("height") is not None
            },
            require_activatable=False,
        )
        updated_pack = self._db.update_persona_visual_pack_manifest(
            pack_id=str(target_pack["id"]),
            persona_id=target_persona_id,
            user_id=user_id,
            manifest=validation.manifest,
            expected_version=int(target_pack["version"]),
        )
        finalized = self._db.update_persona_visual_pack_status(
            pack_id=str(target_pack["id"]),
            persona_id=target_persona_id,
            user_id=user_id,
            status="draft",
            expected_version=int(updated_pack["version"]),
        )
    except Exception:
        for path in copied_file_paths:
            path.unlink(missing_ok=True)
        raise

    assets = self._db.list_persona_visual_assets(
        pack_id=str(target_pack["id"]),
        persona_id=target_persona_id,
        user_id=user_id,
    )
    finalized["assets"] = assets
    finalized["assets_by_id"] = {str(asset["id"]): asset for asset in assets}
    return finalized
```

Adjust exact calls if `get_persona_profile()` uses a different signature; use the existing profile owner helper pattern from the endpoint if needed.

- [ ] **Step 6: Convert manifest validation errors**

Wrap `validate_visual_manifest()` in the duplicate path:

```python
try:
    validation = validate_visual_manifest(...)
except PersonaVisualManifestError as exc:
    raise PersonaVisualServiceError(
        "invalid_manifest",
        str(exc),
        details={"pack_id": pack_id},
    ) from exc
```

- [ ] **Step 7: Run service tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py -q
```

Expected: pass.

- [ ] **Step 8: Commit service duplicate behavior**

```bash
git add tldw_Server_API/app/core/Persona/visual_service.py \
  tldw_Server_API/tests/Persona/test_persona_visual_service.py
git commit -m "Add persona visual pack duplication service"
```

---

### Task 4: Add Duplicate API Contract

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`

- [ ] **Step 1: Write failing API tests**

In `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`, add:

```python
def test_duplicate_visual_pack_to_another_persona_creates_draft(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        source_persona_id = _create_persona(client, name="Source Persona")
        target_persona_id = _create_persona(client, name="Target Persona")
        source_pack = _create_visual_pack(client, source_persona_id, title="Source Visuals")
        asset = _upload_png(client, source_persona_id, source_pack["id"])
        manifest_response = client.patch(
            f"/api/v1/persona/profiles/{source_persona_id}/visual-packs/{source_pack['id']}/manifest",
            json={"manifest": _valid_manifest(asset["id"]), "expected_version": source_pack["version"]},
        )
        assert manifest_response.status_code == 200, manifest_response.text

        response = client.post(
            f"/api/v1/persona/profiles/{source_persona_id}/visual-packs/{source_pack['id']}/duplicate",
            json={"target_persona_id": target_persona_id, "title": "Target Draft"},
        )

        assert response.status_code == 201, response.text
        payload = response.json()
        assert payload["title"] == "Target Draft"
        assert payload["persona_id"] == target_persona_id
        assert payload["status"] == "draft"
        assert payload["parent_pack_id"] == source_pack["id"]
        assert "asset_id_map" not in payload
        assert len(payload["assets"]) == 1
        assert payload["assets"][0]["id"] != asset["id"]
```

Add negative tests:

```python
def test_duplicate_visual_pack_rejects_same_persona_target(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Source Persona")
        source_pack = _create_visual_pack(client, persona_id, title="Source Visuals")

        response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{source_pack['id']}/duplicate",
            json={"target_persona_id": persona_id},
        )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "same_persona_target_unsupported"
```

Also test unauthorized target by creating a target persona under a different user DB/client and expecting `404` with `target_persona_not_found`.

- [ ] **Step 2: Run API tests and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q
```

Expected: duplicate endpoint returns 404 because route does not exist.

- [ ] **Step 3: Add request schema**

In `tldw_Server_API/app/api/v1/schemas/persona.py` near `PersonaVisualPackCreate`:

```python
class PersonaVisualPackDuplicateRequest(BaseModel):
    target_persona_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("target_persona_id")
    @classmethod
    def normalize_target_persona_id(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("target_persona_id is required")
        return normalized
```

- [ ] **Step 4: Import the schema in the endpoint**

Add `PersonaVisualPackDuplicateRequest` to the existing imports from `tldw_Server_API.app.api.v1.schemas.persona`.

- [ ] **Step 5: Extend service error mapping**

In `_persona_visual_service_error_to_http()`:

```python
if exc.code in {
    "pack_not_found",
    "asset_not_found",
    "candidate_not_found",
    "target_persona_not_found",
}:
    status_code = status.HTTP_404_NOT_FOUND
```

Map stale stored-source conditions to conflict:

```python
elif exc.code in {"source_asset_missing", "source_asset_checksum_mismatch"}:
    status_code = status.HTTP_409_CONFLICT
```

Leave `same_persona_target_unsupported` and `invalid_manifest` as `400`.

- [ ] **Step 6: Add duplicate endpoint**

Place after `get_persona_visual_pack()` or before asset upload in `persona.py`:

```python
@router.post(
    "/profiles/{persona_id}/visual-packs/{pack_id}/duplicate",
    response_model=PersonaVisualPackResponse,
    tags=["persona"],
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_rate_limit)],
)
async def duplicate_persona_visual_pack(
    persona_id: str,
    pack_id: str,
    payload: PersonaVisualPackDuplicateRequest,
    _current_user: User = Depends(get_request_user),
    visual_service: PersonaVisualService = Depends(get_persona_visual_service),
) -> PersonaVisualPackResponse:
    """Duplicate a persona visual pack to another same-user persona as a draft."""
    if not is_persona_enabled():
        raise HTTPException(status_code=404, detail="Persona disabled")
    user_id = _require_current_user_id(_current_user)
    try:
        duplicated = await _run_persona_db_call(
            visual_service.duplicate_pack_to_persona,
            source_persona_id=persona_id,
            user_id=user_id,
            pack_id=pack_id,
            target_persona_id=payload.target_persona_id,
            title=payload.title,
        )
        assets = list(duplicated.get("assets") or [])
        return _persona_visual_pack_to_response(duplicated, assets=assets)
    except PersonaVisualServiceError as exc:
        raise _persona_visual_service_error_to_http(exc) from exc
    except (InputError, ConflictError, CharactersRAGDBError) as exc:
        raise _to_http_exception(exc, action="duplicate persona visual pack") from exc
```

- [ ] **Step 7: Run API tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q
```

Expected: pass.

- [ ] **Step 8: Commit API contract**

```bash
git add tldw_Server_API/app/api/v1/schemas/persona.py \
  tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/tests/Persona/test_persona_visuals_api.py
git commit -m "Expose persona visual pack duplication API"
```

---

### Task 5: Add Frontend Service And Types

**Files:**
- Modify: `apps/packages/ui/src/types/persona-visuals.ts`
- Modify: `apps/packages/ui/src/services/persona-visuals.ts`
- Test indirectly in: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [ ] **Step 1: Add frontend types**

In `apps/packages/ui/src/types/persona-visuals.ts`:

```ts
export interface PersonaVisualPackDuplicateRequest {
  target_persona_id: string
  title?: string | null
}

export interface PersonaVisualDuplicateTarget {
  id: string
  name?: string | null
}
```

- [ ] **Step 2: Add API helper imports**

In `apps/packages/ui/src/services/persona-visuals.ts`, import the new types:

```ts
  PersonaVisualDuplicateTarget,
  PersonaVisualPackDuplicateRequest,
```

- [ ] **Step 3: Add duplicate API helper**

In `persona-visuals.ts`:

```ts
export async function duplicatePersonaVisualPack(
  sourcePersonaId: string,
  packId: string,
  payload: PersonaVisualPackDuplicateRequest
): Promise<PersonaVisualPack> {
  return fetchPersonaVisualJson<PersonaVisualPack>(
    packPath(sourcePersonaId, packId, "/duplicate"),
    {
      method: "POST",
      body: payload
    }
  )
}
```

- [ ] **Step 4: Add duplicate target listing helper**

Use the same fetch wrapper so existing `fetchWithAuth` test mocks keep working:

```ts
export async function listPersonaVisualDuplicateTargets(): Promise<
  PersonaVisualDuplicateTarget[]
> {
  const payload = await fetchPersonaVisualJson<unknown>("/api/v1/persona/catalog")
  if (!Array.isArray(payload)) return []
  return payload
    .map((item) => {
      if (!item || typeof item !== "object") return null
      const candidate = item as { id?: unknown; name?: unknown }
      const id = String(candidate.id || "").trim()
      if (!id) return null
      return {
        id,
        name:
          typeof candidate.name === "string" && candidate.name.trim()
            ? candidate.name
            : null
      }
    })
    .filter((item): item is PersonaVisualDuplicateTarget => item !== null)
}
```

- [ ] **Step 5: Run TypeScript-focused frontend test file after UI task**

Defer execution until Task 6 adds UI consumers:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
```

Expected after Task 6: pass.

---

### Task 6: Add VisualPackEditor Duplicate UI

**Files:**
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Modify: `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- Test: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [ ] **Step 1: Write failing UI test for duplicate flow**

In `VisualPackEditor.test.tsx`, add a test near import/export tests:

```tsx
it("duplicates a visual pack to another persona as a draft", async () => {
  const sourcePack = {
    id: "pack-1",
    persona_id: "persona-1",
    title: "Animated pack",
    renderer_type: "sprite_frames",
    status: "active",
    manifest: structuredClone(baseManifest),
    assets: visualAssets,
    version: 3
  }
  const duplicatedPack = {
    ...sourcePack,
    id: "pack-duplicate",
    persona_id: "persona-2",
    title: "Research Buddy copy",
    status: "draft",
    parent_pack_id: "pack-1",
    assets: [{ ...visualAssets[0], id: "asset-copy", pack_id: "pack-duplicate", persona_id: "persona-2" }]
  }
  const openTarget = vi.fn()

  mocks.fetchWithAuth.mockImplementation((path: string, init?: { method?: string; body?: any }) => {
    const method = init?.method || "GET"
    if (path === "/api/v1/persona/profiles/persona-1/visual-packs" && method === "GET") {
      return okResponse([sourcePack])
    }
    if (
      path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/generated-candidates" &&
      method === "GET"
    ) {
      return okResponse({ candidates: [] })
    }
    if (path === "/api/v1/persona/catalog" && method === "GET") {
      return okResponse([
        { id: "persona-1", name: "Source Persona" },
        { id: "persona-2", name: "Research Buddy" }
      ])
    }
    if (
      path === "/api/v1/persona/profiles/persona-1/visual-packs/pack-1/duplicate" &&
      method === "POST"
    ) {
      expect(parseJsonBody(init?.body)).toEqual({
        target_persona_id: "persona-2",
        title: "Research Buddy copy"
      })
      return okResponse(duplicatedPack)
    }
    return Promise.resolve({
      ok: false,
      status: 404,
      error: `Unhandled path: ${path}`,
      json: async () => ({})
    })
  })

  render(
    <VisualPackEditor
      selectedPersonaId="persona-1"
      selectedPersonaName="Source Persona"
      isActive
      onOpenPersonaVisuals={openTarget}
    />
  )

  await waitFor(() =>
    expect(screen.getByTestId("persona-visual-pack-status")).toHaveTextContent("active")
  )
  await waitFor(() =>
    expect(screen.getByTestId("persona-visual-duplicate-target-select")).toHaveTextContent(
      "Research Buddy"
    )
  )
  expect(screen.getByTestId("persona-visual-duplicate-target-select")).not.toHaveTextContent(
    "Source Persona"
  )

  fireEvent.change(screen.getByTestId("persona-visual-duplicate-title-input"), {
    target: { value: "Research Buddy copy" }
  })
  fireEvent.change(screen.getByTestId("persona-visual-duplicate-target-select"), {
    target: { value: "persona-2" }
  })
  fireEvent.click(screen.getByTestId("persona-visual-duplicate-button"))

  await waitFor(() =>
    expect(screen.getByText(/Duplicated as a draft for Research Buddy/)).toBeInTheDocument()
  )
  fireEvent.click(screen.getByTestId("persona-visual-duplicate-open-target"))
  expect(openTarget).toHaveBeenCalledWith("persona-2")
})
```

- [ ] **Step 2: Run the UI test and verify it fails**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
```

Expected: type/test failure because duplicate UI does not exist.

- [ ] **Step 3: Extend `VisualPackEditorProps`**

In `VisualPackEditor.tsx`:

```ts
type VisualPackEditorProps = {
  selectedPersonaId: string
  selectedPersonaName: string
  isActive?: boolean
  onOpenPersonaVisuals?: (personaId: string) => void
}
```

Destructure `onOpenPersonaVisuals`.

- [ ] **Step 4: Import duplicate helpers and icon**

Add `Copy` from `lucide-react`.

Add service imports:

```ts
  duplicatePersonaVisualPack,
  listPersonaVisualDuplicateTargets,
```

Add type import:

```ts
  PersonaVisualDuplicateTarget,
```

- [ ] **Step 5: Add duplicate state**

Near other import/export state:

```ts
const [duplicateTargets, setDuplicateTargets] = React.useState<PersonaVisualDuplicateTarget[]>([])
const [duplicateTargetsLoading, setDuplicateTargetsLoading] = React.useState(false)
const [duplicateTargetId, setDuplicateTargetId] = React.useState("")
const [duplicateTitle, setDuplicateTitle] = React.useState("")
const [duplicatingPack, setDuplicatingPack] = React.useState(false)
const [lastDuplicatedPersonaId, setLastDuplicatedPersonaId] = React.useState("")
```

Add memo:

```ts
const availableDuplicateTargets = React.useMemo(
  () => duplicateTargets.filter((target) => target.id !== selectedPersonaId),
  [duplicateTargets, selectedPersonaId]
)
const selectedDuplicateTarget = availableDuplicateTargets.find(
  (target) => target.id === duplicateTargetId
) ?? null
```

- [ ] **Step 6: Load duplicate targets**

Add a callback/effect similar to `loadPacks()`:

```ts
const loadDuplicateTargets = React.useCallback(async () => {
  if (!isActive || !selectedPersonaId || !selectedPack) return
  setDuplicateTargetsLoading(true)
  try {
    const targets = await listPersonaVisualDuplicateTargets()
    const available = targets.filter((target) => target.id !== selectedPersonaId)
    setDuplicateTargets(targets)
    setDuplicateTargetId((current) =>
      current && available.some((target) => target.id === current)
        ? current
        : available[0]?.id ?? ""
    )
  } catch (loadError) {
    setError(
      loadError instanceof Error
        ? loadError.message
        : "Failed to load persona duplicate targets."
    )
  } finally {
    setDuplicateTargetsLoading(false)
  }
}, [isActive, selectedPersonaId, selectedPack?.id])
```

Run it in an effect:

```ts
React.useEffect(() => {
  void loadDuplicateTargets()
}, [loadDuplicateTargets])
```

When `selectedPack` changes, initialize the title:

```ts
React.useEffect(() => {
  setDuplicateTitle(selectedPack ? `Copy of ${selectedPack.title}` : "")
  setLastDuplicatedPersonaId("")
}, [selectedPack?.id])
```

- [ ] **Step 7: Add duplicate handler**

```ts
const handleDuplicatePack = async () => {
  if (!selectedPersonaId || !selectedPack || !duplicateTargetId) return
  setDuplicatingPack(true)
  setError(null)
  try {
    const duplicated = await duplicatePersonaVisualPack(
      selectedPersonaId,
      selectedPack.id,
      {
        target_persona_id: duplicateTargetId,
        title: duplicateTitle.trim() || `Copy of ${selectedPack.title}`
      }
    )
    setLastDuplicatedPersonaId(duplicated.persona_id)
    const targetName =
      selectedDuplicateTarget?.name || selectedDuplicateTarget?.id || duplicated.persona_id
    setStatusMessage(
      t("sidepanel:personaGarden.visuals.duplicatedDraft", {
        defaultValue: `Duplicated as a draft for ${targetName}. Review and activate it from that persona's Visuals tab.`
      })
    )
  } catch (duplicateError) {
    setError(
      duplicateError instanceof Error
        ? duplicateError.message
        : t("sidepanel:personaGarden.visuals.duplicateError", {
            defaultValue: "Failed to duplicate visual pack."
          })
    )
  } finally {
    setDuplicatingPack(false)
  }
}
```

- [ ] **Step 8: Add duplicate card in the Portability section**

Change the portability grid to allow three cards:

```tsx
<div className="mt-3 grid gap-3 xl:grid-cols-3">
```

Add a new card before export or after import:

```tsx
<div className="rounded border border-border bg-bg p-2">
  <div className="flex flex-wrap items-center justify-between gap-2">
    <Typography.Text strong>Duplicate to persona</Typography.Text>
    <Tag>creates draft</Tag>
  </div>
  <div className="mt-1 text-xs text-text-muted">
    Copy this pack to another persona. It stays a draft until reviewed and activated.
  </div>
  <div className="mt-2 grid gap-2">
    <label className="text-xs text-text-muted">
      <span className="mb-1 block">Target persona</span>
      <select
        data-testid="persona-visual-duplicate-target-select"
        className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
        value={duplicateTargetId}
        disabled={duplicateTargetsLoading || !availableDuplicateTargets.length}
        onChange={(event) => setDuplicateTargetId(event.target.value)}
      >
        {availableDuplicateTargets.map((target) => (
          <option key={target.id} value={target.id}>
            {target.name || target.id}
          </option>
        ))}
      </select>
    </label>
    <label className="text-xs text-text-muted">
      <span className="mb-1 block">Draft title</span>
      <input
        data-testid="persona-visual-duplicate-title-input"
        className="w-full rounded border border-border bg-bg px-2 py-1 text-sm text-text"
        value={duplicateTitle}
        onChange={(event) => setDuplicateTitle(event.target.value)}
      />
    </label>
    <Button
      data-testid="persona-visual-duplicate-button"
      size="small"
      icon={<Copy className="h-3.5 w-3.5" />}
      loading={duplicatingPack}
      disabled={!duplicateTargetId || !selectedPack}
      onClick={() => void handleDuplicatePack()}
    >
      Duplicate as draft
    </Button>
    {lastDuplicatedPersonaId && onOpenPersonaVisuals ? (
      <Button
        data-testid="persona-visual-duplicate-open-target"
        size="small"
        type="link"
        onClick={() => onOpenPersonaVisuals(lastDuplicatedPersonaId)}
      >
        Open target Visuals
      </Button>
    ) : null}
  </div>
  {!availableDuplicateTargets.length ? (
    <div className="mt-2 text-xs text-text-muted">
      Add another persona before duplicating this pack.
    </div>
  ) : null}
</div>
```

- [ ] **Step 9: Wire sidepanel persona switching**

In `apps/packages/ui/src/routes/sidepanel-persona.tsx`, pass:

```tsx
onOpenPersonaVisuals={(personaId) => {
  setSelectedPersonaId(personaId)
  setActiveTab("visuals")
}}
```

to `LazyVisualPackEditor`.

- [ ] **Step 10: Run UI tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
```

Expected: pass.

- [ ] **Step 11: Commit frontend duplicate UI**

```bash
git add apps/packages/ui/src/types/persona-visuals.ts \
  apps/packages/ui/src/services/persona-visuals.ts \
  apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx \
  apps/packages/ui/src/routes/sidepanel-persona.tsx \
  apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
git commit -m "Add persona visual duplicate UI"
```

---

### Task 7: Update Product Documentation And Final Verification

**Files:**
- Modify: `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`
- Modify: `backlog/tasks/task-193 - Write-persona-visual-pack-duplicate-implementation-plan.md` only if executing this plan later in the same branch

- [ ] **Step 1: Update PRD Phase 3 status**

In `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`, update the Phase 3 list item:

```markdown
1. Duplicate pack to another persona. Initial same-user draft duplication is implemented by #1450.
```

If this implementation is not merged yet, phrase it as:

```markdown
1. Duplicate pack to another persona. First implementation target: same-user draft duplication (#1450).
```

- [ ] **Step 2: Run backend focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_manifest_assets.py tldw_Server_API/tests/Persona/test_persona_visual_service.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q
```

Expected: pass.

- [ ] **Step 3: Run frontend focused tests**

Run:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
```

Expected: pass.

- [ ] **Step 4: Run Bandit on touched backend code**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Persona/visual_manifest_assets.py \
  tldw_Server_API/app/core/Persona/visual_service.py \
  tldw_Server_API/app/core/Persona/visual_portability/importer.py \
  tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/app/api/v1/schemas/persona.py \
  tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py \
  -f json -o /tmp/bandit_persona_visual_duplicate.json
```

Expected: no new high or medium findings in touched code. If Bandit is not installed in the environment, document the blocker and do not claim it passed.

- [ ] **Step 5: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 6: Review changed files**

Run:

```bash
git status --short
git diff --stat
```

Expected: only planned files changed.

- [ ] **Step 7: Commit docs and verification notes**

```bash
git add Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
git commit -m "Document persona visual duplicate workflow"
```

If the implementation tasks were not already committed task-by-task, commit all remaining implementation files with a single message:

```bash
git add tldw_Server_API apps/packages/ui Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
git commit -m "Implement persona visual pack duplication"
```

---

## Final Acceptance Checklist

- [ ] Backend duplicate service copies only manifest-referenced assets.
- [ ] Public duplicate endpoint returns `PersonaVisualPackResponse`.
- [ ] Same-persona duplicate is rejected.
- [ ] Target draft is not activated automatically.
- [ ] Source and target active packs are unchanged.
- [ ] UI excludes the current persona from duplicate targets.
- [ ] UI communicates that duplicates are drafts for review.
- [ ] Tests pass:
  - `python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_manifest_assets.py tldw_Server_API/tests/Persona/test_persona_visual_service.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q`
  - `cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
- [ ] Bandit run is recorded or blocker documented.
- [ ] `git diff --check` passes.
