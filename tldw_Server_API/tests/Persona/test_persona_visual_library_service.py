from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Persona.visual_library_service import (
    PersonaVisualLibraryService,
    PersonaVisualLibraryServiceError,
)
from tldw_Server_API.app.core.Persona.visual_service import PersonaVisualService


pytestmark = pytest.mark.unit


def _png_bytes(width: int = 1, height: int = 1) -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (width, height), (20, 140, 220, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _valid_manifest(asset_id: str) -> dict[str, object]:
    states = {
        "idle": {"animation_id": "idle"},
        "listening": {"animation_id": "idle"},
        "thinking": {"animation_id": "idle"},
        "speaking": {"animation_id": "idle"},
        "error": {"animation_id": "idle"},
    }
    return {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": states,
        "animations": {
            "idle": {
                "frames": [{"asset_id": asset_id, "duration_ms": 100}],
                "frame_rate": 1,
            }
        },
    }


def _patch_visuals_dir(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    def _fake_visuals_dir(user_id: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(
        DatabasePaths,
        "get_user_persona_visuals_dir",
        staticmethod(_fake_visuals_dir),
    )


@pytest.fixture()
def db_instance(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "persona_visual_library_service.sqlite", "persona-visual-library-service")
    yield db
    db.close_connection()


@pytest.fixture()
def visual_service(db_instance: CharactersRAGDB) -> PersonaVisualService:
    return PersonaVisualService(db_instance)


@pytest.fixture()
def library_service(
    db_instance: CharactersRAGDB,
    visual_service: PersonaVisualService,
) -> PersonaVisualLibraryService:
    return PersonaVisualLibraryService(db_instance, visual_service=visual_service)


def _create_persona_and_pack(
    db: CharactersRAGDB,
    *,
    user_id: str = "user-1",
    persona_name: str = "Source Persona",
    pack_title: str = "Source Pack",
) -> tuple[str, dict]:
    persona_id = db.create_persona_profile({"user_id": user_id, "name": persona_name})
    pack = db.create_persona_visual_pack(
        persona_id=persona_id,
        user_id=user_id,
        title=pack_title,
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {},
            "animations": {},
        },
    )
    return persona_id, pack


def test_save_pack_lists_idempotent_metadata_without_mutating_source(
    library_service: PersonaVisualLibraryService,
    db_instance: CharactersRAGDB,
) -> None:
    persona_id, pack = _create_persona_and_pack(db_instance, pack_title="Warm Assistant")

    first = library_service.save_pack(
        user_id="user-1",
        source_persona_id=persona_id,
        source_pack_id=pack["id"],
        title="Desk helper",
        notes="Good for long research sessions.",
        tags=["Research", "calm", "research"],
    )
    second = library_service.save_pack(
        user_id="user-1",
        source_persona_id=persona_id,
        source_pack_id=pack["id"],
        title="Warm Assistant",
        notes="Updated notes",
        tags=["calm"],
    )

    assert second["id"] == first["id"]
    assert second["title"] == "Warm Assistant"
    assert second["notes"] == "Updated notes"
    assert second["tags"] == ["calm"]
    assert second["source_available"] is True

    listed = library_service.list_items(user_id="user-1")
    assert [item["id"] for item in listed] == [first["id"]]

    source_after_save = db_instance.get_persona_visual_pack(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
    )
    assert source_after_save["status"] == pack["status"]
    assert source_after_save["version"] == pack["version"]
    assert db_instance.list_persona_visual_assets(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
    ) == []


def test_use_library_item_duplicates_source_to_target_as_draft(
    library_service: PersonaVisualLibraryService,
    visual_service: PersonaVisualService,
    db_instance: CharactersRAGDB,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_visuals_dir(monkeypatch, tmp_path / "visuals")
    user_id = "user-1"
    source_persona_id, source_pack = _create_persona_and_pack(
        db_instance,
        user_id=user_id,
        persona_name="Source Persona",
        pack_title="Source Pack",
    )
    target_persona_id, target_pack = _create_persona_and_pack(
        db_instance,
        user_id=user_id,
        persona_name="Target Persona",
        pack_title="Target Active",
    )
    asset = visual_service.create_asset_from_upload(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=source_pack["id"],
        content=_png_bytes(width=2, height=2),
        mime_type="image/png",
        original_filename="idle.png",
        asset_role="frame",
    )
    source_pack = db_instance.update_persona_visual_pack_manifest(
        pack_id=source_pack["id"],
        persona_id=source_persona_id,
        user_id=user_id,
        manifest=_valid_manifest(asset["id"]),
        expected_version=source_pack["version"],
    )
    db_instance.activate_persona_visual_pack(
        persona_id=source_persona_id,
        user_id=user_id,
        pack_id=source_pack["id"],
    )
    db_instance.activate_persona_visual_pack(
        persona_id=target_persona_id,
        user_id=user_id,
        pack_id=target_pack["id"],
    )
    item = library_service.save_pack(
        user_id=user_id,
        source_persona_id=source_persona_id,
        source_pack_id=source_pack["id"],
        title="Reusable Source Pack",
    )

    duplicated = library_service.use_item_for_persona(
        user_id=user_id,
        item_id=item["id"],
        target_persona_id=target_persona_id,
        title="Target Draft From Library",
    )

    assert duplicated["status"] == "draft"
    assert duplicated["persona_id"] == target_persona_id
    assert duplicated["parent_pack_id"] == source_pack["id"]
    assert duplicated["title"] == "Target Draft From Library"
    assert len(duplicated["assets"]) == 1
    assert duplicated["assets"][0]["id"] != asset["id"]
    assert db_instance.get_active_persona_visual_pack(
        persona_id=source_persona_id,
        user_id=user_id,
    )["id"] == source_pack["id"]
    assert db_instance.get_active_persona_visual_pack(
        persona_id=target_persona_id,
        user_id=user_id,
    )["id"] == target_pack["id"]


def test_stale_library_item_cannot_be_used_but_can_be_deleted(
    library_service: PersonaVisualLibraryService,
    db_instance: CharactersRAGDB,
) -> None:
    persona_id, pack = _create_persona_and_pack(db_instance)
    item = library_service.save_pack(
        user_id="user-1",
        source_persona_id=persona_id,
        source_pack_id=pack["id"],
        title="Reusable helper",
    )
    assert db_instance.soft_delete_persona_visual_pack_with_assets(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="user-1",
    )

    listed = library_service.list_items(user_id="user-1")
    assert listed[0]["id"] == item["id"]
    assert listed[0]["source_available"] is False

    with pytest.raises(PersonaVisualLibraryServiceError) as exc_info:
        library_service.use_item_for_persona(
            user_id="user-1",
            item_id=item["id"],
            target_persona_id="target-persona",
        )
    assert exc_info.value.code == "source_pack_unavailable"

    assert library_service.delete_item(user_id="user-1", item_id=item["id"]) is True
    assert library_service.list_items(user_id="user-1") == []


def test_library_service_rejects_cross_user_source_item_and_target(
    library_service: PersonaVisualLibraryService,
    db_instance: CharactersRAGDB,
) -> None:
    source_persona_id, source_pack = _create_persona_and_pack(db_instance, user_id="user-1")
    other_persona_id, _ = _create_persona_and_pack(db_instance, user_id="user-2")
    item = library_service.save_pack(
        user_id="user-1",
        source_persona_id=source_persona_id,
        source_pack_id=source_pack["id"],
        title="Private helper",
    )

    with pytest.raises(PersonaVisualLibraryServiceError) as exc_info:
        library_service.save_pack(
            user_id="user-2",
            source_persona_id=source_persona_id,
            source_pack_id=source_pack["id"],
            title="Other save",
        )
    assert exc_info.value.code == "source_pack_not_found"

    with pytest.raises(PersonaVisualLibraryServiceError) as exc_info:
        library_service.use_item_for_persona(
            user_id="user-2",
            item_id=item["id"],
            target_persona_id=other_persona_id,
        )
    assert exc_info.value.code == "library_item_not_found"

    with pytest.raises(PersonaVisualLibraryServiceError) as exc_info:
        library_service.use_item_for_persona(
            user_id="user-1",
            item_id=item["id"],
            target_persona_id=other_persona_id,
        )
    assert exc_info.value.code == "target_persona_not_found"
