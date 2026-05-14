from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Persona.visual_manifest_assets import collect_visual_manifest_asset_ids
from tldw_Server_API.app.core.Persona.visual_starter_catalog import (
    PersonaVisualStarterCatalogError,
    PersonaVisualStarterCatalogService,
)
from tldw_Server_API.app.core.Persona.visual_starter_fixtures import (
    DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID,
    PersonaVisualStarterAsset,
    PersonaVisualStarterPack,
)


pytestmark = pytest.mark.unit


def _png_bytes(width: int = 2, height: int = 2) -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (width, height), (48, 96, 160, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def db_instance(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "persona_visual_starter_catalog.sqlite", "persona-visual-starter-test")
    yield db
    db.close_connection()


@pytest.fixture(autouse=True)
def visual_storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "visuals"

    def _fake_visuals_dir(user_id: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(
        DatabasePaths,
        "get_user_persona_visuals_dir",
        staticmethod(_fake_visuals_dir),
    )
    return root


def test_starter_catalog_lists_bundled_sprite_pack(
    db_instance: CharactersRAGDB,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    starters = service.list_starter_packs()

    assert [starter["id"] for starter in starters] == [DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID]
    starter = starters[0]
    assert starter["title"] == "Research Buddy Starter"
    assert starter["renderer_type"] == "sprite_frames"
    assert starter["manifest_version"] == 1
    assert starter["asset_count"] >= 1
    assert starter["total_bytes"] > 0
    assert {"idle", "listening", "thinking", "speaking", "error"}.issubset(
        set(starter["states_offered"])
    )


def test_copy_starter_pack_to_persona_creates_inactive_user_owned_draft(
    db_instance: CharactersRAGDB,
    visual_storage_root: Path,
) -> None:
    user_id = "user-1"
    persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    active_pack = db_instance.create_persona_visual_pack(
        persona_id=persona_id,
        user_id=user_id,
        title="Existing active visual",
        status="draft",
    )
    db_instance.activate_persona_visual_pack(
        persona_id=persona_id,
        user_id=user_id,
        pack_id=active_pack["id"],
    )
    service = PersonaVisualStarterCatalogService(db_instance)

    copied = service.copy_starter_pack_to_persona(
        starter_pack_id=DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID,
        persona_id=persona_id,
        user_id=user_id,
    )

    assert copied["status"] == "draft"
    assert copied["persona_id"] == persona_id
    assert copied["user_id"] == user_id
    assert copied["title"] == "Research Buddy Starter"
    assert copied["provenance"] == "imported"
    assert copied["active_at"] is None
    assert db_instance.get_active_persona_visual_pack(
        persona_id=persona_id,
        user_id=user_id,
    )["id"] == active_pack["id"]

    copied_assets = db_instance.list_persona_visual_assets(
        pack_id=copied["id"],
        persona_id=persona_id,
        user_id=user_id,
    )
    assert copied["assets"] == copied_assets
    assert len(copied_assets) == 1
    assert copied_assets[0]["provenance"] == "imported"
    assert (visual_storage_root / persona_id / copied["id"]).is_dir()

    copied_asset_ids = {str(asset["id"]) for asset in copied_assets}
    assert collect_visual_manifest_asset_ids(copied["manifest"]) == copied_asset_ids
    assert "starter_idle" not in str(copied["manifest"])


def test_copy_starter_pack_rejects_malformed_fixture_manifest(
    db_instance: CharactersRAGDB,
) -> None:
    user_id = "user-1"
    persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    malformed = PersonaVisualStarterPack(
        id="malformed",
        title="Malformed starter",
        description="Invalid test fixture",
        renderer_type="sprite_frames",
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {"idle": {"animation_id": "idle"}},
            "animations": {
                "idle": {
                    "frames": [{"asset_id": "missing_asset", "duration_ms": 100}],
                    "frame_rate": 1,
                }
            },
        },
        assets=(
            PersonaVisualStarterAsset(
                asset_key="starter_idle",
                filename="idle.png",
                mime_type="image/png",
                content=_png_bytes(),
                asset_role="frame",
            ),
        ),
    )
    service = PersonaVisualStarterCatalogService(db_instance, starter_packs=(malformed,))

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.copy_starter_pack_to_persona(
            starter_pack_id="malformed",
            persona_id=persona_id,
            user_id=user_id,
        )

    assert exc_info.value.code == "invalid_starter_manifest"
    assert db_instance.list_persona_visual_packs(persona_id=persona_id, user_id=user_id) == []
