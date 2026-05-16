from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Persona.visual_manifest_assets import (
    collect_visual_manifest_asset_ids,
)
from tldw_Server_API.app.core.Persona.visual_starter_catalog import (
    PersonaVisualStarterCatalogError,
    PersonaVisualStarterCatalogService,
)
from tldw_Server_API.app.core.Persona.visual_starter_fixtures import (
    DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID,
    DEFAULT_PERSONA_VISUAL_STARTER_PACK_IDS,
    LEGACY_PERSONA_VISUAL_STARTER_PACK_ID,
    PersonaVisualStarterAsset,
    PersonaVisualStarterPack,
)


pytestmark = pytest.mark.unit


def _png_bytes(width: int = 2, height: int = 2) -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (width, height), (48, 96, 160, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _manifest_frame_asset_ids(manifest: dict[str, Any]) -> set[str]:
    asset_ids: set[str] = set()
    for animation in manifest.get("animations", {}).values():
        if not isinstance(animation, dict):
            continue
        for frame in animation.get("frames", []):
            if isinstance(frame, dict) and frame.get("asset_id"):
                asset_ids.add(str(frame["asset_id"]))
    return asset_ids


def _state_frame_asset_id(manifest: dict[str, Any], state: str) -> str:
    animation_id = manifest["states"][state]["animation_id"]
    frame = manifest["animations"][animation_id]["frames"][0]
    return str(frame["asset_id"])


@pytest.fixture()
def db_instance(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    db = CharactersRAGDB(
        tmp_path / "persona_visual_starter_catalog.sqlite",
        "persona-visual-starter-test",
    )
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


def test_starter_catalog_lists_bundled_scaffold_packs(
    db_instance: CharactersRAGDB,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    starters = service.list_starter_packs()

    assert [starter["id"] for starter in starters] == list(
        DEFAULT_PERSONA_VISUAL_STARTER_PACK_IDS
    )
    assert LEGACY_PERSONA_VISUAL_STARTER_PACK_ID not in {
        starter["id"] for starter in starters
    }
    starter = starters[0]
    assert starter["title"] == "Research Buddy Basic"
    assert starter["renderer_type"] == "sprite_frames"
    assert starter["manifest_version"] == 1
    assert starter["asset_count"] >= 1
    assert starter["total_bytes"] > 0
    assert {"idle", "listening", "thinking", "speaking", "error"}.issubset(
        set(starter["states_offered"])
    )
    assert all("catalog:scaffold" in starter["tags"] for starter in starters)
    assert all("scaffold" in starter["description"].lower() for starter in starters)
    assert any("tier:intricate" in starter["tags"] for starter in starters)
    assert any("tool.notes_search" in starter["states_offered"] for starter in starters)


def test_get_starter_pack_returns_isolated_manifest_preview(
    db_instance: CharactersRAGDB,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)
    first = service.get_starter_pack(DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID)
    first["manifest"]["states"]["idle"]["animation_id"] = "mutated"

    second = service.get_starter_pack(DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID)

    assert second["manifest"]["states"]["idle"]["animation_id"] == "idle-loop"


def test_get_starter_pack_accepts_legacy_research_buddy_alias(
    db_instance: CharactersRAGDB,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    detail = service.get_starter_pack(LEGACY_PERSONA_VISUAL_STARTER_PACK_ID)

    assert detail["id"] == DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID
    assert detail["title"] == "Research Buddy Basic"


@pytest.mark.parametrize(
    ("starter_pack_id", "custom_states"),
    (
        (
            "action-guide-intricate",
            ("reaction.anticipation", "reaction.success"),
        ),
        (
            "elaborate-persona-intricate",
            ("mood.focused", "tool.media_import"),
        ),
    ),
)
def test_multi_custom_state_scaffolds_use_distinct_variant_assets(
    db_instance: CharactersRAGDB,
    starter_pack_id: str,
    custom_states: tuple[str, ...],
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    detail = service.get_starter_pack(starter_pack_id)
    state_asset_ids = {
        _state_frame_asset_id(detail["manifest"], state)
        for state in custom_states
    }

    assert len(state_asset_ids) == len(custom_states)


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
    assert copied["title"] == "Research Buddy Basic"
    assert copied["provenance"] == "imported"
    assert copied["active_at"] is None
    assert (
        db_instance.get_active_persona_visual_pack(
            persona_id=persona_id,
            user_id=user_id,
        )["id"]
        == active_pack["id"]
    )

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
    assert "neutral" not in str(copied["manifest"])


@pytest.mark.parametrize("starter_pack_id", DEFAULT_PERSONA_VISUAL_STARTER_PACK_IDS)
def test_copy_every_default_scaffold_creates_inactive_user_owned_draft(
    db_instance: CharactersRAGDB,
    starter_pack_id: str,
) -> None:
    user_id = "user-1"
    persona_id = db_instance.create_persona_profile(
        {"user_id": user_id, "name": f"Target {starter_pack_id}"}
    )
    service = PersonaVisualStarterCatalogService(db_instance)
    starter_detail = service.get_starter_pack(starter_pack_id)
    fixture_asset_keys = {
        str(asset["asset_key"])
        for asset in starter_detail["assets"]
    }

    copied = service.copy_starter_pack_to_persona(
        starter_pack_id=starter_pack_id,
        persona_id=persona_id,
        user_id=user_id,
    )

    assert copied["status"] == "draft"
    assert copied["active_at"] is None
    assert copied["title"] == starter_detail["title"]
    assert {"idle", "listening", "thinking", "speaking", "error"}.issubset(
        set(copied["manifest"]["states"])
    )
    copied_assets = db_instance.list_persona_visual_assets(
        pack_id=copied["id"],
        persona_id=persona_id,
        user_id=user_id,
    )
    copied_asset_ids = {str(asset["id"]) for asset in copied_assets}
    assert collect_visual_manifest_asset_ids(copied["manifest"]) == copied_asset_ids
    frame_asset_ids = _manifest_frame_asset_ids(copied["manifest"])
    assert frame_asset_ids == copied_asset_ids
    assert not (fixture_asset_keys & frame_asset_ids)


def test_copy_legacy_research_buddy_alias_creates_default_draft(
    db_instance: CharactersRAGDB,
) -> None:
    user_id = "user-1"
    persona_id = db_instance.create_persona_profile(
        {"user_id": user_id, "name": "Target"}
    )
    service = PersonaVisualStarterCatalogService(db_instance)

    copied = service.copy_starter_pack_to_persona(
        starter_pack_id=LEGACY_PERSONA_VISUAL_STARTER_PACK_ID,
        persona_id=persona_id,
        user_id=user_id,
    )

    assert copied["status"] == "draft"
    assert copied["title"] == "Research Buddy Basic"


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


def test_list_starter_packs_rejects_invalid_fixture_renderer_type(
    db_instance: CharactersRAGDB,
) -> None:
    malformed = PersonaVisualStarterPack(
        id="bad-renderer",
        title="Bad renderer",
        description="Invalid test fixture",
        renderer_type="unknown_renderer",
        manifest={
            "manifest_version": 1,
            "renderer_type": "unknown_renderer",
            "states": {"idle": {"animation_id": "idle"}},
            "animations": {
                "idle": {
                    "frames": [{"asset_id": "starter_idle", "duration_ms": 100}],
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
        service.list_starter_packs()

    assert exc_info.value.code == "invalid_starter_fixture"


def test_list_starter_packs_rejects_duplicate_fixture_asset_keys(
    db_instance: CharactersRAGDB,
) -> None:
    malformed = PersonaVisualStarterPack(
        id="duplicate-asset-key",
        title="Duplicate asset key",
        description="Invalid test fixture",
        renderer_type="sprite_frames",
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {"idle": {"animation_id": "idle"}},
            "animations": {
                "idle": {
                    "frames": [{"asset_id": "starter_idle", "duration_ms": 100}],
                    "frame_rate": 1,
                }
            },
        },
        assets=(
            PersonaVisualStarterAsset(
                asset_key="starter_idle",
                filename="idle-one.png",
                mime_type="image/png",
                content=_png_bytes(),
                asset_role="frame",
            ),
            PersonaVisualStarterAsset(
                asset_key="starter_idle",
                filename="idle-two.png",
                mime_type="image/png",
                content=_png_bytes(),
                asset_role="frame",
            ),
        ),
    )
    service = PersonaVisualStarterCatalogService(db_instance, starter_packs=(malformed,))

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.list_starter_packs()

    assert exc_info.value.code == "invalid_starter_asset"
    assert exc_info.value.details["asset_key"] == "starter_idle"


def test_get_starter_pack_rejects_invalid_fixture_asset_role(
    db_instance: CharactersRAGDB,
) -> None:
    malformed = PersonaVisualStarterPack(
        id="bad-asset-role",
        title="Bad asset role",
        description="Invalid test fixture",
        renderer_type="sprite_frames",
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {"idle": {"animation_id": "idle"}},
            "animations": {
                "idle": {
                    "frames": [{"asset_id": "starter_idle", "duration_ms": 100}],
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
                asset_role="unsupported_role",
            ),
        ),
    )
    service = PersonaVisualStarterCatalogService(db_instance, starter_packs=(malformed,))

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.get_starter_pack("bad-asset-role")

    assert exc_info.value.code == "invalid_starter_fixture"


def test_copy_starter_pack_cleans_up_when_manifest_update_returns_none(
    db_instance: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = "user-1"
    persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    service = PersonaVisualStarterCatalogService(db_instance)
    monkeypatch.setattr(db_instance, "update_persona_visual_pack_manifest", lambda **_: None)

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.copy_starter_pack_to_persona(
            starter_pack_id=DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID,
            persona_id=persona_id,
            user_id=user_id,
        )

    assert exc_info.value.code == "starter_copy_failed"
    assert db_instance.list_persona_visual_packs(persona_id=persona_id, user_id=user_id) == []


def test_copy_starter_pack_cleans_up_when_status_update_returns_none(
    db_instance: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = "user-1"
    persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    service = PersonaVisualStarterCatalogService(db_instance)
    monkeypatch.setattr(db_instance, "update_persona_visual_pack_status", lambda **_: None)

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.copy_starter_pack_to_persona(
            starter_pack_id=DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID,
            persona_id=persona_id,
            user_id=user_id,
        )

    assert exc_info.value.code == "starter_copy_failed"
    assert db_instance.list_persona_visual_packs(persona_id=persona_id, user_id=user_id) == []
