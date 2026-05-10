from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tldw_Server_API.app.api.v1.endpoints.persona import (
    _persona_visual_override_payload_from_tool_result,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.persona_visuals_module import (
    PersonaVisualsModule,
)


pytestmark = pytest.mark.unit


@pytest.fixture()
def chacha_db(tmp_path: Path):
    db_path = tmp_path / "persona_visuals_mcp.sqlite"
    db = CharactersRAGDB(db_path, "persona-visuals-mcp-tests")
    yield db, db_path
    db.close_connection()


class FakeJobManager:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        self.created.append(kwargs)
        return {"id": "job-visual-1", "status": "queued", **kwargs}


def _context(db_path: Path, persona_id: str | None = None) -> SimpleNamespace:
    metadata: dict[str, Any] = {}
    if persona_id:
        metadata["persona_scope"] = {"persona_id": persona_id}
    return SimpleNamespace(
        user_id="1",
        session_id="session-visuals",
        db_paths={"chacha": str(db_path)},
        metadata=metadata,
        persona_scope=metadata.get("persona_scope"),
    )


def _create_persona(db: CharactersRAGDB, *, user_id: str = "1", name: str = "Visual MCP Persona") -> str:
    return db.create_persona_profile({"user_id": user_id, "name": name})


def test_visual_state_override_payload_helper_clamps_and_labels_tool() -> None:
    payload = _persona_visual_override_payload_from_tool_result(
        tool_name="persona_visuals.trigger_state",
        result={
            "ok": True,
            "output": {
                "type": "visual_state_override",
                "persona_id": "persona-1",
                "session_id": "session-1",
                "state": "speaking",
                "duration_ms": 999_999,
                "reason": "demo",
            },
        },
        persona_id="persona-1",
        session_id="session-1",
    )

    assert payload == {
        "type": "visual_state_override",
        "persona_id": "persona-1",
        "session_id": "session-1",
        "state": "speaking",
        "duration_ms": 30_000,
        "reason": "demo",
        "tool": "persona_visuals.trigger_state",
    }
    assert (
        _persona_visual_override_payload_from_tool_result(
            tool_name="notes.search",
            result={"ok": True, "output": {"type": "visual_state_override"}},
            persona_id="persona-1",
            session_id="session-1",
        )
        is None
    )


def test_persona_visuals_module_config_is_present_and_disabled() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    config_path = repo_root / "tldw_Server_API" / "Config_Files" / "mcp_modules.yaml"
    config = yaml.safe_load(config_path.read_text())
    modules = config.get("modules") if isinstance(config, dict) else []
    module = next(
        (item for item in modules if isinstance(item, dict) and item.get("id") == "persona_visuals"),
        None,
    )

    assert module is not None
    assert module["enabled"] is False
    assert (
        module["class"]
        == "tldw_Server_API.app.core.MCP_unified.modules.implementations.persona_visuals_module:PersonaVisualsModule"
    )


@pytest.mark.asyncio
async def test_capabilities_returns_active_and_draft_pack_summaries(chacha_db) -> None:
    db, db_path = chacha_db
    persona_id = _create_persona(db)
    active = db.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="1",
        title="Active Pack",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
        status="active",
    )
    draft = db.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="1",
        title="Draft Pack",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )
    db.create_persona_visual_asset(
        pack_id=active["id"],
        persona_id=persona_id,
        user_id="1",
        asset_role="frame",
        storage_key=f"persona_visuals/{persona_id}/{active['id']}/asset.png",
        original_filename="asset.png",
        mime_type="image/png",
        byte_size=12,
        checksum_sha256="a" * 64,
        width=64,
        height=64,
    )

    module = PersonaVisualsModule(ModuleConfig(name="persona_visuals"))
    result = await module.execute_tool(
        "persona_visuals.capabilities",
        {},
        context=_context(db_path, persona_id),
    )

    assert result["persona_id"] == persona_id
    assert result["active_pack"]["id"] == active["id"]
    assert result["active_pack"]["assets_count"] == 1
    assert [pack["id"] for pack in result["draft_packs"]] == [draft["id"]]
    assert "thinking" in result["states"]


@pytest.mark.asyncio
async def test_library_items_lists_reference_backed_personal_library_entries(chacha_db) -> None:
    db, db_path = chacha_db
    source_persona_id = _create_persona(db, name="Source Persona")
    source_pack = db.create_persona_visual_pack(
        persona_id=source_persona_id,
        user_id="1",
        title="Reusable Source Pack",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )
    item = db.upsert_persona_visual_library_item(
        user_id="1",
        source_persona_id=source_persona_id,
        source_pack_id=source_pack["id"],
        title="Reusable Library Item",
        notes="Use this across personas",
        tags=["Idle", "Reusable"],
    )
    module = PersonaVisualsModule(ModuleConfig(name="persona_visuals"))

    result = await module.execute_tool(
        "persona_visuals.library_items",
        {},
        context=_context(db_path),
    )

    assert result["count"] == 1
    assert result["items"] == [
        {
            "id": item["id"],
            "title": "Reusable Library Item",
            "notes": "Use this across personas",
            "tags": ["idle", "reusable"],
            "source_persona_id": source_persona_id,
            "source_pack_id": source_pack["id"],
            "source_persona_name": "Source Persona",
            "source_pack_title": "Reusable Source Pack",
            "source_pack_version": source_pack["version"],
            "source_current_version": source_pack["version"],
            "source_available": True,
            "source_changed": False,
            "version": item["version"],
        }
    ]


@pytest.mark.asyncio
async def test_library_items_rejects_out_of_range_offset(chacha_db) -> None:
    _db, db_path = chacha_db
    module = PersonaVisualsModule(ModuleConfig(name="persona_visuals"))

    with pytest.raises(ValueError, match="offset must be between 0 and 1000000"):
        await module.execute_tool(
            "persona_visuals.library_items",
            {"offset": 1_000_001},
            context=_context(db_path),
        )


@pytest.mark.asyncio
async def test_trigger_state_requires_context_rejects_unknown_states_and_clamps_duration(chacha_db) -> None:
    db, db_path = chacha_db
    persona_id = _create_persona(db)
    module = PersonaVisualsModule(ModuleConfig(name="persona_visuals"))

    payload = await module.execute_tool(
        "persona_visuals.trigger_state",
        {"state": "speaking", "duration_ms": 99_999, "reason": "mcp_runtime"},
        context=_context(db_path, persona_id),
    )
    assert payload["type"] == "visual_state_override"
    assert payload["persona_id"] == persona_id
    assert payload["session_id"] == "session-visuals"
    assert payload["duration_ms"] == 30_000

    minimum = await module.execute_tool(
        "persona_visuals.trigger_state",
        {"persona_id": persona_id, "state": "thinking", "duration_ms": 1},
        context=_context(db_path),
    )
    assert minimum["duration_ms"] == 100

    with pytest.raises(ValueError, match="Unknown visual state"):
        await module.execute_tool(
            "persona_visuals.trigger_state",
            {"state": "dancing"},
            context=_context(db_path, persona_id),
        )

    with pytest.raises(ValueError, match="Missing persona context"):
        await module.execute_tool(
            "persona_visuals.trigger_state",
            {"state": "speaking"},
            context=_context(db_path),
        )

    with pytest.raises(ValueError, match="Missing user context"):
        await module.execute_tool(
            "persona_visuals.trigger_state",
            {"persona_id": persona_id, "state": "speaking"},
            context=SimpleNamespace(
                session_id="session-visuals",
                db_paths={"chacha": str(db_path)},
                metadata={},
            ),
        )


@pytest.mark.asyncio
async def test_draft_tools_mutate_drafts_without_activating(chacha_db) -> None:
    db, db_path = chacha_db
    persona_id = _create_persona(db)
    module = PersonaVisualsModule(ModuleConfig(name="persona_visuals"))

    created = await module.execute_tool(
        "persona_visuals.create_draft_pack",
        {
            "title": "MCP Draft",
            "manifest": {"manifest_version": 1, "renderer_type": "sprite_frames"},
        },
        context=_context(db_path, persona_id),
    )

    assert created["pack"]["status"] == "draft"
    assert db.get_active_persona_visual_pack(persona_id=persona_id, user_id="1") is None

    updated = await module.execute_tool(
        "persona_visuals.update_manifest",
        {
            "pack_id": created["pack"]["id"],
            "manifest": {
                "manifest_version": 1,
                "renderer_type": "sprite_frames",
                "states": {},
                "animations": {},
            },
            "expected_version": created["pack"]["version"],
        },
        context=_context(db_path, persona_id),
    )

    assert updated["pack"]["status"] == "draft"
    assert updated["pack"]["version"] == created["pack"]["version"] + 1
    assert db.get_active_persona_visual_pack(persona_id=persona_id, user_id="1") is None


@pytest.mark.asyncio
async def test_use_library_item_creates_target_draft_without_activation(chacha_db) -> None:
    db, db_path = chacha_db
    source_persona_id = _create_persona(db, name="Source Persona")
    target_persona_id = _create_persona(db, name="Target Persona")
    source_pack = db.create_persona_visual_pack(
        persona_id=source_persona_id,
        user_id="1",
        title="Reusable Source Pack",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )
    item = db.upsert_persona_visual_library_item(
        user_id="1",
        source_persona_id=source_persona_id,
        source_pack_id=source_pack["id"],
        title="Reusable Library Item",
    )
    module = PersonaVisualsModule(ModuleConfig(name="persona_visuals"))

    result = await module.execute_tool(
        "persona_visuals.use_library_item",
        {"item_id": item["id"], "target_persona_id": target_persona_id, "title": "Target Draft"},
        context=_context(db_path),
    )

    assert result["library_item_id"] == item["id"]
    assert result["review_required"] is True
    assert result["pack"]["persona_id"] == target_persona_id
    assert result["pack"]["title"] == "Target Draft"
    assert result["pack"]["status"] == "draft"
    assert result["pack"]["parent_pack_id"] == source_pack["id"]
    assert db.get_active_persona_visual_pack(persona_id=target_persona_id, user_id="1") is None


@pytest.mark.asyncio
async def test_use_library_item_rejects_missing_target_and_unavailable_sources(chacha_db) -> None:
    db, db_path = chacha_db
    source_persona_id = _create_persona(db, name="Source Persona")
    target_persona_id = _create_persona(db, name="Target Persona")
    source_pack = db.create_persona_visual_pack(
        persona_id=source_persona_id,
        user_id="1",
        title="Reusable Source Pack",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )
    item = db.upsert_persona_visual_library_item(
        user_id="1",
        source_persona_id=source_persona_id,
        source_pack_id=source_pack["id"],
        title="Reusable Library Item",
    )
    module = PersonaVisualsModule(ModuleConfig(name="persona_visuals"))

    with pytest.raises(ValueError, match="Missing target persona context"):
        await module.execute_tool(
            "persona_visuals.use_library_item",
            {"item_id": item["id"]},
            context=_context(db_path),
        )

    with pytest.raises(ValueError, match="target_persona_id cannot be empty"):
        await module.execute_tool(
            "persona_visuals.use_library_item",
            {"item_id": item["id"], "target_persona_id": "   "},
            context=_context(db_path, target_persona_id),
        )

    db.soft_delete_persona_visual_pack_with_assets(
        pack_id=source_pack["id"],
        persona_id=source_persona_id,
        user_id="1",
    )

    with pytest.raises(ValueError, match="Source Persona Visual pack is no longer available"):
        await module.execute_tool(
            "persona_visuals.use_library_item",
            {"item_id": item["id"], "target_persona_id": target_persona_id},
            context=_context(db_path),
        )

    with pytest.raises(ValueError, match="Persona Visual library item not found"):
        await module.execute_tool(
            "persona_visuals.use_library_item",
            {"item_id": "missing-item", "target_persona_id": target_persona_id},
            context=_context(db_path),
        )


@pytest.mark.asyncio
async def test_enqueue_generation_creates_persona_visual_job(chacha_db) -> None:
    db, db_path = chacha_db
    persona_id = _create_persona(db)
    pack = db.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="1",
        title="Generation Draft",
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
    )
    manager = FakeJobManager()
    module = PersonaVisualsModule(
        ModuleConfig(name="persona_visuals", settings={"jobs_manager": manager})
    )

    result = await module.execute_tool(
        "persona_visuals.enqueue_generation",
        {
            "pack_id": pack["id"],
            "prompt": "Create a thoughtful thinking pose",
            "target_state": "thinking",
            "backend": "fake",
        },
        context=_context(db_path, persona_id),
    )

    assert result["job_id"] == "job-visual-1"
    assert result["status"] == "queued"
    assert result["review_required"] is True
    assert manager.created[0]["domain"] == "persona_visuals"
    assert manager.created[0]["payload"]["target_state"] == "thinking"
