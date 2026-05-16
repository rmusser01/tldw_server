from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.vn_scripts import (
    _resolve_accessible_audio_refs,
    router as vn_scripts_router,
)
from tldw_Server_API.app.api.v1.endpoints.vn_capabilities import router as vn_capabilities_router
from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetPackCreate,
    VNAssetReviewRequest,
    VNAssetSlotCreate,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.DB_Management.VNPolicy_DB import (
    LOCAL_DEFAULT_POLICY_DEFINITION,
    STORY_DEFAULT_GENERATION_DEFINITION,
    VNProfileSnapshotRepository,
    VNPolicyProfileStore,
)
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Scripts.templates import list_template_catalog

pytestmark = pytest.mark.integration


@pytest.fixture
def chacha_dbs(tmp_path) -> Iterator[dict[int, CharactersRAGDB]]:
    databases = {
        42: CharactersRAGDB(str(tmp_path / "user-42" / "ChaChaNotes.db"), client_id="vn-scripts-user-42"),
        7: CharactersRAGDB(str(tmp_path / "user-7" / "ChaChaNotes.db"), client_id="vn-scripts-user-7"),
    }
    yield databases
    for database in databases.values():
        database.close_connection()


@pytest.fixture
def current_user() -> dict[str, Any]:
    return {
        "id": 42,
        "username": "user-42",
        "role": "user",
        "roles": ["user"],
        "permissions": [],
        "is_admin": False,
    }


@pytest.fixture
def authnz_pool(tmp_path) -> Iterator[DatabasePool]:
    pool = DatabasePool(Settings(AUTH_MODE="single_user", DATABASE_URL=f"sqlite:///{tmp_path / 'authnz.db'}"))
    yield pool
    asyncio.run(pool.close())


@pytest.fixture
def client(
    chacha_dbs: dict[int, CharactersRAGDB],
    current_user: dict[str, Any],
    authnz_pool: DatabasePool,
) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(vn_capabilities_router, prefix="/api/v1/vn")
    app.include_router(vn_scripts_router, prefix="/api/v1/vn")

    async def override_user() -> User:
        return User(**current_user)

    async def override_chacha_db() -> CharactersRAGDB:
        return chacha_dbs[int(current_user["id"])]

    async def override_db_pool() -> DatabasePool:
        return authnz_pool

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db
    app.dependency_overrides[get_db_pool] = override_db_pool

    with TestClient(app) as test_client:
        yield test_client


def _create_asset_pack(chacha_db: CharactersRAGDB, *, owner_user_id: int = 42) -> tuple[int, str]:
    character_id = chacha_db.add_character_card(
        {
            "name": "Mira",
            "description": "A careful archivist.",
            "personality": "Patient and exacting.",
            "scenario": "Cataloging an orbital library.",
        }
    )
    service = VNAssetPackService(chacha_db, owner_user_id=owner_user_id)
    pack = service.create_pack(VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id))
    slot = service.create_slot(
        pack.id,
        VNAssetSlotCreate(
            asset_type="background",
            slot_key="background.archive.default",
            variant_count=1,
        ),
    )
    repo = VNAssetPacksRepository.initialized(chacha_db)
    item = repo.create_item(
        pack_id=pack.id,
        slot_id=slot.id,
        variant_index=0,
        generated_file_id=1001,
        mime_type="image/png",
    )
    service.review_item(item["id"], VNAssetReviewRequest(review_status="approved"))
    return pack.id, slot.slot_key


def _program(asset_pack_id: int, *, slot_key: str | None = None) -> dict:
    body = [{"op": "narrate", "text": "The archive door hums."}, {"op": "end"}]
    if slot_key is not None:
        body.insert(0, {"op": "set_background", "slot_key": slot_key})
    return {
        "schema_version": "vn_script_program.v1",
        "title": "Archive Door",
        "primary_asset_pack_id": asset_pack_id,
        "entry_label": "start",
        "variables": {},
        "generation_defaults": {"profile_id": "story_default", "persist_model_outputs": True},
        "labels": {"start": body},
    }


def _create_script(client: TestClient, *, asset_pack_id: int) -> int:
    response = client.post(
        "/api/v1/vn/vn-scripts/scripts",
        json={
            "title": "Archive Door",
            "description": "A short route.",
            "primary_asset_pack_id": asset_pack_id,
            "policy_profile_id": "local_default",
            "generation_profile_id": "story_default",
            "content_rating": "general",
        },
    )
    assert response.status_code == 201
    return int(response.json()["id"])


def _contains_key(value: Any, key: str) -> bool:
    """Return true when a nested dict or list contains the requested dictionary key."""
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def test_template_catalog_lists_preview_safe_starter_templates(client: TestClient) -> None:
    response = client.get("/api/v1/vn/vn-scripts/templates")

    assert response.status_code == 200
    payload = response.json()
    template_ids = {item["id"] for item in payload["items"]}
    assert template_ids == {
        "linear_scene",
        "authored_choices",
        "generated_choice_set",
        "scene_update",
        "confirm_gated_generation",
    }
    for item in payload["items"]:
        assert set(item) == {
            "id",
            "label",
            "description",
            "category",
            "recommended_content_rating",
            "required_capabilities",
            "preview",
            "default_title",
            "default_description",
        }
        assert "draft" not in item
        assert "raw_prompt" not in item
        assert "policy_profile_id" not in item
        assert "generation_profile_id" not in item
        assert "generation_profiles" not in item


def test_template_catalog_returns_isolated_preview_payloads() -> None:
    first_catalog = list_template_catalog()
    first_catalog[0]["preview"]["flow"].append("mutated")

    second_catalog = list_template_catalog()

    assert "mutated" not in second_catalog[0]["preview"]["flow"]


def test_authoring_catalog_returns_preview_safe_metadata(client: TestClient) -> None:
    response = client.get("/api/v1/vn/vn-scripts/vn-authoring-catalog")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "vn_script_authoring_catalog.v1"
    assert "script_authoring_catalog" in payload["capability_tokens"]
    assert {operation["op"] for operation in payload["operations"]} >= {"narrate", "generate", "choice"}
    assert {snippet["id"] for snippet in payload["snippets"]} >= {"narration", "generated_choice_set"}
    assert _contains_key(payload, "api_key") is False
    assert _contains_key(payload, "provider") is False
    assert _contains_key(payload, "model") is False
    assert _contains_key(payload, "raw_prompt") is False


def test_snippet_preview_supports_stored_and_supplied_draft(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": _program(asset_pack_id)},
    )
    stored_response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-preview",
        json={
            "snippet_id": "narration",
            "anchor": {"label": "start", "op_index": 0, "mode": "after"},
            "parameters": {"text": "Stored draft line."},
        },
    )
    supplied_draft = _program(asset_pack_id)
    supplied_draft["labels"]["start"][0]["text"] = "Supplied opening."
    supplied_response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-preview",
        json={
            "snippet_id": "narration",
            "anchor": {"label": "start", "op_index": 0, "mode": "after"},
            "parameters": {"text": "Supplied draft line."},
            "draft": supplied_draft,
            "draft_revision": 1,
        },
    )

    assert stored_response.status_code == 200
    stored_payload = stored_response.json()
    assert stored_payload["base_revision"] == 1
    assert stored_payload["draft"]["labels"]["start"][1] == {"op": "narrate", "text": "Stored draft line."}
    assert stored_payload["diagnostics"]["valid"] is True
    assert stored_payload["patch_summary"]["inserted_ops"] == 1
    assert supplied_response.status_code == 200
    supplied_payload = supplied_response.json()
    assert supplied_payload["base_revision"] == 1
    assert supplied_payload["draft"]["labels"]["start"][0]["text"] == "Supplied opening."
    assert supplied_payload["draft"]["labels"]["start"][1] == {"op": "narrate", "text": "Supplied draft line."}


def test_snippet_preview_is_non_mutating(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": _program(asset_pack_id)},
    )
    before = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft").json()

    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-preview",
        json={
            "snippet_id": "narration",
            "anchor": {"label": "start", "mode": "append"},
            "parameters": {"text": "Preview only."},
        },
    )
    after = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft").json()

    assert response.status_code == 200
    assert after["revision"] == before["revision"]
    assert after["draft"] == before["draft"]
    assert after["diagnostics"] == before["diagnostics"]


def test_snippet_apply_requires_revision_persists_patch_and_returns_diagnostics(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": _program(asset_pack_id)},
    )

    missing_revision_response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-apply",
        json={
            "snippet_id": "narration",
            "anchor": {"label": "start", "mode": "append"},
            "parameters": {"text": "Missing revision."},
        },
    )
    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-apply",
        json={
            "if_revision": 1,
            "snippet_id": "narration",
            "anchor": {"label": "start", "op_index": 0, "mode": "after"},
            "parameters": {"text": "Applied line."},
        },
    )
    stored = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft").json()

    assert missing_revision_response.status_code == 422
    assert response.status_code == 200
    payload = response.json()
    assert payload["revision"] == 2
    assert payload["draft"]["labels"]["start"][1] == {"op": "narrate", "text": "Applied line."}
    assert payload["diagnostics"]["valid"] is True
    assert payload["patch_summary"]["inserted_ops"] == 1
    assert stored["revision"] == 2
    assert stored["draft"] == payload["draft"]
    assert stored["diagnostics"] == payload["diagnostics"]


def test_snippet_preview_unknown_snippet_returns_not_found(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": _program(asset_pack_id)},
    )

    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-preview",
        json={
            "snippet_id": "missing_snippet",
            "anchor": {"label": "start", "mode": "append"},
            "parameters": {},
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"]["details"]["reason"] == "snippet_not_found"


def test_snippet_preview_parameter_errors_return_field_path(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": _program(asset_pack_id)},
    )
    too_deep: dict[str, Any] = {}
    cursor = too_deep
    for depth in range(10):
        cursor[f"level_{depth}"] = {}
        cursor = cursor[f"level_{depth}"]
    cases = [
        {"text": "Line.", "unexpected": True},
        {"text": "x" * 9000},
        too_deep,
    ]

    for parameters in cases:
        response = client.post(
            f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-preview",
            json={
                "snippet_id": "narration",
                "anchor": {"label": "start", "mode": "append"},
                "parameters": parameters,
            },
        )
        assert response.status_code == 400
        details = response.json()["detail"]["details"]
        assert details["reason"] == "snippet_parameter_invalid"
        assert details["field_path"].startswith("$.parameters")


def test_snippet_preview_anchor_errors_return_anchor_details(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": _program(asset_pack_id)},
    )
    cases = [
        ({"label": "missing", "mode": "append"}, "snippet_anchor_not_found"),
    ]

    for anchor, expected_reason in cases:
        response = client.post(
            f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-preview",
            json={
                "snippet_id": "narration",
                "anchor": anchor,
                "parameters": {"text": "Line."},
            },
        )
        assert response.status_code == 400
        details = response.json()["detail"]["details"]
        assert details["reason"] == expected_reason
        assert details["anchor"] == anchor


def test_snippet_preview_schema_rejects_invalid_anchor_shape(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": _program(asset_pack_id)},
    )

    invalid_mode = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-preview",
        json={
            "snippet_id": "narration",
            "anchor": {"label": "start", "mode": "beside"},
            "parameters": {"text": "Line."},
        },
    )
    missing_index = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-preview",
        json={
            "snippet_id": "narration",
            "anchor": {"label": "start", "mode": "before"},
            "parameters": {"text": "Line."},
        },
    )

    assert invalid_mode.status_code == 422
    assert missing_index.status_code == 422


def test_snippet_preview_schema_requires_revision_for_supplied_draft(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)

    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-preview",
        json={
            "snippet_id": "narration",
            "anchor": {"label": "start", "mode": "append"},
            "parameters": {"text": "Line."},
            "draft": _program(asset_pack_id),
        },
    )

    assert response.status_code == 422


def test_snippet_apply_stale_revision_returns_current_revision(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": _program(asset_pack_id)},
    )

    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-apply",
        json={
            "if_revision": 0,
            "snippet_id": "narration",
            "anchor": {"label": "start", "mode": "append"},
            "parameters": {"text": "Stale line."},
        },
    )

    assert response.status_code == 409
    details = response.json()["detail"]["details"]
    assert details["reason"] == "draft_revision_conflict"
    assert details["current_revision"] == 1


def test_snippet_apply_stale_revision_conflicts_before_anchor_validation(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    revision_one_draft = _program(asset_pack_id)
    revision_two_draft = {
        **_program(asset_pack_id),
        "entry_label": "renamed",
        "labels": {"renamed": [{"op": "narrate", "text": "Renamed opening."}, {"op": "end"}]},
    }
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": revision_one_draft},
    )
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 1, "draft": revision_two_draft},
    )

    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/snippet-apply",
        json={
            "if_revision": 1,
            "snippet_id": "narration",
            "anchor": {"label": "start", "mode": "append"},
            "parameters": {"text": "Line for stale draft."},
        },
    )

    assert response.status_code == 409
    details = response.json()["detail"]["details"]
    assert details["reason"] == "draft_revision_conflict"
    assert details["current_revision"] == 2


def test_draft_graph_endpoint_returns_stored_graph(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": _program(asset_pack_id)},
    )

    response = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "vn_script_authoring_graph.v1"
    assert payload["graph_semantics_version"] == "vn_script_authoring_graph_edges.v1"
    assert payload["source"] == "stored_draft"
    assert payload["script_id"] == script_id
    assert payload["base_revision"] == 1
    assert payload["version_id"] is None
    assert payload["outline"]["entry_label"] == "start"
    assert payload["outline"]["labels"]
    assert payload["graph"]["nodes"][0]["id"] == "label:start"
    assert payload["validation_context_source"] == "current_draft_context"


def test_graph_preview_accepts_supplied_draft_without_persisting(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    stored = _program(asset_pack_id)
    supplied = _program(asset_pack_id)
    supplied["labels"]["start"][0]["text"] = "Unsaved opening."
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": stored},
    )

    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph-preview",
        json={"draft": supplied, "draft_revision": 1},
    )
    draft_after = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft").json()

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "supplied_draft"
    assert payload["base_revision"] == 1
    assert payload["outline"]["labels"][0]["summary"] == "2 operations."
    assert draft_after["revision"] == 1
    assert draft_after["draft"] == stored


def test_graph_endpoints_use_resolved_authnz_profile_context(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
    authnz_pool: DatabasePool,
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])

    async def seed_profiles() -> None:
        store = VNPolicyProfileStore(authnz_pool)
        await store.initialize()
        await store.create_policy_profile(
            profile_id="custom_local",
            display_name="Custom Local",
            description=None,
            definition=LOCAL_DEFAULT_POLICY_DEFINITION,
            created_by_user_id=42,
        )
        await store.create_generation_profile(
            profile_id="custom_story",
            display_name="Custom Story",
            description=None,
            definition=STORY_DEFAULT_GENERATION_DEFINITION,
            created_by_user_id=42,
        )

    asyncio.run(seed_profiles())
    create_response = client.post(
        "/api/v1/vn/vn-scripts/scripts",
        json={
            "title": "Archive Door",
            "primary_asset_pack_id": asset_pack_id,
            "policy_profile_id": "custom_local",
            "generation_profile_id": "custom_story",
            "content_rating": "general",
        },
    )
    script_id = int(create_response.json()["id"])
    draft = _program(asset_pack_id)
    draft["generation_defaults"]["profile_id"] = "custom_story"
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": draft},
    )

    stored_graph_response = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph")
    preview_graph_response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph-preview",
        json={"draft": draft, "draft_revision": 1},
    )

    assert create_response.status_code == 201
    assert stored_graph_response.status_code == 200
    assert stored_graph_response.json()["validation_diagnostics"]["valid"] is True
    assert preview_graph_response.status_code == 200
    assert preview_graph_response.json()["validation_diagnostics"]["valid"] is True


def test_version_graph_endpoint_returns_published_version_graph(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, slot_key = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": _program(asset_pack_id, slot_key=slot_key)},
    )
    publish_response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/publish",
        json={
            "draft_revision": 1,
            "label": "v1",
            "idempotency_key": "publish-graph-v1",
            "acknowledgements": ["character_safety_missing"],
        },
    )
    version_id = publish_response.json()["version_id"]

    response = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/versions/{version_id}/graph")

    assert publish_response.status_code == 201
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "published_version"
    assert payload["script_id"] == script_id
    assert payload["version_id"] == version_id
    assert payload["base_revision"] is None
    assert payload["validation_context_source"] == "published_version_snapshot"
    assert payload["validation_diagnostics"]["valid"] is True


def test_draft_playtest_endpoint_returns_runtime_readiness(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    draft = _program(asset_pack_id)
    draft["labels"] = {
        "start": [
            {
                "op": "choice",
                "id": "door",
                "choices": [{"id": "open", "text": "Open", "target": "open"}],
            }
        ],
        "open": [{"op": "end"}],
    }
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": draft},
    )

    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/playtest",
        json={"max_steps": 50, "max_paths": 10},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "vn_script_playtest.v1"
    assert payload["source"] == "stored_draft"
    assert payload["runtime_ready"] is True
    assert payload["summary"]["choice_boundary_count"] == 1
    assert payload["summary"]["ending_count"] == 1


def test_draft_playtest_accepts_supplied_draft_without_persisting(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    stored = _program(asset_pack_id)
    supplied = _program(asset_pack_id)
    supplied["labels"]["start"] = [
        {
            "op": "generate",
            "id": "intro",
            "prompt": "Write an intro.",
            "output_schema": "narrative_dialogue",
        }
    ]
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": stored},
    )

    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/playtest",
        json={"draft": supplied, "draft_revision": 1},
    )
    draft_after = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft").json()

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "supplied_draft"
    assert payload["summary"]["generation_boundary_count"] == 1
    assert draft_after["draft"] == stored


def test_version_playtest_endpoint_uses_published_snapshot(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, slot_key = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": _program(asset_pack_id, slot_key=slot_key)},
    )
    publish_response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/publish",
        json={
            "draft_revision": 1,
            "label": "v1",
            "idempotency_key": "publish-playtest-v1",
            "acknowledgements": ["character_safety_missing"],
        },
    )
    version_id = publish_response.json()["version_id"]

    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/versions/{version_id}/playtest",
        json={},
    )

    assert publish_response.status_code == 201
    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "published_version"
    assert payload["version_id"] == version_id
    assert payload["validation_context_source"] == "published_version_snapshot"
    assert payload["runtime_ready"] is True


def test_graph_preview_malformed_supplied_draft_shape_returns_vn_error(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)

    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph-preview",
        json={"draft": ["not", "a", "mapping"]},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["details"]["reason"] == "supplied_draft_invalid_shape"


def test_graph_preview_oversized_supplied_draft_returns_vn_error(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    oversized = {"schema_version": "vn_script_program.v1", "blob": "x" * 1_048_576}

    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph-preview",
        json={"draft": oversized},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["details"]["reason"] == "supplied_draft_too_large"


def test_graph_problems_return_success_with_diagnostics(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    supplied = _program(asset_pack_id)
    supplied["labels"]["start"][1] = {"op": "jump", "target": "missing"}

    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph-preview",
        json={"draft": supplied},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["diagnostics"]["errors"][0]["code"] == "graph_target_missing"
    assert payload["graph"]["edges"][0]["missing_target"] is True
    assert payload["validation_diagnostics"]["valid"] is False


def test_graph_response_does_not_leak_full_op_payloads_or_provider_secrets(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    supplied = _program(asset_pack_id)
    supplied["labels"]["start"] = [
        {
            "op": "generate",
            "prompt": "secret prompt text",
            "raw_prompt": "raw secret prompt",
            "provider": "secret-provider",
            "model": "secret-model",
            "api_key": "sk-secret",
            "provider_config": {"base_url": "https://secret.example"},
            "output_schema": "choice_set",
            "on_generated_choice": "after_generation",
        }
    ]
    supplied["labels"]["after_generation"] = [{"op": "end"}]

    response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/graph-preview",
        json={"draft": supplied},
    )

    assert response.status_code == 200
    payload = response.json()
    operation_nodes = [node for node in payload["graph"]["nodes"] if node["type"] == "operation"]
    assert operation_nodes
    assert all("prompt" not in node for node in operation_nodes)
    assert all("provider_config" not in node for node in operation_nodes)
    serialized = response.text
    for leaked_value in (
        "secret prompt text",
        "raw secret prompt",
        "secret-provider",
        "secret-model",
        "sk-secret",
        "https://secret.example",
    ):
        assert leaked_value not in serialized


def test_vn_capabilities_include_script_authoring_catalog_when_scripts_routes_registered(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/vn/vn-capabilities")

    assert response.status_code == 200
    assert response.json()["features"]["script_authoring_catalog"] is True


def test_create_script_from_template_stores_valid_draft(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])

    response = client.post(
        "/api/v1/vn/vn-scripts/templates/linear_scene/scripts",
        json={
            "title": "Template Route",
            "primary_asset_pack_id": asset_pack_id,
            "content_rating": "general",
        },
    )

    assert response.status_code == 201
    script = response.json()["script"]
    draft_response = response.json()["draft"]
    script_id = script["id"]
    stored_draft = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft").json()
    assert script["title"] == "Template Route"
    assert draft_response["revision"] == 1
    assert draft_response["diagnostics"]["valid"] is True
    assert stored_draft["revision"] == 1
    assert stored_draft["diagnostics"]["valid"] is True
    assert stored_draft["draft"]["primary_asset_pack_id"] == asset_pack_id
    assert _contains_key(stored_draft["draft"], "provider") is False
    assert _contains_key(stored_draft["draft"], "model") is False
    assert _contains_key(stored_draft["draft"], "policy_profile_id") is False


def test_create_script_from_unknown_template_returns_not_found(client: TestClient) -> None:
    response = client.post(
        "/api/v1/vn/vn-scripts/templates/missing-template/scripts",
        json={
            "title": "Template Route",
            "primary_asset_pack_id": 1,
            "content_rating": "general",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"]["details"]["reason"] == "template_not_found"


def test_create_script_from_template_rejects_unknown_profiles(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])

    response = client.post(
        "/api/v1/vn/vn-scripts/templates/linear_scene/scripts",
        json={
            "title": "Template Route",
            "primary_asset_pack_id": asset_pack_id,
            "policy_profile_id": "missing_policy",
            "generation_profile_id": "story_default",
            "content_rating": "general",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["details"]["reason"] == "policy_profile_not_found"


def test_create_script_from_template_does_not_persist_script_when_validation_fails(client: TestClient) -> None:
    response = client.post(
        "/api/v1/vn/vn-scripts/templates/linear_scene/scripts",
        json={
            "title": "Invalid Template Route",
            "primary_asset_pack_id": 999_999,
            "content_rating": "general",
        },
    )
    scripts_response = client.get("/api/v1/vn/vn-scripts/scripts")

    assert response.status_code == 400
    assert response.json()["detail"]["details"]["reason"] == "pack_not_found"
    assert scripts_response.status_code == 200
    assert scripts_response.json()["total"] == 0
    assert scripts_response.json()["items"] == []


def test_generated_choice_template_validates_and_publishes(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    create_response = client.post(
        "/api/v1/vn/vn-scripts/templates/generated_choice_set/scripts",
        json={
            "title": "Generated Choice Route",
            "primary_asset_pack_id": asset_pack_id,
            "content_rating": "general",
        },
    )
    script_id = create_response.json()["script"]["id"]

    diagnostics_response = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/diagnostics")
    publish_response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/publish",
        json={
            "draft_revision": 1,
            "label": "template-v1",
            "idempotency_key": "generated-choice-template-v1",
            "acknowledgements": ["character_safety_missing"],
        },
    )

    assert create_response.status_code == 201
    assert diagnostics_response.status_code == 200
    assert diagnostics_response.json()["diagnostics"]["valid"] is True
    assert publish_response.status_code == 201
    assert publish_response.json()["validation"]["valid"] is True


def test_create_rejects_unknown_profiles(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])

    response = client.post(
        "/api/v1/vn/vn-scripts/scripts",
        json={
            "title": "Archive Door",
            "primary_asset_pack_id": asset_pack_id,
            "policy_profile_id": "missing_policy",
            "generation_profile_id": "story_default",
            "content_rating": "general",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["details"]["reason"] == "policy_profile_not_found"


def test_create_accepts_admin_created_profile_rows(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
    authnz_pool: DatabasePool,
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])

    async def seed_profiles() -> None:
        store = VNPolicyProfileStore(authnz_pool)
        await store.initialize()
        await store.create_policy_profile(
            profile_id="custom_local",
            display_name="Custom Local",
            description=None,
            definition=LOCAL_DEFAULT_POLICY_DEFINITION,
            created_by_user_id=42,
        )
        await store.create_generation_profile(
            profile_id="custom_story",
            display_name="Custom Story",
            description=None,
            definition=STORY_DEFAULT_GENERATION_DEFINITION,
            created_by_user_id=42,
        )

    asyncio.run(seed_profiles())

    response = client.post(
        "/api/v1/vn/vn-scripts/scripts",
        json={
            "title": "Archive Door",
            "primary_asset_pack_id": asset_pack_id,
            "policy_profile_id": "custom_local",
            "generation_profile_id": "custom_story",
            "content_rating": "general",
        },
    )

    assert response.status_code == 201
    assert response.json()["policy_profile_id"] == "custom_local"
    assert response.json()["generation_profile_id"] == "custom_story"


def test_script_api_round_trips_authored_generation_profile_map(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
    authnz_pool: DatabasePool,
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])

    async def seed_profiles() -> None:
        store = VNPolicyProfileStore(authnz_pool)
        await store.initialize()
        await store.create_generation_profile(
            profile_id="choice_profile",
            display_name="Choice Writer",
            description=None,
            definition=STORY_DEFAULT_GENERATION_DEFINITION | {"supported_output_schemas": ["choice_set"]},
            created_by_user_id=42,
        )
        await store.create_generation_profile(
            profile_id="scene_profile",
            display_name="Scene Director",
            description=None,
            definition=STORY_DEFAULT_GENERATION_DEFINITION | {"supported_output_schemas": ["scene_update"]},
            created_by_user_id=42,
        )

    asyncio.run(seed_profiles())

    create_response = client.post(
        "/api/v1/vn/vn-scripts/scripts",
        json={
            "title": "Archive Door",
            "description": "A short route.",
            "primary_asset_pack_id": asset_pack_id,
            "policy_profile_id": "local_default",
            "generation_profile_id": "story_default",
            "generation_profiles": {"choice_writer": "choice_profile"},
            "content_rating": "general",
        },
    )
    script_id = int(create_response.json()["id"])
    patch_response = client.patch(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}",
        json={"generation_profiles": {"choice_writer": "choice_profile", "scene_director": "scene_profile"}},
    )
    detail_response = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}")
    list_response = client.get("/api/v1/vn/vn-scripts/scripts?limit=20&offset=0")

    assert create_response.status_code == 201
    assert create_response.json()["generation_profiles"] == {
        "default": "story_default",
        "choice_writer": "choice_profile",
    }
    assert patch_response.status_code == 200
    assert patch_response.json()["generation_profiles"] == {
        "default": "story_default",
        "choice_writer": "choice_profile",
        "scene_director": "scene_profile",
    }
    assert detail_response.json()["generation_profiles"] == patch_response.json()["generation_profiles"]
    assert list_response.json()["items"][0]["generation_profiles"] == patch_response.json()["generation_profiles"]


def test_script_api_rejects_invalid_generation_profile_map_key(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
    authnz_pool: DatabasePool,
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])

    async def seed_profiles() -> None:
        store = VNPolicyProfileStore(authnz_pool)
        await store.initialize()
        await store.create_generation_profile(
            profile_id="choice_profile",
            display_name="Choice Writer",
            description=None,
            definition=STORY_DEFAULT_GENERATION_DEFINITION | {"supported_output_schemas": ["choice_set"]},
            created_by_user_id=42,
        )

    asyncio.run(seed_profiles())

    response = client.post(
        "/api/v1/vn/vn-scripts/scripts",
        json={
            "title": "Archive Door",
            "primary_asset_pack_id": asset_pack_id,
            "policy_profile_id": "local_default",
            "generation_profile_id": "story_default",
            "generation_profiles": {"Bad Key!": "choice_profile"},
            "content_rating": "general",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["details"]["reason"] == "generation_profile_key_invalid"


def test_script_api_rejects_default_generation_profile_map_key(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])

    response = client.post(
        "/api/v1/vn/vn-scripts/scripts",
        json={
            "title": "Archive Door",
            "primary_asset_pack_id": asset_pack_id,
            "policy_profile_id": "local_default",
            "generation_profile_id": "story_default",
            "generation_profiles": {"default": "story_default"},
            "content_rating": "general",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["details"]["reason"] == "generation_profile_default_reserved"


def test_draft_and_publish_use_custom_authnz_profile_versions(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
    authnz_pool: DatabasePool,
) -> None:
    asset_pack_id, slot_key = _create_asset_pack(chacha_dbs[42])

    async def seed_profiles() -> None:
        store = VNPolicyProfileStore(authnz_pool)
        await store.initialize()
        await store.create_policy_profile(
            profile_id="custom_local",
            display_name="Custom Local",
            description=None,
            definition=LOCAL_DEFAULT_POLICY_DEFINITION,
            created_by_user_id=42,
        )
        await store.update_policy_profile(
            "custom_local",
            display_name="Custom Local V2",
            description=None,
            definition=LOCAL_DEFAULT_POLICY_DEFINITION,
            updated_by_user_id=42,
        )
        generation_definition = dict(STORY_DEFAULT_GENERATION_DEFINITION)
        generation_definition["max_choices"] = 2
        await store.create_generation_profile(
            profile_id="custom_story",
            display_name="Custom Story",
            description=None,
            definition=generation_definition,
            created_by_user_id=42,
        )
        generation_definition["max_choices"] = 3
        await store.update_generation_profile(
            "custom_story",
            display_name="Custom Story V2",
            description=None,
            definition=generation_definition,
            updated_by_user_id=42,
        )

    asyncio.run(seed_profiles())
    create_response = client.post(
        "/api/v1/vn/vn-scripts/scripts",
        json={
            "title": "Archive Door",
            "primary_asset_pack_id": asset_pack_id,
            "policy_profile_id": "custom_local",
            "generation_profile_id": "custom_story",
            "content_rating": "general",
        },
    )
    script_id = int(create_response.json()["id"])
    program = _program(asset_pack_id, slot_key=slot_key)
    program["generation_defaults"]["profile_id"] = "custom_story"
    save_response = client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": program},
    )
    publish_response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/publish",
        json={
            "draft_revision": 1,
            "label": "custom-v2",
            "idempotency_key": "custom-publish-v2",
            "acknowledgements": ["character_safety_missing"],
        },
    )
    snapshots = VNProfileSnapshotRepository.initialized(chacha_dbs[42])

    assert create_response.status_code == 201
    assert save_response.status_code == 200
    assert publish_response.status_code == 201
    policy_snapshot = snapshots.get_profile_snapshot(publish_response.json()["policy_snapshot_id"], owner_user_id=42)
    generation_snapshot = snapshots.get_profile_snapshot(
        publish_response.json()["generation_profile_snapshot_id"],
        owner_user_id=42,
    )
    assert policy_snapshot["profile_id"] == "custom_local"
    assert policy_snapshot["profile_version"] == 2
    assert generation_snapshot["profile_id"] == "custom_story"
    assert generation_snapshot["profile_version"] == 2


def test_script_crud_delete_and_owner_scoping(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
    current_user: dict[str, Any],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    list_response = client.get("/api/v1/vn/vn-scripts/scripts?limit=20&offset=0")
    patch_response = client.patch(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}",
        json={"title": "Archive Door Revised", "status": "draft"},
    )
    get_response = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}")

    current_user["id"] = 7
    other_user_response = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}")
    current_user["id"] = 42
    delete_response = client.delete(f"/api/v1/vn/vn-scripts/scripts/{script_id}")
    deleted_response = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}")

    assert list_response.status_code == 200
    assert list_response.json()["items"][0]["id"] == script_id
    assert get_response.status_code == 200
    assert patch_response.status_code == 200
    assert patch_response.json()["title"] == "Archive Door Revised"
    assert other_user_response.status_code == 404
    assert delete_response.status_code == 204
    assert deleted_response.status_code == 404


def test_draft_save_validate_and_diagnostics(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, _ = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    stale_response = client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 2, "draft": _program(asset_pack_id)},
    )
    save_response = client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": _program(asset_pack_id)},
    )
    get_draft_response = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft")
    invalid_program = _program(asset_pack_id)
    invalid_program["labels"]["start"][1] = {"op": "jump", "target": "missing"}
    validate_response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/validate",
        json={"draft": invalid_program},
    )
    diagnostics_response = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft/diagnostics")

    assert stale_response.status_code == 409
    assert stale_response.json()["detail"]["details"]["reason"] == "draft_revision_conflict"
    assert save_response.status_code == 200
    assert save_response.json()["revision"] == 1
    assert get_draft_response.status_code == 200
    assert get_draft_response.json()["revision"] == 1
    assert validate_response.status_code == 200
    assert validate_response.json()["valid"] is False
    assert validate_response.json()["errors"][0]["code"] == "jump_target_missing"
    assert diagnostics_response.status_code == 200
    assert diagnostics_response.json()["revision"] == 1


def test_publish_versions_manifest_snapshot_and_policy_evaluate(
    client: TestClient,
    chacha_dbs: dict[int, CharactersRAGDB],
) -> None:
    asset_pack_id, slot_key = _create_asset_pack(chacha_dbs[42])
    script_id = _create_script(client, asset_pack_id=asset_pack_id)
    client.put(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/draft",
        json={"if_revision": 0, "draft": _program(asset_pack_id, slot_key=slot_key)},
    )

    publish_response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/publish",
        json={
            "draft_revision": 1,
            "label": "v1",
            "idempotency_key": "publish-v1",
            "acknowledgements": ["character_safety_missing"],
        },
    )
    replay_response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/publish",
        json={
            "draft_revision": 1,
            "label": "v1",
            "idempotency_key": "publish-v1",
            "acknowledgements": ["character_safety_missing"],
        },
    )
    conflict_response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/publish",
        json={
            "draft_revision": 1,
            "label": "v1-conflict",
            "idempotency_key": "publish-v1",
            "acknowledgements": ["character_safety_missing"],
        },
    )

    version_id = publish_response.json()["version_id"]
    versions_response = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/versions")
    version_response = client.get(f"/api/v1/vn/vn-scripts/scripts/{script_id}/versions/{version_id}")
    manifest_response = client.get(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/versions/{version_id}/manifest-snapshot"
    )
    policy_response = client.post(
        f"/api/v1/vn/vn-scripts/scripts/{script_id}/versions/{version_id}/policy/evaluate",
        json={"context": {"character_safety": {"metadata_status": "adult"}}},
    )

    assert publish_response.status_code == 201
    assert replay_response.status_code == 200
    assert replay_response.json() == publish_response.json()
    assert conflict_response.status_code == 409
    assert versions_response.status_code == 200
    assert versions_response.json()["items"][0]["id"] == version_id
    assert version_response.status_code == 200
    assert manifest_response.status_code == 200
    assert manifest_response.json()["manifest"]["assets"]["backgrounds"][0]["slot_key"] == slot_key
    assert policy_response.status_code == 200
    assert policy_response.json()["decision"] == "allow"


@pytest.mark.anyio
async def test_audio_ref_resolver_requires_generated_file_ownership_and_audio_mime() -> None:
    class FakeFilesRepo:
        async def get_files_by_ids(self, file_ids: list[int]) -> list[dict[str, Any]]:
            return [
                {"id": 1, "user_id": 42, "mime_type": "audio/mpeg", "is_deleted": False},
                {"id": 2, "user_id": 7, "mime_type": "audio/mpeg", "is_deleted": False},
                {"id": 3, "user_id": 42, "mime_type": "image/png", "is_deleted": False},
            ]

    program = {
        "media_refs": {
            "valid": {"generated_file_id": 1},
            "wrong_owner": {"generated_file_id": 2},
            "wrong_type": {"generated_file_id": 3},
            "missing": {"generated_file_id": 4},
        }
    }

    resolved = await _resolve_accessible_audio_refs(program, files_repo=FakeFilesRepo(), owner_user_id=42)

    assert resolved == {"valid": {"generated_file_id": 1, "mime_type": "audio/mpeg", "owner_user_id": 42}}
