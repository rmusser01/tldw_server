import sqlite3
from collections.abc import Generator, Iterator
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.vn_assets import router as vn_assets_router
from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetBulkReviewRequest,
    VNAssetCleanupRequest,
    VNAssetGenerationRequest,
    VNAssetPackCreate,
    VNAssetPackUpdate,
    VNAssetPromptPreviewRequest,
    VNAssetReviewRequest,
    VNAssetSlotCreate,
    VNAssetSlotUpdate,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.VN_Assets.constants import (
    ERROR_ITEM_LIMIT_EXCEEDED,
    ERROR_SLOT_VARIANT_LIMIT_EXCEEDED,
)
from tldw_Server_API.app.core.VN_Assets.matrix import expand_starter_matrix
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService


@pytest.fixture
def chacha_db(tmp_path) -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="vn-assets-api-test-client")
    yield database
    database.close_connection()


@pytest.fixture
def character_id(chacha_db: CharactersRAGDB) -> int:
    return chacha_db.add_character_card(
        {
            "name": "Mira",
            "description": "A careful archivist.",
            "personality": "Patient and exacting.",
            "scenario": "Cataloging an orbital library.",
        }
    )


@pytest.fixture
def service(chacha_db: CharactersRAGDB) -> VNAssetPackService:
    return VNAssetPackService(chacha_db, owner_user_id=42)


@pytest.fixture
def current_user_id() -> dict[str, int]:
    return {"value": 42}


@pytest.fixture
def client(
    chacha_db: CharactersRAGDB,
    current_user_id: dict[str, int],
) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(vn_assets_router, prefix="/api/v1/vn")

    async def override_user() -> User:
        user_id = current_user_id["value"]
        return User(id=user_id, username=f"user-{user_id}")

    async def override_chacha_db() -> CharactersRAGDB:
        return chacha_db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {}


@pytest.fixture
def pack_with_items(
    chacha_db: CharactersRAGDB,
    service: VNAssetPackService,
    character_id: int,
) -> SimpleNamespace:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
    repo = VNAssetPacksRepository.initialized(chacha_db)
    approved_item = repo.create_item(
        pack_id=pack.id,
        slot_id=slot.id,
        variant_index=0,
        generated_file_id=1001,
        mime_type="image/png",
    )
    draft_item = repo.create_item(pack_id=pack.id, slot_id=slot.id, variant_index=1)
    service.review_item(approved_item["id"], VNAssetReviewRequest(review_status="approved"))

    return SimpleNamespace(
        id=pack.id,
        approved_item_id=approved_item["id"],
        draft_item_id=draft_item["id"],
    )


def test_create_pack_endpoint_returns_pack(
    client: TestClient,
    auth_headers: dict[str, str],
    character_id: int,
) -> None:
    response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={"title": "Starter", "primary_character_id": character_id},
        headers=auth_headers,
    )

    assert response.status_code == 201
    assert response.json()["primary_character_id"] == character_id


def test_old_top_level_vn_assets_route_is_absent(client: TestClient) -> None:
    response = client.get("/api/v1/vn-assets/packs")

    assert response.status_code == 404


def test_starter_matrices_endpoint_declares_response_model() -> None:
    route = next(
        route
        for route in vn_assets_router.routes
        if getattr(route, "path", "") == "/vn-assets/starter-matrices"
    )

    assert getattr(route, "response_model", None) is not None
    assert route.response_model.__name__ == "VNAssetStarterMatricesResponse"


def test_slot_create_and_update_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        VNAssetSlotCreate(
            asset_type="sprite",
            slot_key="sprite.primary",
            unsupported_field=True,
        )

    with pytest.raises(ValidationError):
        VNAssetSlotUpdate(unsupported_field=True)


@pytest.mark.parametrize("variant_count", [True, 1.0, "1"])
def test_create_pack_rejects_coerced_starter_variant_count(
    client: TestClient,
    auth_headers: dict[str, str],
    character_id: int,
    variant_count: object,
) -> None:
    response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={
            "title": "Starter",
            "primary_character_id": character_id,
            "apply_starter_matrix": True,
            "starter_matrix_variant_count": variant_count,
        },
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_manifest_endpoint_omits_unapproved_items(
    client: TestClient,
    auth_headers: dict[str, str],
    pack_with_items: SimpleNamespace,
) -> None:
    response = client.get(
        f"/api/v1/vn/vn-assets/packs/{pack_with_items.id}/manifest",
        headers=auth_headers,
    )

    assert response.status_code == 200
    item_ids = [
        item["item_id"]
        for collection in response.json()["assets"].values()
        for item in collection
    ]
    assert pack_with_items.approved_item_id in item_ids
    assert pack_with_items.draft_item_id not in item_ids


def test_preferred_endpoint_requires_existing_approved_item(
    client: TestClient,
    auth_headers: dict[str, str],
    pack_with_items: SimpleNamespace,
) -> None:
    draft_response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_with_items.id}/items/{pack_with_items.draft_item_id}/preferred",
        headers=auth_headers,
    )
    approved_response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_with_items.id}/items/{pack_with_items.approved_item_id}/preferred",
        headers=auth_headers,
    )

    assert draft_response.status_code == 400
    assert draft_response.json()["detail"] == "preferred_item_must_be_approved"
    assert approved_response.status_code == 200
    assert approved_response.json()["preferred"] is True


def test_pack_endpoints_deny_cross_user_access(
    client: TestClient,
    auth_headers: dict[str, str],
    current_user_id: dict[str, int],
    character_id: int,
) -> None:
    create_response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={"title": "Starter", "primary_character_id": character_id},
        headers=auth_headers,
    )
    pack_id = create_response.json()["id"]

    current_user_id["value"] = 7

    assert client.get(f"/api/v1/vn/vn-assets/packs/{pack_id}", headers=auth_headers).status_code == 404
    assert client.patch(
        f"/api/v1/vn/vn-assets/packs/{pack_id}",
        json={"title": "Not Mine"},
        headers=auth_headers,
    ).status_code == 404
    assert client.delete(f"/api/v1/vn/vn-assets/packs/{pack_id}", headers=auth_headers).status_code == 404


def test_validation_limit_errors_map_to_400(
    client: TestClient,
    auth_headers: dict[str, str],
    character_id: int,
) -> None:
    create_response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={"title": "Starter", "primary_character_id": character_id},
        headers=auth_headers,
    )
    pack_id = create_response.json()["id"]

    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/matrix/apply",
        json={"matrix_key": "starter", "overrides": {"variant_count": 7}},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == ERROR_SLOT_VARIANT_LIMIT_EXCEEDED


def test_delete_parent_slot_with_dependents_returns_controlled_conflict(
    client: TestClient,
    auth_headers: dict[str, str],
    character_id: int,
) -> None:
    pack_response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={"title": "Starter", "primary_character_id": character_id},
        headers=auth_headers,
    )
    pack_id = pack_response.json()["id"]
    matrix_response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/matrix/apply",
        json={"matrix_key": "starter", "overrides": {"variant_count": 1}},
        headers=auth_headers,
    )
    slots = matrix_response.json()
    dependent_slot = next(slot for slot in slots if slot["depends_on_slot_id"] is not None)
    parent_slot_id = dependent_slot["depends_on_slot_id"]

    response = client.delete(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/slots/{parent_slot_id}",
        headers=auth_headers,
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "slot_has_dependents"


def test_duplicate_slot_key_update_returns_controlled_conflict(
    client: TestClient,
    auth_headers: dict[str, str],
    character_id: int,
) -> None:
    pack_response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={"title": "Starter", "primary_character_id": character_id},
        headers=auth_headers,
    )
    pack_id = pack_response.json()["id"]
    first = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/slots",
        json={"asset_type": "sprite", "slot_key": "sprite.first"},
        headers=auth_headers,
    ).json()
    second = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/slots",
        json={"asset_type": "sprite", "slot_key": "sprite.second"},
        headers=auth_headers,
    ).json()

    response = client.patch(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/slots/{second['id']}",
        json={"slot_key": first["slot_key"]},
        headers=auth_headers,
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "slot_already_exists"


@pytest.mark.parametrize("variant_count", [True, 1.0, "1"])
def test_slot_endpoints_reject_coerced_variant_counts(
    client: TestClient,
    auth_headers: dict[str, str],
    character_id: int,
    variant_count: object,
) -> None:
    pack_response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={"title": "Starter", "primary_character_id": character_id},
        headers=auth_headers,
    )
    pack_id = pack_response.json()["id"]
    slot = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/slots",
        json={"asset_type": "sprite", "slot_key": "sprite.valid"},
        headers=auth_headers,
    ).json()

    create_response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/slots",
        json={
            "asset_type": "sprite",
            "slot_key": f"sprite.invalid.{type(variant_count).__name__}",
            "variant_count": variant_count,
        },
        headers=auth_headers,
    )
    update_response = client.patch(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/slots/{slot['id']}",
        json={"variant_count": variant_count},
        headers=auth_headers,
    )

    assert create_response.status_code == 422
    assert update_response.status_code == 422


def test_slot_dependency_update_rejects_self_and_cycles(
    client: TestClient,
    auth_headers: dict[str, str],
    character_id: int,
) -> None:
    pack_response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={"title": "Starter", "primary_character_id": character_id},
        headers=auth_headers,
    )
    pack_id = pack_response.json()["id"]
    first = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/slots",
        json={"asset_type": "background", "slot_key": "background.first"},
        headers=auth_headers,
    ).json()
    second = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/slots",
        json={
            "asset_type": "depth_companion",
            "slot_key": "depth.first",
            "depends_on_slot_id": first["id"],
            "required_for_runtime": False,
            "variant_count": 0,
        },
        headers=auth_headers,
    ).json()

    self_response = client.patch(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/slots/{first['id']}",
        json={"depends_on_slot_id": first["id"]},
        headers=auth_headers,
    )
    cycle_response = client.patch(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/slots/{first['id']}",
        json={"depends_on_slot_id": second["id"]},
        headers=auth_headers,
    )

    assert self_response.status_code == 400
    assert self_response.json()["detail"] == "slot_dependency_self"
    assert cycle_response.status_code == 400
    assert cycle_response.json()["detail"] == "slot_dependency_cycle"


def test_prompt_preview_rejects_unknown_budget_keys(
    client: TestClient,
    auth_headers: dict[str, str],
    character_id: int,
) -> None:
    pack_response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={"title": "Starter", "primary_character_id": character_id},
        headers=auth_headers,
    )
    pack_id = pack_response.json()["id"]
    slot = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/slots",
        json={"asset_type": "sprite", "slot_key": "sprite.first"},
        headers=auth_headers,
    ).json()

    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/prompt-preview",
        json={"slot_id": slot["id"], "budgets": {"unknown": 10}},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_prompt_budget"


def test_prompt_preview_combines_pack_and_slot_negative_prompts(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(
            title="Preview Pack",
            primary_character_id=character_id,
            negative_prompt="low quality",
        )
    )
    slot = service.create_slot(
        pack.id,
        VNAssetSlotCreate(
            asset_type="sprite",
            slot_key="sprite.first",
            negative_prompt_template="extra fingers",
        ),
    )

    preview = service.preview_prompt(pack.id, VNAssetPromptPreviewRequest(slot_id=slot.id))

    assert "low quality" in preview.negative_prompt
    assert "extra fingers" in preview.negative_prompt
    assert "Negative prompt:" not in preview.prompt


@pytest.mark.parametrize("budget_value", [True, 1.0, "1"])
def test_prompt_preview_rejects_coerced_budget_values(
    client: TestClient,
    auth_headers: dict[str, str],
    character_id: int,
    budget_value: object,
) -> None:
    pack_response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={"title": "Starter", "primary_character_id": character_id},
        headers=auth_headers,
    )
    pack_id = pack_response.json()["id"]
    slot = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/slots",
        json={"asset_type": "sprite", "slot_key": "sprite.first"},
        headers=auth_headers,
    ).json()

    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/prompt-preview",
        json={"slot_id": slot["id"], "budgets": {"total": budget_value}},
        headers=auth_headers,
    )

    assert response.status_code == 422


@pytest.mark.parametrize("variant_count", [None, {}, [], "many", 1.5, 6.9])
def test_matrix_apply_rejects_malformed_variant_count(
    client: TestClient,
    auth_headers: dict[str, str],
    character_id: int,
    variant_count: object,
) -> None:
    pack_response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={"title": "Starter", "primary_character_id": character_id},
        headers=auth_headers,
    )
    pack_id = pack_response.json()["id"]

    response = client.post(
        f"/api/v1/vn/vn-assets/packs/{pack_id}/matrix/apply",
        json={"matrix_key": "starter", "overrides": {"variant_count": variant_count}},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_matrix_variant_count"


def test_generation_budget_rejects_malformed_planned_output_count(
    client: TestClient,
    auth_headers: dict[str, str],
    character_id: int,
) -> None:
    create_response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={
            "title": "Starter",
            "primary_character_id": character_id,
            "generation_budget": {"planned_output_count": {}},
        },
        headers=auth_headers,
    )
    pack_response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={"title": "Starter", "primary_character_id": character_id},
        headers=auth_headers,
    )
    pack_id = pack_response.json()["id"]
    update_response = client.patch(
        f"/api/v1/vn/vn-assets/packs/{pack_id}",
        json={"generation_budget": {"planned_output_count": []}},
        headers=auth_headers,
    )

    assert create_response.status_code == 400
    assert create_response.json()["detail"] == "invalid_generation_budget"
    assert update_response.status_code == 400
    assert update_response.json()["detail"] == "invalid_generation_budget"


def test_generation_budget_rejects_float_planned_output_count(
    client: TestClient,
    auth_headers: dict[str, str],
    character_id: int,
) -> None:
    create_response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={
            "title": "Starter",
            "primary_character_id": character_id,
            "generation_budget": {"planned_output_count": 300.9},
        },
        headers=auth_headers,
    )
    pack_response = client.post(
        "/api/v1/vn/vn-assets/packs",
        json={"title": "Starter", "primary_character_id": character_id},
        headers=auth_headers,
    )
    pack_id = pack_response.json()["id"]
    update_response = client.patch(
        f"/api/v1/vn/vn-assets/packs/{pack_id}",
        json={"generation_budget": {"planned_output_count": 300.9}},
        headers=auth_headers,
    )

    assert create_response.status_code == 400
    assert create_response.json()["detail"] == "invalid_generation_budget"
    assert update_response.status_code == 400
    assert update_response.json()["detail"] == "invalid_generation_budget"


def test_service_creates_pack_from_existing_character(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(
            title="Starter Pack",
            primary_character_id=character_id,
            description="Base VN assets",
            style_prompt="watercolor",
        )
    )

    assert pack.id > 0
    assert pack.owner_user_id == 42
    assert pack.title == "Starter Pack"
    assert pack.primary_character_id == character_id
    assert pack.style_prompt == "watercolor"
    assert pack.planned_output_count == 0


def test_service_rejects_missing_character(service: VNAssetPackService) -> None:
    with pytest.raises(ValueError, match="primary_character_not_found"):
        service.create_pack(
            VNAssetPackCreate(title="Missing Character Pack", primary_character_id=9999)
        )


def test_apply_starter_matrix_returns_slots_and_estimated_planned_count(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )

    slots = service.apply_matrix(pack.id, "starter", {"variant_count": 2})
    refreshed = service.get_pack(pack.id)

    assert len(slots) == 8
    assert refreshed.planned_output_count == 12
    assert all(slot.pack_id == pack.id for slot in slots)
    assert any(slot.asset_type == "depth_companion" and slot.variant_count == 0 for slot in slots)


def test_patch_slot_variant_count_within_default_max(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]

    updated = service.update_slot(slot.id, VNAssetSlotUpdate(variant_count=6))

    assert updated.variant_count == 6
    assert service.get_pack(pack.id).planned_output_count == 11


def test_direct_slot_update_rejects_self_and_cyclic_dependencies(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    parent = service.create_slot(
        pack.id,
        VNAssetSlotCreate(asset_type="background", slot_key="background.parent"),
    )
    child = service.create_slot(
        pack.id,
        VNAssetSlotCreate(
            asset_type="depth_companion",
            slot_key="depth.parent",
            depends_on_slot_id=parent.id,
            required_for_runtime=False,
            variant_count=0,
        ),
    )

    with pytest.raises(ValueError, match="slot_dependency_self"):
        service.update_slot(parent.id, VNAssetSlotUpdate(depends_on_slot_id=parent.id))
    with pytest.raises(ValueError, match="slot_dependency_cycle"):
        service.update_slot(parent.id, VNAssetSlotUpdate(depends_on_slot_id=child.id))


def test_service_rejects_pack_over_300_planned_generated_items(
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    service = VNAssetPackService(chacha_db, owner_user_id=42, item_limit=5)
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )

    with pytest.raises(ValueError, match=ERROR_ITEM_LIMIT_EXCEEDED):
        service.apply_matrix(pack.id, "starter", {"variant_count": 1})


def test_service_rejects_matrix_over_slot_variant_limit(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )

    with pytest.raises(ValueError, match=ERROR_SLOT_VARIANT_LIMIT_EXCEEDED):
        service.apply_matrix(pack.id, "starter", {"variant_count": 7})


def test_create_pack_rejects_generation_budget_over_limit_without_partial_pack(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    with pytest.raises(ValueError, match=ERROR_ITEM_LIMIT_EXCEEDED):
        service.create_pack(
            VNAssetPackCreate(
                title="Too Large",
                primary_character_id=character_id,
                generation_budget={"planned_output_count": 301},
            )
        )

    assert service.list_packs() == []


def test_create_pack_with_starter_matrix_rejects_over_limit_without_partial_pack(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    with pytest.raises(ValidationError):
        VNAssetPackCreate(
            title="Too Large",
            primary_character_id=character_id,
            apply_starter_matrix=True,
            starter_matrix_variant_count=7,
        )

    assert service.list_packs() == []


def test_slot_update_rejects_variant_count_over_default_slot_max() -> None:
    with pytest.raises(ValidationError):
        VNAssetSlotUpdate(variant_count=7)


def test_review_transitions_support_approved_rejected_and_hidden(
    chacha_db: CharactersRAGDB,
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
    repo = VNAssetPacksRepository.initialized(chacha_db)
    approved_item = repo.create_item(pack_id=pack.id, slot_id=slot.id, variant_index=0)
    rejected_item = repo.create_item(pack_id=pack.id, slot_id=slot.id, variant_index=1)
    hidden_item = repo.create_item(pack_id=pack.id, slot_id=slot.id, variant_index=2)

    assert service.review_item(approved_item["id"], VNAssetReviewRequest(review_status="approved")).review_status == "approved"
    assert service.review_item(rejected_item["id"], VNAssetReviewRequest(review_status="rejected")).review_status == "rejected"
    assert service.review_item(hidden_item["id"], VNAssetReviewRequest(review_status="hidden")).review_status == "hidden"
    assert service.review_item(hidden_item["id"], VNAssetReviewRequest(review_status="draft")).review_status == "draft"


def test_review_request_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        VNAssetReviewRequest(review_status="pending")


def test_review_preferred_requires_approved_status_and_clears_slot_siblings(
    chacha_db: CharactersRAGDB,
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
    repo = VNAssetPacksRepository.initialized(chacha_db)
    first_item = repo.create_item(pack_id=pack.id, slot_id=slot.id, variant_index=0)
    second_item = repo.create_item(pack_id=pack.id, slot_id=slot.id, variant_index=1)

    with pytest.raises(ValueError, match="preferred_item_must_be_approved"):
        service.review_item(
            first_item["id"],
            VNAssetReviewRequest(review_status="rejected", preferred=True),
        )

    first_review = service.review_item(
        first_item["id"],
        VNAssetReviewRequest(review_status="approved", preferred=True),
    )
    second_review = service.review_item(
        second_item["id"],
        VNAssetReviewRequest(review_status="approved", preferred=True),
    )

    assert first_review.preferred is True
    assert second_review.preferred is True
    assert repo.get_item(first_item["id"])["preferred"] == 0
    assert repo.get_item(second_item["id"])["preferred"] == 1


def test_repository_create_item_enforces_pack_slot_and_preferred_invariants(
    chacha_db: CharactersRAGDB,
    service: VNAssetPackService,
    character_id: int,
) -> None:
    first_pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    first_slot = service.apply_matrix(first_pack.id, "starter", {"variant_count": 1})[0]
    second_pack = service.create_pack(
        VNAssetPackCreate(title="Other Pack", primary_character_id=character_id)
    )
    repo = VNAssetPacksRepository.initialized(chacha_db)

    with pytest.raises(ValueError, match="slot_not_in_pack"):
        repo.create_item(pack_id=second_pack.id, slot_id=first_slot.id, variant_index=0)

    with pytest.raises(ValueError, match="preferred_item_must_be_approved"):
        repo.create_item(
            pack_id=first_pack.id,
            slot_id=first_slot.id,
            variant_index=0,
            preferred=True,
        )

    first_item = repo.create_item(
        pack_id=first_pack.id,
        slot_id=first_slot.id,
        variant_index=0,
        review_status="approved",
        preferred=True,
    )
    second_item = repo.create_item(
        pack_id=first_pack.id,
        slot_id=first_slot.id,
        variant_index=1,
        review_status="approved",
        preferred=True,
    )

    assert repo.get_item(first_item["id"])["preferred"] == 0
    assert repo.get_item(second_item["id"])["preferred"] == 1


def test_pack_update_rejects_direct_status_write() -> None:
    with pytest.raises(ValidationError):
        VNAssetPackUpdate(status="ready")


@pytest.mark.parametrize("variant_count", [True, 1.0, "1"])
def test_generation_request_rejects_coerced_variant_count(variant_count: object) -> None:
    with pytest.raises(ValidationError):
        VNAssetGenerationRequest(variant_count=variant_count)


@pytest.mark.parametrize("budget_value", [True, 1.0, "1"])
def test_prompt_preview_request_rejects_coerced_budget_values(budget_value: object) -> None:
    with pytest.raises(ValidationError):
        VNAssetPromptPreviewRequest(slot_id=1, budgets={"total": budget_value})


def test_manifest_returns_approved_only_items(
    chacha_db: CharactersRAGDB,
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
    repo = VNAssetPacksRepository.initialized(chacha_db)
    approved_item = repo.create_item(
        pack_id=pack.id,
        slot_id=slot.id,
        variant_index=0,
        generated_file_id=1001,
        mime_type="image/png",
    )
    draft_item = repo.create_item(pack_id=pack.id, slot_id=slot.id, variant_index=1)
    service.review_item(approved_item["id"], VNAssetReviewRequest(review_status="approved"))

    manifest = service.build_manifest(pack.id)

    asset_ids = [
        item["item_id"]
        for collection in manifest.assets.values()
        for item in collection
    ]
    assert asset_ids == [approved_item["id"]]
    assert draft_item["id"] not in asset_ids


def test_readiness_treats_optional_failures_as_warnings_not_blocking_errors(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    slots = service.apply_matrix(pack.id, "starter", {"variant_count": 1})
    optional_slot = next(slot for slot in slots if slot.required_for_runtime is False)
    required_slot = next(slot for slot in slots if slot.required_for_runtime is True)
    service.update_slot(optional_slot.id, VNAssetSlotUpdate(status="failed", last_error="depth failed"))
    service.update_slot(required_slot.id, VNAssetSlotUpdate(status="approved"))

    readiness = service.get_readiness(pack.id)

    assert readiness.ready is False
    assert f"optional_slot_failed:{optional_slot.id}" in readiness.warnings
    assert f"required_slot_not_ready:{optional_slot.id}" not in readiness.errors


def test_pack_update_enforces_item_limit_when_slots_exist(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    service.apply_matrix(pack.id, "starter", {"variant_count": 1})

    with pytest.raises(ValueError, match=ERROR_ITEM_LIMIT_EXCEEDED):
        service.update_pack(
            pack.id,
            VNAssetPackUpdate(generation_budget={"planned_output_count": 301}),
        )


def test_apply_matrix_preflights_duplicate_slot_keys_without_partial_writes(
    chacha_db: CharactersRAGDB,
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    duplicate_depth_slot = next(
        slot
        for slot in expand_starter_matrix(primary_character_id=character_id, variant_count=1)
        if slot.depends_on_slot_key is not None
    )
    repo = VNAssetPacksRepository.initialized(chacha_db)
    repo.create_slot(
        pack_id=pack.id,
        asset_type=duplicate_depth_slot.asset_type,
        slot_key=duplicate_depth_slot.slot_key,
        labels=duplicate_depth_slot.labels,
        variant_count=duplicate_depth_slot.variant_count,
        required_for_runtime=duplicate_depth_slot.required_for_runtime,
    )

    with pytest.raises(ValueError, match="slot_already_exists"):
        service.apply_matrix(pack.id, "starter", {"variant_count": 1})

    slot_keys = [slot["slot_key"] for slot in repo.list_slots(pack.id)]
    assert slot_keys == [duplicate_depth_slot.slot_key]


def test_apply_matrix_rolls_back_when_later_slot_insert_fails(
    chacha_db: CharactersRAGDB,
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    failing_slot_key = f"cg.{character_id}.opening"
    conn = chacha_db.get_connection()
    conn.execute(
        f"""
        CREATE TRIGGER vn_asset_test_fail_cg_insert
        BEFORE INSERT ON vn_asset_slots
        WHEN NEW.slot_key = '{failing_slot_key}'
        BEGIN
            SELECT RAISE(ABORT, 'forced_matrix_failure');
        END
        """
    )

    try:
        with pytest.raises(sqlite3.IntegrityError, match="forced_matrix_failure"):
            service.apply_matrix(pack.id, "starter", {"variant_count": 1})
    finally:
        conn.execute("DROP TRIGGER IF EXISTS vn_asset_test_fail_cg_insert")

    assert failing_slot_key not in {slot["slot_key"] for slot in service.repo.list_slots(pack.id)}
    assert service.repo.list_slots(pack.id) == []


def test_service_scopes_pack_access_to_owner(
    chacha_db: CharactersRAGDB,
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    other_service = VNAssetPackService(chacha_db, owner_user_id=7)

    assert other_service.list_packs() == []
    with pytest.raises(ValueError, match="pack_not_found"):
        other_service.get_pack(pack.id)
    with pytest.raises(ValueError, match="pack_not_found"):
        other_service.update_pack(pack.id, VNAssetPackUpdate(title="Not Mine"))
    with pytest.raises(ValueError, match="pack_not_found"):
        other_service.soft_delete_pack(pack.id)


def test_service_rejects_cross_owner_slot_and_item_updates(
    chacha_db: CharactersRAGDB,
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
    repo = VNAssetPacksRepository.initialized(chacha_db)
    item = repo.create_item(pack_id=pack.id, slot_id=slot.id, variant_index=0)
    other_service = VNAssetPackService(chacha_db, owner_user_id=7)

    with pytest.raises(ValueError, match="slot_not_found"):
        other_service.update_slot(slot.id, VNAssetSlotUpdate(variant_count=2))
    with pytest.raises(ValueError, match="item_not_found"):
        other_service.review_item(
            item["id"],
            VNAssetReviewRequest(review_status="approved"),
        )


def test_bulk_review_validates_all_items_before_mutating(
    chacha_db: CharactersRAGDB,
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
    repo = VNAssetPacksRepository.initialized(chacha_db)
    item = repo.create_item(pack_id=pack.id, slot_id=slot.id, variant_index=0)

    with pytest.raises(ValueError, match="item_not_found"):
        service.bulk_review_items(
            VNAssetBulkReviewRequest(
                item_ids=[item["id"], item["id"] + 9999],
                review_status="approved",
            )
        )

    assert repo.get_item(item["id"])["review_status"] == "draft"


def test_stale_approved_slot_status_does_not_make_pack_ready(
    service: VNAssetPackService,
    character_id: int,
) -> None:
    pack = service.create_pack(
        VNAssetPackCreate(title="Starter Pack", primary_character_id=character_id)
    )
    slot = service.apply_matrix(pack.id, "starter", {"variant_count": 1})[0]
    service.update_slot(slot.id, VNAssetSlotUpdate(status="approved"))

    readiness = service.get_readiness(pack.id)

    assert readiness.ready is False
    assert f"required_slot_not_ready:{slot.id}" in readiness.errors


def test_cleanup_request_requires_explicit_approved_opt_in() -> None:
    request = VNAssetCleanupRequest()

    assert request.dry_run is True
    assert request.include_approved is False
    assert request.confirmation_text is None
    assert request.confirmation_token is None
