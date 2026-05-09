from collections.abc import Generator, Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.vn_play import router as vn_play_router
from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
    VNAssetPackCreate,
    VNAssetReviewRequest,
)
from tldw_Server_API.app.api.v1.schemas.vn_play_schemas import (
    VNPlaySessionCreate,
    VNPlayTurnRequest,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService


@pytest.fixture
def chacha_db(tmp_path) -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="vn-play-api-test-client")
    yield database
    database.close_connection()


@pytest.fixture
def character_id(chacha_db: CharactersRAGDB) -> int:
    existing_characters, _ = chacha_db.query_character_cards(limit=100)
    for character in existing_characters:
        chacha_db.soft_delete_character_card(int(character["id"]), int(character["version"]))
    return chacha_db.add_character_card(
        {
            "name": "Mira",
            "description": "A careful archivist.",
            "personality": "Patient and exacting.",
            "scenario": "Cataloging an orbital library.",
            "image": "data:image/png;base64,abc123",
        }
    )


@pytest.fixture
def ready_pack_id() -> int:
    return 10


@pytest.fixture
def client(chacha_db: CharactersRAGDB) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(vn_play_router, prefix="/api/v1")

    async def override_user() -> User:
        return User(id=42, username="user-42")

    async def override_chacha_db() -> CharactersRAGDB:
        return chacha_db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def session_id(client: TestClient, character_id: int, ready_pack_id: int) -> int:
    response = client.post(
        "/api/v1/vn-play/sessions",
        json={
            "mode": "freeform",
            "title": "Library night",
            "primary_character_id": character_id,
            "vn_asset_pack_id": ready_pack_id,
        },
    )
    assert response.status_code == 201
    return int(response.json()["id"])


def _add_character(
    chacha_db: CharactersRAGDB,
    *,
    name: str,
    description: str = "A careful archivist.",
) -> int:
    character_id = chacha_db.add_character_card(
        {
            "name": name,
            "description": description,
            "personality": "Patient and exacting.",
            "scenario": "Cataloging an orbital library.",
            "tags": ["guide", "archive"],
            "image": "data:image/png;base64,abc123",
        }
    )
    assert character_id is not None
    return int(character_id)


def _create_pack(
    chacha_db: CharactersRAGDB,
    *,
    owner_user_id: int,
    character_id: int,
    title: str = "Mira - Archive Pack",
    content_rating: str = "general",
    ready: bool = True,
) -> int:
    service = VNAssetPackService(chacha_db, owner_user_id=owner_user_id)
    pack = service.create_pack(
        VNAssetPackCreate(
            title=title,
            primary_character_id=character_id,
            content_rating=content_rating,
        )
    )
    slots = service.apply_matrix(pack.id, "starter", {"variant_count": 1})
    if ready:
        repo = VNAssetPacksRepository.initialized(chacha_db)
        for index, slot in enumerate(slots, start=1):
            item = repo.create_item(
                pack_id=pack.id,
                slot_id=slot.id,
                variant_index=0,
                generated_file_id=1000 + index,
                mime_type="image/png",
            )
            service.review_item(item["id"], VNAssetReviewRequest(review_status="approved"))
    return pack.id


def test_turn_request_requires_exactly_one_input_field() -> None:
    VNPlayTurnRequest(input_text="hello", client_scene_version=0, idempotency_key="k")

    with pytest.raises(ValidationError):
        VNPlayTurnRequest(
            input_text="hello",
            choice_id="choice-1",
            client_scene_version=0,
            idempotency_key="k",
        )


def test_turn_request_rejects_missing_input_field() -> None:
    with pytest.raises(ValidationError):
        VNPlayTurnRequest(client_scene_version=0, idempotency_key="k")


def test_create_session_defaults_linked_chat_to_read_only() -> None:
    request = VNPlaySessionCreate(
        mode="freeform",
        title="Test",
        primary_character_id=1,
        vn_asset_pack_id=2,
    )

    assert request.linked_chat_mode == "read_only_context"


def test_create_session_endpoint_returns_scene_state(
    client: TestClient,
    ready_pack_id: int,
    character_id: int,
) -> None:
    response = client.post(
        "/api/v1/vn-play/sessions",
        json={
            "mode": "freeform",
            "title": "Library night",
            "primary_character_id": character_id,
            "vn_asset_pack_id": ready_pack_id,
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["mode"] == "freeform"
    assert body["scene_state"]["scene_version"] == 0


def test_setup_options_returns_selector_safe_character_and_pack(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=character_id,
    )

    response = client.get(
        f"/api/v1/vn-play/setup-options?selected_character_id={character_id}"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selected_character"]["id"] == character_id
    assert body["selected_character"]["has_image"] is True
    assert body["characters"][0]["name"] == "Mira"
    assert body["asset_packs"][0]["title"] == "Mira - Archive Pack"
    assert body["asset_packs"][0]["ready"] is True
    assert body["asset_packs"][0]["compatibility"]["status"] == "compatible"
    assert body["asset_packs"][0]["warning_summary"]["requires_acknowledgement"] is False
    assert body["pagination"]["characters"]["limit"] == 25
    assert body["pagination"]["asset_packs"]["limit"] == 25
    assert "image" not in body["characters"][0]
    assert "image_base64" not in body["characters"][0]


def test_setup_options_preserves_selected_character_outside_page(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    selected_character_id = _add_character(
        chacha_db,
        name="Zara",
        description="A pilot who knows the archive station.",
    )

    response = client.get(
        "/api/v1/vn-play/setup-options"
        f"?character_limit=1&character_offset=0&selected_character_id={selected_character_id}"
    )

    assert response.status_code == 200
    body = response.json()
    page_character_ids = {item["id"] for item in body["characters"]}
    assert body["selected_character"]["id"] == selected_character_id
    assert selected_character_id not in page_character_ids
    assert character_id in page_character_ids


def test_setup_options_warns_for_unready_and_incompatible_packs(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    other_character_id = _add_character(chacha_db, name="Zara")
    _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=other_character_id,
        title="Zara Draft Pack",
        content_rating="mature",
        ready=False,
    )

    response = client.get(
        "/api/v1/vn-play/setup-options"
        f"?selected_character_id={character_id}&content_rating=general"
    )

    assert response.status_code == 200
    pack = response.json()["asset_packs"][0]
    warning_codes = {warning["code"] for warning in pack["warning_summary"]["warnings"]}
    assert pack["ready"] is False
    assert pack["compatibility"]["status"] == "different_character"
    assert pack["warning_summary"]["requires_acknowledgement"] is True
    assert "pack_character_mismatch" in warning_codes
    assert "pack_not_ready" in warning_codes
    assert "content_rating_mismatch" in warning_codes


def test_setup_options_marks_untrusted_import_provenance(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    pack_id = _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=character_id,
    )
    repo = VNAssetPacksRepository.initialized(chacha_db)
    preview = repo.create_import_preview(
        owner_user_id=42,
        job_id="preview-job",
        status="completed",
        archive_path="test-artifacts/preview.vnpack",
    )
    repo.create_import_journal(
        owner_user_id=42,
        preview_id=int(preview["id"]),
        job_id="import-job",
        status="completed",
        stage="completed",
        trust_mode="untrusted_import",
        target_mode="create_new",
        target_pack_id=pack_id,
        completed_at="2026-05-09T00:00:00Z",
    )

    response = client.get(
        f"/api/v1/vn-play/setup-options?selected_character_id={character_id}"
    )

    assert response.status_code == 200
    pack = response.json()["asset_packs"][0]
    warning_codes = {warning["code"] for warning in pack["warning_summary"]["warnings"]}
    assert pack["trust_level"] == "untrusted_import"
    assert pack["trust_source"] == "latest_import_journal"
    assert "pack_untrusted_import" in warning_codes


def test_setup_options_degrades_to_unknown_when_provenance_lookup_fails(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=character_id,
    )

    def failing_provenance_lookup(
        self: VNAssetPacksRepository,
        *,
        owner_user_id: int,
        pack_ids: list[int],
    ) -> dict[int, dict[str, Any]]:
        raise RuntimeError("journal lookup unavailable")

    monkeypatch.setattr(
        VNAssetPacksRepository,
        "latest_completed_import_provenance_by_pack_ids",
        failing_provenance_lookup,
    )

    response = client.get(
        f"/api/v1/vn-play/setup-options?selected_character_id={character_id}"
    )

    assert response.status_code == 200
    pack = response.json()["asset_packs"][0]
    assert pack["trust_level"] == "unknown"
    assert pack["trust_source"] == "unknown"


def test_setup_options_evaluates_readiness_only_for_returned_pack_page(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_pack_id = _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=character_id,
        title="Mira - First Pack",
    )
    second_pack_id = _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=character_id,
        title="Mira - Second Pack",
    )
    original_get_readiness = VNAssetPackService.get_readiness
    readiness_calls: list[int] = []

    def tracking_get_readiness(
        self: VNAssetPackService,
        pack_id: int,
    ) -> Any:
        readiness_calls.append(pack_id)
        return original_get_readiness(self, pack_id)

    monkeypatch.setattr(VNAssetPackService, "get_readiness", tracking_get_readiness)

    response = client.get(
        "/api/v1/vn-play/setup-options"
        f"?selected_character_id={character_id}&pack_limit=1"
    )

    assert response.status_code == 200
    body = response.json()
    assert [pack["id"] for pack in body["asset_packs"]] == [first_pack_id]
    assert second_pack_id not in readiness_calls
    assert readiness_calls == [first_pack_id]
    assert body["pagination"]["asset_packs"]["has_more"] is True


def test_setup_options_degrades_when_pack_readiness_fails(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=character_id,
        title="Mira - Fragile Pack",
    )

    def failing_get_readiness(
        self: VNAssetPackService,
        pack_id: int,
    ) -> Any:
        raise RuntimeError("readiness backend unavailable")

    monkeypatch.setattr(VNAssetPackService, "get_readiness", failing_get_readiness)

    response = client.get(
        f"/api/v1/vn-play/setup-options?selected_character_id={character_id}"
    )

    assert response.status_code == 200
    pack = response.json()["asset_packs"][0]
    warning_codes = {warning["code"] for warning in pack["warning_summary"]["warnings"]}
    assert pack["ready"] is False
    assert pack["readiness_status"] == "unknown"
    assert "readiness_unavailable" in warning_codes


def test_setup_options_returns_global_empty_states_for_empty_user(
    client: TestClient,
    chacha_db: CharactersRAGDB,
) -> None:
    characters, _ = chacha_db.query_character_cards(limit=100)
    for character in characters:
        chacha_db.soft_delete_character_card(int(character["id"]), int(character["version"]))

    response = client.get("/api/v1/vn-play/setup-options")

    assert response.status_code == 200
    empty_states = {state["code"]: state for state in response.json()["empty_states"]}
    assert empty_states["no_characters"]["scope"] == "global"
    assert empty_states["no_asset_packs"]["scope"] == "global"


def test_turn_endpoint_rejects_stale_scene_version(
    client: TestClient,
    session_id: int,
) -> None:
    first = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/turn",
        json={"input_text": "Hello", "client_scene_version": 0, "idempotency_key": "a"},
    )
    assert first.status_code == 200

    stale = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/turn",
        json={"input_text": "Again", "client_scene_version": 0, "idempotency_key": "b"},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"] == "stale_scene_version"


def test_delete_session_endpoint_soft_deletes(
    client: TestClient,
    session_id: int,
) -> None:
    response = client.delete(f"/api/v1/vn-play/sessions/{session_id}")

    assert response.status_code == 204
    assert client.get(f"/api/v1/vn-play/sessions/{session_id}").status_code == 404
