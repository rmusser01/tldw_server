from collections.abc import Generator, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.vn_play import router as vn_play_router
from tldw_Server_API.app.api.v1.schemas.vn_play_schemas import (
    VNPlaySessionCreate,
    VNPlayTurnRequest,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def chacha_db(tmp_path) -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="vn-play-api-test-client")
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
