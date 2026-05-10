from collections.abc import Generator, Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.vn_play import router as vn_play_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.DB_Management.VNPlay_DB import VNPlayRepository
from tldw_Server_API.app.core.VN_Play.service import (
    DeterministicVNPlayTurnAdapter,
    VNPlayConflictError,
    VNPlayService,
)


@pytest.fixture
def chacha_db(tmp_path) -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="vn-play-save-slot-test")
    yield database
    database.close_connection()


@pytest.fixture
def service(chacha_db: CharactersRAGDB) -> VNPlayService:
    return VNPlayService(
        repo=VNPlayRepository.initialized(chacha_db),
        owner_user_id=42,
        adapter=DeterministicVNPlayTurnAdapter(),
    )


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


def _ready_session(service: VNPlayService):
    return service.create_session(
        mode="freeform",
        title="Library night",
        primary_character_id=1,
        vn_asset_pack_id=10,
        content_rating="general",
        seed="seed-1",
        settings={},
    )


def _pack_id(chacha_db: CharactersRAGDB) -> int:
    repo = VNAssetPacksRepository(chacha_db)
    pack = repo.create_pack(
        owner_user_id=42,
        primary_character_id=1,
        title="Library Pack",
    )
    return int(pack["id"])


@pytest.mark.asyncio
async def test_save_slot_create_list_patch_delete_and_restore(
    service: VNPlayService,
) -> None:
    session = _ready_session(service)
    await service.submit_turn(
        session.id,
        input_text="First",
        client_scene_version=0,
        idempotency_key="save-slot-first",
    )

    created = service.create_save_slot(
        session.id,
        slot_key="manual-1",
        title="After first",
        metadata={"kind": "manual"},
        idempotency_key="save-slot-create",
    )
    replayed = service.create_save_slot(
        session.id,
        slot_key="manual-1",
        title="After first",
        metadata={"kind": "manual"},
        idempotency_key="save-slot-create",
    )

    assert created["replayed"] is False
    assert replayed == {**created, "replayed": True}
    assert created["slot_key"] == "manual-1"
    assert created["checkpoint_id"] is not None
    assert service.list_save_slots(session.id)[0]["id"] == created["id"]

    updated = service.update_save_slot(
        session.id,
        int(created["id"]),
        title="Renamed",
        metadata={"kind": "manual", "favorite": True},
    )
    assert updated["title"] == "Renamed"
    assert updated["metadata"] == {"kind": "manual", "favorite": True}

    await service.submit_turn(
        session.id,
        input_text="Second",
        client_scene_version=1,
        idempotency_key="save-slot-second",
    )
    restored = service.restore_save_slot(
        session.id,
        int(created["id"]),
        client_scene_version=2,
        idempotency_key="save-slot-restore",
    )
    restored_again = service.restore_save_slot(
        session.id,
        int(created["id"]),
        client_scene_version=2,
        idempotency_key="save-slot-restore",
    )

    assert restored["status"] == "completed"
    assert restored["save_slot_id"] == int(created["id"])
    assert restored["scene_version"] == 3
    assert restored_again == {**restored, "replayed": True}

    service.delete_save_slot(session.id, int(created["id"]))
    assert service.list_save_slots(session.id) == []
    deleted = service.get_save_slot(session.id, int(created["id"]), include_deleted=True)
    assert deleted["deleted"] is True


@pytest.mark.asyncio
async def test_save_slot_idempotency_conflict_and_stale_restore_are_stable(
    service: VNPlayService,
) -> None:
    session = _ready_session(service)
    await service.submit_turn(
        session.id,
        input_text="First",
        client_scene_version=0,
        idempotency_key="save-slot-conflict-first",
    )
    service.create_save_slot(
        session.id,
        slot_key="manual-1",
        title="After first",
        metadata={},
        idempotency_key="save-slot-conflict-create",
    )

    with pytest.raises(VNPlayConflictError, match="idempotency_key_conflict"):
        service.create_save_slot(
            session.id,
            slot_key="manual-2",
            title="Different",
            metadata={},
            idempotency_key="save-slot-conflict-create",
        )

    slot = service.list_save_slots(session.id)[0]
    await service.submit_turn(
        session.id,
        input_text="Second",
        client_scene_version=1,
        idempotency_key="save-slot-conflict-second",
    )

    with pytest.raises(VNPlayConflictError, match="stale_scene_version"):
        service.restore_save_slot(
            session.id,
            int(slot["id"]),
            client_scene_version=1,
            idempotency_key="save-slot-stale-restore",
        )
    with pytest.raises(VNPlayConflictError, match="stale_scene_version"):
        service.restore_save_slot(
            session.id,
            int(slot["id"]),
            client_scene_version=1,
            idempotency_key="save-slot-stale-restore",
        )


def test_save_slot_api_create_read_patch_delete_and_restore(
    client: TestClient,
    chacha_db: CharactersRAGDB,
) -> None:
    pack_id = _pack_id(chacha_db)
    session_response = client.post(
        "/api/v1/vn-play/sessions",
        json={
            "mode": "freeform",
            "title": "Library night",
            "primary_character_id": 1,
            "vn_asset_pack_id": pack_id,
        },
    )
    assert session_response.status_code == 201
    session_id = int(session_response.json()["id"])
    turn_response = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/turn",
        json={
            "input_text": "First",
            "client_scene_version": 0,
            "idempotency_key": "api-save-slot-first",
        },
    )
    assert turn_response.status_code == 200

    create_response = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/save-slots",
        json={
            "slot_key": "manual-1",
            "title": "After first",
            "metadata": {"kind": "manual"},
            "idempotency_key": "api-save-slot-create",
        },
    )
    assert create_response.status_code == 201
    slot = create_response.json()
    assert slot["slot_key"] == "manual-1"

    list_response = client.get(f"/api/v1/vn-play/sessions/{session_id}/save-slots")
    assert list_response.status_code == 200
    assert [item["id"] for item in list_response.json()] == [slot["id"]]

    read_response = client.get(
        f"/api/v1/vn-play/sessions/{session_id}/save-slots/{slot['id']}"
    )
    assert read_response.status_code == 200
    assert read_response.json()["title"] == "After first"

    patch_response = client.patch(
        f"/api/v1/vn-play/sessions/{session_id}/save-slots/{slot['id']}",
        json={"title": "Renamed", "metadata": {"kind": "manual", "favorite": True}},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["title"] == "Renamed"

    second_turn = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/turn",
        json={
            "input_text": "Second",
            "client_scene_version": 1,
            "idempotency_key": "api-save-slot-second",
        },
    )
    assert second_turn.status_code == 200
    restore_response = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/save-slots/{slot['id']}/restore",
        json={"client_scene_version": 2, "idempotency_key": "api-save-slot-restore"},
    )
    assert restore_response.status_code == 200
    assert restore_response.json()["save_slot_id"] == slot["id"]
    assert restore_response.json()["scene_version"] == 3

    delete_response = client.delete(
        f"/api/v1/vn-play/sessions/{session_id}/save-slots/{slot['id']}"
    )
    assert delete_response.status_code == 204
    assert client.get(f"/api/v1/vn-play/sessions/{session_id}/save-slots").json() == []
