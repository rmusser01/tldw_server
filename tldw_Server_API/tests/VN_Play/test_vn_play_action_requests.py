from collections.abc import Generator

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPlay_DB import VNPlayRepository
from tldw_Server_API.app.core.VN_Play.constants import (
    ERROR_ACTION_REQUEST_ABANDONED,
    ERROR_RESTORE_ACTION_IN_PROGRESS,
    ERROR_STALE_SCENE_VERSION,
    ERROR_TURN_IN_PROGRESS,
    SESSION_ACTION_STATUS_PENDING,
    TURN_STATUS_ABANDONED,
    TURN_STATUS_COMPLETED,
    TURN_STATUS_MODEL_CALLING,
    TURN_STATUS_MODEL_FAILED,
)
from tldw_Server_API.app.core.VN_Play.models import TurnResult
from tldw_Server_API.app.core.VN_Play.service import (
    DeterministicVNPlayTurnAdapter,
    VNPlayConflictError,
    VNPlayService,
    VNPlayTurnContext,
    VNPlayTurnError,
    _payload_hash,
)


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-play-action-request-test")
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
def failing_service(chacha_db: CharactersRAGDB) -> VNPlayService:
    class FailingAdapter:
        async def generate_turn(self, context: VNPlayTurnContext) -> TurnResult:
            raise RuntimeError("provider unavailable")

    return VNPlayService(
        repo=VNPlayRepository.initialized(chacha_db),
        owner_user_id=42,
        adapter=FailingAdapter(),
    )


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


def _turn_hash(session_id: int, input_payload: dict[str, str]) -> str:
    return _payload_hash({"session_id": session_id, "input": input_payload})


@pytest.mark.asyncio
async def test_stale_turn_request_is_persisted_as_abandoned_before_execution(
    service: VNPlayService,
) -> None:
    session = _ready_session(service)
    await service.submit_turn(
        session.id,
        input_text="First",
        client_scene_version=0,
        idempotency_key="first",
    )

    with pytest.raises(VNPlayConflictError, match=ERROR_STALE_SCENE_VERSION):
        await service.submit_turn(
            session.id,
            input_text="Second",
            client_scene_version=0,
            idempotency_key="stale-second",
        )

    stale_request = service.repo.get_turn_request_by_key(
        session_id=session.id,
        owner_user_id=42,
        idempotency_key="stale-second",
    )
    assert stale_request is not None
    assert stale_request["status"] == TURN_STATUS_ABANDONED
    assert stale_request["error"] == {"code": ERROR_STALE_SCENE_VERSION}


@pytest.mark.asyncio
async def test_duplicate_in_flight_turn_key_replays_conflict(
    service: VNPlayService,
) -> None:
    session = _ready_session(service)
    input_payload = {"input_text": "Hello"}
    turn_request = service.repo.create_turn_request(
        session_id=session.id,
        owner_user_id=42,
        idempotency_key="in-flight",
        request_payload_hash=_turn_hash(session.id, input_payload),
        base_scene_version=0,
        status=TURN_STATUS_MODEL_CALLING,
    )
    service.repo.update_session(
        session.id,
        {"active_turn_request_id": int(turn_request["id"])},
        owner_user_id=42,
    )

    with pytest.raises(VNPlayConflictError, match=ERROR_TURN_IN_PROGRESS):
        await service.submit_turn(
            session.id,
            input_text="Hello",
            client_scene_version=0,
            idempotency_key="in-flight",
        )


@pytest.mark.asyncio
async def test_abandoned_turn_request_replays_stable_error(
    service: VNPlayService,
) -> None:
    session = _ready_session(service)
    request_hash = _turn_hash(session.id, {"input_text": "Hello"})
    turn_request = service.repo.create_turn_request(
        session_id=session.id,
        owner_user_id=42,
        idempotency_key="abandoned",
        request_payload_hash=request_hash,
        base_scene_version=0,
        status=TURN_STATUS_ABANDONED,
    )
    service.repo.update_turn_request(
        int(turn_request["id"]),
        {
            "error": {"code": ERROR_ACTION_REQUEST_ABANDONED},
        },
        owner_user_id=42,
    )

    with pytest.raises(VNPlayTurnError, match=ERROR_ACTION_REQUEST_ABANDONED):
        await service.submit_turn(
            session.id,
            input_text="Hello",
            client_scene_version=0,
            idempotency_key="abandoned",
        )


@pytest.mark.asyncio
async def test_completed_turn_request_replays_stored_response(
    service: VNPlayService,
) -> None:
    session = _ready_session(service)
    first = await service.submit_turn(
        session.id,
        input_text="Hello",
        client_scene_version=0,
        idempotency_key="completed",
    )

    second = await service.submit_turn(
        session.id,
        input_text="Hello",
        client_scene_version=0,
        idempotency_key="completed",
    )

    assert first.status == TURN_STATUS_COMPLETED
    assert second.turn_request_id == first.turn_request_id
    assert second.status == first.status
    assert second.events == first.events


@pytest.mark.asyncio
async def test_failed_turn_request_replays_stable_failure_error(
    failing_service: VNPlayService,
) -> None:
    session = _ready_session(failing_service)

    with pytest.raises(VNPlayTurnError, match=TURN_STATUS_MODEL_FAILED):
        await failing_service.submit_turn(
            session.id,
            input_text="Break",
            client_scene_version=0,
            idempotency_key="failed",
        )

    with pytest.raises(VNPlayTurnError, match=TURN_STATUS_MODEL_FAILED):
        await failing_service.submit_turn(
            session.id,
            input_text="Break",
            client_scene_version=0,
            idempotency_key="failed",
        )


def test_duplicate_active_session_action_key_does_not_abandon_original(
    service: VNPlayService,
) -> None:
    session = _ready_session(service)
    action = service.repo.create_session_action(
        session_id=session.id,
        owner_user_id=42,
        action_type="save_slot_restore",
        idempotency_key="active-action",
        request_payload_hash=_payload_hash({"action_type": "save_slot_restore"}),
    )
    assert service.repo.try_acquire_session_action_lock(
        session_id=session.id,
        owner_user_id=42,
        action_id=int(action["id"]),
        expected_scene_version=0,
    )

    with pytest.raises(VNPlayConflictError, match=ERROR_RESTORE_ACTION_IN_PROGRESS):
        service._validate_restore_can_start(
            session_id=session.id,
            action_id=int(action["id"]),
            expected_scene_version=0,
        )

    stored_action = service.repo.get_session_action(int(action["id"]), owner_user_id=42)
    stored_session = service.get_session(session.id)
    assert stored_action is not None
    assert stored_action["status"] == SESSION_ACTION_STATUS_PENDING
    assert stored_session.active_session_action_id == int(action["id"])
