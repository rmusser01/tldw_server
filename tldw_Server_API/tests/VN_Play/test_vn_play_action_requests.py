from collections.abc import Generator, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPlay_DB import VNPlayRepository
from tldw_Server_API.app.core.VN_Play.adapters import (
    VNGenerationCallRequest,
    VNGenerationCallResult,
)
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
    MODE_SCRIPTED_STORY,
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


class CountingGenerationAdapter:
    def __init__(self) -> None:
        self.calls: list[VNGenerationCallRequest] = []

    async def generate(self, request: VNGenerationCallRequest) -> VNGenerationCallResult:
        self.calls.append(request)
        return VNGenerationCallResult(
            raw_content='{"schema":"narrative_dialogue","narrative":[{"text":"Generated line."}]}',
            usage_metadata={"total_tokens": 12},
            response_metadata={"model": "fake-model"},
        )


def _scripted_session(service: VNPlayService):
    row = service.repo.create_session(
        owner_user_id=42,
        mode=MODE_SCRIPTED_STORY,
        title="Scripted library",
        primary_character_id=1,
        vn_asset_pack_id=10,
        content_rating="general",
        seed="seed-1",
        script_id=100,
        script_version_id=200,
        script_manifest_snapshot_id=300,
        script_policy_snapshot_id=400,
        script_generation_profile_snapshot_id=500,
        script_position={"label": "start", "index": 0},
    )
    return service.get_session(int(row["id"]))


def _generation_call_kwargs(session_id: int, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "client_scene_version": 0,
        "idempotency_key": "generate-key",
        "generation_point_key": "start:0:generate",
        "output_schema": "narrative_dialogue",
        "generation_profile_key": "default",
        "generation_profile_snapshot_id": 500,
        "profile_snapshot": {
            "definition": {
                "provider": "fake",
                "model": "fake-model",
                "moderation_required": False,
            }
        },
        "messages": [{"role": "user", "content": "Write the next beat."}],
        "request_kind": "automatic",
        "opcode_snapshot": {"type": "generate", "output_schema": "narrative_dialogue"},
        "prompt_fingerprint": "prompt-1",
    }
    payload.update(overrides)
    return payload


def _generation_call_hash(service: VNPlayService, kwargs: Mapping[str, Any]) -> str:
    return service._generation_call_payload_hash(
        action_kind="execute",
        client_scene_version=int(kwargs["client_scene_version"]),
        generation_point_key=str(kwargs["generation_point_key"]),
        output_schema=str(kwargs["output_schema"]),
        generation_profile_key=str(kwargs["generation_profile_key"]),
        generation_profile_snapshot_id=int(kwargs["generation_profile_snapshot_id"]),
        request_kind=str(kwargs["request_kind"]),
        opcode_snapshot=kwargs.get("opcode_snapshot"),
        prompt_fingerprint=kwargs.get("prompt_fingerprint"),
        messages=kwargs.get("messages"),
    )


def _create_existing_generation_action(
    service: VNPlayService,
    *,
    session_id: int,
    idempotency_key: str,
    request_payload_hash: str,
    request_status: str,
    action_status: str,
    provider_call_started_at: str | None = None,
    lease_expires_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    generation = service.repo.get_or_create_generation(
        session_id=session_id,
        owner_user_id=42,
        generation_point_key="start:0:generate",
        output_schema="narrative_dialogue",
        generation_profile_key="default",
        generation_profile_snapshot_id=500,
        script_id=100,
        script_version_id=200,
        status="in_progress",
    )
    request = service.repo.create_generation_request(
        session_id=session_id,
        owner_user_id=42,
        generation_id=int(generation["id"]),
        request_kind="automatic",
        client_scene_version=0,
        status=request_status,
        opcode_snapshot={"type": "generate", "output_schema": "narrative_dialogue"},
        prompt_fingerprint="prompt-1",
    )
    request_updates: dict[str, Any] = {}
    if provider_call_started_at is not None:
        request_updates["provider_call_started_at"] = provider_call_started_at
    if lease_expires_at is not None:
        request_updates["lease_expires_at"] = lease_expires_at
    if request_updates:
        updated = service.repo.update_generation_request(
            int(request["id"]),
            request_updates,
            owner_user_id=42,
        )
        assert updated is not None
        request = updated
    action = service.repo.create_generation_action(
        session_id=session_id,
        owner_user_id=42,
        action_kind="execute",
        idempotency_key=idempotency_key,
        request_payload_hash=request_payload_hash,
        generation_id=int(generation["id"]),
        generation_request_id=int(request["id"]),
        status=action_status,
    )
    service.repo.update_generation_request(
        int(request["id"]),
        {"execute_action_id": int(action["id"])},
        owner_user_id=42,
    )
    return generation, request, action


@pytest.mark.asyncio
async def test_duplicate_generation_request_in_progress_does_not_call_provider(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = CountingGenerationAdapter()
    service = VNPlayService(
        repo=VNPlayRepository.initialized(chacha_db),
        owner_user_id=42,
        adapter=DeterministicVNPlayTurnAdapter(),
        generation_adapter=adapter,
    )
    session = _scripted_session(service)
    kwargs = _generation_call_kwargs(session.id, idempotency_key="in-progress")
    request_hash = _generation_call_hash(service, kwargs)
    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    _create_existing_generation_action(
        service,
        session_id=session.id,
        idempotency_key="in-progress",
        request_payload_hash=request_hash,
        request_status="in_progress",
        action_status="in_progress",
        provider_call_started_at=datetime.now(timezone.utc).isoformat(),
        lease_expires_at=future,
    )

    with pytest.raises(VNPlayConflictError, match="generation_request_in_progress"):
        await service.execute_script_generation_call(**kwargs)

    assert adapter.calls == []


@pytest.mark.asyncio
async def test_generation_request_reclaims_same_key_before_provider_start(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = CountingGenerationAdapter()
    service = VNPlayService(
        repo=VNPlayRepository.initialized(chacha_db),
        owner_user_id=42,
        adapter=DeterministicVNPlayTurnAdapter(),
        generation_adapter=adapter,
    )
    session = _scripted_session(service)
    kwargs = _generation_call_kwargs(session.id, idempotency_key="reclaim")
    request_hash = _generation_call_hash(service, kwargs)
    _create_existing_generation_action(
        service,
        session_id=session.id,
        idempotency_key="reclaim",
        request_payload_hash=request_hash,
        request_status="in_progress",
        action_status="in_progress",
    )

    response = await service.execute_script_generation_call(**kwargs)

    assert response["status"] == "completed"
    assert response["replayed"] is False
    assert response["public_output"]["narrative"][0]["text"] == "Generated line."
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_generation_request_stale_provider_lease_is_abandoned(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = CountingGenerationAdapter()
    service = VNPlayService(
        repo=VNPlayRepository.initialized(chacha_db),
        owner_user_id=42,
        adapter=DeterministicVNPlayTurnAdapter(),
        generation_adapter=adapter,
    )
    session = _scripted_session(service)
    kwargs = _generation_call_kwargs(session.id, idempotency_key="stale-lease")
    request_hash = _generation_call_hash(service, kwargs)
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    _, request, action = _create_existing_generation_action(
        service,
        session_id=session.id,
        idempotency_key="stale-lease",
        request_payload_hash=request_hash,
        request_status="in_progress",
        action_status="in_progress",
        provider_call_started_at=(datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
        lease_expires_at=past,
    )

    with pytest.raises(VNPlayConflictError, match="generation_attempt_abandoned"):
        await service.execute_script_generation_call(**kwargs)

    abandoned_request = service.repo.get_generation_request(int(request["id"]), owner_user_id=42)
    abandoned_action = service.repo.get_generation_action(int(action["id"]), owner_user_id=42)
    assert abandoned_request is not None
    assert abandoned_action is not None
    assert abandoned_request["status"] == "abandoned"
    assert abandoned_request["public_error_code"] == "generation_attempt_abandoned"
    assert abandoned_action["status"] == "abandoned"
    assert abandoned_action["public_error_code"] == "generation_attempt_abandoned"
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_completed_generation_request_replays_stored_response_without_provider_call(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = CountingGenerationAdapter()
    service = VNPlayService(
        repo=VNPlayRepository.initialized(chacha_db),
        owner_user_id=42,
        adapter=DeterministicVNPlayTurnAdapter(),
        generation_adapter=adapter,
    )
    session = _scripted_session(service)
    kwargs = _generation_call_kwargs(session.id, idempotency_key="completed")
    first = await service.execute_script_generation_call(**kwargs)

    service.repo.update_session(session.id, {"scene_version": 1}, owner_user_id=42)
    adapter.calls.clear()
    replayed = await service.execute_script_generation_call(**kwargs)

    assert replayed["replayed"] is True
    assert replayed["generation_revision_id"] == first["generation_revision_id"]
    assert replayed["public_output"] == first["public_output"]
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_stale_generation_scene_version_rejects_before_provider_call(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = CountingGenerationAdapter()
    service = VNPlayService(
        repo=VNPlayRepository.initialized(chacha_db),
        owner_user_id=42,
        adapter=DeterministicVNPlayTurnAdapter(),
        generation_adapter=adapter,
    )
    session = _scripted_session(service)
    service.repo.update_session(session.id, {"scene_version": 2}, owner_user_id=42)

    with pytest.raises(VNPlayConflictError, match=ERROR_STALE_SCENE_VERSION):
        await service.execute_script_generation_call(
            **_generation_call_kwargs(session.id, client_scene_version=1)
        )

    assert adapter.calls == []


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
