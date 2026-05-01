from collections.abc import Generator

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPlay_DB import VNPlayRepository
from tldw_Server_API.app.core.VN_Play.service import (
    DeterministicVNPlayTurnAdapter,
    VNPlayConflictError,
    VNPlayService,
    VNPlayTurnError,
)


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-play-turn-test-client")
    yield database
    database.close_connection()


@pytest.fixture
def service(chacha_db: CharactersRAGDB) -> VNPlayService:
    repo = VNPlayRepository.initialized(chacha_db)
    return VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=DeterministicVNPlayTurnAdapter(),
    )


@pytest.fixture
def service_with_failing_adapter(chacha_db: CharactersRAGDB) -> VNPlayService:
    class FailingAdapter:
        async def generate_turn(self, context):
            raise RuntimeError("provider unavailable")

    repo = VNPlayRepository.initialized(chacha_db)
    return VNPlayService(repo=repo, owner_user_id=42, adapter=FailingAdapter())


@pytest.fixture
def ready_session(service: VNPlayService):
    return service.create_session(
        mode="freeform",
        title="Library night",
        primary_character_id=1,
        vn_asset_pack_id=10,
        content_rating="general",
        seed="seed-1",
        settings={},
    )


@pytest.fixture
def failing_ready_session(service_with_failing_adapter: VNPlayService):
    return service_with_failing_adapter.create_session(
        mode="freeform",
        title="Library night",
        primary_character_id=1,
        vn_asset_pack_id=10,
        content_rating="general",
        seed="seed-1",
        settings={},
    )


@pytest.mark.asyncio
async def test_duplicate_completed_turn_returns_stored_response(
    service: VNPlayService,
    ready_session,
) -> None:
    first = await service.submit_turn(
        ready_session.id,
        input_text="Hello",
        client_scene_version=0,
        idempotency_key="turn-1",
    )
    second = await service.submit_turn(
        ready_session.id,
        input_text="Hello",
        client_scene_version=0,
        idempotency_key="turn-1",
    )

    assert second.turn_request_id == first.turn_request_id
    assert second.events == first.events


@pytest.mark.asyncio
async def test_same_idempotency_key_different_payload_conflicts(
    service: VNPlayService,
    ready_session,
) -> None:
    await service.submit_turn(
        ready_session.id,
        input_text="Hello",
        client_scene_version=0,
        idempotency_key="turn-1",
    )

    with pytest.raises(VNPlayConflictError, match="idempotency_key_conflict"):
        await service.submit_turn(
            ready_session.id,
            input_text="Different",
            client_scene_version=0,
            idempotency_key="turn-1",
        )


@pytest.mark.asyncio
async def test_stale_scene_version_conflicts(
    service: VNPlayService,
    ready_session,
) -> None:
    await service.submit_turn(
        ready_session.id,
        input_text="First",
        client_scene_version=0,
        idempotency_key="first",
    )

    with pytest.raises(VNPlayConflictError, match="stale_scene_version"):
        await service.submit_turn(
            ready_session.id,
            input_text="Second",
            client_scene_version=0,
            idempotency_key="second",
        )


@pytest.mark.asyncio
async def test_model_failure_marks_turn_failed_and_clears_lock(
    service_with_failing_adapter: VNPlayService,
    failing_ready_session,
) -> None:
    with pytest.raises(VNPlayTurnError):
        await service_with_failing_adapter.submit_turn(
            failing_ready_session.id,
            input_text="Break",
            client_scene_version=0,
            idempotency_key="fail-1",
        )

    session = service_with_failing_adapter.get_session(failing_ready_session.id)
    assert session.active_turn_request_id is None
