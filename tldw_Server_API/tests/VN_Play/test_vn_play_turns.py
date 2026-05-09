from collections.abc import Generator

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.DB_Management.VNPlay_DB import VNPlayRepository
from tldw_Server_API.app.core.VN_Play import adapters as vn_play_adapters
from tldw_Server_API.app.core.VN_Play.models import SceneState, TurnResult
from tldw_Server_API.app.core.VN_Play.parser import VNPlayParseError, parse_model_turn
from tldw_Server_API.app.core.VN_Play.service import (
    DeterministicVNPlayTurnAdapter,
    VNPlayConflictError,
    VNPlayService,
    VNPlaySession,
    VNPlayTurnContext,
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


def make_turn_context(mode: str = "freeform") -> VNPlayTurnContext:
    return VNPlayTurnContext(
        session=VNPlaySession(
            id=1,
            owner_user_id=42,
            mode=mode,
            title="Library night",
            status="active",
            primary_character_id=1,
            additional_character_ids=[],
            linked_chat_id=None,
            vn_asset_pack_id=10,
            asset_manifest_version=None,
            source_world_book_ids=[],
            content_rating="general",
            trust_level="local",
            linked_chat_mode="read_only_context",
            seed="seed-1",
            settings={},
            scene_version=0,
            active_turn_request_id=None,
        ),
        input_payload={"input_text": "Hello"},
        scene_state=SceneState(location_key="library", mood="quiet"),
        recent_events=[],
        turn_request_id=1,
    )


class VisualDirectiveAdapter:
    async def generate_turn(self, context):
        return TurnResult(
            narrative_text="The library appears.",
            dialogue=[{"speaker": "Narrator", "text": "The library appears."}],
            visual_directives=[
                {"asset_type": "background", "labels": {"location": "library"}},
                {"asset_type": "sprite", "labels": {"emotion": "happy"}},
            ],
            scene_updates={"location_key": "library"},
        )


class MissingVisualDirectiveAdapter:
    async def generate_turn(self, context):
        return TurnResult(
            narrative_text="The library remains quiet.",
            dialogue=[{"speaker": "Narrator", "text": "The library remains quiet."}],
            visual_directives=[
                {"asset_type": "sprite", "labels": {"emotion": "angry"}},
            ],
            scene_updates={"location_key": "library"},
        )


def create_visual_pack(chacha_db: CharactersRAGDB) -> tuple[int, int, int, int]:
    character_id = chacha_db.add_character_card(
        {
            "name": "Mira",
            "description": "A careful archivist.",
            "personality": "Patient and exacting.",
            "scenario": "Cataloging an orbital library.",
        }
    )
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(
        owner_user_id=42,
        primary_character_id=int(character_id),
        title="Mira - Archive Pack",
    )
    background_slot = repo.create_slot(
        pack_id=int(pack["id"]),
        asset_type="background",
        slot_key="background.library",
        labels={"location": "library"},
    )
    sprite_slot = repo.create_slot(
        pack_id=int(pack["id"]),
        asset_type="sprite",
        slot_key="sprite.happy",
        labels={"emotion": "happy"},
    )
    background_item = repo.create_item(
        pack_id=int(pack["id"]),
        slot_id=int(background_slot["id"]),
        generated_file_id=1001,
        mime_type="image/png",
        review_status="approved",
    )
    sprite_item = repo.create_item(
        pack_id=int(pack["id"]),
        slot_id=int(sprite_slot["id"]),
        generated_file_id=1002,
        mime_type="image/png",
        review_status="approved",
    )
    return (
        int(character_id),
        int(pack["id"]),
        int(background_item["id"]),
        int(sprite_item["id"]),
    )


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
async def test_turn_applies_visual_directives_to_scene_state(
    chacha_db: CharactersRAGDB,
) -> None:
    character_id, pack_id, background_item_id, sprite_item_id = create_visual_pack(chacha_db)
    service = VNPlayService(
        repo=VNPlayRepository.initialized(chacha_db),
        owner_user_id=42,
        adapter=VisualDirectiveAdapter(),
    )
    session = service.create_session(
        mode="freeform",
        title="Library night",
        primary_character_id=character_id,
        vn_asset_pack_id=pack_id,
        seed="seed-1",
    )

    first = await service.submit_turn(
        session.id,
        input_text="Look around",
        client_scene_version=0,
        idempotency_key="visual-turn-1",
    )

    event_types = [event["event_type"] for event in first.events]
    assert "visual_directive_requested" in event_types
    assert event_types.count("visual_directive_applied") == 2
    state = service.repo.get_scene_state(session.id, owner_user_id=42)
    assert state is not None
    assert state["current_background_item_id"] == background_item_id
    assert state["active_sprite_items"][0]["item_id"] == sprite_item_id

    second = await service.submit_turn(
        session.id,
        input_text="Look around",
        client_scene_version=0,
        idempotency_key="visual-turn-1",
    )
    assert second.events == first.events


@pytest.mark.asyncio
async def test_turn_rejects_unresolved_visual_directive_without_failing_text_turn(
    chacha_db: CharactersRAGDB,
) -> None:
    character_id, pack_id, _, _ = create_visual_pack(chacha_db)
    service = VNPlayService(
        repo=VNPlayRepository.initialized(chacha_db),
        owner_user_id=42,
        adapter=MissingVisualDirectiveAdapter(),
    )
    session = service.create_session(
        mode="freeform",
        title="Library night",
        primary_character_id=character_id,
        vn_asset_pack_id=pack_id,
        seed="seed-1",
    )

    response = await service.submit_turn(
        session.id,
        input_text="Look around",
        client_scene_version=0,
        idempotency_key="missing-visual-turn-1",
    )

    event_types = [event["event_type"] for event in response.events]
    assert response.status == "completed"
    assert "model_turn" in event_types
    assert "visual_directive_rejected" in event_types
    assert response.warnings[0]["reason"] == "asset_not_found"
    state = service.repo.get_scene_state(session.id, owner_user_id=42)
    assert state is not None
    assert state["warnings"][0]["reason"] == "asset_not_found"
    assert len(state["warnings"]) == 1
    assert service.get_session(session.id).active_turn_request_id is None


@pytest.mark.asyncio
async def test_turn_records_resolver_error_without_failing_text_turn(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character_id, pack_id, _, _ = create_visual_pack(chacha_db)
    service = VNPlayService(
        repo=VNPlayRepository.initialized(chacha_db),
        owner_user_id=42,
        adapter=VisualDirectiveAdapter(),
    )
    session = service.create_session(
        mode="freeform",
        title="Library night",
        primary_character_id=character_id,
        vn_asset_pack_id=pack_id,
        seed="seed-1",
    )

    def fail_resolution(*args, **kwargs):
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.VN_Play.service.resolve_scene_directives",
        fail_resolution,
    )

    response = await service.submit_turn(
        session.id,
        input_text="Look around",
        client_scene_version=0,
        idempotency_key="resolver-error-turn-1",
    )

    rejected_events = [
        event
        for event in response.events
        if event["event_type"] == "visual_directive_rejected"
    ]
    assert response.status == "completed"
    assert len(rejected_events) == 2
    assert rejected_events[0]["event_payload"]["reason"] == "resolver_error"
    assert rejected_events[0]["event_payload"]["directive"]["asset_type"] == "background"
    assert response.warnings[0]["error"] == "resolver exploded"
    assert service.get_session(session.id).active_turn_request_id is None


@pytest.mark.asyncio
async def test_scene_enrichment_omits_sprites_missing_from_approved_manifest(
    chacha_db: CharactersRAGDB,
) -> None:
    character_id, pack_id, _, sprite_item_id = create_visual_pack(chacha_db)
    service = VNPlayService(
        repo=VNPlayRepository.initialized(chacha_db),
        owner_user_id=42,
        adapter=VisualDirectiveAdapter(),
    )
    session = service.create_session(
        mode="freeform",
        title="Library night",
        primary_character_id=character_id,
        vn_asset_pack_id=pack_id,
        seed="seed-1",
    )
    await service.submit_turn(
        session.id,
        input_text="Look around",
        client_scene_version=0,
        idempotency_key="approved-visual-turn-1",
    )

    VNAssetPacksRepository.initialized(chacha_db).update_item_review(
        sprite_item_id,
        review_status="rejected",
    )

    state = service.get_enriched_scene_state(session.id)

    assert state is not None
    assert state["active_sprite_items"][0]["item_id"] == sprite_item_id
    assert state["active_sprites"] == []


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


def test_parse_structured_turn_result() -> None:
    result = parse_model_turn(
        {
            "narration": "The library lights flicker.",
            "dialogue": [{"speaker": "Mira", "text": "Stay close."}],
            "scene_directives": {"background": {"labels": {"location": "library"}}},
            "choices": [
                {"id": "choice-1", "text": "Inspect the shelves"},
                {"id": "choice-2", "text": "Call out softly"},
            ],
            "summary": "Mira enters the library.",
        },
        mode="story",
    )

    assert result.narration.startswith("The library")
    assert result.choices[0].text == "Inspect the shelves"
    assert result.scene_updates["location_key"] == "library"


def test_story_parser_requires_two_to_five_choices() -> None:
    with pytest.raises(VNPlayParseError):
        parse_model_turn({"narration": "No choice", "choices": []}, mode="story")


@pytest.mark.asyncio
async def test_chat_adapter_calls_existing_chat_service(monkeypatch) -> None:
    captured = {}

    async def fake_chat_call(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"narration":"Hi",'
                            '"dialogue":[{"speaker":"Mira","text":"Hello."}],'
                            '"choices":[{"id":"a","text":"A"},{"id":"b","text":"B"}],'
                            '"summary":"Greeting"}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(vn_play_adapters, "perform_chat_api_call_async", fake_chat_call)
    adapter = vn_play_adapters.ChatVNPlayTurnAdapter(provider="openai", model="gpt-test")

    result = await adapter.generate_turn(context=make_turn_context(mode="freeform"))

    assert result.dialogue[0].text == "Hello."
    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-test"
    assert captured["stream"] is False
