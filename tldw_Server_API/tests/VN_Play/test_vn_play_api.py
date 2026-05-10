import json
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
    VNAssetReadinessResponse,
    VNAssetReviewRequest,
)
from tldw_Server_API.app.api.v1.schemas.vn_play_schemas import (
    VNPlaySessionCreate,
    VNPlayTurnRequest,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.DB_Management.VNPlay_DB import VNPlayRepository
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Play.models import TurnResult
from tldw_Server_API.app.core.VN_Play.service import VNPlayService, VNPlayTurnContext
from tldw_Server_API.app.core.VN_Scripts.service import VNScriptService


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


def _publish_script_version(
    chacha_db: CharactersRAGDB,
    *,
    owner_user_id: int,
    pack_id: int,
    title: str = "Archive Door Script",
    program: dict[str, Any] | None = None,
) -> dict[str, Any]:
    service = VNScriptService(chacha_db, owner_user_id=owner_user_id)
    script = service.create_script(
        title=title,
        primary_asset_pack_id=pack_id,
        content_rating="general",
    )
    program = program or {
        "schema_version": "vn_script_program.v1",
        "entry_label": "start",
        "primary_asset_pack_id": pack_id,
        "variables": {},
        "labels": {
            "start": [
                {"op": "narrate", "text": "The archive door hums."},
                {"op": "end"},
            ]
        },
    }
    draft = service.get_draft(int(script["id"]))
    service.replace_draft(
        int(script["id"]),
        if_revision=int(draft["revision"]),
        draft=program,
    )
    return service.publish_script(
        int(script["id"]),
        draft_revision=int(draft["revision"]) + 1,
        label="v1",
        idempotency_key=f"publish-{script['id']}",
        acknowledgements=["character_safety_missing"],
    )


class _VisualDirectiveAdapter:
    async def generate_turn(self, context: VNPlayTurnContext) -> TurnResult:
        return TurnResult(
            narrative_text="The library appears.",
            dialogue=[{"speaker": "Narrator", "text": "The library appears."}],
            visual_directives=[
                {"asset_type": "background", "labels": {"location": "library"}},
                {"asset_type": "sprite", "labels": {"emotion": "happy"}},
            ],
            scene_updates={"location_key": "library"},
        )


def _create_visual_pack(chacha_db: CharactersRAGDB) -> tuple[int, int, int, int]:
    character_id = _add_character(chacha_db, name="Visual Mira")
    repo = VNAssetPacksRepository.initialized(chacha_db)
    pack = repo.create_pack(
        owner_user_id=42,
        primary_character_id=character_id,
        title="Visual Mira - Archive Pack",
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
        generated_file_id=2001,
        mime_type="image/png",
        review_status="approved",
    )
    sprite_item = repo.create_item(
        pack_id=int(pack["id"]),
        slot_id=int(sprite_slot["id"]),
        generated_file_id=2002,
        mime_type="image/png",
        review_status="approved",
    )
    return (
        character_id,
        int(pack["id"]),
        int(background_item["id"]),
        int(sprite_item["id"]),
    )


def _create_story_session_with_choice(
    chacha_db: CharactersRAGDB,
    *,
    character_id: int,
    pack_id: int,
) -> int:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(repo=repo, owner_user_id=42)
    session = service.create_session(
        mode="story",
        title="Archive Door",
        primary_character_id=character_id,
        vn_asset_pack_id=pack_id,
        content_rating="general",
        seed="story-seed",
    )
    choice_presented = repo.append_event(
        session_id=session.id,
        owner_user_id=42,
        event_type="choice_presented",
        event_payload={
            "choices": [{"id": "open", "text": "Open the door"}],
            "scene_version": 1,
        },
        source="runtime",
    )
    repo.set_scene_state(
        session_id=session.id,
        owner_user_id=42,
        last_event_id=int(choice_presented["id"]),
        visible_choices=[{"id": "open", "text": "Open the door"}],
        scene_version=1,
    )
    repo.update_session(session.id, {"scene_version": 1}, owner_user_id=42)
    return session.id


def _create_story_session_with_completed_branch(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    *,
    character_id: int,
    pack_id: int,
    idempotency_key: str,
) -> tuple[int, int]:
    story_session_id = _create_story_session_with_choice(
        chacha_db,
        character_id=character_id,
        pack_id=pack_id,
    )
    turn_response = client.post(
        f"/api/v1/vn-play/sessions/{story_session_id}/turn",
        json={
            "choice_id": "open",
            "client_scene_version": 1,
            "idempotency_key": idempotency_key,
        },
    )
    assert turn_response.status_code == 200
    branch_id = int(turn_response.json()["current_scene"]["active_branch_node_id"])
    return story_session_id, branch_id


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


def test_create_session_rejects_coerced_script_ids() -> None:
    with pytest.raises(ValidationError):
        VNPlaySessionCreate(
            mode="scripted_story",
            title="Test",
            primary_character_id=1,
            vn_asset_pack_id=2,
            script_id="1",
            script_version_id=1,
        )

    with pytest.raises(ValidationError):
        VNPlaySessionCreate(
            mode="scripted_story",
            title="Test",
            primary_character_id=1,
            vn_asset_pack_id=2,
            script_id=1,
            script_version_id="1",
        )


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


def test_story_choice_turn_returns_branch_state(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    ready_pack_id: int,
) -> None:
    story_session_id = _create_story_session_with_choice(
        chacha_db,
        character_id=character_id,
        pack_id=ready_pack_id,
    )

    response = client.post(
        f"/api/v1/vn-play/sessions/{story_session_id}/turn",
        json={
            "choice_id": "open",
            "client_scene_version": 1,
            "idempotency_key": "api-story-choice-1",
        },
    )

    assert response.status_code == 200
    body = response.json()
    active_branch_node_id = body["current_scene"]["active_branch_node_id"]
    assert body["status"] == "completed"
    assert body["scene_version"] == 2
    assert isinstance(active_branch_node_id, int)
    choice_selected = next(
        event for event in body["events"] if event["event_type"] == "choice_selected"
    )
    assert choice_selected["branch_node_id"] == active_branch_node_id
    assert body["current_scene"]["visible_choices"] == []


def test_story_start_endpoint_starts_model_story_session(
    client: TestClient,
    character_id: int,
    ready_pack_id: int,
) -> None:
    session_response = client.post(
        "/api/v1/vn-play/sessions",
        json={
            "mode": "story",
            "title": "Model story",
            "primary_character_id": character_id,
            "vn_asset_pack_id": ready_pack_id,
        },
    )
    assert session_response.status_code == 201
    session_id = int(session_response.json()["id"])

    response = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/story/start",
        json={"client_scene_version": 0, "idempotency_key": "api-story-start"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["scene_version"] == 1
    assert body["session"]["mode"] == "story"


def test_story_unknown_choice_returns_invalid_choice_id(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    ready_pack_id: int,
) -> None:
    story_session_id = _create_story_session_with_choice(
        chacha_db,
        character_id=character_id,
        pack_id=ready_pack_id,
    )

    response = client.post(
        f"/api/v1/vn-play/sessions/{story_session_id}/turn",
        json={
            "choice_id": "unknown",
            "client_scene_version": 1,
            "idempotency_key": "api-story-choice-unknown",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_choice_id"


def test_story_retry_completed_turn_returns_not_failed(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    ready_pack_id: int,
) -> None:
    story_session_id = _create_story_session_with_choice(
        chacha_db,
        character_id=character_id,
        pack_id=ready_pack_id,
    )
    turn_response = client.post(
        f"/api/v1/vn-play/sessions/{story_session_id}/turn",
        json={
            "choice_id": "open",
            "client_scene_version": 1,
            "idempotency_key": "api-story-choice-completed",
        },
    )
    assert turn_response.status_code == 200

    response = client.post(
        f"/api/v1/vn-play/sessions/{story_session_id}/retry-last-turn",
        json={
            "client_scene_version": 2,
            "idempotency_key": "api-story-retry-completed",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "retry_last_turn_not_failed"


def test_branch_list_keeps_branch_path_list_shape(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    ready_pack_id: int,
) -> None:
    story_session_id = _create_story_session_with_choice(
        chacha_db,
        character_id=character_id,
        pack_id=ready_pack_id,
    )
    turn_response = client.post(
        f"/api/v1/vn-play/sessions/{story_session_id}/turn",
        json={
            "choice_id": "open",
            "client_scene_version": 1,
            "idempotency_key": "api-story-choice-branch-path",
        },
    )
    assert turn_response.status_code == 200

    response = client.get(f"/api/v1/vn-play/sessions/{story_session_id}/branches")

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body[0]["branch_path"], list)
    assert body[0]["branch_path"][0]["choice_id"] == "open"


def test_branch_navigation_returns_active_path_and_restore_capability(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    ready_pack_id: int,
) -> None:
    story_session_id, branch_id = _create_story_session_with_completed_branch(
        client,
        chacha_db,
        character_id=character_id,
        pack_id=ready_pack_id,
        idempotency_key="api-branch-navigation-choice",
    )

    response = client.get(
        f"/api/v1/vn-play/sessions/{story_session_id}/branch-navigation"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == story_session_id
    assert body["active_branch_node_id"] == branch_id
    assert body["active_path"] == [
        {
            "branch_id": branch_id,
            "branch_label": "Open the door",
            "choice_id": "open",
            "choice_text": "Open the door",
            "depth": 1,
        }
    ]
    branch = body["branches"][0]
    assert branch["branch_id"] == branch_id
    assert branch["is_active"] is True
    assert branch["is_on_active_path"] is True
    assert branch["restore"]["supported"] is True
    assert branch["restore"]["default_target"] == "branch_latest"
    assert branch["restore"]["targets"]["branch_latest"]["event_id"] is not None
    assert branch["restore"]["targets"]["choice_point"]["event_id"] is not None


def test_branch_events_filter_keeps_existing_list_response_shape(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    ready_pack_id: int,
) -> None:
    story_session_id, branch_id = _create_story_session_with_completed_branch(
        client,
        chacha_db,
        character_id=character_id,
        pack_id=ready_pack_id,
        idempotency_key="api-branch-events-choice",
    )

    response = client.get(
        f"/api/v1/vn-play/sessions/{story_session_id}/events",
        params={
            "branch_id": branch_id,
            "limit": 2,
            "include_descendants": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert 1 <= len(body) <= 2
    assert {event["branch_node_id"] for event in body} == {branch_id}


def test_unfiltered_events_preserve_legacy_unbounded_default(
    client: TestClient,
    session_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_limit: list[int | None] = []

    def list_events_with_metadata(
        self: VNPlayService,
        session_id: int,
        *,
        branch_id: int | None = None,
        after_sequence: int | None = None,
        limit: int | None = None,
        include_descendants: bool = False,
    ) -> dict[str, Any]:
        captured_limit.append(limit)
        return {
            "events": [
                {
                    "id": index,
                    "session_id": session_id,
                    "owner_user_id": self.owner_user_id,
                    "sequence_number": index,
                    "event_type": "model_turn",
                    "event_payload": {"text": f"Event {index}"},
                    "source": "model",
                    "created_at": "2026-05-09T00:00:00Z",
                }
                for index in range(1, 102)
            ],
            "warnings": [],
        }

    monkeypatch.setattr(
        VNPlayService,
        "list_events_with_metadata",
        list_events_with_metadata,
    )

    response = client.get(f"/api/v1/vn-play/sessions/{session_id}/events")

    assert response.status_code == 200
    assert captured_limit == [None]
    assert len(response.json()) == 101


def test_branch_events_default_to_bounded_limit(
    client: TestClient,
    session_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_limit: list[int | None] = []

    def list_events_with_metadata(
        self: VNPlayService,
        session_id: int,
        *,
        branch_id: int | None = None,
        after_sequence: int | None = None,
        limit: int | None = None,
        include_descendants: bool = False,
    ) -> dict[str, Any]:
        captured_limit.append(limit)
        return {"events": [], "warnings": []}

    monkeypatch.setattr(
        VNPlayService,
        "list_events_with_metadata",
        list_events_with_metadata,
    )

    response = client.get(
        f"/api/v1/vn-play/sessions/{session_id}/events",
        params={"branch_id": 7},
    )

    assert response.status_code == 200
    assert captured_limit == [100]


def test_branch_events_warning_header_uses_compact_json(
    client: TestClient,
    session_id: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning = {
        "code": "branch_interval_replay_limit_exceeded",
        "severity": "warning",
        "message": "Branch replay was capped.",
        "recoverable": True,
        "branch_id": 7,
    }

    def list_events_with_warning(
        self: VNPlayService,
        session_id: int,
        *,
        branch_id: int | None = None,
        after_sequence: int | None = None,
        limit: int | None = None,
        include_descendants: bool = False,
    ) -> dict[str, Any]:
        return {
            "events": [
                {
                    "id": 100,
                    "session_id": session_id,
                    "owner_user_id": self.owner_user_id,
                    "sequence_number": 4,
                    "event_type": "model_turn",
                    "event_payload": {"text": "Filtered"},
                    "source": "model",
                    "branch_node_id": branch_id,
                    "created_at": "2026-05-09T00:00:00Z",
                }
            ],
            "warnings": [warning],
        }

    monkeypatch.setattr(
        VNPlayService,
        "list_events_with_metadata",
        list_events_with_warning,
    )

    response = client.get(
        f"/api/v1/vn-play/sessions/{session_id}/events",
        params={
            "branch_id": 7,
            "after_sequence": 3,
            "limit": 2,
            "include_descendants": True,
        },
    )

    assert response.status_code == 200
    assert response.json()[0]["branch_node_id"] == 7
    assert json.loads(response.headers["X-VN-Play-Warnings"]) == [warning]
    assert ": " not in response.headers["X-VN-Play-Warnings"]
    assert ", " not in response.headers["X-VN-Play-Warnings"]


def test_branch_restore_returns_response_shape_and_replays_idempotently(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    ready_pack_id: int,
) -> None:
    story_session_id, branch_id = _create_story_session_with_completed_branch(
        client,
        chacha_db,
        character_id=character_id,
        pack_id=ready_pack_id,
        idempotency_key="api-branch-restore-choice",
    )
    payload = {
        "client_scene_version": 2,
        "idempotency_key": "api-branch-restore",
    }

    first = client.post(
        f"/api/v1/vn-play/sessions/{story_session_id}/branches/{branch_id}/restore",
        json=payload,
    )
    second = client.post(
        f"/api/v1/vn-play/sessions/{story_session_id}/branches/{branch_id}/restore",
        json=payload,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_body = first.json()
    second_body = second.json()
    assert first_body["status"] == "completed"
    assert first_body["replayed"] is False
    assert first_body["branch_id"] == branch_id
    assert first_body["target"] == "branch_latest"
    assert first_body["scene_version"] == 3
    assert first_body["session"]["id"] == story_session_id
    assert first_body["current_scene"]["scene_version"] == 3
    assert first_body["branch_navigation"]["active_branch_node_id"] == branch_id
    assert second_body == {**first_body, "replayed": True}


def test_branch_restore_stale_scene_version_returns_conflict(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    ready_pack_id: int,
) -> None:
    story_session_id, branch_id = _create_story_session_with_completed_branch(
        client,
        chacha_db,
        character_id=character_id,
        pack_id=ready_pack_id,
        idempotency_key="api-branch-restore-stale-choice",
    )

    response = client.post(
        f"/api/v1/vn-play/sessions/{story_session_id}/branches/{branch_id}/restore",
        json={
            "client_scene_version": 1,
            "idempotency_key": "api-branch-restore-stale",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "stale_scene_version"


def test_branch_restore_missing_branch_returns_branch_not_found(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
    ready_pack_id: int,
) -> None:
    story_session_id = _create_story_session_with_choice(
        chacha_db,
        character_id=character_id,
        pack_id=ready_pack_id,
    )

    response = client.post(
        f"/api/v1/vn-play/sessions/{story_session_id}/branches/9999/restore",
        json={
            "client_scene_version": 1,
            "idempotency_key": "api-branch-restore-missing",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "branch_not_found"


def test_checkpoint_restore_passes_idempotency_key_and_replays(
    client: TestClient,
    session_id: int,
) -> None:
    first_turn = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/turn",
        json={
            "input_text": "First",
            "client_scene_version": 0,
            "idempotency_key": "api-checkpoint-first-turn",
        },
    )
    assert first_turn.status_code == 200
    checkpoint = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/checkpoint",
        json={"label": "First"},
    )
    assert checkpoint.status_code == 200
    second_turn = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/turn",
        json={
            "input_text": "Second",
            "client_scene_version": 1,
            "idempotency_key": "api-checkpoint-second-turn",
        },
    )
    assert second_turn.status_code == 200
    restore_payload = {
        "checkpoint_id": checkpoint.json()["id"],
        "client_scene_version": second_turn.json()["scene_version"],
        "idempotency_key": "api-checkpoint-restore",
    }

    first_restore = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/restore",
        json=restore_payload,
    )
    assert first_restore.status_code == 200
    first_restore_body = first_restore.json()
    assert first_restore_body["scene_version"] == 3

    third_turn = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/turn",
        json={
            "input_text": "Third",
            "client_scene_version": first_restore_body["scene_version"],
            "idempotency_key": "api-checkpoint-third-turn",
        },
    )
    assert third_turn.status_code == 200
    assert third_turn.json()["scene_version"] == 4

    second_restore = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/restore",
        json=restore_payload,
    )

    assert second_restore.status_code == 200
    assert second_restore.json() == first_restore_body


def test_checkpoint_restore_rejects_stale_client_scene_version(
    client: TestClient,
    session_id: int,
) -> None:
    first_turn = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/turn",
        json={
            "input_text": "First",
            "client_scene_version": 0,
            "idempotency_key": "api-checkpoint-stale-first-turn",
        },
    )
    assert first_turn.status_code == 200
    checkpoint = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/checkpoint",
        json={"label": "First"},
    )
    assert checkpoint.status_code == 200
    second_turn = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/turn",
        json={
            "input_text": "Second",
            "client_scene_version": 1,
            "idempotency_key": "api-checkpoint-stale-second-turn",
        },
    )
    assert second_turn.status_code == 200

    stale_restore = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/restore",
        json={
            "checkpoint_id": checkpoint.json()["id"],
            "client_scene_version": 1,
            "idempotency_key": "api-checkpoint-stale-restore",
        },
    )

    assert stale_restore.status_code == 409
    assert stale_restore.json()["detail"]["code"] == "stale_scene_version"


@pytest.mark.asyncio
async def test_session_response_includes_resolved_scene_assets(
    client: TestClient,
    chacha_db: CharactersRAGDB,
) -> None:
    character_id, pack_id, background_item_id, sprite_item_id = _create_visual_pack(chacha_db)
    service = VNPlayService(
        repo=VNPlayRepository.initialized(chacha_db),
        owner_user_id=42,
        adapter=_VisualDirectiveAdapter(),
    )
    session = service.create_session(
        mode="freeform",
        title="Visual library",
        primary_character_id=character_id,
        vn_asset_pack_id=pack_id,
        seed="seed-1",
    )
    await service.submit_turn(
        session.id,
        input_text="Look around",
        client_scene_version=0,
        idempotency_key="visual-api-turn-1",
    )

    response = client.get(f"/api/v1/vn-play/sessions/{session.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["scene_state"]["background"]["item_id"] == background_item_id
    assert body["scene_state"]["background"]["content_url"].endswith(
        f"/items/{background_item_id}/content"
    )
    assert body["scene_state"]["active_sprites"][0]["item_id"] == sprite_item_id
    assert body["scene_state"]["active_sprites"][0]["content_url"].endswith(
        f"/items/{sprite_item_id}/content"
    )


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


def test_setup_options_returns_script_versions_for_scripted_story(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    pack_id = _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=character_id,
    )
    published = _publish_script_version(
        chacha_db,
        owner_user_id=42,
        pack_id=pack_id,
    )

    response = client.get(
        "/api/v1/vn-play/setup-options"
        f"?mode=scripted_story&selected_character_id={character_id}"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["defaults"]["mode"] == "scripted_story"
    assert body["defaults"]["script_version_id"] == published["version_id"]
    assert body["defaults"]["asset_pack_id"] == pack_id
    assert body["defaults"]["policy_profile_id"] == "local_default"
    assert body["defaults"]["generation_profile_id"] == "story_default"
    script_option = body["script_versions"][0]
    assert script_option["id"] == published["version_id"]
    assert script_option["asset_pack_id"] == pack_id
    assert script_option["ready"] is True
    assert script_option["recommended"] is False
    assert script_option["warning_summary"]["requires_acknowledgement"] is True
    warning_codes = {
        warning["code"]
        for warning in script_option["warning_summary"]["warnings"]
    }
    assert "character_safety_missing" in warning_codes


def test_setup_options_returns_script_empty_state_for_scripted_story(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/vn-play/setup-options?mode=scripted_story")

    assert response.status_code == 200
    empty_states = {state["code"]: state for state in response.json()["empty_states"]}
    assert empty_states["no_script_versions"]["scope"] == "global"


def test_scripted_story_session_pins_script_version_snapshots_and_policy_ack(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    pack_id = _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=character_id,
    )
    published = _publish_script_version(
        chacha_db,
        owner_user_id=42,
        pack_id=pack_id,
    )
    request_body = {
        "mode": "scripted_story",
        "title": "Scripted archive door",
        "primary_character_id": character_id,
        "vn_asset_pack_id": pack_id,
        "script_id": published["script_id"],
        "script_version_id": published["version_id"],
    }

    blocked = client.post("/api/v1/vn-play/sessions", json=request_body)

    assert blocked.status_code == 400
    assert blocked.json()["detail"]["code"] == "script_session_acknowledgement_required"

    response = client.post(
        "/api/v1/vn-play/sessions",
        json={**request_body, "acknowledgements": ["character_safety_missing"]},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["mode"] == "scripted_story"
    assert body["script_id"] == published["script_id"]
    assert body["script_version_id"] == published["version_id"]
    assert body["script_manifest_snapshot_id"] == published["manifest_snapshot_id"]
    assert body["script_policy_snapshot_id"] == published["policy_snapshot_id"]
    assert body["script_generation_profile_snapshot_id"] == published[
        "generation_profile_snapshot_id"
    ]
    assert "progress_token" in body["script_position"]
    assert "label" not in body["script_position"]


def test_scripted_story_session_rejects_request_metadata_mismatch(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    pack_id = _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=character_id,
    )
    published = _publish_script_version(
        chacha_db,
        owner_user_id=42,
        pack_id=pack_id,
    )
    other_character_id = _add_character(chacha_db, name="Other Mira")
    base_body = {
        "mode": "scripted_story",
        "title": "Scripted archive door",
        "primary_character_id": character_id,
        "vn_asset_pack_id": pack_id,
        "script_id": published["script_id"],
        "script_version_id": published["version_id"],
        "acknowledgements": ["character_safety_missing"],
    }

    character_mismatch = client.post(
        "/api/v1/vn-play/sessions",
        json={**base_body, "primary_character_id": other_character_id},
    )
    rating_mismatch = client.post(
        "/api/v1/vn-play/sessions",
        json={**base_body, "content_rating": "mature"},
    )

    assert character_mismatch.status_code == 400
    assert character_mismatch.json()["detail"]["code"] == "script_primary_character_mismatch"
    assert rating_mismatch.status_code == 400
    assert rating_mismatch.json()["detail"]["code"] == "script_content_rating_mismatch"


def test_scripted_story_runtime_starts_and_advances_visible_choice(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    pack_id = _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=character_id,
    )
    program = {
        "schema_version": "vn_script_program.v1",
        "entry_label": "start",
        "primary_asset_pack_id": pack_id,
        "variables": {
            "trust": {"type": "integer", "default": 0, "public": True},
            "secret": {"type": "string", "default": "sealed", "public": False},
            "roll": {"type": "integer", "default": 0, "public": True},
        },
        "labels": {
            "start": [
                {"op": "narrate", "text": "The archive door hums."},
                {"op": "say", "speaker": "Mira", "text": "Which way?"},
                {"op": "set", "var": "trust", "value": 1},
                {"op": "random", "id": "door-roll", "var": "roll", "min": 1, "max": 6},
                {
                    "op": "choice",
                    "id": "door",
                    "choices": [
                        {"id": "open", "text": "Open it", "target": "open"},
                        {"id": "wait", "text": "Wait", "target": "wait"},
                    ],
                },
            ],
            "open": [
                {"op": "narrate", "text": "The door slides open."},
                {"op": "end"},
            ],
            "wait": [
                {"op": "narrate", "text": "The door keeps humming."},
                {"op": "end"},
            ],
        },
    }
    published = _publish_script_version(
        chacha_db,
        owner_user_id=42,
        pack_id=pack_id,
        program=program,
    )
    session_response = client.post(
        "/api/v1/vn-play/sessions",
        json={
            "mode": "scripted_story",
            "title": "Scripted archive door",
            "primary_character_id": character_id,
            "vn_asset_pack_id": pack_id,
            "script_id": published["script_id"],
            "script_version_id": published["version_id"],
            "acknowledgements": ["character_safety_missing"],
        },
    )
    assert session_response.status_code == 201
    session_id = int(session_response.json()["id"])

    turn_bypass = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/turn",
        json={
            "input_text": "Bypass the script",
            "client_scene_version": 0,
            "idempotency_key": "script-turn-bypass",
        },
    )
    assert turn_bypass.status_code == 400
    assert turn_bypass.json()["detail"]["code"] == "scripted_story_turn_not_allowed"

    alias_session_response = client.post(
        "/api/v1/vn-play/sessions",
        json={
            "mode": "scripted_story",
            "title": "Scripted archive door alias",
            "primary_character_id": character_id,
            "vn_asset_pack_id": pack_id,
            "script_id": published["script_id"],
            "script_version_id": published["version_id"],
            "acknowledgements": ["character_safety_missing"],
        },
    )
    assert alias_session_response.status_code == 201
    alias_session_id = int(alias_session_response.json()["id"])
    alias_start = client.post(
        f"/api/v1/vn-play/sessions/{alias_session_id}/script/advance",
        json={"client_scene_version": 0, "idempotency_key": "start-script-alias"},
    )
    assert alias_start.status_code == 200
    assert alias_start.json()["script_state"]["waiting_choice"]["id"] == "door"
    assert "target" not in alias_start.json()["script_state"]["waiting_choice"]["choices"][0]

    start = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/story/start",
        json={"client_scene_version": 0, "idempotency_key": "start-script"},
    )

    assert start.status_code == 200
    start_body = start.json()
    assert start_body["scene_version"] == 1
    assert "progress_token" in start_body["script_state"]["position"]
    assert "label" not in start_body["script_state"]["position"]
    assert start_body["script_state"]["variables"]["trust"] == 1
    assert 1 <= start_body["script_state"]["variables"]["roll"] <= 6
    assert "secret" not in start_body["script_state"]["variables"]
    assert [
        choice["id"] for choice in start_body["current_scene"]["visible_choices"]
    ] == ["open", "wait"]
    assert "target" not in start_body["current_scene"]["visible_choices"][0]
    scene_changed = next(
        event for event in start_body["events"] if event["event_type"] == "scene_state_changed"
    )
    assert scene_changed["event_payload"]["random_results"][0]["id"] == "door-roll"

    state = client.get(f"/api/v1/vn-play/sessions/{session_id}/script/state")
    assert state.status_code == 200
    state_body = state.json()
    assert state_body["waiting_choice"]["id"] == "door"
    assert "label" not in state_body["position"]

    debug_state = client.get(
        f"/api/v1/vn-play/sessions/{session_id}/script/debug-state"
    )
    assert debug_state.status_code == 200
    debug_body = debug_state.json()
    assert debug_body["script_version_id"] == published["version_id"]
    assert debug_body["position"]["label"] == "start"
    assert debug_body["position"]["index"] == 4
    assert debug_body["program"]["entry_label"] == "start"
    openapi = client.get("/openapi.json").json()
    debug_schema = openapi["paths"][
        "/api/v1/vn-play/sessions/{session_id}/script/debug-state"
    ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert debug_schema["$ref"].endswith("/VNPlayScriptDebugStateResponse")

    replay = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/story/start",
        json={"client_scene_version": 0, "idempotency_key": "start-script"},
    )
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True
    assert replay.json()["scene_version"] == 1
    assert replay.json()["script_state"]["variables"]["roll"] == start_body[
        "script_state"
    ]["variables"]["roll"]

    blocked_advance = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/script/advance",
        json={"client_scene_version": 1, "idempotency_key": "blocked-at-choice"},
    )
    assert blocked_advance.status_code == 409
    assert blocked_advance.json()["detail"]["code"] == "script_advance_blocked"

    choice = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/script/choices/open",
        json={"client_scene_version": 1, "idempotency_key": "choose-open"},
    )

    assert choice.status_code == 200
    choice_body = choice.json()
    assert choice_body["scene_version"] == 2
    assert "label" not in choice_body["script_state"]["position"]
    assert choice_body["script_state"]["ended"] is True
    assert choice_body["script_state"]["variables"]["trust"] == 1
    assert choice_body["current_scene"]["visible_choices"] == []

    ended_advance = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/script/advance",
        json={"client_scene_version": 2, "idempotency_key": "blocked-at-end"},
    )
    assert ended_advance.status_code == 409
    assert ended_advance.json()["detail"]["code"] == "script_ended"


def test_scripted_story_save_slot_restore_restores_script_cursor(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    pack_id = _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=character_id,
    )
    program = {
        "schema_version": "vn_script_program.v1",
        "entry_label": "start",
        "primary_asset_pack_id": pack_id,
        "variables": {},
        "labels": {
            "start": [
                {
                    "op": "choice",
                    "id": "door",
                    "choices": [
                        {"id": "open", "text": "Open it", "target": "open"},
                        {"id": "wait", "text": "Wait", "target": "wait"},
                    ],
                },
            ],
            "open": [{"op": "end"}],
            "wait": [{"op": "end"}],
        },
    }
    published = _publish_script_version(
        chacha_db,
        owner_user_id=42,
        pack_id=pack_id,
        program=program,
    )
    session_response = client.post(
        "/api/v1/vn-play/sessions",
        json={
            "mode": "scripted_story",
            "title": "Scripted save slot",
            "primary_character_id": character_id,
            "vn_asset_pack_id": pack_id,
            "script_id": published["script_id"],
            "script_version_id": published["version_id"],
            "acknowledgements": ["character_safety_missing"],
        },
    )
    assert session_response.status_code == 201
    session_id = int(session_response.json()["id"])
    assert client.post(
        f"/api/v1/vn-play/sessions/{session_id}/script/advance",
        json={"client_scene_version": 0, "idempotency_key": "restore-start"},
    ).status_code == 200
    save_response = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/save-slots",
        json={
            "slot_key": "choice",
            "title": "At choice",
            "idempotency_key": "script-save-slot",
        },
    )
    assert save_response.status_code == 201
    save_slot_id = int(save_response.json()["id"])
    assert client.post(
        f"/api/v1/vn-play/sessions/{session_id}/script/choices/open",
        json={"client_scene_version": 1, "idempotency_key": "restore-open"},
    ).status_code == 200

    restore_response = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/save-slots/{save_slot_id}/restore",
        json={"client_scene_version": 2, "idempotency_key": "restore-choice"},
    )
    assert restore_response.status_code == 200
    debug_state = client.get(
        f"/api/v1/vn-play/sessions/{session_id}/script/debug-state"
    ).json()
    assert debug_state["position"]["label"] == "start"
    assert debug_state["position"]["waiting_choice_id"] == "door"

    wait_response = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/script/choices/wait",
        json={"client_scene_version": 3, "idempotency_key": "restore-wait"},
    )
    assert wait_response.status_code == 200
    assert wait_response.json()["script_state"]["ended"] is True


def test_scripted_story_generate_and_regenerate_persist_replayable_events(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    pack_id = _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=character_id,
    )
    program = {
        "schema_version": "vn_script_program.v1",
        "entry_label": "start",
        "primary_asset_pack_id": pack_id,
        "variables": {},
        "labels": {
            "start": [
                {
                    "op": "generate",
                    "id": "intro-beat",
                    "prompt": "Introduce the archive door",
                    "text": "The generated archive beat appears.",
                    "regeneration_text": "The archive beat is regenerated from an explicit script variant.",
                },
                {"op": "end"},
            ]
        },
    }
    published = _publish_script_version(
        chacha_db,
        owner_user_id=42,
        pack_id=pack_id,
        program=program,
    )
    session_response = client.post(
        "/api/v1/vn-play/sessions",
        json={
            "mode": "scripted_story",
            "title": "Scripted generation",
            "primary_character_id": character_id,
            "vn_asset_pack_id": pack_id,
            "script_id": published["script_id"],
            "script_version_id": published["version_id"],
            "acknowledgements": ["character_safety_missing"],
        },
    )
    assert session_response.status_code == 201
    session_id = int(session_response.json()["id"])

    generated = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/script/advance",
        json={"client_scene_version": 0, "idempotency_key": "generate-intro"},
    )
    assert generated.status_code == 200
    generation_event = next(
        event for event in generated.json()["events"] if event["event_type"] == "model_turn"
    )
    generation_results = generation_event["event_payload"]["generation_results"]
    assert generation_results[0]["id"] == "intro-beat"
    assert generation_results[0]["regenerated"] is False
    assert generation_results[0]["model_invoked"] is False
    assert generation_results[0]["source"] == "script_literal"

    replayed = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/script/advance",
        json={"client_scene_version": 0, "idempotency_key": "generate-intro"},
    )
    assert replayed.status_code == 200
    assert replayed.json()["replayed"] is True
    assert replayed.json()["events"] == generated.json()["events"]

    regenerated = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/script/regenerate",
        json={"client_scene_version": 1, "idempotency_key": "regenerate-intro"},
    )
    assert regenerated.status_code == 200
    regenerated_event = next(
        event for event in regenerated.json()["events"] if event["event_type"] == "model_turn"
    )
    assert regenerated_event["event_payload"]["generation_results"][0]["id"] == "intro-beat"
    assert regenerated_event["event_payload"]["generation_results"][0]["regenerated"] is True
    assert regenerated.json()["scene_version"] == 2


def test_scripted_story_generate_without_literal_text_is_rejected(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    pack_id = _create_pack(
        chacha_db,
        owner_user_id=42,
        character_id=character_id,
    )
    program = {
        "schema_version": "vn_script_program.v1",
        "entry_label": "start",
        "primary_asset_pack_id": pack_id,
        "variables": {},
        "labels": {
            "start": [
                {
                    "op": "generate",
                    "id": "intro-beat",
                    "prompt": "Introduce the archive door",
                }
            ]
        },
    }
    published = _publish_script_version(
        chacha_db,
        owner_user_id=42,
        pack_id=pack_id,
        program=program,
    )
    session_response = client.post(
        "/api/v1/vn-play/sessions",
        json={
            "mode": "scripted_story",
            "title": "Scripted generation",
            "primary_character_id": character_id,
            "vn_asset_pack_id": pack_id,
            "script_id": published["script_id"],
            "script_version_id": published["version_id"],
            "acknowledgements": ["character_safety_missing"],
        },
    )
    assert session_response.status_code == 201
    session_id = int(session_response.json()["id"])

    generated = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/script/advance",
        json={"client_scene_version": 0, "idempotency_key": "generate-intro-no-literal"},
    )

    assert generated.status_code == 400
    assert generated.json()["detail"]["code"] == "script_generation_unavailable"


def test_setup_options_uses_lightweight_character_selector_queries(
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

    def fail_full_character_query(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("setup options should not load full character rows")

    def query_selector_rows(
        self: CharactersRAGDB,
        *,
        query: str | None = None,
        include_deleted: bool = False,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        return (
            [
                {
                    "id": character_id,
                    "name": "Mira",
                    "description": "A careful archivist.",
                    "tags": ["guide", "archive"],
                    "extensions": {"tldw": {"favorite": True}},
                    "deleted": False,
                    "has_image": True,
                }
            ],
            1,
        )

    def get_selector_row(
        self: CharactersRAGDB,
        selected_character_id: int,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        assert selected_character_id == character_id
        return {
            "id": character_id,
            "name": "Mira",
            "description": "A careful archivist.",
            "tags": ["guide", "archive"],
            "extensions": {"tldw": {"favorite": True}},
            "deleted": False,
            "has_image": True,
        }

    monkeypatch.setattr(CharactersRAGDB, "query_character_cards", fail_full_character_query)
    monkeypatch.setattr(CharactersRAGDB, "get_character_card_by_id", fail_full_character_query)
    monkeypatch.setattr(
        CharactersRAGDB,
        "query_character_setup_options",
        query_selector_rows,
        raising=False,
    )
    monkeypatch.setattr(
        CharactersRAGDB,
        "get_character_setup_option_by_id",
        get_selector_row,
        raising=False,
    )

    response = client.get(
        f"/api/v1/vn-play/setup-options?selected_character_id={character_id}"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selected_character"]["has_image"] is True
    assert body["characters"][0]["favorite"] is True


def test_setup_options_description_preview_respects_max_length(
    client: TestClient,
    chacha_db: CharactersRAGDB,
    character_id: int,
) -> None:
    long_description = " ".join(["archivist"] * 40)
    chacha_db.update_character_card(
        character_id,
        {"description": long_description},
        expected_version=1,
    )

    response = client.get("/api/v1/vn-play/setup-options")

    assert response.status_code == 200
    description_preview = response.json()["characters"][0]["description_preview"]
    assert description_preview.endswith("...")
    assert len(description_preview) <= 160


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


def test_setup_options_pack_listing_does_not_compute_planned_output_count(
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

    def fail_planned_output_count(self: VNAssetPackService, pack_id: int) -> int:
        raise AssertionError("setup pack listing should not scan slots for planned counts")

    monkeypatch.setattr(
        VNAssetPackService,
        "_planned_output_count",
        fail_planned_output_count,
    )

    response = client.get(
        f"/api/v1/vn-play/setup-options?selected_character_id={character_id}"
    )

    assert response.status_code == 200
    assert response.json()["asset_packs"][0]["title"] == "Mira - Archive Pack"


def test_setup_options_missing_required_assets_warning_uses_structured_errors(
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

    def readiness_with_false_positive_text(
        self: VNAssetPackService,
        pack_id: int,
    ) -> VNAssetReadinessResponse:
        return VNAssetReadinessResponse(
            ready=True,
            status="ready",
            warnings=["Required assets are present, none are missing."],
            errors=[],
        )

    monkeypatch.setattr(
        VNAssetPackService,
        "get_readiness",
        readiness_with_false_positive_text,
    )

    response = client.get(
        f"/api/v1/vn-play/setup-options?selected_character_id={character_id}"
    )

    assert response.status_code == 200
    warning_codes = {
        warning["code"]
        for warning in response.json()["asset_packs"][0]["warning_summary"]["warnings"]
    }
    assert "pack_missing_required_assets" not in warning_codes

    def readiness_with_structured_missing_required_slot(
        self: VNAssetPackService,
        pack_id: int,
    ) -> VNAssetReadinessResponse:
        return VNAssetReadinessResponse(
            ready=False,
            status="not_ready",
            warnings=[],
            errors=["required_slot_not_ready:123"],
        )

    monkeypatch.setattr(
        VNAssetPackService,
        "get_readiness",
        readiness_with_structured_missing_required_slot,
    )

    structured_response = client.get(
        f"/api/v1/vn-play/setup-options?selected_character_id={character_id}"
    )

    assert structured_response.status_code == 200
    structured_warning_codes = {
        warning["code"]
        for warning in structured_response.json()["asset_packs"][0]["warning_summary"]["warnings"]
    }
    assert "pack_missing_required_assets" in structured_warning_codes


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
    assert stale.json()["detail"]["code"] == "stale_scene_version"

    stale_retry = client.post(
        f"/api/v1/vn-play/sessions/{session_id}/turn",
        json={"input_text": "Again", "client_scene_version": 0, "idempotency_key": "b"},
    )
    assert stale_retry.status_code == 409
    assert stale_retry.json()["detail"]["code"] == "stale_scene_version"


def test_delete_session_endpoint_soft_deletes(
    client: TestClient,
    session_id: int,
) -> None:
    response = client.delete(f"/api/v1/vn-play/sessions/{session_id}")

    assert response.status_code == 204
    assert client.get(f"/api/v1/vn-play/sessions/{session_id}").status_code == 404
