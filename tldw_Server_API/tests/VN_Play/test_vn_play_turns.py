import inspect
from collections.abc import Generator, Mapping, Sequence
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.DB_Management.VNPlay_DB import VNPlayRepository
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Play import adapters as vn_play_adapters
from tldw_Server_API.app.core.VN_Play.constants import (
    BRANCH_RESTORE_TARGET_CHOICE_POINT,
    STORY_BRANCH_LABEL_MAX_LENGTH,
)
from tldw_Server_API.app.core.VN_Play.models import (
    SceneState,
    TurnResult,
    VisualDirectiveResolution,
)
from tldw_Server_API.app.core.VN_Play.parser import VNPlayParseError, parse_model_turn
from tldw_Server_API.app.core.VN_Play.service import (
    DeterministicVNPlayTurnAdapter,
    VNPlayConflictError,
    VNPlayNotFoundError,
    VNPlayService,
    VNPlaySession,
    VNPlayTurnContext,
    VNPlayTurnError,
    _parent_choice_event_id,
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
        async def generate_turn(self, context: VNPlayTurnContext) -> TurnResult:
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


class MissingVisualDirectiveAdapter:
    async def generate_turn(self, context: VNPlayTurnContext) -> TurnResult:
        return TurnResult(
            narrative_text="The library remains quiet.",
            dialogue=[{"speaker": "Narrator", "text": "The library remains quiet."}],
            visual_directives=[
                {"asset_type": "sprite", "labels": {"emotion": "angry"}},
            ],
            scene_updates={"location_key": "library"},
        )


class InspectingStoryAdapter:
    def __init__(self, repo: VNPlayRepository, owner_user_id: int) -> None:
        self.repo = repo
        self.owner_user_id = owner_user_id
        self.seen_contexts: list[VNPlayTurnContext] = []

    async def generate_turn(self, context: VNPlayTurnContext) -> TurnResult:
        self.seen_contexts.append(context)
        persisted = self.repo.get_scene_state(
            context.session.id,
            owner_user_id=self.owner_user_id,
        )
        assert persisted is not None
        assert persisted["active_branch_node_id"] is not None
        assert persisted["visible_choices"] == []
        return TurnResult(
            narrative_text="The door opens.",
            dialogue=[{"speaker": "Narrator", "text": "The door opens."}],
            choices=[
                {"id": "inside", "text": "Step inside"},
                {"id": "wait", "text": "Wait outside"},
            ],
        )


class StoryVisualDirectiveAdapter:
    async def generate_turn(self, context: VNPlayTurnContext) -> TurnResult:
        return TurnResult(
            narrative_text="The door opens onto the library.",
            dialogue=[{"speaker": "Narrator", "text": "The door opens."}],
            visual_directives=[
                {"asset_type": "background", "labels": {"location": "library"}},
            ],
            choices=[
                {"id": "inside", "text": "Step inside"},
                {"id": "wait", "text": "Wait outside"},
            ],
            scene_updates={"location_key": "library"},
        )


class CountingStoryAdapter:
    def __init__(self) -> None:
        self.seen_contexts: list[VNPlayTurnContext] = []

    async def generate_turn(self, context: VNPlayTurnContext) -> TurnResult:
        self.seen_contexts.append(context)
        return TurnResult(narrative_text="Unused")


class FailingStoryAdapter:
    async def generate_turn(self, context: VNPlayTurnContext) -> TurnResult:
        raise RuntimeError("story provider unavailable")


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


def create_story_session_with_visible_choice(
    service: VNPlayService,
    repo: VNPlayRepository,
    *,
    choice_text: str = "Open the door",
) -> VNPlaySession:
    owner_user_id = service.owner_user_id
    session = service.create_session(
        mode="story",
        title="Door",
        primary_character_id=1,
        vn_asset_pack_id=10,
        content_rating="general",
        seed="seed-story",
        settings={},
    )
    choice_presented = repo.append_event(
        session_id=session.id,
        owner_user_id=owner_user_id,
        event_type="choice_presented",
        event_payload={
            "choices": [{"id": "open", "text": choice_text}],
            "scene_version": 1,
        },
        source="runtime",
    )
    repo.set_scene_state(
        session_id=session.id,
        owner_user_id=owner_user_id,
        last_event_id=int(choice_presented["id"]),
        visible_choices=[{"id": "open", "text": choice_text}],
        scene_version=1,
    )
    repo.update_session(session.id, {"scene_version": 1}, owner_user_id=owner_user_id)
    return service.get_session(session.id)


async def create_two_level_story_branches(
    service: VNPlayService,
    repo: VNPlayRepository,
) -> tuple[VNPlaySession, dict[str, Any], dict[str, Any]]:
    session = create_story_session_with_visible_choice(service, repo)
    await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="story-open",
    )
    first_branch = service.list_branches(session.id)[0]

    await service.submit_turn(
        session.id,
        choice_id="inside",
        client_scene_version=2,
        idempotency_key="story-inside",
    )
    branches = service.list_branches(session.id)
    second_branch = next(
        branch for branch in branches if int(branch["id"]) != int(first_branch["id"])
    )
    return service.get_session(session.id), first_branch, second_branch


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
async def test_story_choice_creates_branch_and_choice_selected_before_model(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    adapter = InspectingStoryAdapter(repo, owner_user_id=42)
    service = VNPlayService(repo=repo, owner_user_id=42, adapter=adapter)
    session = create_story_session_with_visible_choice(service, repo)

    response = await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="story-choice-1",
    )

    event_types = [event["event_type"] for event in response.events]
    assert event_types[:2] == ["turn_started", "choice_selected"]
    assert adapter.seen_contexts[0].scene_state.active_branch_node_id is not None
    assert adapter.seen_contexts[0].input_payload == {
        "choice_id": "open",
        "choice": {"id": "open", "text": "Open the door"},
    }

    branches = service.list_branches(session.id)
    assert len(branches) == 1
    assert branches[0]["branch_path"][0]["choice_id"] == "open"

    state = repo.get_scene_state(session.id, owner_user_id=42)
    assert state is not None
    assert state["active_branch_node_id"] == branches[0]["id"]
    assert state["visible_choices"] == [
        {"id": "inside", "text": "Step inside"},
        {"id": "wait", "text": "Wait outside"},
    ]


@pytest.mark.asyncio
async def test_branch_navigation_service_returns_active_path(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session = create_story_session_with_visible_choice(service, repo)

    await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="story-navigation-open",
    )

    branch = service.list_branches(session.id)[0]
    navigation = service.get_branch_navigation(session.id)

    assert navigation["mode"] == "story"
    assert navigation["active_branch_node_id"] == branch["id"]
    assert [step["branch_id"] for step in navigation["active_path"]] == [branch["id"]]
    assert navigation["branches"][0]["restore"]["supported"] is True


def test_freeform_branch_navigation_returns_empty_without_error(
    service: VNPlayService,
    ready_session,
) -> None:
    navigation = service.get_branch_navigation(ready_session.id)

    assert navigation["mode"] == "freeform"
    assert navigation["active_path"] == []
    assert navigation["branches"] == []
    assert navigation["warnings"] == []


@pytest.mark.asyncio
async def test_list_events_with_metadata_filters_direct_branch_only(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session, first_branch, second_branch = await create_two_level_story_branches(
        service,
        repo,
    )

    result = service.list_events_with_metadata(
        session.id,
        branch_id=int(first_branch["id"]),
        include_descendants=False,
    )

    assert result["warnings"] == []
    assert result["events"]
    assert any(event["branch_node_id"] == first_branch["id"] for event in result["events"])
    assert not any(event["branch_node_id"] == second_branch["id"] for event in result["events"])
    assert not any(
        event["event_type"] == "choice_selected"
        and event["event_payload"].get("choice_id") == "inside"
        for event in result["events"]
    )


@pytest.mark.asyncio
async def test_list_events_with_metadata_include_descendants(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session, first_branch, second_branch = await create_two_level_story_branches(
        service,
        repo,
    )

    result = service.list_events_with_metadata(
        session.id,
        branch_id=int(first_branch["id"]),
        include_descendants=True,
    )

    assert result["warnings"] == []
    branch_node_ids = {event["branch_node_id"] for event in result["events"]}
    assert first_branch["id"] in branch_node_ids
    assert second_branch["id"] in branch_node_ids
    assert any(
        event["event_type"] == "choice_selected"
        and event["event_payload"].get("choice_id") == "inside"
        for event in result["events"]
    )


@pytest.mark.asyncio
async def test_list_events_with_metadata_rejects_missing_or_foreign_branch(
    service: VNPlayService,
    ready_session,
    chacha_db: CharactersRAGDB,
) -> None:
    with pytest.raises(VNPlayNotFoundError, match="branch_not_found"):
        service.list_events_with_metadata(ready_session.id, branch_id=9999)

    other_repo = VNPlayRepository.initialized(chacha_db)
    other_service = VNPlayService(
        repo=other_repo,
        owner_user_id=77,
        adapter=InspectingStoryAdapter(other_repo, owner_user_id=77),
    )
    foreign_session = create_story_session_with_visible_choice(other_service, other_repo)
    await other_service.submit_turn(
        foreign_session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="foreign-open",
    )
    foreign_branch_id = int(other_service.list_branches(foreign_session.id)[0]["id"])

    with pytest.raises(VNPlayNotFoundError, match="branch_not_found"):
        service.list_events_with_metadata(ready_session.id, branch_id=foreign_branch_id)


@pytest.mark.asyncio
async def test_list_events_with_metadata_honors_after_sequence_and_limit(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session = create_story_session_with_visible_choice(service, repo)
    response = await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="story-branch-page",
    )
    branch_id = service.list_branches(session.id)[0]["id"]
    branch_events = [
        event for event in response.events if event["branch_node_id"] == branch_id
    ]

    result = service.list_events_with_metadata(
        session.id,
        branch_id=int(branch_id),
        after_sequence=int(branch_events[0]["sequence_number"]),
        limit=2,
    )

    assert [event["id"] for event in result["events"]] == [
        event["id"] for event in branch_events[1:3]
    ]


@pytest.mark.asyncio
async def test_list_events_with_metadata_uses_tagged_branch_query(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session = create_story_session_with_visible_choice(service, repo)
    response = await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="story-branch-tagged-query",
    )
    branch_id = int(service.list_branches(session.id)[0]["id"])
    branch_events = [
        event for event in response.events if event["branch_node_id"] == branch_id
    ]
    original_branch_query = repo.list_events_for_branch_nodes
    branch_query_calls: list[dict[str, Any]] = []

    def blocked_full_history(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("branch-filtered event reads should use branch-node query")

    def tracking_branch_query(
        session_id: int,
        branch_node_ids: Sequence[int],
        *,
        after_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        branch_query_calls.append(
            {
                "session_id": session_id,
                "branch_node_ids": list(branch_node_ids),
                "after_sequence": after_sequence,
                "limit": limit,
            }
        )
        return original_branch_query(
            session_id,
            branch_node_ids,
            after_sequence=after_sequence,
            limit=limit,
        )

    monkeypatch.setattr(repo, "list_events", blocked_full_history)
    monkeypatch.setattr(repo, "list_events_for_branch_nodes", tracking_branch_query)

    result = service.list_events_with_metadata(
        session.id,
        branch_id=branch_id,
        after_sequence=int(branch_events[0]["sequence_number"]),
        limit=2,
    )

    assert branch_query_calls == [
        {
            "session_id": session.id,
            "branch_node_ids": [branch_id],
            "after_sequence": int(branch_events[0]["sequence_number"]),
            "limit": 2,
        }
    ]
    assert result["warnings"] == []
    assert [event["id"] for event in result["events"]] == [
        event["id"] for event in branch_events[1:3]
    ]


@pytest.mark.asyncio
async def test_story_completion_events_after_selected_branch_are_tagged(
    chacha_db: CharactersRAGDB,
) -> None:
    character_id, pack_id, _, _ = create_visual_pack(chacha_db)
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=StoryVisualDirectiveAdapter(),
    )
    session = create_story_session_with_visible_choice(service, repo)
    service.repo.update_session(
        session.id,
        {"primary_character_id": character_id, "vn_asset_pack_id": pack_id},
        owner_user_id=42,
    )

    response = await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="story-tagged-completion",
    )

    branch_id = service.list_branches(session.id)[0]["id"]
    expected_event_types = {
        "model_turn",
        "visual_directive_requested",
        "visual_directive_applied",
        "choice_presented",
        "scene_state_changed",
        "turn_completed",
    }
    tagged = [
        event
        for event in response.events
        if event["event_type"] in expected_event_types
    ]
    assert {event["event_type"] for event in tagged} == expected_event_types
    assert {event["branch_node_id"] for event in tagged} == {branch_id}


@pytest.mark.asyncio
async def test_freeform_completion_events_remain_untagged(
    service: VNPlayService,
    ready_session,
) -> None:
    response = await service.submit_turn(
        ready_session.id,
        input_text="Look around",
        client_scene_version=0,
        idempotency_key="freeform-untagged",
    )

    assert {event["branch_node_id"] for event in response.events} == {None}


@pytest.mark.asyncio
async def test_story_choice_branch_labels_are_bounded(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=DeterministicVNPlayTurnAdapter(),
    )
    long_choice_text = "Open " + ("the sealed archive door " * 20)
    session = create_story_session_with_visible_choice(
        service,
        repo,
        choice_text=long_choice_text,
    )

    await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="story-choice-long-label",
    )

    branch = service.list_branches(session.id)[0]
    expected_label = long_choice_text[:STORY_BRANCH_LABEL_MAX_LENGTH]
    assert branch["branch_label"] == expected_label
    assert branch["branch_path"][0]["choice_text"] == expected_label


@pytest.mark.asyncio
async def test_freeform_turn_defers_full_event_history_until_after_input_events(
    service: VNPlayService,
    ready_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_list_events = service.repo.list_events
    unbounded_event_counts: list[int] = []

    def tracking_list_events(
        session_id: int,
        *,
        after_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if after_sequence is None and limit is None:
            cursor = service.repo.db.execute_query(
                "SELECT COUNT(*) AS count FROM vn_play_events WHERE session_id = ?",
                (session_id,),
            )
            unbounded_event_counts.append(int(cursor.fetchone()["count"]))
        return original_list_events(
            session_id,
            after_sequence=after_sequence,
            limit=limit,
        )

    monkeypatch.setattr(service.repo, "list_events", tracking_list_events)

    await service.submit_turn(
        ready_session.id,
        input_text="Look around",
        client_scene_version=0,
        idempotency_key="freeform-query-deferral",
    )

    assert unbounded_event_counts
    assert unbounded_event_counts[0] >= 3


def test_story_choice_repository_signature_keeps_required_params_first() -> None:
    parameters = list(
        inspect.signature(
            VNPlayRepository.record_story_choice_selection,
        ).parameters
    )

    assert parameters.index("branch_label") < parameters.index(
        "expected_scene_last_event_id"
    )
    assert parameters.index("branch_path") < parameters.index(
        "expected_scene_last_event_id"
    )


@pytest.mark.asyncio
async def test_story_unknown_choice_id_fails_before_model(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    adapter = InspectingStoryAdapter(repo, owner_user_id=42)
    service = VNPlayService(repo=repo, owner_user_id=42, adapter=adapter)
    session = create_story_session_with_visible_choice(service, repo)

    with pytest.raises(VNPlayTurnError, match="invalid_choice_id"):
        await service.submit_turn(
            session.id,
            choice_id="locked",
            client_scene_version=1,
            idempotency_key="story-choice-invalid",
        )

    assert adapter.seen_contexts == []


@pytest.mark.asyncio
async def test_freeform_choice_id_is_not_allowed(
    service: VNPlayService,
    ready_session,
) -> None:
    with pytest.raises(VNPlayTurnError, match="choice_not_allowed"):
        await service.submit_turn(
            ready_session.id,
            choice_id="open",
            client_scene_version=0,
            idempotency_key="freeform-choice",
        )


@pytest.mark.asyncio
async def test_story_input_text_is_not_allowed(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    adapter = InspectingStoryAdapter(repo, owner_user_id=42)
    service = VNPlayService(repo=repo, owner_user_id=42, adapter=adapter)
    session = create_story_session_with_visible_choice(service, repo)

    with pytest.raises(VNPlayTurnError, match="choice_not_allowed"):
        await service.submit_turn(
            session.id,
            input_text="Open the door",
            client_scene_version=1,
            idempotency_key="story-input-text",
        )

    assert adapter.seen_contexts == []


@pytest.mark.asyncio
async def test_story_custom_action_remains_non_branching(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=DeterministicVNPlayTurnAdapter(),
    )
    session = create_story_session_with_visible_choice(service, repo)

    response = await service.submit_turn(
        session.id,
        custom_action={"verb": "inspect", "target": "door"},
        client_scene_version=1,
        idempotency_key="story-custom-action",
    )

    event_types = [event["event_type"] for event in response.events]
    assert event_types[:2] == ["turn_started", "user_turn"]
    assert "choice_selected" not in event_types
    assert service.list_branches(session.id) == []


@pytest.mark.asyncio
async def test_story_choice_disappearing_after_validation_fails_before_model(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    adapter = CountingStoryAdapter()
    service = VNPlayService(repo=repo, owner_user_id=42, adapter=adapter)
    session = create_story_session_with_visible_choice(service, repo)
    original_try_acquire = repo.try_acquire_turn_lock

    def acquire_and_hide_choice(*args: Any, **kwargs: Any) -> bool:
        acquired = original_try_acquire(*args, **kwargs)
        if acquired:
            state = repo.get_scene_state(session.id, owner_user_id=42)
            assert state is not None
            repo.set_scene_state(
                session_id=session.id,
                owner_user_id=42,
                last_event_id=state["last_event_id"],
                visible_choices=[],
                scene_version=state["scene_version"],
            )
        return acquired

    monkeypatch.setattr(repo, "try_acquire_turn_lock", acquire_and_hide_choice)

    with pytest.raises(VNPlayTurnError, match="invalid_choice_id"):
        await service.submit_turn(
            session.id,
            choice_id="open",
            client_scene_version=1,
            idempotency_key="story-choice-hidden",
        )

    assert adapter.seen_contexts == []
    assert service.list_branches(session.id) == []
    event_types = [event["event_type"] for event in repo.list_events(session.id)]
    assert "turn_started" not in event_types
    assert "choice_selected" not in event_types
    assert service.get_session(session.id).active_turn_request_id is None
    turn = repo.get_turn_request_by_key(
        session_id=session.id,
        owner_user_id=42,
        idempotency_key="story-choice-hidden",
    )
    assert turn is not None
    assert turn["status"] == "abandoned"
    assert turn["error"]["code"] == "invalid_choice_id"


def test_parent_choice_lookup_stays_within_restore_and_scene_state_window() -> None:
    events = [
        {
            "id": 1,
            "sequence_number": 1,
            "event_type": "choice_presented",
            "event_payload": {"choices": [{"id": "stale", "text": "Old door"}]},
        },
        {
            "id": 2,
            "sequence_number": 2,
            "event_type": "session_restored",
            "event_payload": {"scene_version": 1},
        },
        {
            "id": 3,
            "sequence_number": 3,
            "event_type": "choice_presented",
            "event_payload": {"choices": [{"id": "open", "text": "Open"}]},
        },
        {
            "id": 4,
            "sequence_number": 4,
            "event_type": "choice_presented",
            "event_payload": {"choices": [{"id": "future", "text": "Future"}]},
        },
    ]

    assert _parent_choice_event_id(events, 3, "open") == 3
    assert _parent_choice_event_id(events, 3, "stale") is None
    assert _parent_choice_event_id(events, 3, "future") is None


@pytest.mark.asyncio
async def test_retry_failed_story_choice_reuses_original_branch(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    failing = VNPlayService(repo=repo, owner_user_id=42, adapter=FailingStoryAdapter())
    session = create_story_session_with_visible_choice(failing, repo)

    with pytest.raises(VNPlayTurnError, match="model_failed"):
        await failing.submit_turn(
            session.id,
            choice_id="open",
            client_scene_version=1,
            idempotency_key="story-fail-1",
        )

    branches_before = failing.list_branches(session.id)
    retry_adapter = InspectingStoryAdapter(repo, owner_user_id=42)
    retrying = VNPlayService(repo=repo, owner_user_id=42, adapter=retry_adapter)

    response = await retrying.retry_last_turn(
        session.id,
        client_scene_version=1,
        idempotency_key="story-retry-1",
    )

    assert response.status == "completed"
    assert retrying.list_branches(session.id) == branches_before
    events = retrying.list_events(session.id)
    assert [event["event_type"] for event in events].count("choice_selected") == 1
    assert retry_adapter.seen_contexts[0].input_payload["choice_id"] == "open"
    assert (
        retry_adapter.seen_contexts[0].input_payload["branch_node_id"]
        == branches_before[0]["id"]
    )

    with pytest.raises(VNPlayTurnError, match="retry_last_turn_not_failed"):
        await retrying.retry_last_turn(
            session.id,
            client_scene_version=2,
            idempotency_key="story-retry-after-success",
        )


@pytest.mark.asyncio
async def test_retry_completed_story_choice_is_not_failed(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session = create_story_session_with_visible_choice(service, repo)
    await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="story-choice-completed",
    )

    with pytest.raises(VNPlayTurnError, match="retry_last_turn_not_failed"):
        await service.retry_last_turn(
            session.id,
            client_scene_version=2,
            idempotency_key="retry-completed",
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
async def test_branch_latest_restore_advances_scene_version_and_replays_duplicate(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session = create_story_session_with_visible_choice(service, repo)
    await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="restore-latest-open-turn",
    )
    branch = service.list_branches(session.id)[0]
    pre_restore_target_event_id = service.get_branch_navigation(session.id)["branches"][
        0
    ]["event_range"]["latest_event_id"]
    events_before = service.list_events(session.id)
    restore_events_before = [
        event for event in events_before if event["event_type"] == "session_restored"
    ]

    first = service.restore_branch(
        session.id,
        branch_id=int(branch["id"]),
        client_scene_version=2,
        idempotency_key="restore-latest-open",
    )
    second = service.restore_branch(
        session.id,
        branch_id=int(branch["id"]),
        client_scene_version=2,
        idempotency_key="restore-latest-open",
    )

    assert first["status"] == "completed"
    assert second == {**first, "replayed": True}
    assert first["scene_version"] == 3
    assert first["target_event_id"] == pre_restore_target_event_id
    restore_events_after = [
        event for event in service.list_events(session.id) if event["event_type"] == "session_restored"
    ]
    assert len(restore_events_after) == len(restore_events_before) + 1
    assert service.get_session(session.id).scene_version == 3


@pytest.mark.asyncio
async def test_choice_point_restore_returns_parent_choices_and_parent_branch(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session, first_branch, second_branch = await create_two_level_story_branches(
        service,
        repo,
    )

    response = service.restore_branch(
        session.id,
        branch_id=int(second_branch["id"]),
        client_scene_version=3,
        idempotency_key="restore-choice-point-inside",
        target=BRANCH_RESTORE_TARGET_CHOICE_POINT,
    )

    assert response["status"] == "completed"
    assert response["target"] == "choice_point"
    assert response["scene_version"] == 4
    assert response["current_scene"]["visible_choices"] == [
        {"id": "inside", "text": "Step inside"},
        {"id": "wait", "text": "Wait outside"},
    ]
    assert response["current_scene"]["active_branch_node_id"] == first_branch["id"]
    assert response["current_scene"]["active_branch_node_id"] != second_branch["id"]
    assert response["branch_navigation"]["active_branch_node_id"] == first_branch["id"]


@pytest.mark.asyncio
async def test_branch_restore_rejects_stale_scene_version(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session = create_story_session_with_visible_choice(service, repo)
    await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="restore-stale-open-turn",
    )
    branch = service.list_branches(session.id)[0]

    with pytest.raises(VNPlayConflictError, match="stale_scene_version"):
        service.restore_branch(
            session.id,
            branch_id=int(branch["id"]),
            client_scene_version=1,
            idempotency_key="restore-stale-open",
        )


@pytest.mark.asyncio
async def test_branch_restore_rejects_active_turn(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session = create_story_session_with_visible_choice(service, repo)
    await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="restore-active-turn-open-turn",
    )
    branch = service.list_branches(session.id)[0]
    turn = repo.create_turn_request(
        session_id=session.id,
        owner_user_id=42,
        idempotency_key="active-turn-marker",
        request_payload_hash="active-turn-marker",
        base_scene_version=2,
    )
    repo.update_session(
        session.id,
        {"active_turn_request_id": int(turn["id"])},
        owner_user_id=42,
    )

    with pytest.raises(VNPlayConflictError, match="turn_in_progress"):
        service.restore_branch(
            session.id,
            branch_id=int(branch["id"]),
            client_scene_version=2,
            idempotency_key="restore-active-turn",
        )


@pytest.mark.asyncio
async def test_branch_restore_recovers_expired_active_turn_lock(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session = create_story_session_with_visible_choice(service, repo)
    await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="restore-expired-lock-open-turn",
    )
    branch = service.list_branches(session.id)[0]
    stale_turn = repo.create_turn_request(
        session_id=session.id,
        owner_user_id=42,
        idempotency_key="restore-expired-lock-marker",
        request_payload_hash="restore-expired-lock-marker",
        base_scene_version=2,
        status="model_calling",
    )
    repo.update_turn_request(
        stale_turn["id"],
        {"locked_until": "2000-01-01 00:00:00", "lease_owner": "worker-1"},
        owner_user_id=42,
    )
    repo.update_session(
        session.id,
        {"active_turn_request_id": int(stale_turn["id"])},
        owner_user_id=42,
    )

    response = service.restore_branch(
        session.id,
        branch_id=int(branch["id"]),
        client_scene_version=2,
        idempotency_key="restore-expired-lock",
    )

    assert response["status"] == "completed"
    assert response["scene_version"] == 3
    recovered = repo.get_turn_request(stale_turn["id"])
    assert recovered["status"] == "abandoned"
    assert recovered["error"] == {"code": "turn_lock_abandoned"}
    assert service.get_session(session.id).active_turn_request_id is None


@pytest.mark.asyncio
async def test_branch_restore_rejects_active_restore_action(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session = create_story_session_with_visible_choice(service, repo)
    await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="restore-active-action-open-turn",
    )
    branch = service.list_branches(session.id)[0]
    action = repo.create_session_action(
        session_id=session.id,
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="active-restore-marker",
        request_payload_hash="active-restore-marker",
    )
    repo.update_session(
        session.id,
        {"active_session_action_id": int(action["id"])},
        owner_user_id=42,
    )

    with pytest.raises(VNPlayConflictError, match="restore_action_in_progress"):
        service.restore_branch(
            session.id,
            branch_id=int(branch["id"]),
            client_scene_version=2,
            idempotency_key="restore-active-action",
        )


@pytest.mark.asyncio
async def test_turn_submission_rejects_active_restore_action(
    service: VNPlayService,
    ready_session,
) -> None:
    action = service.repo.create_session_action(
        session_id=ready_session.id,
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="active-restore-before-turn",
        request_payload_hash="active-restore-before-turn",
    )
    service.repo.update_session(
        ready_session.id,
        {"active_session_action_id": int(action["id"])},
        owner_user_id=42,
    )

    with pytest.raises(VNPlayConflictError, match="restore_action_in_progress"):
        await service.submit_turn(
            ready_session.id,
            input_text="Hello",
            client_scene_version=0,
            idempotency_key="turn-during-restore",
        )


@pytest.mark.asyncio
async def test_branch_restore_same_key_different_target_conflicts(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session = create_story_session_with_visible_choice(service, repo)
    await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="restore-conflict-open-turn",
    )
    branch = service.list_branches(session.id)[0]
    service.restore_branch(
        session.id,
        branch_id=int(branch["id"]),
        client_scene_version=2,
        idempotency_key="restore-conflict-key",
    )

    with pytest.raises(VNPlayConflictError, match="idempotency_key_conflict"):
        service.restore_branch(
            session.id,
            branch_id=int(branch["id"]),
            client_scene_version=2,
            idempotency_key="restore-conflict-key",
            target=BRANCH_RESTORE_TARGET_CHOICE_POINT,
        )


def test_branch_restore_rejects_freeform_session(
    service: VNPlayService,
    ready_session,
) -> None:
    with pytest.raises(VNPlayConflictError, match="branch_restore_not_allowed"):
        service.restore_branch(
            ready_session.id,
            branch_id=1,
            client_scene_version=0,
            idempotency_key="restore-freeform",
        )


def test_branch_restore_target_failure_clears_active_restore_action(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(repo=repo, owner_user_id=42)
    session = create_story_session_with_visible_choice(service, repo)
    branch = repo.create_branch(
        session_id=session.id,
        owner_user_id=42,
        parent_event_id=None,
        branch_label="Detached",
        branch_path=[{"choice_id": "detached", "choice_text": "Detached"}],
    )

    def fail_separate_lock_clear(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("separate lock clear should not be used")

    monkeypatch.setattr(repo, "clear_session_action_lock", fail_separate_lock_clear)

    with pytest.raises(VNPlayConflictError, match="branch_restore_target_unavailable"):
        service.restore_branch(
            session.id,
            branch_id=int(branch["id"]),
            client_scene_version=1,
            idempotency_key="restore-detached-choice-point",
            target=BRANCH_RESTORE_TARGET_CHOICE_POINT,
        )

    assert service.get_session(session.id).active_session_action_id is None


@pytest.mark.asyncio
async def test_failed_restore_retry_preserves_terminal_error(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session = create_story_session_with_visible_choice(service, repo)
    await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="restore-failed-open-turn",
    )
    branch = service.list_branches(session.id)[0]

    def fail_restore_commit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("simulated restore write failure")

    monkeypatch.setattr(repo, "commit_session_restore_action", fail_restore_commit)

    with pytest.raises(RuntimeError, match="simulated restore write failure"):
        service.restore_branch(
            session.id,
            branch_id=int(branch["id"]),
            client_scene_version=2,
            idempotency_key="restore-failed-terminal",
        )

    failed_action = repo.get_session_action_by_key(
        session_id=session.id,
        owner_user_id=42,
        idempotency_key="restore-failed-terminal",
    )
    assert failed_action is not None
    assert failed_action["status"] == "failed"
    assert failed_action["error"] == {
        "code": "internal_error",
        "error_type": "RuntimeError",
    }

    with pytest.raises(VNPlayConflictError, match="internal_error"):
        service.restore_branch(
            session.id,
            branch_id=int(branch["id"]),
            client_scene_version=2,
            idempotency_key="restore-failed-terminal",
        )

    retried_action = repo.get_session_action(int(failed_action["id"]), owner_user_id=42)
    assert retried_action["status"] == "failed"
    assert retried_action["error"] == failed_action["error"]


@pytest.mark.asyncio
async def test_abandoned_restore_retry_preserves_terminal_error(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=InspectingStoryAdapter(repo, owner_user_id=42),
    )
    session = create_story_session_with_visible_choice(service, repo)
    await service.submit_turn(
        session.id,
        choice_id="open",
        client_scene_version=1,
        idempotency_key="restore-abandoned-open-turn",
    )
    branch = service.list_branches(session.id)[0]

    with pytest.raises(VNPlayConflictError, match="stale_scene_version"):
        service.restore_branch(
            session.id,
            branch_id=int(branch["id"]),
            client_scene_version=1,
            idempotency_key="restore-abandoned-terminal",
        )

    abandoned_action = repo.get_session_action_by_key(
        session_id=session.id,
        owner_user_id=42,
        idempotency_key="restore-abandoned-terminal",
    )
    assert abandoned_action is not None
    assert abandoned_action["status"] == "abandoned"
    assert abandoned_action["error"] == {"code": "stale_scene_version"}

    with pytest.raises(VNPlayConflictError, match="stale_scene_version"):
        service.restore_branch(
            session.id,
            branch_id=int(branch["id"]),
            client_scene_version=1,
            idempotency_key="restore-abandoned-terminal",
        )

    retried_action = repo.get_session_action(int(abandoned_action["id"]), owner_user_id=42)
    assert retried_action["status"] == "abandoned"
    assert retried_action["error"] == abandoned_action["error"]


@pytest.mark.asyncio
async def test_checkpoint_restore_duplicate_same_key_replays_response(
    service: VNPlayService,
    ready_session,
) -> None:
    await service.submit_turn(
        ready_session.id,
        input_text="First",
        client_scene_version=0,
        idempotency_key="checkpoint-replay-first",
    )
    checkpoint = service.create_checkpoint(ready_session.id, label="First")
    await service.submit_turn(
        ready_session.id,
        input_text="Second",
        client_scene_version=1,
        idempotency_key="checkpoint-replay-second",
    )

    first = service.restore_checkpoint(
        ready_session.id,
        int(checkpoint["id"]),
        client_scene_version=2,
        idempotency_key="checkpoint-replay",
    )
    second = service.restore_checkpoint(
        ready_session.id,
        int(checkpoint["id"]),
        client_scene_version=2,
        idempotency_key="checkpoint-replay",
    )

    assert first["status"] == "completed"
    assert second == {**first, "replayed": True}
    restore_events = [
        event for event in service.list_events(ready_session.id) if event["event_type"] == "session_restored"
    ]
    assert len(restore_events) == 1


@pytest.mark.asyncio
async def test_checkpoint_restore_same_key_different_checkpoint_conflicts(
    service: VNPlayService,
    ready_session,
) -> None:
    await service.submit_turn(
        ready_session.id,
        input_text="First",
        client_scene_version=0,
        idempotency_key="checkpoint-conflict-first",
    )
    first_checkpoint = service.create_checkpoint(ready_session.id, label="First")
    await service.submit_turn(
        ready_session.id,
        input_text="Second",
        client_scene_version=1,
        idempotency_key="checkpoint-conflict-second",
    )
    second_checkpoint = service.create_checkpoint(ready_session.id, label="Second")
    service.restore_checkpoint(
        ready_session.id,
        int(first_checkpoint["id"]),
        client_scene_version=2,
        idempotency_key="checkpoint-conflict",
    )

    with pytest.raises(VNPlayConflictError, match="idempotency_key_conflict"):
        service.restore_checkpoint(
            ready_session.id,
            int(second_checkpoint["id"]),
            client_scene_version=2,
            idempotency_key="checkpoint-conflict",
        )


@pytest.mark.asyncio
async def test_checkpoint_restore_advances_scene_version_by_one(
    service: VNPlayService,
    ready_session,
) -> None:
    await service.submit_turn(
        ready_session.id,
        input_text="First",
        client_scene_version=0,
        idempotency_key="checkpoint-version-first",
    )
    checkpoint = service.create_checkpoint(ready_session.id, label="First")
    await service.submit_turn(
        ready_session.id,
        input_text="Second",
        client_scene_version=1,
        idempotency_key="checkpoint-version-second",
    )

    response = service.restore_checkpoint(
        ready_session.id,
        int(checkpoint["id"]),
        client_scene_version=2,
        idempotency_key="checkpoint-version",
    )

    assert response["scene_version"] == 3
    assert response["current_scene"]["scene_version"] == 3
    assert service.get_session(ready_session.id).scene_version == 3


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

    def fail_resolution(
        manifest: Mapping[str, Any],
        directives: Sequence[Mapping[str, Any]],
        *,
        seed: str,
    ) -> list[VisualDirectiveResolution]:
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
    assert response.warnings[0]["error_type"] == "RuntimeError"
    assert "error" not in response.warnings[0]
    assert service.get_session(session.id).active_turn_request_id is None


@pytest.mark.asyncio
async def test_scene_enrichment_reuses_turn_manifest_cache(
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
    build_calls: list[int] = []
    original_build_manifest = VNAssetPackService.build_manifest

    def counted_build_manifest(
        self: VNAssetPackService,
        requested_pack_id: int,
    ) -> Any:
        build_calls.append(requested_pack_id)
        return original_build_manifest(self, requested_pack_id)

    monkeypatch.setattr(VNAssetPackService, "build_manifest", counted_build_manifest)

    await service.submit_turn(
        session.id,
        input_text="Look around",
        client_scene_version=0,
        idempotency_key="cached-manifest-turn-1",
    )
    state = service.get_enriched_scene_state(session.id)

    assert state is not None
    assert state["background"] is not None
    assert build_calls == [pack_id]


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

    fresh_service = VNPlayService(
        repo=VNPlayRepository.initialized(chacha_db),
        owner_user_id=42,
        adapter=VisualDirectiveAdapter(),
    )
    state = fresh_service.get_enriched_scene_state(session.id)

    assert state is not None
    assert state["active_sprite_items"][0]["item_id"] == sprite_item_id
    assert state["active_sprites"] == []


def test_scene_enrichment_warning_uses_safe_error_type(
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

    def fail_manifest(self: VNAssetPackService, requested_pack_id: int) -> object:
        raise RuntimeError("/private/path/secret.db")

    monkeypatch.setattr(VNAssetPackService, "build_manifest", fail_manifest)

    state = service.get_enriched_scene_state(session.id)

    assert state is not None
    assert state["warnings"][0]["reason"] == "manifest_unavailable"
    assert state["warnings"][0]["error_type"] == "RuntimeError"
    assert "error" not in state["warnings"][0]


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

    with pytest.raises(VNPlayConflictError, match="stale_scene_version"):
        await service.submit_turn(
            ready_session.id,
            input_text="Second",
            client_scene_version=0,
            idempotency_key="second",
        )


@pytest.mark.asyncio
async def test_submit_turn_recovers_expired_active_turn_lock(
    service: VNPlayService,
    ready_session,
) -> None:
    repo = service.repo
    stale_turn = repo.create_turn_request(
        session_id=ready_session.id,
        owner_user_id=42,
        idempotency_key="stale-active-turn",
        request_payload_hash="stale-active-turn",
        base_scene_version=0,
        status="model_calling",
    )
    repo.update_turn_request(
        stale_turn["id"],
        {"locked_until": "2000-01-01 00:00:00", "lease_owner": "worker-1"},
        owner_user_id=42,
    )
    repo.update_session(
        ready_session.id,
        {"active_turn_request_id": int(stale_turn["id"])},
        owner_user_id=42,
    )

    response = await service.submit_turn(
        ready_session.id,
        input_text="After crash",
        client_scene_version=0,
        idempotency_key="after-crash",
    )

    assert response.status == "completed"
    recovered = repo.get_turn_request(stale_turn["id"])
    assert recovered["status"] == "abandoned"
    assert recovered["error"] == {"code": "turn_lock_abandoned"}
    session = service.get_session(ready_session.id)
    assert session.scene_version == 1
    assert session.active_turn_request_id is None


@pytest.mark.asyncio
async def test_submit_turn_preserves_fresh_active_turn_lock(
    service: VNPlayService,
    ready_session,
) -> None:
    repo = service.repo
    fresh_turn = repo.create_turn_request(
        session_id=ready_session.id,
        owner_user_id=42,
        idempotency_key="fresh-active-turn",
        request_payload_hash="fresh-active-turn",
        base_scene_version=0,
        status="model_calling",
    )
    repo.update_turn_request(
        fresh_turn["id"],
        {"locked_until": "2999-01-01 00:00:00", "lease_owner": "worker-1"},
        owner_user_id=42,
    )
    repo.update_session(
        ready_session.id,
        {"active_turn_request_id": int(fresh_turn["id"])},
        owner_user_id=42,
    )

    with pytest.raises(VNPlayConflictError, match="turn_in_progress"):
        await service.submit_turn(
            ready_session.id,
            input_text="Still busy",
            client_scene_version=0,
            idempotency_key="still-busy",
        )

    assert repo.get_turn_request(fresh_turn["id"])["status"] == "model_calling"
    assert service.get_session(ready_session.id).active_turn_request_id == fresh_turn["id"]


@pytest.mark.asyncio
async def test_model_failure_marks_turn_failed_and_clears_lock(
    service_with_failing_adapter: VNPlayService,
    failing_ready_session,
) -> None:
    with pytest.raises(VNPlayTurnError, match="model_failed"):
        await service_with_failing_adapter.submit_turn(
            failing_ready_session.id,
            input_text="Break",
            client_scene_version=0,
            idempotency_key="fail-1",
        )

    with pytest.raises(VNPlayTurnError, match="model_failed"):
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
