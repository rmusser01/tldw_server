from collections.abc import Generator, Mapping
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPlay_DB import VNPlayRepository
from tldw_Server_API.app.core.DB_Management.VNPolicy_DB import VNProfileSnapshotRepository
from tldw_Server_API.app.core.VN_Play.adapters import (
    VNGenerationAdapterError,
    VNGenerationCallRequest,
    VNGenerationCallResult,
)
from tldw_Server_API.app.core.VN_Play.branch_navigation import filter_branch_events
from tldw_Server_API.app.core.VN_Play.constants import MODE_SCRIPTED_STORY
from tldw_Server_API.app.core.VN_Play.models import VisualDirectiveResolution
from tldw_Server_API.app.core.VN_Play.service import (
    DeterministicVNPlayTurnAdapter,
    VNPlayConflictError,
    VNPlayService,
    VNPlayTurnError,
)


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-play-scripted-generation-test")
    yield database
    database.close_connection()


class ScriptedGenerationAdapter:
    def __init__(self, raw_content: str | None = None, error_code: str | None = None) -> None:
        self.raw_content = raw_content or (
            '{"schema":"narrative_dialogue","narrative":[{"text":"Generated archive beat."}]}'
        )
        self.error_code = error_code
        self.calls: list[VNGenerationCallRequest] = []

    async def generate(self, request: VNGenerationCallRequest) -> VNGenerationCallResult:
        self.calls.append(request)
        if self.error_code is not None:
            raise VNGenerationAdapterError(self.error_code)
        return VNGenerationCallResult(
            raw_content=self.raw_content,
            usage_metadata={"total_tokens": 18},
            response_metadata={"model": "fake-model"},
        )


def _profile_snapshot(
    chacha_db: CharactersRAGDB,
    *,
    batch_cap: int = 1,
    supported_output_schemas: list[str] | None = None,
) -> dict[str, Any]:
    return VNProfileSnapshotRepository.initialized(chacha_db).create_profile_snapshot(
        owner_user_id=42,
        snapshot_type="generation",
        profile_id="story_default",
        profile_version=1,
        resource_type="script_version",
        resource_id=200,
        definition={
            "provider": "fake",
            "model": "fake-model",
            "moderation_required": False,
            "automatic_generation_batch_cap": batch_cap,
            "supported_output_schemas": supported_output_schemas
            or ["narrative_dialogue", "scene_update", "choice_set"],
        },
    )


def _scripted_service(
    chacha_db: CharactersRAGDB,
    *,
    program: Mapping[str, Any],
    adapter: ScriptedGenerationAdapter | None = None,
    profile_snapshot: Mapping[str, Any] | None = None,
) -> tuple[VNPlayService, Any, ScriptedGenerationAdapter, dict[str, Any]]:
    generation_adapter = adapter or ScriptedGenerationAdapter()
    snapshot = dict(profile_snapshot or _profile_snapshot(chacha_db))
    repo = VNPlayRepository.initialized(chacha_db)
    service = VNPlayService(
        repo=repo,
        owner_user_id=42,
        adapter=DeterministicVNPlayTurnAdapter(),
        generation_adapter=generation_adapter,
    )
    row = repo.create_session(
        owner_user_id=42,
        mode=MODE_SCRIPTED_STORY,
        title="Scripted generation",
        primary_character_id=1,
        vn_asset_pack_id=10,
        content_rating="general",
        seed="seed-1",
        script_id=100,
        script_version_id=200,
        script_manifest_snapshot_id=300,
        script_policy_snapshot_id=400,
        script_generation_profile_snapshot_id=int(snapshot["id"]),
        script_position={"label": "start", "index": 0},
    )
    version = {
        "id": 200,
        "script_id": 100,
        "generation_profile_snapshot_id": int(snapshot["id"]),
        "generation_profile_snapshots": {"default": int(snapshot["id"])},
        "program": dict(program),
    }
    service._script_version_for_session = lambda session: version  # type: ignore[method-assign]
    service._build_pack_manifest = lambda pack_id: {"assets": {}, "pack_id": pack_id}  # type: ignore[method-assign]
    return service, service.get_session(int(row["id"])), generation_adapter, snapshot


def _program_with_generates(*opcodes: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "vn_script_program.v1",
        "entry_label": "start",
        "primary_asset_pack_id": 10,
        "variables": {},
        "labels": {"start": [dict(opcode) for opcode in opcodes]},
    }


@pytest.mark.asyncio
async def test_automatic_script_generation_creates_revision_and_advances_scene(
    chacha_db: CharactersRAGDB,
) -> None:
    service, session, adapter, _ = _scripted_service(
        chacha_db,
        program=_program_with_generates(
            {
                "op": "generate",
                "id": "intro",
                "prompt": "Write the intro",
                "output_schema": "narrative_dialogue",
            }
        ),
    )

    response = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="auto-generate",
    )

    assert response["scene_version"] == 1
    model_event = next(event for event in response["events"] if event["event_type"] == "model_turn")
    assert model_event["event_payload"]["narrative_text"] == "Generated archive beat."
    generation = model_event["event_payload"]["generation_results"][0]
    assert generation["id"] == "intro"
    assert generation["model_invoked"] is True
    assert generation["revision_id"] >= 1
    assert len(adapter.calls) == 1
    persisted_generation = service.repo.get_generation_by_point(
        session_id=session.id,
        owner_user_id=42,
        generation_point_key="start:0:intro",
    )
    assert persisted_generation is not None
    assert persisted_generation["active_revision_id"] == generation["revision_id"]


@pytest.mark.asyncio
async def test_scene_update_generation_persists_visual_resolution_outcomes(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedGenerationAdapter(
        raw_content=(
            '{"schema":"scene_update",'
            '"narrative":[{"text":"The archive lights wake."}],'
            '"visual_directives":['
            '{"asset_type":"background","slot_key":"archive"},'
            '{"asset_type":"sprite","slot_key":"missing"}'
            "]}"
        )
    )
    service, session, _, _ = _scripted_service(
        chacha_db,
        adapter=adapter,
        program=_program_with_generates(
            {
                "op": "generate",
                "id": "scene",
                "prompt": "Set the scene",
                "output_schema": "scene_update",
            }
        ),
    )

    def resolve_visuals(
        manifest: Mapping[str, Any],
        directives: list[Mapping[str, Any]],
        *,
        seed: str,
    ) -> list[VisualDirectiveResolution]:
        return [
            VisualDirectiveResolution(
                applied=True,
                directive=dict(directives[0]),
                item={
                    "item_id": 11,
                    "asset_type": "background",
                    "slot_key": "archive",
                    "file_id": 101,
                },
            ),
            VisualDirectiveResolution(
                applied=False,
                directive=dict(directives[1]),
                reason="asset_not_found",
            ),
        ]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.VN_Play.service.resolve_scene_directives",
        resolve_visuals,
    )

    response = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="scene-update-generate",
    )

    generation = next(
        event["event_payload"]["generation_results"][0]
        for event in response["events"]
        if event["event_type"] == "model_turn"
    )
    revision = service.repo.get_generation_revision(
        int(generation["revision_id"]),
        owner_user_id=42,
    )
    assert revision is not None
    assert revision["applied_visuals"][0]["item"]["slot_key"] == "archive"
    assert revision["rejected_visuals"][0]["reason"] == "asset_not_found"


@pytest.mark.asyncio
async def test_confirmation_gated_generation_pauses_without_model_call(
    chacha_db: CharactersRAGDB,
) -> None:
    service, session, adapter, _ = _scripted_service(
        chacha_db,
        program=_program_with_generates(
            {
                "op": "generate",
                "id": "confirm-me",
                "prompt": "Write after confirmation",
                "requires_user_confirm": True,
            }
        ),
    )

    response = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="confirm-generate",
    )

    assert response["scene_version"] == 1
    assert response["script_state"]["position"]["waiting_reason"] == "generation_confirmation"
    assert adapter.calls == []
    generation = service.repo.get_generation_by_point(
        session_id=session.id,
        owner_user_id=42,
        generation_point_key="start:0:confirm-me",
    )
    assert generation is not None
    request = service.repo.get_generation_request(
        int(generation["latest_request_id"]),
        owner_user_id=42,
    )
    assert request is not None
    assert request["status"] == "pending_confirmation"


def _waiting_generation_request_id(service: VNPlayService, session_id: int) -> int:
    position = service.get_session(session_id).script_position
    waiting = position.get("waiting_generation_confirmation")
    assert isinstance(waiting, Mapping)
    return int(waiting["generation_request_id"])


@pytest.mark.asyncio
async def test_cancel_generation_confirmation_with_on_cancel_runs_authored_branch(
    chacha_db: CharactersRAGDB,
) -> None:
    service, session, adapter, _ = _scripted_service(
        chacha_db,
        program={
            "schema_version": "vn_script_program.v1",
            "entry_label": "start",
            "primary_asset_pack_id": 10,
            "variables": {},
            "labels": {
                "start": [
                    {
                        "op": "generate",
                        "id": "confirm-me",
                        "prompt": "Write after confirmation",
                        "requires_user_confirm": True,
                        "on_cancel": "cancelled",
                    }
                ],
                "cancelled": [
                    {"op": "narrate", "text": "The generation was cancelled."},
                    {"op": "end"},
                ],
            },
        },
    )
    first = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="cancel-with-branch-pause",
    )
    request_id = _waiting_generation_request_id(service, session.id)

    response = service.cancel_script_generation_request(
        session.id,
        generation_request_id=request_id,
        client_scene_version=int(first["scene_version"]),
        idempotency_key="cancel-with-branch",
    )

    assert adapter.calls == []
    assert response["scene_version"] == 2
    assert response["script_state"]["ended"] is True
    model_event = next(event for event in response["events"] if event["event_type"] == "model_turn")
    assert model_event["event_payload"]["narrative_text"] == "The generation was cancelled."
    request = service.repo.get_generation_request(request_id, owner_user_id=42)
    assert request is not None
    assert request["status"] == "canceled"
    assert request["cancel_action_id"] is not None


@pytest.mark.asyncio
async def test_cancel_generation_confirmation_without_on_cancel_leaves_stable_state(
    chacha_db: CharactersRAGDB,
) -> None:
    service, session, adapter, _ = _scripted_service(
        chacha_db,
        program=_program_with_generates(
            {
                "op": "generate",
                "id": "confirm-me",
                "prompt": "Write after confirmation",
                "requires_user_confirm": True,
            }
        ),
    )
    first = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="cancel-no-branch-pause",
    )
    request_id = _waiting_generation_request_id(service, session.id)

    response = service.cancel_script_generation_request(
        session.id,
        generation_request_id=request_id,
        client_scene_version=int(first["scene_version"]),
        idempotency_key="cancel-no-branch",
    )

    assert adapter.calls == []
    assert response["scene_version"] == 2
    assert response["script_state"]["position"]["waiting_reason"] == "generation_canceled"
    assert "waiting_generation_confirmation" not in service.get_session(session.id).script_position
    request = service.repo.get_generation_request(request_id, owner_user_id=42)
    assert request is not None
    assert request["status"] == "canceled"

    with pytest.raises(VNPlayConflictError, match="script_advance_blocked"):
        await service.advance_script(
            session.id,
            client_scene_version=int(response["scene_version"]),
            idempotency_key="advance-after-canceled-generation",
        )


@pytest.mark.asyncio
async def test_cancel_generation_invalid_on_cancel_does_not_mutate_request(
    chacha_db: CharactersRAGDB,
) -> None:
    service, session, adapter, _ = _scripted_service(
        chacha_db,
        program=_program_with_generates(
            {
                "op": "generate",
                "id": "pending",
                "prompt": "Wait for confirmation",
                "requires_user_confirm": True,
                "on_cancel": "missing_label",
            }
        ),
    )
    await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="cancel-invalid-branch-start",
    )
    request_id = _waiting_generation_request_id(service, session.id)

    with pytest.raises(VNPlayTurnError):
        service.cancel_script_generation_request(
            session.id,
            generation_request_id=request_id,
            client_scene_version=1,
            idempotency_key="cancel-invalid-branch",
        )

    assert adapter.calls == []
    request = service.repo.get_generation_request(request_id, owner_user_id=42)
    assert request is not None
    assert request["status"] == "pending_confirmation"


@pytest.mark.asyncio
async def test_generation_batch_cap_pauses_before_second_auto_generation(
    chacha_db: CharactersRAGDB,
) -> None:
    service, session, adapter, _ = _scripted_service(
        chacha_db,
        program=_program_with_generates(
            {"op": "generate", "id": "first", "prompt": "First"},
            {"op": "generate", "id": "second", "prompt": "Second"},
        ),
    )

    response = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="batch-first",
    )

    assert response["scene_version"] == 1
    assert response["script_state"]["position"]["waiting_reason"] == "generation_batch_limit"
    assert service.get_session(session.id).script_position["index"] == 1
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_generation_model_failure_persists_failed_revision_without_advancing(
    chacha_db: CharactersRAGDB,
) -> None:
    failing_adapter = ScriptedGenerationAdapter(error_code="provider_unavailable")
    service, session, adapter, _ = _scripted_service(
        chacha_db,
        adapter=failing_adapter,
        program=_program_with_generates({"op": "generate", "id": "fail", "prompt": "Fail"}),
    )

    with pytest.raises(VNPlayTurnError, match="provider_unavailable"):
        await service.advance_script(
            session.id,
            client_scene_version=0,
            idempotency_key="failed-generate",
        )

    assert len(adapter.calls) == 1
    assert service.get_session(session.id).scene_version == 0
    generation = service.repo.get_generation_by_point(
        session_id=session.id,
        owner_user_id=42,
        generation_point_key="start:0:fail",
    )
    assert generation is not None
    request = service.repo.get_generation_request(
        int(generation["latest_request_id"]),
        owner_user_id=42,
    )
    assert request is not None
    assert request["status"] == "failed"
    revisions = service.repo.list_generation_revisions(
        session_id=session.id,
        owner_user_id=42,
        generation_id=int(generation["id"]),
    )
    assert revisions[0]["status"] == "failed"
    assert revisions[0]["public_error_code"] == "provider_unavailable"


@pytest.mark.asyncio
async def test_regenerate_generation_creates_revision_history_and_activates_new_revision(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = ScriptedGenerationAdapter(
        raw_content='{"schema":"narrative_dialogue","narrative":[{"text":"First beat."}]}'
    )
    service, session, _, _ = _scripted_service(
        chacha_db,
        adapter=adapter,
        program=_program_with_generates(
            {"op": "generate", "id": "intro", "prompt": "Write the intro"}
        ),
    )
    first = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="regen-first",
    )
    generation = service.repo.get_generation_by_point(
        session_id=session.id,
        owner_user_id=42,
        generation_point_key="start:0:intro",
    )
    assert generation is not None
    first_revision_id = int(generation["active_revision_id"])
    adapter.raw_content = '{"schema":"narrative_dialogue","narrative":[{"text":"Second beat."}]}'

    response = await service.regenerate_script_generation(
        session.id,
        generation_id=int(generation["id"]),
        client_scene_version=int(first["scene_version"]),
        idempotency_key="regen-second",
    )

    updated_generation = service.repo.get_generation(int(generation["id"]), owner_user_id=42)
    assert updated_generation is not None
    assert int(updated_generation["active_revision_id"]) != first_revision_id
    revisions = service.repo.list_generation_revisions(
        session_id=session.id,
        owner_user_id=42,
        generation_id=int(generation["id"]),
    )
    assert [revision["status"] for revision in revisions] == ["succeeded", "succeeded"]
    assert int(revisions[0]["id"]) == int(updated_generation["active_revision_id"])
    model_event = next(event for event in response["events"] if event["event_type"] == "model_turn")
    assert model_event["event_payload"]["narrative_text"] == "Second beat."
    assert response["script_state"]["position"]["progress_token"]


@pytest.mark.asyncio
async def test_activate_generation_revision_switches_active_output_without_rewriting_history(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = ScriptedGenerationAdapter(
        raw_content='{"schema":"choice_set","choices":[{"id":"first","text":"First choice"}]}'
    )
    service, session, _, _ = _scripted_service(
        chacha_db,
        adapter=adapter,
        program={
            "schema_version": "vn_script_program.v1",
            "entry_label": "start",
            "primary_asset_pack_id": 10,
            "variables": {},
            "labels": {
                "start": [
                    {
                        "op": "generate",
                        "id": "choice",
                        "prompt": "Choice",
                        "output_schema": "choice_set",
                        "on_generated_choice": "after",
                    }
                ],
                "after": [{"op": "end"}],
            },
        },
    )
    first = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="activate-first",
    )
    generation = service.repo.get_generation_by_point(
        session_id=session.id,
        owner_user_id=42,
        generation_point_key="start:0:choice",
    )
    assert generation is not None
    first_revision_id = int(generation["active_revision_id"])
    adapter.raw_content = '{"schema":"choice_set","choices":[{"id":"second","text":"Second choice"}]}'
    second = await service.regenerate_script_generation(
        session.id,
        generation_id=int(generation["id"]),
        client_scene_version=int(first["scene_version"]),
        idempotency_key="activate-second",
    )
    updated_generation = service.repo.get_generation(int(generation["id"]), owner_user_id=42)
    assert updated_generation is not None
    second_revision_id = int(updated_generation["active_revision_id"])

    response = service.activate_script_generation_revision(
        session.id,
        generation_id=int(generation["id"]),
        revision_id=first_revision_id,
        client_scene_version=int(second["scene_version"]),
        idempotency_key="activate-first-revision",
    )

    assert second_revision_id != first_revision_id
    reactivated_generation = service.repo.get_generation(int(generation["id"]), owner_user_id=42)
    assert reactivated_generation is not None
    assert int(reactivated_generation["active_revision_id"]) == first_revision_id
    assert response["script_state"]["waiting_choice"]["choices"][0]["id"] == "first"
    event_types = [event["event_type"] for event in service.repo.list_events(session.id)]
    assert event_types.count("model_turn") == 2
    assert "script_generation_revision_activated" in event_types


@pytest.mark.asyncio
async def test_activate_generation_revision_is_blocked_after_downstream_material_events(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = ScriptedGenerationAdapter(
        raw_content='{"schema":"choice_set","choices":[{"id":"first","text":"First choice"}]}'
    )
    service, session, _, _ = _scripted_service(
        chacha_db,
        adapter=adapter,
        program={
            "schema_version": "vn_script_program.v1",
            "entry_label": "start",
            "primary_asset_pack_id": 10,
            "variables": {},
            "labels": {
                "start": [
                    {
                        "op": "generate",
                        "id": "choice",
                        "prompt": "Choice",
                        "output_schema": "choice_set",
                        "on_generated_choice": "after",
                    }
                ],
                "after": [{"op": "narrate", "text": "Committed downstream text."}, {"op": "end"}],
            },
        },
    )
    first = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="block-activate-first",
    )
    generation = service.repo.get_generation_by_point(
        session_id=session.id,
        owner_user_id=42,
        generation_point_key="start:0:choice",
    )
    assert generation is not None
    first_revision_id = int(generation["active_revision_id"])
    adapter.raw_content = '{"schema":"choice_set","choices":[{"id":"second","text":"Second choice"}]}'
    second = await service.regenerate_script_generation(
        session.id,
        generation_id=int(generation["id"]),
        client_scene_version=int(first["scene_version"]),
        idempotency_key="block-activate-second",
    )
    selected = await service.choose_script_option(
        session.id,
        choice_id="second",
        client_scene_version=int(second["scene_version"]),
        idempotency_key="commit-generated-choice",
    )

    with pytest.raises(VNPlayConflictError, match="revision_activation_blocked"):
        service.activate_script_generation_revision(
            session.id,
            generation_id=int(generation["id"]),
            revision_id=first_revision_id,
            client_scene_version=int(selected["scene_version"]),
            idempotency_key="blocked-activate-first",
        )

    unchanged_generation = service.repo.get_generation(int(generation["id"]), owner_user_id=42)
    assert unchanged_generation is not None
    assert int(unchanged_generation["active_revision_id"]) != first_revision_id


@pytest.mark.asyncio
async def test_regenerate_generation_is_blocked_after_downstream_material_events(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = ScriptedGenerationAdapter(
        raw_content='{"schema":"choice_set","choices":[{"id":"first","text":"First choice"}]}'
    )
    service, session, _, _ = _scripted_service(
        chacha_db,
        adapter=adapter,
        program={
            "schema_version": "vn_script_program.v1",
            "entry_label": "start",
            "primary_asset_pack_id": 10,
            "variables": {},
            "labels": {
                "start": [
                    {
                        "op": "generate",
                        "id": "choice",
                        "prompt": "Choice",
                        "output_schema": "choice_set",
                        "on_generated_choice": "after",
                    }
                ],
                "after": [{"op": "narrate", "text": "Committed downstream text."}, {"op": "end"}],
            },
        },
    )
    first = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="block-regen-first",
    )
    generation = service.repo.get_generation_by_point(
        session_id=session.id,
        owner_user_id=42,
        generation_point_key="start:0:choice",
    )
    assert generation is not None
    active_revision_id = int(generation["active_revision_id"])
    selected = await service.choose_script_option(
        session.id,
        choice_id="first",
        client_scene_version=int(first["scene_version"]),
        idempotency_key="commit-before-regen",
    )

    with pytest.raises(VNPlayConflictError, match="revision_activation_blocked"):
        await service.regenerate_script_generation(
            session.id,
            generation_id=int(generation["id"]),
            client_scene_version=int(selected["scene_version"]),
            idempotency_key="blocked-regenerate",
        )

    assert len(adapter.calls) == 1
    unchanged_generation = service.repo.get_generation(int(generation["id"]), owner_user_id=42)
    assert unchanged_generation is not None
    assert int(unchanged_generation["active_revision_id"]) == active_revision_id


@pytest.mark.asyncio
async def test_activate_narrative_revision_exposes_active_public_output(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = ScriptedGenerationAdapter(
        raw_content='{"schema":"narrative_dialogue","narrative":[{"text":"First public beat."}]}'
    )
    service, session, _, _ = _scripted_service(
        chacha_db,
        adapter=adapter,
        program=_program_with_generates({"op": "generate", "id": "intro", "prompt": "Intro"}),
    )
    first = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="narrative-activate-first",
    )
    generation = service.repo.get_generation_by_point(
        session_id=session.id,
        owner_user_id=42,
        generation_point_key="start:0:intro",
    )
    assert generation is not None
    first_revision_id = int(generation["active_revision_id"])
    adapter.raw_content = (
        '{"schema":"narrative_dialogue","narrative":[{"text":"Second public beat."}]}'
    )
    second = await service.regenerate_script_generation(
        session.id,
        generation_id=int(generation["id"]),
        client_scene_version=int(first["scene_version"]),
        idempotency_key="narrative-activate-second",
    )

    response = service.activate_script_generation_revision(
        session.id,
        generation_id=int(generation["id"]),
        revision_id=first_revision_id,
        client_scene_version=int(second["scene_version"]),
        idempotency_key="narrative-reactivate-first",
    )

    active_generation = response["script_state"]["active_generation"]
    assert active_generation["revision_id"] == first_revision_id
    assert active_generation["public_output"]["narrative"][0]["text"] == "First public beat."


@pytest.mark.asyncio
async def test_checkpoint_restore_restores_active_generation_revision_map(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = ScriptedGenerationAdapter(
        raw_content='{"schema":"narrative_dialogue","narrative":[{"text":"First checkpoint beat."}]}'
    )
    service, session, _, _ = _scripted_service(
        chacha_db,
        adapter=adapter,
        program=_program_with_generates(
            {"op": "generate", "id": "checkpointed", "prompt": "Checkpoint this"}
        ),
    )
    first = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="checkpoint-first",
    )
    generation = service.repo.get_generation_by_point(
        session_id=session.id,
        owner_user_id=42,
        generation_point_key="start:0:checkpointed",
    )
    assert generation is not None
    first_revision_id = int(generation["active_revision_id"])
    checkpoint = service.create_checkpoint(session.id, label="Before regen")
    adapter.raw_content = (
        '{"schema":"narrative_dialogue","narrative":[{"text":"Second checkpoint beat."}]}'
    )
    second = await service.regenerate_script_generation(
        session.id,
        generation_id=int(generation["id"]),
        client_scene_version=int(first["scene_version"]),
        idempotency_key="checkpoint-second",
    )
    changed_generation = service.repo.get_generation(int(generation["id"]), owner_user_id=42)
    assert changed_generation is not None
    assert int(changed_generation["active_revision_id"]) != first_revision_id

    service.restore_checkpoint(
        session.id,
        int(checkpoint["id"]),
        client_scene_version=int(second["scene_version"]),
        idempotency_key="restore-checkpoint-revision-map",
    )

    restored_generation = service.repo.get_generation(int(generation["id"]), owner_user_id=42)
    assert restored_generation is not None
    assert int(restored_generation["active_revision_id"]) == first_revision_id
    first_revision = service.repo.get_generation_revision(first_revision_id, owner_user_id=42)
    assert first_revision is not None
    assert int(restored_generation["latest_request_id"]) == int(
        first_revision["generation_request_id"]
    )


@pytest.mark.asyncio
async def test_generated_choice_selection_jumps_to_authored_on_generated_choice(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = ScriptedGenerationAdapter(
        raw_content=(
            '{"schema":"choice_set",'
            '"lead_in":"Mira studies your reaction.",'
            '"choices":[{"id":"ask_map","text":"Ask about the map","metadata":{"tone":"curious"}}]}'
        )
    )
    service, session, _, _ = _scripted_service(
        chacha_db,
        adapter=adapter,
        program={
            "schema_version": "vn_script_program.v1",
            "entry_label": "start",
            "primary_asset_pack_id": 10,
            "variables": {
                "last_generated_choice.id": {"public": True},
                "last_generated_choice.text": {"public": True},
                "last_generated_choice.metadata": {"public": True},
            },
            "labels": {
                "start": [
                    {
                        "op": "generate",
                        "id": "dynamic-choice",
                        "prompt": "Offer a dynamic choice",
                        "output_schema": "choice_set",
                        "on_generated_choice": "generated_branch",
                    }
                ],
                "generated_branch": [
                    {"op": "narrate", "text": "The authored generated-choice branch runs."},
                    {"op": "end"},
                ],
            },
        },
    )

    first = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="choice-set-generate",
    )

    waiting_choice = first["script_state"]["waiting_choice"]
    assert waiting_choice is not None
    public_choice = waiting_choice["choices"][0]
    assert public_choice == {
        "id": "ask_map",
        "text": "Ask about the map",
        "source": "generated",
        "generation_id": public_choice["generation_id"],
        "revision_id": public_choice["revision_id"],
    }
    assert "target" not in public_choice
    assert first["current_scene"]["visible_choices"][0] == public_choice

    second = await service.choose_script_option(
        session.id,
        choice_id="ask_map",
        client_scene_version=1,
        idempotency_key="select-generated-choice",
    )

    assert second["script_state"]["ended"] is True
    model_event = next(event for event in second["events"] if event["event_type"] == "model_turn")
    assert model_event["event_payload"]["narrative_text"] == "The authored generated-choice branch runs."
    variables = second["script_state"]["variables"]
    assert variables["last_generated_choice.id"] == "ask_map"
    assert variables["last_generated_choice.text"] == "Ask about the map"
    assert variables["last_generated_choice.metadata"] == {"tone": "curious"}


@pytest.mark.asyncio
async def test_generated_choice_metadata_is_stored_in_branch_events(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = ScriptedGenerationAdapter(
        raw_content=(
            '{"schema":"choice_set","choices":['
            '{"id":"follow_clue","text":"Follow the clue","metadata":{"risk":"high"}}]}'
        )
    )
    service, session, _, _ = _scripted_service(
        chacha_db,
        adapter=adapter,
        program={
            "schema_version": "vn_script_program.v1",
            "entry_label": "start",
            "primary_asset_pack_id": 10,
            "variables": {},
            "labels": {
                "start": [
                    {
                        "op": "generate",
                        "id": "choice",
                        "prompt": "Choice",
                        "output_schema": "choice_set",
                        "on_generated_choice": "after",
                    }
                ],
                "after": [
                    {"op": "narrate", "text": "Followed."},
                    {
                        "op": "generate",
                        "id": "literal-branch-beat",
                        "prompt": "Literal branch beat",
                        "narrative_text": "Generated branch beat.",
                    },
                    {"op": "narrate", "text": "Continued."},
                    {"op": "end"},
                ],
            },
        },
    )

    first = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="branch-metadata-generate",
    )
    generation_id = first["script_state"]["waiting_choice"]["choices"][0]["generation_id"]
    revision_id = first["script_state"]["waiting_choice"]["choices"][0]["revision_id"]

    await service.choose_script_option(
        session.id,
        choice_id="follow_clue",
        client_scene_version=1,
        idempotency_key="branch-metadata-select",
    )
    await service.advance_script(
        session.id,
        client_scene_version=2,
        idempotency_key="branch-metadata-continue",
    )

    all_events = service.repo.list_events(session.id)
    choice_event = next(
        event
        for event in all_events
        if event["event_type"] == "choice_selected"
        and event["event_payload"].get("choice_id") == "follow_clue"
    )
    assert choice_event["event_payload"]["generated_choice"] == {
        "generation_id": generation_id,
        "revision_id": revision_id,
        "choice_id": "follow_clue",
    }
    assert choice_event["event_payload"]["choice"]["metadata"] == {"risk": "high"}
    assert "raw_output" not in choice_event["event_payload"]
    assert "raw_prompt" not in choice_event["event_payload"]
    navigation = service.get_branch_navigation(session.id)
    branch = navigation["branches"][0]
    branch_id = branch["branch_id"]
    assert branch["generated_choice"] == {
        "generation_id": generation_id,
        "revision_id": revision_id,
        "choice_id": "follow_clue",
    }
    assert branch["branch_path"][-1]["generated_choice"] == branch["generated_choice"]
    assert navigation["active_path"][0]["generated_choice"] == branch["generated_choice"]
    assert "metadata" not in branch["generated_choice"]

    branch_events = [
        event
        for event in all_events
        if event["event_type"] in {"choice_selected", "model_turn", "scene_state_changed"}
        and event.get("branch_node_id") == branch_id
    ]
    assert [event["event_type"] for event in branch_events] == [
        "choice_selected",
        "model_turn",
        "scene_state_changed",
        "model_turn",
        "scene_state_changed",
    ]
    for event in branch_events:
        assert event["event_payload"]["branch_node_id"] == branch_id

    filtered_events, warnings = filter_branch_events(
        branch_id=branch_id,
        branches=service.repo.list_branches(session.id),
        events=all_events,
        replay_limit=1,
    )
    assert {warning["code"] for warning in warnings} == {
        "branch_interval_replay_limit_exceeded",
    }
    assert [event["event_type"] for event in filtered_events] == [
        "choice_selected",
        "model_turn",
        "scene_state_changed",
        "model_turn",
        "scene_state_changed",
    ]


@pytest.mark.asyncio
async def test_generated_choice_from_inactive_revision_cannot_be_selected(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = ScriptedGenerationAdapter(
        raw_content='{"schema":"choice_set","choices":[{"id":"active","text":"Active choice"}]}'
    )
    service, session, _, _ = _scripted_service(
        chacha_db,
        adapter=adapter,
        program={
            "schema_version": "vn_script_program.v1",
            "entry_label": "start",
            "primary_asset_pack_id": 10,
            "variables": {},
            "labels": {
                "start": [
                    {
                        "op": "generate",
                        "id": "choice",
                        "prompt": "Choice",
                        "output_schema": "choice_set",
                        "on_generated_choice": "after",
                    }
                ],
                "after": [{"op": "end"}],
            },
        },
    )
    first = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="inactive-choice-generate",
    )
    active_choice = first["script_state"]["waiting_choice"]["choices"][0]
    inactive_revision = service.repo.create_generation_revision(
        session_id=session.id,
        owner_user_id=42,
        generation_id=int(active_choice["generation_id"]),
        generation_request_id=1,
        status="succeeded",
        output_schema="choice_set",
        public_output={
            "schema": "choice_set",
            "choices": [{"id": "inactive", "text": "Inactive choice"}],
        },
    )

    position = dict(service.get_session(session.id).script_position)
    stale_choices = [dict(choice) for choice in position["waiting_choices"]]
    stale_choices.append(
        {
            "id": "inactive",
            "text": "Inactive choice",
            "source": "generated",
            "generation_id": int(active_choice["generation_id"]),
            "revision_id": int(inactive_revision["id"]),
            "generation_point_key": "start:0:choice",
            "target": "after",
        }
    )
    position["waiting_choices"] = stale_choices
    service.repo.update_session(
        session.id,
        {"script_position": position},
        owner_user_id=42,
    )

    with pytest.raises(VNPlayTurnError, match="invalid_choice_id"):
        await service.choose_script_option(
            session.id,
            choice_id="inactive",
            client_scene_version=1,
            idempotency_key="inactive-choice-select",
        )


@pytest.mark.asyncio
async def test_generated_choice_not_in_active_revision_cannot_be_selected(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = ScriptedGenerationAdapter(
        raw_content='{"schema":"choice_set","choices":[{"id":"active","text":"Active choice"}]}'
    )
    service, session, _, _ = _scripted_service(
        chacha_db,
        adapter=adapter,
        program={
            "schema_version": "vn_script_program.v1",
            "entry_label": "start",
            "primary_asset_pack_id": 10,
            "variables": {},
            "labels": {
                "start": [
                    {
                        "op": "generate",
                        "id": "choice",
                        "prompt": "Choice",
                        "output_schema": "choice_set",
                        "on_generated_choice": "after",
                    }
                ],
                "after": [{"op": "end"}],
            },
        },
    )
    first = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="tampered-choice-generate",
    )
    active_choice = first["script_state"]["waiting_choice"]["choices"][0]

    position = dict(service.get_session(session.id).script_position)
    tampered_choices = [dict(choice) for choice in position["waiting_choices"]]
    tampered_choices.append(
        {
            "id": "fabricated",
            "text": "Fabricated choice",
            "source": "generated",
            "generation_id": int(active_choice["generation_id"]),
            "revision_id": int(active_choice["revision_id"]),
            "generation_point_key": "start:0:choice",
            "target": "after",
        }
    )
    position["waiting_choices"] = tampered_choices
    service.repo.update_session(
        session.id,
        {"script_position": position},
        owner_user_id=42,
    )

    with pytest.raises(VNPlayTurnError, match="invalid_choice_id"):
        await service.choose_script_option(
            session.id,
            choice_id="fabricated",
            client_scene_version=1,
            idempotency_key="tampered-choice-select",
        )


@pytest.mark.asyncio
async def test_generated_choice_target_must_match_generation_opcode(
    chacha_db: CharactersRAGDB,
) -> None:
    adapter = ScriptedGenerationAdapter(
        raw_content='{"schema":"choice_set","choices":[{"id":"active","text":"Active choice"}]}'
    )
    service, session, _, _ = _scripted_service(
        chacha_db,
        adapter=adapter,
        program={
            "schema_version": "vn_script_program.v1",
            "entry_label": "start",
            "primary_asset_pack_id": 10,
            "variables": {},
            "labels": {
                "start": [
                    {
                        "op": "generate",
                        "id": "choice",
                        "prompt": "Choice",
                        "output_schema": "choice_set",
                        "on_generated_choice": "after",
                    }
                ],
                "after": [{"op": "end"}],
                "other": [{"op": "end"}],
            },
        },
    )
    first = await service.advance_script(
        session.id,
        client_scene_version=0,
        idempotency_key="tampered-target-generate",
    )
    active_choice = first["script_state"]["waiting_choice"]["choices"][0]

    position = dict(service.get_session(session.id).script_position)
    tampered_choices = [dict(choice) for choice in position["waiting_choices"]]
    tampered_choices[0]["target"] = "other"
    position["waiting_choices"] = tampered_choices
    service.repo.update_session(
        session.id,
        {"script_position": position},
        owner_user_id=42,
    )

    with pytest.raises(VNPlayTurnError, match="invalid_choice_id"):
        await service.choose_script_option(
            session.id,
            choice_id=str(active_choice["id"]),
            client_scene_version=1,
            idempotency_key="tampered-target-select",
        )
