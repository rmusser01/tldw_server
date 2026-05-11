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
from tldw_Server_API.app.core.VN_Play.constants import MODE_SCRIPTED_STORY
from tldw_Server_API.app.core.VN_Play.models import VisualDirectiveResolution
from tldw_Server_API.app.core.VN_Play.service import (
    DeterministicVNPlayTurnAdapter,
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
            "supported_output_schemas": ["narrative_dialogue", "scene_update"],
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
