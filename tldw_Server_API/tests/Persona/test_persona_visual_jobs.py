from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenResult


pytestmark = pytest.mark.unit


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeJobsManager:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        self.created.append(kwargs)
        return {"id": 42, "status": "queued", **kwargs}


def _create_persona_and_pack(db: CharactersRAGDB) -> tuple[str, dict[str, Any]]:
    persona_id = db.create_persona_profile({"user_id": "1", "name": "Visual Job Persona"})
    pack = db.create_persona_visual_pack(
        persona_id=persona_id,
        user_id="1",
        title="Generated Draft",
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {},
            "animations": {},
        },
    )
    return persona_id, pack


def test_create_persona_visual_generation_job_uses_domain_and_idempotency_key() -> None:
    from tldw_Server_API.app.core.Persona.visual_jobs import (
        PERSONA_VISUALS_DOMAIN,
        PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE,
        create_generate_candidate_job,
        visual_generate_candidate_idempotency_key,
    )

    manager = FakeJobsManager()

    job = create_generate_candidate_job(
        manager,
        user_id="user-1",
        persona_id="persona-1",
        pack_id="pack-1",
        prompt="make a thinking pose",
        target_state="thinking",
        backend="fake",
    )

    assert job["domain"] == PERSONA_VISUALS_DOMAIN
    assert job["queue"] == "generation"
    assert job["job_type"] == PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE
    assert job["owner_user_id"] == "user-1"
    assert job["idempotency_key"] == visual_generate_candidate_idempotency_key(
        user_id="user-1",
        persona_id="persona-1",
        pack_id="pack-1",
        prompt="make a thinking pose",
        target_state="thinking",
        backend="fake",
    )
    assert manager.created[0]["payload"] == {
        "user_id": "user-1",
        "persona_id": "persona-1",
        "pack_id": "pack-1",
        "prompt": "make a thinking pose",
        "target_state": "thinking",
        "backend": "fake",
    }


def test_create_persona_visual_generation_job_idempotency_distinguishes_prompt_and_backend() -> None:
    from tldw_Server_API.app.core.Persona.visual_jobs import create_generate_candidate_job

    manager = FakeJobsManager()

    create_generate_candidate_job(
        manager,
        user_id="user-1",
        persona_id="persona-1",
        pack_id="pack-1",
        prompt="make a thinking pose",
        target_state="thinking",
        backend="fake-a",
    )
    create_generate_candidate_job(
        manager,
        user_id="user-1",
        persona_id="persona-1",
        pack_id="pack-1",
        prompt="make a speaking pose",
        target_state="thinking",
        backend="fake-a",
    )
    create_generate_candidate_job(
        manager,
        user_id="user-1",
        persona_id="persona-1",
        pack_id="pack-1",
        prompt="make a thinking pose",
        target_state="thinking",
        backend="fake-b",
    )

    keys = [created["idempotency_key"] for created in manager.created]
    assert len(set(keys)) == 3


def test_create_persona_visual_generation_job_carries_recipe_intent_and_request_id() -> None:
    from tldw_Server_API.app.core.Persona.visual_jobs import create_generate_candidate_job

    manager = FakeJobsManager()
    recipe_intent = {
        "starter_pack_id": "research-buddy-basic",
        "recipe_output": "required_state_loops",
        "correlation_id": "recipe-request-1",
        "user_prompt": "make the speaking loop upbeat",
        "identity_brief": "Simple readable buddy.",
        "neutral_anchor": "Create one front-facing neutral pose.",
        "static_sheet": "Keep expression frames separate from timed loops.",
        "review_checks": ["neutral_identity_consistency"],
    }

    first = create_generate_candidate_job(
        manager,
        user_id="user-1",
        persona_id="persona-1",
        pack_id="pack-1",
        prompt="effective recipe prompt",
        target_state="speaking",
        backend="fake",
        request_id="recipe-request-1",
        recipe_intent=recipe_intent,
    )
    second = create_generate_candidate_job(
        manager,
        user_id="user-1",
        persona_id="persona-1",
        pack_id="pack-1",
        prompt="effective recipe prompt",
        target_state="speaking",
        backend="fake",
        request_id="recipe-request-2",
        recipe_intent={**recipe_intent, "correlation_id": "recipe-request-2"},
    )

    assert first["idempotency_key"] == second["idempotency_key"]
    assert manager.created[0]["request_id"] == "recipe-request-1"
    assert manager.created[0]["payload"] == {
        "user_id": "user-1",
        "persona_id": "persona-1",
        "pack_id": "pack-1",
        "prompt": "effective recipe prompt",
        "target_state": "speaking",
        "backend": "fake",
        "request_id": "recipe-request-1",
        "recipe_intent": recipe_intent,
    }
    assert (
        manager.created[0]["idempotency_key"]
        != create_generate_candidate_job(
            manager,
            user_id="user-1",
            persona_id="persona-1",
            pack_id="pack-1",
            prompt="different effective recipe prompt",
            target_state="speaking",
            backend="fake",
            request_id="recipe-request-1",
            recipe_intent=recipe_intent,
        )["idempotency_key"]
    )


def test_create_persona_visual_pack_export_job_includes_options_digest() -> None:
    from tldw_Server_API.app.core.Persona.visual_jobs import (
        PERSONA_VISUALS_DOMAIN,
        PERSONA_VISUAL_PACK_EXPORT_JOB_TYPE,
        create_visual_pack_export_job,
        visual_pack_export_idempotency_key,
    )

    manager = FakeJobsManager()

    job = create_visual_pack_export_job(
        manager,
        user_id="user-1",
        persona_id="persona-1",
        pack_id="pack-1",
        portability_job_id="export-row-1",
        request_id="request-1",
        options={"strict": True},
    )

    expected_key = visual_pack_export_idempotency_key(
        user_id="user-1",
        persona_id="persona-1",
        pack_id="pack-1",
        request_id="request-1",
        options={"strict": True},
    )
    assert job["domain"] == PERSONA_VISUALS_DOMAIN
    assert job["queue"] == "default"
    assert job["job_type"] == PERSONA_VISUAL_PACK_EXPORT_JOB_TYPE
    assert job["owner_user_id"] == "user-1"
    assert job["batch_group"] == "persona_visuals:user:user-1:persona:persona-1:pack:pack-1:portability:export:request-1"
    assert job["idempotency_key"] == expected_key
    assert expected_key != visual_pack_export_idempotency_key(
        user_id="user-1",
        persona_id="persona-1",
        pack_id="pack-1",
        request_id="request-1",
        options={"strict": False},
    )
    assert manager.created[0]["payload"] == {
        "user_id": "user-1",
        "persona_id": "persona-1",
        "pack_id": "pack-1",
        "portability_job_id": "export-row-1",
        "request_id": "request-1",
        "options": {"strict": True},
    }


def test_create_persona_visual_pack_import_preview_job_uses_archive_digest() -> None:
    from tldw_Server_API.app.core.Persona.visual_jobs import (
        PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
        create_visual_pack_import_preview_job,
        visual_pack_import_preview_idempotency_key,
    )

    manager = FakeJobsManager()

    job = create_visual_pack_import_preview_job(
        manager,
        user_id="user-1",
        preview_id="preview-1",
        archive_path="archives/pack.tldw-persona-vpack",
        request_id="request-1",
        target_persona_id="persona-2",
    )

    expected_key = visual_pack_import_preview_idempotency_key(
        user_id="user-1",
        preview_id="preview-1",
        archive_path="archives/pack.tldw-persona-vpack",
        request_id="request-1",
    )
    assert job["queue"] == "default"
    assert job["job_type"] == PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE
    assert job["batch_group"] == "persona_visuals:user:user-1:portability:import-preview:preview-1:request-1"
    assert job["idempotency_key"] == expected_key
    assert expected_key != visual_pack_import_preview_idempotency_key(
        user_id="user-1",
        preview_id="preview-1",
        archive_path="archives/other.tldw-persona-vpack",
        request_id="request-1",
    )
    assert manager.created[0]["payload"] == {
        "user_id": "user-1",
        "preview_id": "preview-1",
        "archive_path": "archives/pack.tldw-persona-vpack",
        "request_id": "request-1",
        "target_persona_id": "persona-2",
    }


def test_create_persona_visual_pack_import_commit_job_uses_preview_group() -> None:
    from tldw_Server_API.app.core.Persona.visual_jobs import (
        PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE,
        create_visual_pack_import_commit_job,
        visual_pack_import_commit_idempotency_key,
    )

    manager = FakeJobsManager()

    job = create_visual_pack_import_commit_job(
        manager,
        user_id="user-1",
        preview_id="preview-1",
        portability_job_id="commit-row-1",
        request_id="request-1",
        target_persona_id="persona-2",
        trust_mode="untrusted_import",
        target_mode="replace_draft",
        target_pack_id="draft-pack-1",
        title="Replacement Pack",
    )

    expected_key = visual_pack_import_commit_idempotency_key(
        user_id="user-1",
        preview_id="preview-1",
        request_id="request-1",
        trust_mode="untrusted_import",
        target_mode="replace_draft",
        target_pack_id="draft-pack-1",
        title="Replacement Pack",
        conflict_choice_explicit=False,
    )
    assert job["queue"] == "default"
    assert job["job_type"] == PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE
    assert job["batch_group"] == "persona_visuals:user:user-1:portability:import-commit:preview-1:request-1"
    assert job["idempotency_key"] == expected_key
    assert expected_key != visual_pack_import_commit_idempotency_key(
        user_id="user-1",
        preview_id="preview-1",
        request_id="request-1",
        trust_mode="untrusted_import",
        target_mode="create_new",
    )
    assert expected_key != visual_pack_import_commit_idempotency_key(
        user_id="user-1",
        preview_id="preview-1",
        request_id="request-1",
        trust_mode="untrusted_import",
        target_mode="replace_draft",
        target_pack_id="draft-pack-2",
        title="Replacement Pack",
        conflict_choice_explicit=False,
    )
    assert expected_key != visual_pack_import_commit_idempotency_key(
        user_id="user-1",
        preview_id="preview-1",
        request_id="request-1",
        trust_mode="untrusted_import",
        target_mode="replace_draft",
        target_pack_id="draft-pack-1",
        title="Replacement Pack",
        conflict_choice_explicit=True,
    )
    assert manager.created[0]["payload"] == {
        "user_id": "user-1",
        "preview_id": "preview-1",
        "portability_job_id": "commit-row-1",
        "request_id": "request-1",
        "target_persona_id": "persona-2",
        "trust_mode": "untrusted_import",
        "target_mode": "replace_draft",
        "target_pack_id": "draft-pack-1",
        "title": "Replacement Pack",
    }


@pytest.fixture()
def persona_visual_db(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "persona_visual_jobs.sqlite", client_id="persona-visual-jobs-tests")
    yield db
    db.close_connection()


@pytest.fixture(autouse=True)
def visual_storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "visuals"

    def _fake_visuals_dir(user_id: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(
        DatabasePaths,
        "get_user_persona_visuals_dir",
        staticmethod(_fake_visuals_dir),
    )
    return root


@pytest.mark.asyncio
async def test_generation_worker_fails_when_image_backend_unavailable(
    persona_visual_db: CharactersRAGDB,
) -> None:
    from tldw_Server_API.app.core.Persona.visual_jobs import (
        PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        PersonaVisualGenerationWorker,
    )

    persona_id, pack = _create_persona_and_pack(persona_visual_db)

    class NoBackendRegistry:
        def resolve_backend(self, requested: str | None) -> None:
            return None

    worker = PersonaVisualGenerationWorker(
        db=persona_visual_db,
        image_registry=NoBackendRegistry(),
    )

    with pytest.raises(ValueError, match="image_backend_unavailable"):
        await worker.handle_job_async(
            {
                "id": 100,
                "job_type": PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE,
                "payload": {
                    "user_id": "1",
                    "persona_id": persona_id,
                    "pack_id": pack["id"],
                    "prompt": "make a thinking pose",
                    "target_state": "thinking",
                    "backend": None,
                },
            }
        )


@pytest.mark.asyncio
async def test_generation_worker_stores_generated_asset_and_candidate(
    persona_visual_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Persona.visual_jobs import (
        PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs_worker import (
        PersonaVisualGenerationWorker,
    )

    persona_id, pack = _create_persona_and_pack(persona_visual_db)
    generated_requests: list[Any] = []

    class FakeAdapter:
        def generate(self, request: Any) -> ImageGenResult:
            generated_requests.append(request)
            content = _png_bytes()
            return ImageGenResult(
                content=content,
                content_type="image/png",
                bytes_len=len(content),
            )

    class FakeRegistry:
        def resolve_backend(self, requested: str | None) -> str:
            assert requested == "fake"
            return "fake"

        def get_adapter(self, name: str) -> FakeAdapter:
            assert name == "fake"
            return FakeAdapter()

    import tldw_Server_API.app.core.Persona.visual_jobs_worker as worker_module

    offloaded_call_names: list[str] = []
    real_to_thread = worker_module.asyncio.to_thread

    async def recording_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        offloaded_call_names.append(getattr(func, "__name__", repr(func)))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(worker_module.asyncio, "to_thread", recording_to_thread)

    worker = PersonaVisualGenerationWorker(
        db=persona_visual_db,
        image_registry=FakeRegistry(),
    )

    result = await worker.handle_job_async(
        {
            "id": 101,
            "job_type": PERSONA_VISUAL_GENERATE_CANDIDATE_JOB_TYPE,
            "payload": {
                "user_id": "1",
                "persona_id": persona_id,
                "pack_id": pack["id"],
                "prompt": "make a thinking pose",
                "target_state": "thinking",
                "backend": "fake",
                "request_id": "request-worker-1",
                "recipe_intent": {
                    "starter_pack_id": "starter-basic",
                    "recipe_output": "static_sheet",
                    "correlation_id": "corr-worker-1",
                    "identity_brief": "small helpful buddy",
                    "neutral_anchor": "front-facing neutral pose",
                    "static_sheet": "static talking variants",
                    "review_checks": ["consistent silhouette", "transparent background"],
                    "user_prompt": "raw user direction should not be copied",
                },
            },
        }
    )

    assert result["status"] == "candidate_created"
    assert generated_requests[0].backend == "fake"
    assert generated_requests[0].width == 1024
    assets = persona_visual_db.list_persona_visual_assets(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="1",
    )
    assert len(assets) == 1
    assert assets[0]["asset_role"] == "generated_candidate"
    candidates = persona_visual_db.list_persona_visual_candidates(
        pack_id=pack["id"],
        persona_id=persona_id,
        user_id="1",
    )
    assert len(candidates) == 1
    assert candidates[0]["job_id"] == "101"
    assert candidates[0]["generated_asset_ids"] == [assets[0]["id"]]
    assert candidates[0]["proposed_manifest_patch"]["states"]["thinking"]["animation_id"]
    provenance = candidates[0]["generation_provenance"]
    assert provenance["schema_version"] == 1
    assert provenance["generation_mode"] == "recipe_backed"
    assert provenance["request_id"] == "request-worker-1"
    assert provenance["job_id"] == "101"
    assert provenance["backend"] == "fake"
    assert provenance["target_state"] == "thinking"
    assert provenance["recipe"]["starter_pack_id"] == "starter-basic"
    assert provenance["recipe"]["recipe_output"] == "static_sheet"
    assert provenance["recipe"]["correlation_id"] == "corr-worker-1"
    assert provenance["recipe"]["review_checks"] == [
        "consistent silhouette",
        "transparent background",
    ]
    assert provenance["recipe"]["user_prompt_included"] is True
    serialized_provenance = json.dumps(provenance)
    assert "make a thinking pose" not in serialized_provenance
    assert "raw user direction should not be copied" not in serialized_provenance
    assert {"get_persona_visual_pack", "generate"} <= set(offloaded_call_names)
    assert any(name == "_persist_generated_candidate" for name in offloaded_call_names)
