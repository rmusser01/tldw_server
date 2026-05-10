from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, check_rate_limit, get_request_user
from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Persona.visual_library_service import PersonaVisualLibraryServiceError


pytestmark = pytest.mark.unit

fastapi_app = FastAPI()
fastapi_app.include_router(persona_ep.router, prefix="/api/v1/persona")


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (1, 1), (0, 255, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _valid_manifest(asset_id: str) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {
            "idle": {"animation_id": "idle"},
            "listening": {"animation_id": "idle"},
            "thinking": {"animation_id": "idle"},
            "speaking": {"animation_id": "idle"},
            "error": {"animation_id": "idle"},
        },
        "animations": {
            "idle": {
                "frames": [{"asset_id": asset_id, "duration_ms": 100}],
                "frame_rate": 1,
            }
        },
    }


def _client_for_user(user_id: int, db: CharactersRAGDB) -> TestClient:
    async def override_user() -> User:
        return User(
            id=user_id,
            username=f"persona-user-{user_id}",
            email=None,
            is_active=True,
        )

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    fastapi_app.dependency_overrides[check_rate_limit] = lambda: None
    return TestClient(fastapi_app)


@pytest.fixture()
def persona_db(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "persona_visuals_api.sqlite", client_id="persona-visuals-api-tests")
    yield db
    db.close_connection()


@pytest.fixture(autouse=True)
def visual_storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "visuals"
    outputs_root = tmp_path / "outputs"

    def _fake_visuals_dir(user_id: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _fake_temp_outputs_dir(user_id: str) -> Path:
        path = outputs_root / str(user_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    monkeypatch.setattr(
        DatabasePaths,
        "get_user_persona_visuals_dir",
        staticmethod(_fake_visuals_dir),
    )
    monkeypatch.setattr(
        DatabasePaths,
        "get_user_temp_outputs_dir",
        staticmethod(_fake_temp_outputs_dir),
    )
    yield root
    fastapi_app.dependency_overrides.clear()


def _create_persona(client: TestClient, *, name: str) -> str:
    response = client.post("/api/v1/persona/profiles", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _create_visual_pack(client: TestClient, persona_id: str, *, title: str = "Pack") -> dict:
    response = client.post(
        f"/api/v1/persona/profiles/{persona_id}/visual-packs",
        json={
            "title": title,
            "manifest": {
                "manifest_version": 1,
                "renderer_type": "sprite_frames",
                "states": {},
                "animations": {},
            },
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _upload_png(client: TestClient, persona_id: str, pack_id: str) -> dict:
    response = client.post(
        f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/assets",
        data={"asset_role": "frame"},
        files={"file": ("idle.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 201, response.text
    return response.json()


class FakeJobManager:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.jobs_by_id: dict[int, dict] = {}
        self.cancelled: list[tuple[int, str | None]] = []

    def create_job(self, **kwargs):
        job_id = 9001 + len(self.created)
        self.created.append(kwargs)
        job = {"id": job_id, "status": "queued", **kwargs}
        self.jobs_by_id[job_id] = job
        return job

    def get_job(self, job_id: int):
        return self.jobs_by_id.get(int(job_id))

    def cancel_job(self, job_id: int, reason: str | None = None):
        job = self.jobs_by_id.get(int(job_id))
        if job is None:
            return False
        job["status"] = "cancelled"
        self.cancelled.append((int(job_id), reason))
        return True


class FakeImageRegistry:
    """Small image registry double for persona visual readiness tests."""

    def __init__(
        self,
        *,
        enabled_backends: list[str] | None = None,
        default_backend: str | None = None,
        adapter_available: bool = True,
        raise_on_resolve: bool = False,
    ) -> None:
        self.enabled_backends = enabled_backends or []
        self.default_backend = default_backend
        self.adapter_available = adapter_available
        self.raise_on_resolve = raise_on_resolve

    def list_backend_names(self, *, include_disabled: bool = False) -> list[str]:
        return list(self.enabled_backends)

    def resolve_backend(self, requested: str | None) -> str | None:
        if self.raise_on_resolve:
            raise RuntimeError("backend resolution failed")
        name = (requested or self.default_backend or "").strip()
        if not name or name not in self.enabled_backends:
            return None
        return name

    def get_adapter(self, name: str):
        if name not in self.enabled_backends or not self.adapter_available:
            return None
        return object()


def test_create_list_and_activate_visual_pack(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Visual API Persona")
        pack = _create_visual_pack(client, persona_id)
        asset = _upload_png(client, persona_id, pack["id"])

        manifest_response = client.patch(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/manifest",
            json={"manifest": _valid_manifest(asset["id"]), "expected_version": pack["version"]},
        )
        assert manifest_response.status_code == 200, manifest_response.text

        activated = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/activate"
        )
        assert activated.status_code == 200, activated.text
        assert activated.json()["status"] == "active"

        listed = client.get(f"/api/v1/persona/profiles/{persona_id}/visual-packs")
        assert listed.status_code == 200, listed.text
        payload = listed.json()
        assert [item["id"] for item in payload] == [pack["id"]]
        assert payload[0]["status"] == "active"
        assert payload[0]["assets"][0]["id"] == asset["id"]


def test_duplicate_visual_pack_to_another_persona_creates_draft(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        source_persona_id = _create_persona(client, name="Source Persona")
        target_persona_id = _create_persona(client, name="Target Persona")
        source_pack = _create_visual_pack(client, source_persona_id, title="Source Visuals")
        asset = _upload_png(client, source_persona_id, source_pack["id"])
        manifest_response = client.patch(
            f"/api/v1/persona/profiles/{source_persona_id}/visual-packs/{source_pack['id']}/manifest",
            json={"manifest": _valid_manifest(asset["id"]), "expected_version": source_pack["version"]},
        )
        assert manifest_response.status_code == 200, manifest_response.text

        response = client.post(
            f"/api/v1/persona/profiles/{source_persona_id}/visual-packs/{source_pack['id']}/duplicate",
            json={"target_persona_id": target_persona_id, "title": "Target Draft"},
        )

        assert response.status_code == 201, response.text
        payload = response.json()
        assert payload["title"] == "Target Draft"
        assert payload["persona_id"] == target_persona_id
        assert payload["status"] == "draft"
        assert payload["parent_pack_id"] == source_pack["id"]
        assert "asset_id_map" not in payload
        assert len(payload["assets"]) == 1
        assert payload["assets"][0]["id"] != asset["id"]


def test_duplicate_visual_pack_rejects_same_persona_target(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Source Persona")
        source_pack = _create_visual_pack(client, persona_id, title="Source Visuals")

        response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{source_pack['id']}/duplicate",
            json={"target_persona_id": persona_id},
        )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "same_persona_target_unsupported"


def test_duplicate_visual_pack_rejects_other_user_target(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(2, persona_db) as other_client:
        other_persona_id = _create_persona(other_client, name="Other User Target")
    with _client_for_user(1, persona_db) as client:
        source_persona_id = _create_persona(client, name="Source Persona")
        source_pack = _create_visual_pack(client, source_persona_id, title="Source Visuals")

        response = client.post(
            f"/api/v1/persona/profiles/{source_persona_id}/visual-packs/{source_pack['id']}/duplicate",
            json={"target_persona_id": other_persona_id},
        )

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "target_persona_not_found"


def test_visual_library_save_list_update_and_delete(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Library Source Persona")
        pack = _create_visual_pack(client, persona_id, title="Library Source Pack")

        saved = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/library",
            json={
                "title": "Desk helper",
                "notes": "Good for focused research.",
                "tags": ["Research", "calm", "research"],
            },
        )
        assert saved.status_code == 201, saved.text
        item = saved.json()
        assert item["source_persona_id"] == persona_id
        assert item["source_pack_id"] == pack["id"]
        assert item["source_available"] is True
        assert item["tags"] == ["research", "calm"]

        listed = client.get("/api/v1/persona/visual-library")
        assert listed.status_code == 200, listed.text
        assert [entry["id"] for entry in listed.json()["items"]] == [item["id"]]

        updated = client.patch(
            f"/api/v1/persona/visual-library/{item['id']}",
            json={"title": "Updated helper", "notes": None, "tags": ["focus"]},
        )
        assert updated.status_code == 200, updated.text
        assert updated.json()["title"] == "Updated helper"
        assert updated.json()["notes"] is None
        assert updated.json()["tags"] == ["focus"]

        resaved = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/library",
            json={},
        )
        assert resaved.status_code == 201, resaved.text
        assert resaved.json()["id"] == item["id"]
        assert resaved.json()["title"] == "Updated helper"
        assert resaved.json()["notes"] is None
        assert resaved.json()["tags"] == ["focus"]

        deleted = client.delete(f"/api/v1/persona/visual-library/{item['id']}")
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == {"status": "deleted", "item_id": item["id"]}

        listed_after_delete = client.get("/api/v1/persona/visual-library")
        assert listed_after_delete.status_code == 200
        assert listed_after_delete.json()["items"] == []


def test_visual_library_use_creates_target_draft_without_activation(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        source_persona_id = _create_persona(client, name="Library Use Source")
        target_persona_id = _create_persona(client, name="Library Use Target")
        source_pack = _create_visual_pack(client, source_persona_id, title="Source Library Pack")
        target_pack = _create_visual_pack(client, target_persona_id, title="Target Active Pack")
        asset = _upload_png(client, source_persona_id, source_pack["id"])
        target_asset = _upload_png(client, target_persona_id, target_pack["id"])
        manifest_response = client.patch(
            f"/api/v1/persona/profiles/{source_persona_id}/visual-packs/{source_pack['id']}/manifest",
            json={"manifest": _valid_manifest(asset["id"]), "expected_version": source_pack["version"]},
        )
        assert manifest_response.status_code == 200, manifest_response.text
        source_pack = manifest_response.json()
        target_manifest_response = client.patch(
            f"/api/v1/persona/profiles/{target_persona_id}/visual-packs/{target_pack['id']}/manifest",
            json={"manifest": _valid_manifest(target_asset["id"]), "expected_version": target_pack["version"]},
        )
        assert target_manifest_response.status_code == 200, target_manifest_response.text
        target_pack = target_manifest_response.json()
        source_active = client.post(
            f"/api/v1/persona/profiles/{source_persona_id}/visual-packs/{source_pack['id']}/activate"
        )
        assert source_active.status_code == 200, source_active.text
        target_active = client.post(
            f"/api/v1/persona/profiles/{target_persona_id}/visual-packs/{target_pack['id']}/activate"
        )
        assert target_active.status_code == 200, target_active.text
        saved = client.post(
            f"/api/v1/persona/profiles/{source_persona_id}/visual-packs/{source_pack['id']}/library",
            json={"title": "Reusable library pack"},
        )
        assert saved.status_code == 201, saved.text
        item_id = saved.json()["id"]

        used = client.post(
            f"/api/v1/persona/visual-library/{item_id}/use",
            json={"target_persona_id": target_persona_id, "title": "Target Draft From Library"},
        )

        assert used.status_code == 201, used.text
        payload = used.json()
        assert payload["title"] == "Target Draft From Library"
        assert payload["persona_id"] == target_persona_id
        assert payload["status"] == "draft"
        assert payload["parent_pack_id"] == source_pack["id"]
        assert len(payload["assets"]) == 1
        assert payload["assets"][0]["id"] != asset["id"]
        assert persona_db.get_active_persona_visual_pack(
            persona_id=source_persona_id,
            user_id="1",
        )["id"] == source_pack["id"]
        assert persona_db.get_active_persona_visual_pack(
            persona_id=target_persona_id,
            user_id="1",
        )["id"] == target_pack["id"]


def test_visual_library_stale_source_returns_409_but_delete_succeeds(
    persona_db: CharactersRAGDB,
) -> None:
    with _client_for_user(1, persona_db) as client:
        source_persona_id = _create_persona(client, name="Stale Library Source")
        target_persona_id = _create_persona(client, name="Stale Library Target")
        source_pack = _create_visual_pack(client, source_persona_id, title="Soon Stale Pack")
        saved = client.post(
            f"/api/v1/persona/profiles/{source_persona_id}/visual-packs/{source_pack['id']}/library",
            json={"title": "Reusable stale pack"},
        )
        assert saved.status_code == 201, saved.text
        item_id = saved.json()["id"]
        assert persona_db.soft_delete_persona_visual_pack_with_assets(
            pack_id=source_pack["id"],
            persona_id=source_persona_id,
            user_id="1",
        )

        listed = client.get("/api/v1/persona/visual-library")
        assert listed.status_code == 200, listed.text
        assert listed.json()["items"][0]["source_available"] is False

        used = client.post(
            f"/api/v1/persona/visual-library/{item_id}/use",
            json={"target_persona_id": target_persona_id},
        )
        assert used.status_code == 409
        assert used.json()["detail"]["code"] == "source_pack_unavailable"

        deleted = client.delete(f"/api/v1/persona/visual-library/{item_id}")
        assert deleted.status_code == 200, deleted.text


def test_visual_library_rejects_cross_user_item_source_and_target(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(2, persona_db) as other_client:
        other_persona_id = _create_persona(other_client, name="Other Library Target")
    with _client_for_user(1, persona_db) as client:
        source_persona_id = _create_persona(client, name="Owner Library Source")
        source_pack = _create_visual_pack(client, source_persona_id, title="Owner Library Pack")
        saved = client.post(
            f"/api/v1/persona/profiles/{source_persona_id}/visual-packs/{source_pack['id']}/library",
            json={"title": "Private library item"},
        )
        assert saved.status_code == 201, saved.text
        item_id = saved.json()["id"]

        other_target = client.post(
            f"/api/v1/persona/visual-library/{item_id}/use",
            json={"target_persona_id": other_persona_id},
        )
        assert other_target.status_code == 404
        assert other_target.json()["detail"]["code"] == "target_persona_not_found"

    with _client_for_user(2, persona_db) as other_client:
        save_other_source = other_client.post(
            f"/api/v1/persona/profiles/{source_persona_id}/visual-packs/{source_pack['id']}/library",
            json={"title": "Not mine"},
        )
        assert save_other_source.status_code == 404
        assert save_other_source.json()["detail"]["code"] == "source_pack_not_found"

        listed = other_client.get("/api/v1/persona/visual-library")
        assert listed.status_code == 200
        assert listed.json()["items"] == []

        update_other_item = other_client.patch(
            f"/api/v1/persona/visual-library/{item_id}",
            json={"title": "Not mine"},
        )
        assert update_other_item.status_code == 404
        assert update_other_item.json()["detail"]["code"] == "library_item_not_found"

        use_other_item = other_client.post(
            f"/api/v1/persona/visual-library/{item_id}/use",
            json={"target_persona_id": other_persona_id},
        )
        assert use_other_item.status_code == 404
        assert use_other_item.json()["detail"]["code"] == "library_item_not_found"


def test_visual_library_error_mapper_returns_403_for_forbidden() -> None:
    exc = PersonaVisualLibraryServiceError(
        "forbidden",
        "Library item does not belong to the current user.",
        details={"item_id": "library-1"},
    )

    http_exc = persona_ep._persona_visual_library_service_error_to_http(exc)

    assert http_exc.status_code == 403
    assert http_exc.detail == {
        "code": "forbidden",
        "message": "Library item does not belong to the current user.",
        "details": {"item_id": "library-1"},
    }


def test_upload_rejects_unsupported_mime_type(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Visual Upload Persona")
        pack = _create_visual_pack(client, persona_id)

        response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/assets",
            files={"file": ("bad.txt", b"not an image", "text/plain")},
        )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "unsupported_mime_type"


def test_upload_rejects_oversized_asset_before_service_call(persona_db: CharactersRAGDB) -> None:
    from tldw_Server_API.app.core.Persona.visual_service import MAX_VISUAL_UPLOAD_BYTES

    class UnexpectedService:
        def create_asset_from_upload(self, **_kwargs):
            raise AssertionError("oversized upload reached visual service")

    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_service] = lambda: UnexpectedService()
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Oversized Visual Upload Persona")
        pack = _create_visual_pack(client, persona_id)

        response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/assets",
            data={"asset_role": "frame"},
            files={
                "file": (
                    "too-large.png",
                    BytesIO(b"x" * (MAX_VISUAL_UPLOAD_BYTES + 1)),
                    "image/png",
                )
            },
        )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "upload_too_large"


def test_activation_rejects_manifest_without_required_states(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Invalid Manifest Persona")
        pack = _create_visual_pack(client, persona_id)

        response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/activate"
        )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "invalid_manifest"


def test_other_user_cannot_access_pack(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Owner Persona")
        pack = _create_visual_pack(client, persona_id)

    with _client_for_user(2, persona_db) as other_client:
        response = other_client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}"
        )

    assert response.status_code == 404


def test_accept_and_reject_generated_candidates(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Candidate Persona")
        pack = _create_visual_pack(client, persona_id)
        asset = _upload_png(client, persona_id, pack["id"])
        accepted = persona_db.create_persona_visual_candidate(
            pack_id=pack["id"],
            persona_id=persona_id,
            user_id="1",
            job_id="job-1",
            proposed_manifest_patch={
                "states": {"thinking": {"animation_id": "generated-thinking"}},
                "animations": {
                    "generated-thinking": {
                        "asset_ids": [asset["id"]],
                        "frame_rate": 1,
                        "loop": True,
                    }
                },
            },
            generated_asset_ids=[asset["id"]],
            prompt="make thinking",
        )
        rejected = persona_db.create_persona_visual_candidate(
            pack_id=pack["id"],
            persona_id=persona_id,
            user_id="1",
            job_id="job-2",
            proposed_manifest_patch={},
            generated_asset_ids=[],
            prompt="make another",
        )

        accepted_response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/candidates/{accepted['id']}/review",
            json={"status": "accepted"},
        )
        rejected_response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/candidates/{rejected['id']}/review",
            json={"status": "rejected", "failure_reason": "not useful"},
        )

        assert accepted_response.status_code == 200, accepted_response.text
        assert accepted_response.json()["status"] == "accepted"
        detail = client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}"
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["status"] == "draft"
        assert detail.json()["manifest"]["states"]["thinking"]["animation_id"] == "generated-thinking"
        assert rejected_response.status_code == 200, rejected_response.text
        assert rejected_response.json()["status"] == "rejected"
        assert rejected_response.json()["failure_reason"] == "not useful"


def test_list_generated_candidates_returns_preview_asset_urls(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Candidate List Persona")
        pack = _create_visual_pack(client, persona_id)
        asset = _upload_png(client, persona_id, pack["id"])
        candidate = persona_db.create_persona_visual_candidate(
            pack_id=pack["id"],
            persona_id=persona_id,
            user_id="1",
            job_id="job-preview",
            proposed_manifest_patch={},
            generated_asset_ids=[asset["id"]],
            prompt="make preview",
        )

        response = client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/generated-candidates"
        )

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["candidates"][0]["id"] == candidate["id"]
        assert payload["candidates"][0]["generated_assets"][0]["id"] == asset["id"]
        assert payload["candidates"][0]["generated_assets"][0]["url"].endswith(
            f"/visual-packs/{pack['id']}/assets/{asset['id']}/content"
        )
        detail = client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/generated-candidates/{candidate['id']}"
        )
        assert detail.status_code == 200, detail.text
        assert detail.json()["generated_assets"][0]["id"] == asset["id"]


def test_create_generation_job_for_visual_pack(persona_db: CharactersRAGDB) -> None:
    manager = FakeJobManager()
    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_job_manager] = lambda: manager
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Generation Job Persona")
        pack = _create_visual_pack(client, persona_id)

        response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/generation-jobs",
            json={
                "prompt": "make a speaking pose",
                "target_state": "speaking",
                "backend": "fake",
            },
        )

        assert response.status_code == 200, response.text
        assert response.json()["job_id"] == "9001"
        assert manager.created[0]["domain"] == "persona_visuals"
        assert manager.created[0]["payload"]["target_state"] == "speaking"


def test_create_generation_job_rejects_other_user_pack(persona_db: CharactersRAGDB) -> None:
    manager = FakeJobManager()
    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_job_manager] = lambda: manager
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Generation Owner Persona")
        pack = _create_visual_pack(client, persona_id)

    with _client_for_user(2, persona_db) as other_client:
        response = other_client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/generation-jobs",
            json={"prompt": "steal pack", "target_state": "idle"},
        )

    assert response.status_code == 404
    assert manager.created == []


def test_visual_generation_readiness_reports_disabled_worker(
    persona_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PERSONA_VISUAL_GENERATION_WORKER_ENABLED", raising=False)
    monkeypatch.setenv("PERSONA_VISUAL_GENERATION_JOBS_QUEUE", "persona-generation")
    monkeypatch.setattr(
        persona_ep,
        "get_image_generation_registry",
        lambda: FakeImageRegistry(
            enabled_backends=["openrouter"],
            default_backend="openrouter",
        ),
        raising=False,
    )

    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Readiness Worker Persona")
        pack = _create_visual_pack(client, persona_id)

        response = client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/generation-readiness"
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["available"] is False
    assert payload["worker_enabled"] is False
    assert payload["queue"] == "persona-generation"
    assert payload["image_backend_available"] is True
    assert payload["default_backend"] == "openrouter"
    assert payload["enabled_backends"] == ["openrouter"]
    assert payload["reasons"] == ["jobs_worker_disabled"]


def test_visual_generation_readiness_reports_missing_image_provider(
    persona_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONA_VISUAL_GENERATION_WORKER_ENABLED", "1")
    monkeypatch.setattr(
        persona_ep,
        "get_image_generation_registry",
        lambda: FakeImageRegistry(enabled_backends=[], default_backend=None),
        raising=False,
    )

    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Readiness Provider Persona")
        pack = _create_visual_pack(client, persona_id)

        response = client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/generation-readiness"
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["available"] is False
    assert payload["worker_enabled"] is True
    assert payload["image_backend_available"] is False
    assert payload["default_backend"] is None
    assert payload["enabled_backends"] == []
    assert payload["reasons"] == ["image_backend_unavailable"]


def test_visual_generation_readiness_reports_available_backend(
    persona_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONA_VISUAL_GENERATION_WORKER_ENABLED", "true")
    monkeypatch.setattr(
        persona_ep,
        "get_image_generation_registry",
        lambda: FakeImageRegistry(
            enabled_backends=["openrouter", "novita"],
            default_backend="openrouter",
        ),
        raising=False,
    )

    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Readiness Ready Persona")
        pack = _create_visual_pack(client, persona_id)

        response = client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/generation-readiness?backend=novita"
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["available"] is True
    assert payload["worker_enabled"] is True
    assert payload["image_backend_available"] is True
    assert payload["default_backend"] == "openrouter"
    assert payload["requested_backend"] == "novita"
    assert payload["requested_backend_available"] is True
    assert payload["enabled_backends"] == ["openrouter", "novita"]
    assert payload["reasons"] == []


def test_visual_generation_readiness_reports_adapter_unavailable(
    persona_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONA_VISUAL_GENERATION_WORKER_ENABLED", "true")
    monkeypatch.setattr(
        persona_ep,
        "get_image_generation_registry",
        lambda: FakeImageRegistry(
            enabled_backends=["openrouter"],
            default_backend="openrouter",
            adapter_available=False,
        ),
        raising=False,
    )

    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Readiness Adapter Persona")
        pack = _create_visual_pack(client, persona_id)

        response = client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/generation-readiness"
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["available"] is False
    assert payload["worker_enabled"] is True
    assert payload["image_backend_available"] is False
    assert payload["default_backend"] == "openrouter"
    assert payload["enabled_backends"] == ["openrouter"]
    assert payload["reasons"] == ["image_adapter_unavailable"]


def test_visual_generation_readiness_fails_closed_when_dependency_check_errors(
    persona_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONA_VISUAL_GENERATION_WORKER_ENABLED", "true")
    monkeypatch.setenv("PERSONA_VISUAL_GENERATION_JOBS_QUEUE", "persona-generation")
    monkeypatch.setattr(
        persona_ep,
        "get_image_generation_registry",
        lambda: FakeImageRegistry(
            enabled_backends=["openrouter"],
            default_backend="openrouter",
            raise_on_resolve=True,
        ),
        raising=False,
    )

    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Readiness Dependency Persona")
        pack = _create_visual_pack(client, persona_id)

        response = client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/generation-readiness?backend=openrouter"
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["available"] is False
    assert payload["worker_enabled"] is True
    assert payload["queue"] == "persona-generation"
    assert payload["image_backend_available"] is False
    assert payload["default_backend"] is None
    assert payload["requested_backend"] == "openrouter"
    assert payload["requested_backend_available"] is False
    assert payload["enabled_backends"] == []
    assert payload["reasons"] == ["dependency_check_failed"]


def test_start_visual_pack_export_creates_portability_job(persona_db: CharactersRAGDB) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs import PERSONA_VISUAL_PACK_EXPORT_JOB_TYPE

    manager = FakeJobManager()
    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_job_manager] = lambda: manager
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Export API Persona")
        pack = _create_visual_pack(client, persona_id)

        response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/export",
            json={"request_id": "req-export", "strict": False},
        )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["job_id"] == "9001"
    assert payload["operation"] == "export"
    assert payload["persona_id"] == persona_id
    assert payload["pack_id"] == pack["id"]
    assert payload["status"] == "queued"
    assert payload["stage"] == "queued"
    assert manager.created[0]["job_type"] == PERSONA_VISUAL_PACK_EXPORT_JOB_TYPE
    assert manager.created[0]["payload"]["portability_job_id"] == ""
    repo = PersonaVisualPortabilityRepository.initialized(persona_db)
    row = repo.get_portability_job_by_job_id("9001", owner_user_id="1")
    assert row is not None
    assert row["operation"] == "export"
    assert row["persona_id"] == persona_id
    assert row["pack_id"] == pack["id"]


def test_visual_pack_export_status_download_and_scope(persona_db: CharactersRAGDB) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_portability.constants import (
        PERSONA_VISUAL_PACK_EXTENSION,
    )

    manager = FakeJobManager()
    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_job_manager] = lambda: manager
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Export Download Persona")
        pack = _create_visual_pack(client, persona_id)
        started = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/export",
            json={"request_id": "req-download"},
        )
        assert started.status_code == 202, started.text
        job_id = started.json()["job_id"]

        archive_root = DatabasePaths.get_user_temp_outputs_dir("1") / "persona_visual_packs"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive_path = archive_root / f"done{PERSONA_VISUAL_PACK_EXTENSION}"
        archive_path.write_bytes(b"portable visual archive")
        repo = PersonaVisualPortabilityRepository.initialized(persona_db)
        repo.update_portability_job(
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "archive_path": str(archive_path),
                "archive_sha256": "a" * 64,
                "canonical_payload_fingerprint": "b" * 64,
            },
            owner_user_id="1",
        )

        status_response = client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/exports/{job_id}"
        )
        download_response = client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/exports/{job_id}/download"
        )

    assert status_response.status_code == 200, status_response.text
    assert status_response.json()["status"] == "completed"
    assert status_response.json()["archive_sha256"] == "a" * 64
    assert download_response.status_code == 200, download_response.text
    assert download_response.content == b"portable visual archive"

    with _client_for_user(2, persona_db) as other_client:
        denied = other_client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/exports/{job_id}"
        )

    assert denied.status_code == 404


def test_cancel_visual_pack_export_updates_job_and_portability_row(
    persona_db: CharactersRAGDB,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )

    manager = FakeJobManager()
    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_job_manager] = lambda: manager
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Export Cancel Persona")
        pack = _create_visual_pack(client, persona_id)
        started = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/export",
            json={"request_id": "req-cancel"},
        )
        assert started.status_code == 202, started.text
        job_id = started.json()["job_id"]

        response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/exports/{job_id}/cancel"
        )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelled"
    assert manager.cancelled == [(int(job_id), "persona_visual_pack_export_cancel_requested")]
    repo = PersonaVisualPortabilityRepository.initialized(persona_db)
    row = repo.get_portability_job_by_job_id(job_id, owner_user_id="1")
    assert row is not None
    assert row["status"] == "cancelled"
    assert row["stage"] == "cancelled"


def test_start_visual_pack_export_rejects_other_user_pack(persona_db: CharactersRAGDB) -> None:
    manager = FakeJobManager()
    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_job_manager] = lambda: manager
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Export Owner Persona")
        pack = _create_visual_pack(client, persona_id)

    with _client_for_user(2, persona_db) as other_client:
        response = other_client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/export",
            json={"request_id": "req-denied"},
        )

    assert response.status_code == 404
    assert manager.created == []


def test_start_import_preview_creates_preview_job_without_mutating_packs(
    persona_db: CharactersRAGDB,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs import (
        PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE,
    )
    from tldw_Server_API.app.core.Persona.visual_portability.constants import (
        PERSONA_VISUAL_PACK_EXTENSION,
    )

    manager = FakeJobManager()
    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_job_manager] = lambda: manager
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Import Preview Persona")
        _create_visual_pack(client, persona_id)
        packs_before = client.get(f"/api/v1/persona/profiles/{persona_id}/visual-packs").json()

        response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/import-previews",
            files={
                "archive": (
                    f"pack{PERSONA_VISUAL_PACK_EXTENSION}",
                    b"portable visual archive",
                    "application/zip",
                )
            },
        )
        packs_after = client.get(f"/api/v1/persona/profiles/{persona_id}/visual-packs").json()

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["job_id"] == "9001"
    assert payload["operation"] == "import_preview"
    assert payload["target_persona_id"] == persona_id
    assert payload["status"] == "queued"
    assert manager.created[0]["job_type"] == PERSONA_VISUAL_PACK_IMPORT_PREVIEW_JOB_TYPE
    repo = PersonaVisualPortabilityRepository.initialized(persona_db)
    preview = repo.get_import_preview(payload["preview_id"], owner_user_id="1")
    assert preview is not None
    assert preview["target_persona_id"] == persona_id
    assert packs_after == packs_before


def test_import_preview_status_is_scoped(persona_db: CharactersRAGDB) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_portability.constants import (
        PERSONA_VISUAL_PACK_EXTENSION,
    )

    manager = FakeJobManager()
    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_job_manager] = lambda: manager
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Import Preview Status Persona")
        started = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/import-previews",
            files={
                "archive": (
                    f"pack{PERSONA_VISUAL_PACK_EXTENSION}",
                    b"portable visual archive",
                    "application/zip",
                )
            },
        )
        assert started.status_code == 202, started.text
        payload = started.json()
        repo = PersonaVisualPortabilityRepository.initialized(persona_db)
        repo.update_import_preview(
            payload["preview_id"],
            {
                "status": "completed",
                "stage": "completed",
                "schema_version": "tldw.persona_visual_pack.v1",
                "bundle_summary": {"pack_title": "Imported Visuals"},
                "target_warnings": [],
            },
            owner_user_id="1",
        )
        repo.update_portability_job(
            payload["job_id"],
            {"status": "completed", "stage": "completed"},
            owner_user_id="1",
        )

        status_response = client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/import-previews/{payload['preview_id']}"
        )

    assert status_response.status_code == 200, status_response.text
    body = status_response.json()
    assert body["status"] == "completed"
    assert body["bundle_summary"] == {"pack_title": "Imported Visuals"}
    assert body["target_warnings"] == []

    with _client_for_user(2, persona_db) as other_client:
        denied = other_client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/import-previews/{payload['preview_id']}"
        )

    assert denied.status_code == 404


def test_cancel_import_preview_updates_preview_and_portability_row(
    persona_db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_portability.constants import (
        PERSONA_VISUAL_PACK_EXTENSION,
    )

    manager = FakeJobManager()
    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_job_manager] = lambda: manager
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Import Preview Cancel Persona")
        archive_path = tmp_path / f"queued{PERSONA_VISUAL_PACK_EXTENSION}"
        archive_path.write_bytes(b"portable visual archive")
        repo = PersonaVisualPortabilityRepository.initialized(persona_db)
        preview = repo.create_import_preview(
            owner_user_id="1",
            job_id="9001",
            status="queued",
            stage="queued",
            archive_path=str(archive_path),
            target_persona_id=persona_id,
        )
        repo.create_portability_job(
            owner_user_id="1",
            job_id="9001",
            operation="import_preview",
            status="queued",
            stage="queued",
            preview_id=preview["id"],
            archive_path=str(archive_path),
        )
        manager.jobs_by_id[9001] = {"id": 9001, "status": "queued"}

        response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/import-previews/{preview['id']}/cancel"
        )

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelled"
    assert response.json()["stage"] == "cancelled"
    assert manager.cancelled == [(9001, "persona_visual_pack_import_preview_cancel_requested")]
    updated_preview = repo.get_import_preview(preview["id"], owner_user_id="1")
    updated_job = repo.get_portability_job_by_job_id("9001", owner_user_id="1")
    assert updated_preview is not None
    assert updated_preview["status"] == "cancelled"
    assert updated_preview["stage"] == "cancelled"
    assert updated_job is not None
    assert updated_job["status"] == "cancelled"
    assert updated_job["stage"] == "cancelled"


def test_delete_import_preview_removes_staged_archive(
    persona_db: CharactersRAGDB,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_portability.constants import (
        PERSONA_VISUAL_PACK_EXTENSION,
    )

    manager = FakeJobManager()
    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_job_manager] = lambda: manager
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Import Preview Delete Persona")
        archive_root = DatabasePaths.get_user_temp_outputs_dir("1") / "persona_visual_pack_import_previews"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive_path = archive_root / f"delete-me{PERSONA_VISUAL_PACK_EXTENSION}"
        archive_path.write_bytes(b"portable visual archive")
        repo = PersonaVisualPortabilityRepository.initialized(persona_db)
        preview = repo.create_import_preview(
            owner_user_id="1",
            job_id="9001",
            status="failed",
            stage="failed",
            archive_path=str(archive_path),
            target_persona_id=persona_id,
        )
        repo.create_portability_job(
            owner_user_id="1",
            job_id="9001",
            operation="import_preview",
            status="failed",
            stage="failed",
            preview_id=preview["id"],
            archive_path=str(archive_path),
        )
        manager.jobs_by_id[9001] = {"id": 9001, "status": "failed"}

        response = client.delete(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/import-previews/{preview['id']}"
        )

    assert response.status_code == 204, response.text
    assert not archive_path.exists()
    updated_preview = repo.get_import_preview(preview["id"], owner_user_id="1")
    updated_job = repo.get_portability_job_by_job_id("9001", owner_user_id="1")
    assert updated_preview is not None
    assert updated_preview["status"] == "deleted"
    assert updated_preview["stage"] == "deleted"
    assert updated_job is not None
    assert updated_job["status"] == "cancelled"
    assert updated_job["stage"] == "deleted"


def test_start_import_commit_creates_jobs_backed_portability_row(
    persona_db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )
    from tldw_Server_API.app.core.Persona.visual_jobs import (
        PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE,
    )

    manager = FakeJobManager()
    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_job_manager] = lambda: manager
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Import Commit Persona")
        archive_path = tmp_path / "commit.tldw-persona-vpack"
        archive_path.write_bytes(b"portable visual archive")
        repo = PersonaVisualPortabilityRepository.initialized(persona_db)
        preview = repo.create_import_preview(
            owner_user_id="1",
            job_id="preview-job-1",
            status="completed",
            stage="completed",
            archive_path=str(archive_path),
            archive_sha256="a" * 64,
            canonical_payload_fingerprint="b" * 64,
            schema_version="tldw.persona_visual_pack.v1",
            target_persona_id=persona_id,
        )

        response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/import-previews/{preview['id']}/commit",
            json={"trust_mode": "untrusted_import", "target_mode": "create_new"},
        )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["job_id"] == "9001"
    assert body["operation"] == "import_commit"
    assert body["preview_id"] == preview["id"]
    assert body["target_persona_id"] == persona_id
    assert body["status"] == "queued"
    assert body["stage"] == "queued"
    assert manager.created[0]["job_type"] == PERSONA_VISUAL_PACK_IMPORT_COMMIT_JOB_TYPE
    row = repo.get_portability_job_by_job_id("9001", owner_user_id="1")
    assert row is not None
    assert row["operation"] == "import_commit"
    assert row["preview_id"] == preview["id"]
    assert row["persona_id"] == persona_id


def test_import_commit_status_is_scoped(persona_db: CharactersRAGDB, tmp_path: Path) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )

    manager = FakeJobManager()
    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_job_manager] = lambda: manager
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Import Commit Status Persona")
        archive_path = tmp_path / "commit-status.tldw-persona-vpack"
        archive_path.write_bytes(b"portable visual archive")
        repo = PersonaVisualPortabilityRepository.initialized(persona_db)
        preview = repo.create_import_preview(
            owner_user_id="1",
            job_id="preview-job-1",
            status="completed",
            stage="completed",
            archive_path=str(archive_path),
            archive_sha256="a" * 64,
            canonical_payload_fingerprint="b" * 64,
            schema_version="tldw.persona_visual_pack.v1",
            target_persona_id=persona_id,
        )
        started = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/import-previews/{preview['id']}/commit",
            json={"trust_mode": "trusted_restore", "target_mode": "create_new"},
        )
        assert started.status_code == 202, started.text
        job_id = started.json()["job_id"]
        imported_pack = _create_visual_pack(client, persona_id, title="Imported Pack")
        repo.update_portability_job(
            job_id,
            {
                "status": "completed",
                "stage": "completed",
                "pack_id": imported_pack["id"],
                "progress": {"asset_count": 1},
            },
            owner_user_id="1",
        )

        status_response = client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/imports/{job_id}"
        )

    assert status_response.status_code == 200, status_response.text
    body = status_response.json()
    assert body["status"] == "completed"
    assert body["pack_id"] == imported_pack["id"]
    assert body["progress"] == {"asset_count": 1}

    with _client_for_user(2, persona_db) as other_client:
        denied = other_client.get(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/imports/{job_id}"
        )

    assert denied.status_code == 404


def test_start_import_commit_rejects_incomplete_preview(
    persona_db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
        PersonaVisualPortabilityRepository,
    )

    manager = FakeJobManager()
    fastapi_app.dependency_overrides[persona_ep.get_persona_visual_job_manager] = lambda: manager
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Incomplete Import Commit Persona")
        repo = PersonaVisualPortabilityRepository.initialized(persona_db)
        preview = repo.create_import_preview(
            owner_user_id="1",
            job_id="preview-job-1",
            status="queued",
            archive_path=str(tmp_path / "not-ready.tldw-persona-vpack"),
            target_persona_id=persona_id,
        )

        response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/import-previews/{preview['id']}/commit",
            json={"trust_mode": "untrusted_import", "target_mode": "create_new"},
        )

    assert response.status_code == 409
    assert manager.created == []


def test_deactivate_visual_pack_reverts_to_derived_buddy(persona_db: CharactersRAGDB) -> None:
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Deactivate Visual API Persona")
        pack = _create_visual_pack(client, persona_id)
        asset = _upload_png(client, persona_id, pack["id"])
        client.patch(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/manifest",
            json={"manifest": _valid_manifest(asset["id"]), "expected_version": pack["version"]},
        )
        activated = client.post(
            f"/api/v1/persona/profiles/{persona_id}/visual-packs/{pack['id']}/activate"
        )
        assert activated.status_code == 200, activated.text

        deactivated = client.post(f"/api/v1/persona/profiles/{persona_id}/visual-packs/deactivate")

        assert deactivated.status_code == 200, deactivated.text
        assert deactivated.json() == {"status": "deactivated", "persona_id": persona_id}
        assert persona_db.get_active_persona_visual_pack(persona_id=persona_id, user_id="1") is None
