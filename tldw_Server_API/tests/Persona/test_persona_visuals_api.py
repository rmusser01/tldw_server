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

    def _fake_visuals_dir(user_id: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(
        DatabasePaths,
        "get_user_persona_visuals_dir",
        staticmethod(_fake_visuals_dir),
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
        accepted = persona_db.create_persona_visual_candidate(
            pack_id=pack["id"],
            persona_id=persona_id,
            user_id="1",
            job_id="job-1",
            proposed_manifest_patch={"states": {"thinking": {"animation_id": "think"}}},
            generated_asset_ids=["asset-1"],
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
        assert rejected_response.status_code == 200, rejected_response.text
        assert rejected_response.json()["status"] == "rejected"
        assert rejected_response.json()["failure_reason"] == "not useful"


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
