"""Integration tests for web clipper API endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_rate_limiter_dep
from tldw_Server_API.app.api.v1.endpoints import web_clipper as web_clipper_endpoint
from tldw_Server_API.app.api.v1.schemas.web_clipper_schemas import WebClipperSaveResponse
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_web_clipper_db(tmp_path):
    db_path = tmp_path / "web_clipper_integration.db"
    db = CharactersRAGDB(str(db_path), client_id="web_clipper_user")
    db.upsert_workspace("ws-1", "Research Workspace")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    class _NoopRateLimiter:
        async def check_user_rate_limit(self, *_args, **_kwargs):
            return True, {}

    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_db_dep():
        return db

    fastapi_app = FastAPI()
    fastapi_app.include_router(web_clipper_endpoint.router, prefix="/api/v1/web-clipper")

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_db_dep
    fastapi_app.dependency_overrides[get_rate_limiter_dep] = lambda: _NoopRateLimiter()
    fastapi_app.state.web_clipper_db = db

    with TestClient(fastapi_app) as client:
        yield client

    fastapi_app.dependency_overrides.clear()
    db.close_connection()


def _save_payload(*, clip_id: str = "clip-123", destination_mode: str = "both", include_attachment: bool = False):
    attachments = []
    if include_attachment:
        attachments.append(
            {
                "slot": "page-screenshot",
                "file_name": "page-screenshot.txt",
                "media_type": "text/plain",
                "text_content": "captured attachment payload",
            }
        )
    payload = {
        "clip_id": clip_id,
        "clip_type": "article",
        "source_url": "https://example.com/story",
        "source_title": "Example Story",
        "destination_mode": destination_mode,
        "note": {
            "title": "Example Story",
            "comment": "Saved from the browser clipper.",
            "keywords": ["example"],
        },
        "content": {
            "visible_body": "Alpha paragraph.",
            "full_extract": "Alpha paragraph.\n\nBeta paragraph.",
            "selected_text": "Alpha paragraph.",
        },
        "attachments": attachments,
        "enhancements": {"run_ocr": False, "run_vlm": False},
        "capture_metadata": {"fallback_path": ["article"]},
    }
    if destination_mode in {"workspace", "both"}:
        payload["workspace"] = {"workspace_id": "ws-1"}
    return payload


def test_web_clipper_save_and_status_flow(client_with_web_clipper_db: TestClient):
    client = client_with_web_clipper_db

    response = client.post("/api/v1/web-clipper/save", json=_save_payload(include_attachment=True))
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["status"] == "saved"
    assert data["note"]["id"] == "clip-123"
    assert data["workspace_placement"]["workspace_id"] == "ws-1"
    assert len(data["attachments"]) == 1

    status_response = client.get("/api/v1/web-clipper/clip-123")
    assert status_response.status_code == 200, status_response.text
    status_data = status_response.json()
    assert status_data["note"]["id"] == "clip-123"
    assert len(status_data["workspace_placements"]) == 1
    assert len(status_data["attachments"]) == 1


def test_web_clipper_save_retry_reuses_note_and_attachment(client_with_web_clipper_db: TestClient):
    client = client_with_web_clipper_db

    first = client.post("/api/v1/web-clipper/save", json=_save_payload(include_attachment=True))
    second = client.post("/api/v1/web-clipper/save", json=_save_payload(include_attachment=True))

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["note"]["id"] == second.json()["note"]["id"] == "clip-123"
    assert first.json()["workspace_placement"]["workspace_note_id"] == second.json()["workspace_placement"]["workspace_note_id"]

    status_response = client.get("/api/v1/web-clipper/clip-123")
    assert status_response.status_code == 200, status_response.text
    assert len(status_response.json()["attachments"]) == 1


def test_web_clipper_enrichment_conflict_flow(client_with_web_clipper_db: TestClient):
    client = client_with_web_clipper_db
    db = client.app.state.web_clipper_db

    save_response = client.post("/api/v1/web-clipper/save", json=_save_payload(destination_mode="note"))
    assert save_response.status_code == 200, save_response.text
    saved = save_response.json()

    current_note = db.get_note_by_id("clip-123")
    assert current_note is not None
    db.update_note(
        note_id="clip-123",
        update_data={"content": f"{current_note['content']}\n\nUser edit."},
        expected_version=int(current_note["version"]),
    )

    enrichment_response = client.post(
        "/api/v1/web-clipper/clip-123/enrichments",
        json={
            "clip_id": "clip-123",
            "enrichment_type": "ocr",
            "status": "complete",
            "inline_summary": "Captured text summary.",
            "structured_payload": {"raw_text": "Captured text summary."},
            "source_note_version": saved["note"]["version"],
        },
    )

    assert enrichment_response.status_code == 200, enrichment_response.text
    enrichment_data = enrichment_response.json()
    assert enrichment_data["inline_applied"] is False
    assert enrichment_data["conflict_reason"] == "source_note_version_mismatch"

    status_response = client.get("/api/v1/web-clipper/clip-123")
    assert status_response.status_code == 200, status_response.text
    assert status_response.json()["analysis"]["ocr"]["structured_payload"]["raw_text"] == "Captured text summary."


def test_web_clipper_rate_limiter_failure_returns_503(client_with_web_clipper_db: TestClient):
    client = client_with_web_clipper_db

    class _ExplodingRateLimiter:
        async def check_user_rate_limit(self, *_args, **_kwargs):
            raise RuntimeError("rate limiter unavailable")

    client.app.dependency_overrides[get_rate_limiter_dep] = lambda: _ExplodingRateLimiter()

    response = client.post("/api/v1/web-clipper/save", json=_save_payload())

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "Rate limiter unavailable"


def test_web_clipper_save_returns_500_for_failed_canonical_save_payload(
    client_with_web_clipper_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    client = client_with_web_clipper_db

    def _failed_save(*_args, **_kwargs):
        return WebClipperSaveResponse(
            clip_id="clip-123",
            status="failed",
            note=None,
            workspace_placement=None,
            attachments=[],
            warnings=["Canonical note save failed: write failed at /private/clipper.db"],
            note_id="clip-123",
            workspace_placement_saved=False,
            workspace_placement_count=0,
        )

    monkeypatch.setattr(web_clipper_endpoint.WebClipperService, "save_clip", _failed_save)

    response = client.post("/api/v1/web-clipper/save", json=_save_payload())

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Canonical note save failed."
