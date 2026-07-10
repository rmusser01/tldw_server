"""API tests for chat document preflight and upload draft handoff."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit


@pytest.fixture()
def document_upload_app(tmp_path, monkeypatch):
    """Build an isolated app backed by a temporary shared draft database."""
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
    from tldw_Server_API.app.api.v1.endpoints.media import document_upload_processing
    from tldw_Server_API.app.core.Ingestion_Media_Processing.document_upload_drafts import (
        DocumentUploadDraftStore,
    )

    draft_store = DocumentUploadDraftStore(db_path=tmp_path / "document-upload-drafts.db")
    monkeypatch.setattr(
        document_upload_processing,
        "get_document_upload_draft_store",
        lambda: draft_store,
    )
    app = FastAPI()
    app.include_router(document_upload_processing.router, prefix="/api/v1/media")
    app.dependency_overrides[get_request_user] = lambda: type("User", (), {"id": 1})()
    try:
        yield app, document_upload_processing
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def client(document_upload_app):
    """Return a test client for the isolated document upload app."""
    app, _document_upload_processing = document_upload_app
    return TestClient(app)


def test_document_upload_preflight_pdf_ocr_available(client, document_upload_app, monkeypatch):
    _app, document_upload_processing = document_upload_app
    monkeypatch.setattr(
        document_upload_processing,
        "_list_ocr_backends",
        lambda: {"tesseract": {"available": True}},
    )

    response = client.post(
        "/api/v1/media/document-upload/preflight",
        json={
            "files": [
                {
                    "client_id": "file-1",
                    "filename": "scan.pdf",
                    "mime_type": "application/pdf",
                    "size_bytes": 1024,
                }
            ]
        },
    )

    assert response.status_code == 200
    item = response.json()["files"][0]
    assert item["client_id"] == "file-1"
    assert item["media_type"] == "pdf"
    assert item["modes"]["add_to_chat"]["available"] is True
    assert item["modes"]["ocr_pages"]["available"] is True
    assert item["modes"]["ingest_to_library"]["available"] is True
    assert item["default_mode"] == "add_to_chat"
    assert item["requires_send_time_estimate"] is True


def test_document_upload_preflight_docx_blocks_ocr_with_reason(client):
    response = client.post(
        "/api/v1/media/document-upload/preflight",
        json={
            "files": [
                {
                    "client_id": "file-1",
                    "filename": "notes.docx",
                    "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "size_bytes": 1024,
                }
            ]
        },
    )

    assert response.status_code == 200
    item = response.json()["files"][0]
    assert item["media_type"] == "document"
    assert item["modes"]["add_to_chat"]["available"] is True
    assert item["modes"]["ingest_to_library"]["available"] is True
    assert item["modes"]["ocr_pages"] == {
        "available": False,
        "status": "unavailable",
        "reason": "OCR unavailable: server cannot render .DOCX pages",
    }


def test_document_upload_preflight_pdf_ocr_unavailable_without_backend(
    client,
    document_upload_app,
    monkeypatch,
):
    _app, document_upload_processing = document_upload_app
    monkeypatch.setattr(document_upload_processing, "_list_ocr_backends", lambda: {})

    response = client.post(
        "/api/v1/media/document-upload/preflight",
        json={
            "files": [
                {
                    "client_id": "file-1",
                    "filename": "scan.pdf",
                    "mime_type": "application/pdf",
                    "size_bytes": 1024,
                }
            ]
        },
    )

    item = response.json()["files"][0]
    assert item["modes"]["ocr_pages"] == {
        "available": False,
        "status": "unavailable",
        "reason": "OCR unavailable: no OCR backend configured",
    }


def test_document_upload_preflight_unsupported_file_disables_all_modes(client):
    response = client.post(
        "/api/v1/media/document-upload/preflight",
        json={
            "files": [
                {
                    "client_id": "file-1",
                    "filename": "archive.zip",
                    "mime_type": "application/zip",
                    "size_bytes": 1024,
                }
            ]
        },
    )

    item = response.json()["files"][0]
    assert item["media_type"] == "unsupported"
    assert item["default_mode"] is None
    assert all(mode["available"] is False for mode in item["modes"].values())
    assert {mode["status"] for mode in item["modes"].values()} == {"unavailable"}


def test_document_upload_preflight_marks_size_page_and_token_limits_blocked(
    client,
    document_upload_app,
    monkeypatch,
):
    _app, document_upload_processing = document_upload_app
    monkeypatch.setattr(
        document_upload_processing,
        "_list_ocr_backends",
        lambda: {"tesseract": {"available": True}},
    )

    response = client.post(
        "/api/v1/media/document-upload/preflight",
        json={
            "files": [
                {
                    "client_id": "oversized",
                    "filename": "huge.pdf",
                    "mime_type": "application/pdf",
                    "size_bytes": document_upload_processing.DEFAULT_MAX_CHAT_UPLOAD_BYTES + 1,
                },
                {
                    "client_id": "too-many-pages",
                    "filename": "long.pdf",
                    "mime_type": "application/pdf",
                    "size_bytes": 1024,
                    "page_count": document_upload_processing.DEFAULT_MAX_CHAT_UPLOAD_PAGES + 1,
                },
                {
                    "client_id": "too-many-tokens",
                    "filename": "long.md",
                    "mime_type": "text/markdown",
                    "size_bytes": 1024,
                    "estimated_tokens": document_upload_processing.DEFAULT_MAX_DIRECT_CHAT_TOKENS + 1,
                },
            ]
        },
    )

    items = {item["client_id"]: item for item in response.json()["files"]}
    assert items["oversized"]["modes"]["add_to_chat"]["status"] == "blocked"
    assert "exceeds 20 MB limit" in items["oversized"]["modes"]["add_to_chat"]["reason"]
    assert items["too-many-pages"]["modes"]["add_to_chat"]["status"] == "blocked"
    assert "exceeds 200 page limit" in items["too-many-pages"]["modes"]["ocr_pages"]["reason"]
    assert items["too-many-tokens"]["modes"]["add_to_chat"]["status"] == "blocked"
    assert "exceeds 24000 token direct-chat limit" in items["too-many-tokens"]["modes"]["add_to_chat"]["reason"]
    assert items["too-many-tokens"]["modes"]["ingest_to_library"]["available"] is True
    assert items["too-many-tokens"]["default_mode"] == "ingest_to_library"


def test_document_upload_draft_owner_read_delete_and_expiry(document_upload_app, monkeypatch):
    app, document_upload_processing = document_upload_app
    first_client = TestClient(app)

    created_at = datetime(2026, 7, 9, tzinfo=timezone.utc)
    draft_store = document_upload_processing.get_document_upload_draft_store()
    monkeypatch.setattr(draft_store, "_clock", lambda: created_at)
    create_response = first_client.post(
        "/api/v1/media/document-upload/drafts",
        json={
            "payload": {
                "draft": "summarize this",
                "files": [
                    {
                        "client_id": "file-1",
                        "filename": "notes.md",
                        "content": "data:text/markdown;base64,I25vdGVz",
                        "processing_mode": "add_to_chat",
                    }
                ],
            }
        },
    )

    assert create_response.status_code == 200
    draft_id = create_response.json()["draft_id"]
    assert (
        create_response.json()["expires_at"]
        == (created_at + timedelta(seconds=document_upload_processing.DRAFT_TTL_SECONDS)).isoformat()
    )

    read_response = first_client.get(f"/api/v1/media/document-upload/drafts/{draft_id}")
    assert read_response.status_code == 200
    assert read_response.json()["payload"]["draft"] == "summarize this"

    app.dependency_overrides[document_upload_processing.get_request_user] = lambda: type("User", (), {"id": 2})()
    other_user_response = TestClient(app).get(f"/api/v1/media/document-upload/drafts/{draft_id}")
    assert other_user_response.status_code == 404

    app.dependency_overrides[document_upload_processing.get_request_user] = lambda: type("User", (), {"id": 1})()
    monkeypatch.setattr(
        draft_store,
        "_clock",
        lambda: created_at + timedelta(seconds=document_upload_processing.DRAFT_TTL_SECONDS + 1),
    )
    expired_response = TestClient(app).get(f"/api/v1/media/document-upload/drafts/{draft_id}")
    assert expired_response.status_code == 404

    monkeypatch.setattr(draft_store, "_clock", lambda: created_at)
    second_draft = (
        TestClient(app)
        .post(
            "/api/v1/media/document-upload/drafts",
            json={"payload": {"draft": "delete me", "files": []}},
        )
        .json()["draft_id"]
    )
    delete_response = TestClient(app).delete(f"/api/v1/media/document-upload/drafts/{second_draft}")
    assert delete_response.status_code == 204
    assert TestClient(app).get(f"/api/v1/media/document-upload/drafts/{second_draft}").status_code == 404


def test_document_upload_draft_rejects_oversized_payload(
    document_upload_app,
    monkeypatch,
):
    app, document_upload_processing = document_upload_app
    draft_store = document_upload_processing.get_document_upload_draft_store()
    monkeypatch.setattr(
        draft_store,
        "max_payload_bytes",
        32,
    )

    response = TestClient(app).post(
        "/api/v1/media/document-upload/drafts",
        json={"payload": {"draft": "x" * 128, "files": []}},
    )

    assert response.status_code == 413


def test_document_upload_draft_rejects_per_owner_quota(
    document_upload_app,
    monkeypatch,
):
    app, document_upload_processing = document_upload_app
    created_at = datetime(2026, 7, 9, tzinfo=timezone.utc)
    draft_store = document_upload_processing.get_document_upload_draft_store()
    monkeypatch.setattr(draft_store, "_clock", lambda: created_at)
    monkeypatch.setattr(draft_store, "max_drafts_per_owner", 1)

    first_response = TestClient(app).post(
        "/api/v1/media/document-upload/drafts",
        json={"payload": {"draft": "first", "files": []}},
    )
    second_response = TestClient(app).post(
        "/api/v1/media/document-upload/drafts",
        json={"payload": {"draft": "second", "files": []}},
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 429


def test_document_upload_router_registered():
    from tldw_Server_API.app.api.v1.endpoints import media

    route_paths = {route.path for route in media.router.routes}
    assert "/document-upload/preflight" in route_paths
    assert "/document-upload/drafts" in route_paths
