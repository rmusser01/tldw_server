import json

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import chatbooks as chatbooks_endpoints
from tldw_Server_API.app.api.v1.schemas.chatbook_schemas import (
    ChatbookImportSourceFormat,
    ImportChatbookRequest,
)
from tldw_Server_API.tests.Chatbooks.test_chatbooks_api_error_and_preview_mapping import _make_app


class _OpenWebUIDBPreviewService:
    db = None

    def preview_openwebui_db(self, file_path: str):
        self.preview_path = file_path
        return {
            "user_count": 1,
            "users": [
                {
                    "source_user_id": "user-a",
                    "display_label": "Alice",
                    "email": "alice@example.test",
                    "chat_count": 2,
                    "folder_count": 1,
                    "message_count": 4,
                    "branched_chat_count": 1,
                    "duplicate_chat_count": 0,
                    "archived_chat_count": 0,
                    "pinned_chat_count": 1,
                    "attachment_reference_count": 1,
                    "warning_count": 0,
                    "warnings": [],
                }
            ],
            "warnings": [],
        }, None


class _OpenWebUIDBImportService:
    db = None

    def __init__(self) -> None:
        self.called_kwargs = None

    async def import_chatbook(self, **kwargs):
        self.called_kwargs = kwargs
        return True, "ok", {
            "selected_user_id": kwargs["selected_openwebui_user_id"],
            "selected_user_label": "Alice",
            "imported_chats": 1,
            "skipped_chats": 0,
            "failed_chats": 0,
            "imported_messages": 2,
            "skipped_messages": 0,
            "duplicate_chats": 0,
            "warnings": [],
        }


def test_schema_accepts_openwebui_db_source_format_and_selected_user():
    request = ImportChatbookRequest(
        source_format="openwebui_db",
        selected_openwebui_user_id="user-a",
    )

    assert ChatbookImportSourceFormat.OPENWEBUI_DB.value == "openwebui_db"
    assert request.source_format == ChatbookImportSourceFormat.OPENWEBUI_DB
    assert request.selected_openwebui_user_id == "user-a"


def test_preview_chatbook_route_has_rbac_rate_limit():
    route = next(
        route
        for route in chatbooks_endpoints.router.routes
        if isinstance(route, APIRoute) and route.path == "/chatbooks/preview" and "POST" in route.methods
    )
    resources = [
        getattr(dependency.call, "_tldw_rate_limit_resource", None)
        for dependency in route.dependant.dependencies
    ]

    assert "chatbooks.preview" in resources


def test_preview_openwebui_db_source_format_skips_archive_validation(monkeypatch):
    service = _OpenWebUIDBPreviewService()
    app = _make_app(service)

    def _fail_zip_validation(_path: str):
        raise AssertionError("OpenWebUI DB preview must not validate as a ZIP archive")

    monkeypatch.setattr(chatbooks_endpoints.ChatbookValidator, "validate_zip_file", _fail_zip_validation)

    files = {"file": ("webui.db", b"SQLite format 3\x00fake-db", "application/octet-stream")}
    data = {"source_format": "openwebui_db"}

    with TestClient(app) as client:
        response = client.post("/api/v1/chatbooks/preview", files=files, data=data)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_format"] == "openwebui_db"
    assert body["manifest"] is None
    assert body["openwebui_preview"] is None
    assert body["openwebui_db_preview"]["user_count"] == 1
    assert body["openwebui_db_preview"]["users"][0]["source_user_id"] == "user-a"


def test_import_openwebui_db_requires_selected_user_before_service_call():
    service = _OpenWebUIDBImportService()
    app = _make_app(service)
    files = {"file": ("webui.db", b"SQLite format 3\x00fake-db", "application/octet-stream")}
    data = {
        "source_format": "openwebui_db",
        "import_media": "false",
        "import_embeddings": "false",
        "async_mode": "false",
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/chatbooks/import", files=files, data=data)

    assert response.status_code == 400
    assert response.json()["detail"] == "selected_openwebui_user_id is required for OpenWebUI DB imports"
    assert service.called_kwargs is None


def test_import_openwebui_db_passes_selected_user_to_service(monkeypatch):
    service = _OpenWebUIDBImportService()
    app = _make_app(service)

    def _fail_zip_validation(_path: str):
        raise AssertionError("OpenWebUI DB import must not validate as a ZIP archive")

    monkeypatch.setattr(chatbooks_endpoints.ChatbookValidator, "validate_zip_file", _fail_zip_validation)

    files = {"file": ("webui.sqlite", b"SQLite format 3\x00fake-db", "application/octet-stream")}
    data = {
        "source_format": "openwebui_db",
        "selected_openwebui_user_id": "user-a",
        "conflict_resolution": "rename",
        "prefix_imported": "true",
        "import_media": "false",
        "import_embeddings": "false",
        "async_mode": "false",
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/chatbooks/import", files=files, data=data)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source_format"] == "openwebui_db"
    assert body["openwebui_result"] is None
    assert body["openwebui_db_result"]["selected_user_id"] == "user-a"
    assert body["openwebui_db_result"]["imported_chats"] == 1
    assert service.called_kwargs["source_format"] == "openwebui_db"
    assert service.called_kwargs["selected_openwebui_user_id"] == "user-a"
    assert service.called_kwargs["conflict_resolution"].value == "rename"
    assert service.called_kwargs["prefix_imported"] is True


def test_openwebui_db_source_rejects_json_filename_before_service_call():
    service = _OpenWebUIDBImportService()
    app = _make_app(service)
    files = {"file": ("openwebui.json", json.dumps([]).encode("utf-8"), "application/json")}
    data = {
        "source_format": "openwebui_db",
        "selected_openwebui_user_id": "user-a",
        "import_media": "false",
        "import_embeddings": "false",
        "async_mode": "false",
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/chatbooks/import", files=files, data=data)

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]
    assert service.called_kwargs is None
