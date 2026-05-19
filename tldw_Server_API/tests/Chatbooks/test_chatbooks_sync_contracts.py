import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.schemas.chatbook_schemas import (
    CreateChatbookResponse,
)
from tldw_Server_API.app.core.Chatbooks.chatbook_models import ConflictResolution, ContentType
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.tests.Chatbooks.test_chatbooks_export_sync import client_override


def _make_import_archive(base_dir: Path, *, note_id: str = "note-1", note_title: str = "Imported Note") -> Path:
    archive_path = base_dir / "sample.chatbook"
    manifest = {
        "version": "1.0.0",
        "name": "Import Sample",
        "description": "Sync import contract fixture",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "content_items": [
            {
                "id": note_id,
                "type": "note",
                "title": note_title,
                "file_path": f"content/notes/note_{note_id}.md",
            }
        ],
        "configuration": {},
        "statistics": {},
        "metadata": {},
        "user_info": {"user_id": "sync-user"},
    }
    note_body = f"---\ntitle: {note_title}\n---\nBody"

    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr(f"content/notes/note_{note_id}.md", note_body)

    return archive_path


def _make_import_upload() -> tuple[bytes, str]:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "version": "1.0",
                    "name": "Upload Import Sample",
                    "description": "Upload fixture",
                    "content_items": [],
                }
            ),
        )
    return payload.getvalue(), "upload.chatbook"


class _ImportDB:
    def __init__(self):
        self.created_notes: list[tuple[str, str]] = []

    def get_connection(self):
        class _Conn:
            def execute(self, *_args, **_kwargs):
                return None

            def close(self):
                return None

        return _Conn()

    def execute_query(self, *_args, **_kwargs):
        class _Cursor:
            def fetchall(self):
                return []

        return _Cursor()

    def add_note(self, title: str, content: str):
        self.created_notes.append((title, content))
        return "new-note-id"


@pytest.mark.unit
def test_create_chatbook_response_schema_omits_file_path():
    assert "file_path" not in CreateChatbookResponse.model_fields


@pytest.mark.unit
def test_continue_export_rejects_async_mode_before_service_call(
    client_override: TestClient,
    monkeypatch,
):
    called = False

    async def _unexpected_continue(*_args, **_kwargs):
        nonlocal called
        called = True
        return True, "unexpected", "unused"

    monkeypatch.setattr(ChatbookService, "continue_chatbook_export", _unexpected_continue)
    opaque_cursor = "-".join(("cursor", "1"))

    response = client_override.post(
        "/api/v1/chatbooks/export/continue",
        json={
            "export_id": "export-1",
            "continuations": [{"evaluation_id": "eval-1", "continuation_token": opaque_cursor}],
            "async_mode": True,
        },
    )

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Async continuation exports are not supported"
    assert called is False


@pytest.mark.unit
def test_sync_import_endpoint_returns_imported_items_and_warnings(
    client_override: TestClient,
    monkeypatch,
):
    async def _fake_import_chatbook(*_args, **_kwargs):
        return True, "Import completed", {"imported_items": {"note": 1}, "warnings": ["renamed note"]}

    monkeypatch.setattr(ChatbookService, "import_chatbook", _fake_import_chatbook)
    upload_bytes, filename = _make_import_upload()

    response = client_override.post(
        "/api/v1/chatbooks/import",
        files={"file": (filename, upload_bytes, "application/zip")},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "success": True,
        "message": "Import completed",
        "job_id": None,
        "imported_items": {"note": 1},
        "warnings": ["renamed note"],
    }


@pytest.mark.unit
def test_sync_import_endpoint_propagates_wrapper_result(
    client_override: TestClient,
    monkeypatch,
):
    captured = {}

    def _fake_import_chatbook_sync(
        self,
        file_path,
        content_selections,
        conflict_resolution,
        prefix_imported,
        import_media,
        import_embeddings,
    ):
        captured.update(
            {
                "file_path": file_path,
                "content_selections": content_selections,
                "conflict_resolution": conflict_resolution,
                "prefix_imported": prefix_imported,
                "import_media": import_media,
                "import_embeddings": import_embeddings,
            }
        )
        return True, "Import completed", {"imported_items": {"note": 2}, "warnings": ["from-wrapper"]}

    monkeypatch.setattr(ChatbookService, "_import_chatbook_sync", _fake_import_chatbook_sync)
    upload_bytes, filename = _make_import_upload()

    response = client_override.post(
        "/api/v1/chatbooks/import",
        files={"file": (filename, upload_bytes, "application/zip")},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "success": True,
        "message": "Import completed",
        "job_id": None,
        "imported_items": {"note": 2},
        "warnings": ["from-wrapper"],
    }
    assert captured["import_media"] is False
    assert captured["import_embeddings"] is False


@pytest.mark.unit
def test_sync_import_service_counts_renamed_items_as_imported(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    service = ChatbookService(user_id="sync-user", db=_ImportDB())
    archive_path = _make_import_archive(service.import_dir, note_title="Conflicting Title")

    monkeypatch.setattr(service, "_get_note_by_title", lambda _title: {"id": "existing-note"})
    monkeypatch.setattr(service, "_generate_unique_name", lambda title, _kind: f"{title} (Imported)")

    success, message, result = service._import_chatbook_sync(
        file_path=str(archive_path),
        content_selections={ContentType.NOTE: ["note-1"]},
        conflict_resolution=ConflictResolution.RENAME,
        prefix_imported=False,
        import_media=False,
        import_embeddings=False,
    )

    assert success is True
    assert message == "Successfully imported 1/1 items"
    assert result["imported_items"]["note"] == 1
    assert result["warnings"] == []


@pytest.mark.unit
def test_sync_import_service_skipped_conflicts_do_not_increment_imported_items(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    service = ChatbookService(user_id="sync-user", db=_ImportDB())
    archive_path = _make_import_archive(service.import_dir, note_title="Existing Title")

    monkeypatch.setattr(service, "_get_note_by_title", lambda _title: {"id": "existing-note"})

    success, message, result = service._import_chatbook_sync(
        file_path=str(archive_path),
        content_selections={ContentType.NOTE: ["note-1"]},
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        import_media=False,
        import_embeddings=False,
    )

    assert success is True
    assert message == "Import completed: All 1 items were skipped"
    assert result["imported_items"].get("note", 0) == 0
    assert result["warnings"] == []


@pytest.mark.unit
def test_remove_job_routes_describe_terminal_states_and_broaden_rejection_message(
    client_override: TestClient,
    monkeypatch,
):
    monkeypatch.setattr(ChatbookService, "delete_export_job", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(ChatbookService, "delete_import_job", lambda *_args, **_kwargs: False)

    export_response = client_override.delete("/api/v1/chatbooks/export/jobs/export-1/remove")
    import_response = client_override.delete("/api/v1/chatbooks/import/jobs/import-1/remove")
    openapi = client_override.get("/openapi.json")

    assert export_response.status_code == 400, export_response.text
    assert import_response.status_code == 400, import_response.text
    assert export_response.json()["detail"] == "Only terminal export jobs can be removed"
    assert import_response.json()["detail"] == "Only terminal import jobs can be removed"

    paths = openapi.json()["paths"]
    export_description = paths["/api/v1/chatbooks/export/jobs/{job_id}/remove"]["delete"]["description"].lower()
    import_description = paths["/api/v1/chatbooks/import/jobs/{job_id}/remove"]["delete"]["description"].lower()
    assert "completed, cancelled, failed, or expired" in export_description
    assert "completed, cancelled, or failed" in import_description
