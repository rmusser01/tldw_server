import json
import os
import shutil
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from importlib import import_module, reload
from tldw_Server_API.app.api.v1.endpoints import items as items_endpoint
from tldw_Server_API.app.api.v1.endpoints import outputs as outputs_endpoint
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.backends.factory import close_all_backends
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError
from tldw_Server_API.app.api.v1.endpoints.outputs import _resolve_output_path_for_user, _strip_html_for_tts


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.debugs: list[str] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.errors.append(message.format(*args) if args else message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(message.format(*args) if args else message)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append(message.format(*args) if args else message)


def _insert_output_row_raw(
    cdb: CollectionsDatabase,
    *,
    title: str,
    format_: str,
    storage_path: str,
    retention_until: str | None = None,
) -> int:
    now = datetime.utcnow().replace(microsecond=0).isoformat()
    res = cdb.backend.execute(
        "INSERT INTO outputs (user_id, job_id, run_id, type, title, format, storage_path, metadata_json, created_at, media_item_id, chatbook_path, deleted, deleted_at, retention_until) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?)",
        (
            cdb.user_id,
            None,
            None,
            "newsletter_markdown",
            title,
            format_,
            storage_path,
            None,
            now,
            None,
            None,
            retention_until,
        ),
    )
    return int(res.lastrowid or 0)


@pytest.fixture()
def client_with_user(monkeypatch):
    async def override_user():
        return User(id=123, username="tester", email=None, is_active=True)

    # Use full app profile for Collections/outputs endpoints
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")

    # Force per-user DB dir into project Databases/ for sandbox write allowance
    base_dir = Path.cwd() / "Databases" / "test_user_dbs"
    close_all_backends()
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    app = None
    try:
        # Reload app after env vars to honor minimal test mode changes
        mod = import_module("tldw_Server_API.app.main")
        mod = reload(mod)
        app = getattr(mod, "app")
        app.dependency_overrides[get_request_user] = override_user
        with TestClient(app) as client:
            yield client
    finally:
        if app is not None:
            app.dependency_overrides.clear()
        close_all_backends()
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def test_items_endpoint_minimal(client_with_user):

    client = client_with_user
    r = client.get("/api/v1/items", params={"ids": [1, 2]})
    assert r.status_code == 200, r.text
    data = r.json()
    assert "items" in data and isinstance(data["items"], list)


def test_items_endpoint_uses_collections_layer(client_with_user):

    client = client_with_user
    collections_db = CollectionsDatabase.for_user(user_id=123)
    collections_db.upsert_content_item(
        origin="watchlist",
        origin_type="rss",
        origin_id=1,
        url="https://example.com/story",
        canonical_url="https://example.com/story",
        domain="example.com",
        title="Story Headline",
        summary="Summary text for story",
        content_hash=hashlib.sha256(b"Story Headline").hexdigest(),
        word_count=3,
        published_at="2024-01-01T00:00:00Z",
        tags=["news"],
        metadata={"test": True},
        media_id=456,
        job_id=99,
        run_id=100,
        source_id=200,
    )

    r = client.get("/api/v1/items", params={"page": 1, "size": 5, "origin": "watchlist", "q": "Story"})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["total"] >= 1
    assert any(item["title"] == "Story Headline" for item in payload["items"])
    assert all(item["type"] == "watchlist" for item in payload["items"])
    assert all("status" in item for item in payload["items"])
    assert all("favorite" in item for item in payload["items"])

    r = client.get("/api/v1/items", params={"origin": "reading"})
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_items_get_by_id(client_with_user):

    client = client_with_user
    r = client.post(
        "/api/v1/reading/save",
        json={
            "url": "https://example.com/article",
            "title": "Example Article",
            "content": "Inline article content used for tests.",
            "tags": ["demo"],
        },
    )
    assert r.status_code == 200, r.text
    item_id = r.json()["id"]

    r = client.get(f"/api/v1/items/{item_id}")
    assert r.status_code == 200, r.text
    item = r.json()
    assert item["id"] == item_id
    assert item["title"] == "Example Article"
    assert item["status"] == "saved"
    assert item["favorite"] is False


def test_items_endpoint_maps_media_input_error_to_400(client_with_user, monkeypatch):

    client = client_with_user

    def _raise_input_error(*args, **kwargs):
        _ = (args, kwargs)
        raise InputError("invalid items query")

    monkeypatch.setattr(items_endpoint, "search_media", _raise_input_error)

    r = client.get("/api/v1/items", params={"q": "budget"})
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == "invalid items query"


def test_items_get_by_id_maps_media_input_error_to_400(client_with_user, monkeypatch):

    client = client_with_user

    def _raise_input_error(*args, **kwargs):
        _ = (args, kwargs)
        raise InputError("invalid item lookup")

    monkeypatch.setattr(items_endpoint, "search_media", _raise_input_error)

    r = client.get("/api/v1/items/99999")
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == "invalid item lookup"


@pytest.mark.asyncio
async def test_items_list_collections_failure_log_is_sanitized(monkeypatch):
    class _FailingCollectionsDB:
        def list_content_items(self, **kwargs: Any):
            raise RuntimeError("collections backend exploded at /private/items.db")

    logger = _LoggerStub()
    monkeypatch.setattr(items_endpoint, "logger", logger)

    with pytest.raises(HTTPException) as exc_info:
        await items_endpoint.list_items(
            ids=None,
            q=None,
            tags=None,
            domain=None,
            date_from=None,
            date_to=None,
            status_filter=None,
            favorite=None,
            origin=None,
            job_id=None,
            run_id=None,
            page=1,
            size=20,
            current_user=User(id=123, username="tester", email=None, is_active=True),
            db=object(),
            collections_db=_FailingCollectionsDB(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "items_query_failed"
    assert logger.errors == ["collections items query failed"]
    logged = "\n".join(logger.errors)
    assert "collections backend exploded" not in logged
    assert "/private/items.db" not in logged


@pytest.mark.asyncio
async def test_items_list_media_fallback_failure_log_is_sanitized(monkeypatch):
    class _EmptyCollectionsDB:
        def list_content_items(self, **kwargs: Any):
            return [], 0

    def _raise_runtime_error(*args: Any, **kwargs: Any):
        raise RuntimeError("media backend exploded at /private/media.db")

    logger = _LoggerStub()
    monkeypatch.setattr(items_endpoint, "logger", logger)
    monkeypatch.setattr(items_endpoint, "search_media", _raise_runtime_error)

    with pytest.raises(HTTPException) as exc_info:
        await items_endpoint.list_items(
            ids=None,
            q=None,
            tags=None,
            domain=None,
            date_from=None,
            date_to=None,
            status_filter=None,
            favorite=None,
            origin=None,
            job_id=None,
            run_id=None,
            page=1,
            size=20,
            current_user=User(id=123, username="tester", email=None, is_active=True),
            db=object(),
            collections_db=_EmptyCollectionsDB(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "items_query_failed"
    assert logger.errors == ["items list failed"]
    logged = "\n".join(logger.errors)
    assert "media backend exploded" not in logged
    assert "/private/media.db" not in logged


@pytest.mark.asyncio
async def test_items_get_collections_fetch_failure_log_is_sanitized(monkeypatch):
    class _FailingCollectionsDB:
        def get_content_item(self, item_id: int):
            raise RuntimeError("collections backend exploded at /private/items.db")

    logger = _LoggerStub()
    monkeypatch.setattr(items_endpoint, "logger", logger)

    with pytest.raises(HTTPException) as exc_info:
        await items_endpoint.get_item(
            item_id=1,
            current_user=User(id=123, username="tester", email=None, is_active=True),
            db=object(),
            collections_db=_FailingCollectionsDB(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "item_fetch_failed"
    assert logger.errors == ["collections item fetch failed"]
    logged = "\n".join(logger.errors)
    assert "collections backend exploded" not in logged
    assert "/private/items.db" not in logged


@pytest.mark.asyncio
async def test_items_get_collections_media_id_fetch_failure_log_is_sanitized(monkeypatch):
    class _FailingCollectionsDB:
        def get_content_item(self, item_id: int):
            raise KeyError(item_id)

        def get_content_item_by_media_id(self, media_id: int):
            raise RuntimeError("collections backend exploded at /private/items.db")

    logger = _LoggerStub()
    monkeypatch.setattr(items_endpoint, "logger", logger)

    with pytest.raises(HTTPException) as exc_info:
        await items_endpoint.get_item(
            item_id=1,
            current_user=User(id=123, username="tester", email=None, is_active=True),
            db=object(),
            collections_db=_FailingCollectionsDB(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "item_fetch_failed"
    assert logger.errors == ["collections item fetch by media_id failed"]
    logged = "\n".join(logger.errors)
    assert "collections backend exploded" not in logged
    assert "/private/items.db" not in logged


@pytest.mark.asyncio
async def test_items_get_media_fetch_failure_log_is_sanitized(monkeypatch):
    class _MissingCollectionsDB:
        def get_content_item(self, item_id: int):
            raise KeyError(item_id)

        def get_content_item_by_media_id(self, media_id: int):
            raise KeyError(media_id)

    def _raise_runtime_error(*args: Any, **kwargs: Any):
        raise RuntimeError("media backend exploded at /private/media.db")

    logger = _LoggerStub()
    monkeypatch.setattr(items_endpoint, "logger", logger)
    monkeypatch.setattr(items_endpoint, "search_media", _raise_runtime_error)

    with pytest.raises(HTTPException) as exc_info:
        await items_endpoint.get_item(
            item_id=1,
            current_user=User(id=123, username="tester", email=None, is_active=True),
            db=object(),
            collections_db=_MissingCollectionsDB(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "item_fetch_failed"
    assert logger.errors == ["media item fetch failed"]
    logged = "\n".join(logger.errors)
    assert "media backend exploded" not in logged
    assert "/private/media.db" not in logged


def test_outputs_normalize_storage_path_update_failure_log_is_sanitized(monkeypatch):
    def _raise_update_failure(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("output backend exploded at /private/outputs.db")

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "normalize_output_storage_path", lambda *_args: "normalized/output.md")
    monkeypatch.setattr(outputs_endpoint, "update_output_artifact_db", _raise_update_failure)

    with pytest.raises(HTTPException) as exc_info:
        outputs_endpoint._normalize_output_storage_path_for_user(
            cdb=object(),
            user_id=123,
            output_id=777,
            storage_path="legacy/output.md",
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "db_update_failed"
    assert logger.errors == ["outputs storage_path normalization update failed"]
    logged = "\n".join(logger.errors)
    assert "777" not in logged
    assert "output backend exploded" not in logged
    assert "/private/outputs.db" not in logged


@pytest.mark.asyncio
async def test_outputs_create_render_failure_log_is_sanitized(monkeypatch):
    class _Template:
        id = 1
        name = "Template"
        body = "{{ broken }}"
        type = "newsletter_markdown"
        format = "md"

    class _CollectionsDB:
        def get_output_template(self, _template_id: int):
            return _Template()

    def _raise_render_failure(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("render backend exploded at /private/output-template.md")

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "render_output_template", _raise_render_failure)

    with pytest.raises(HTTPException) as exc_info:
        await outputs_endpoint.create_output(
            payload=outputs_endpoint.OutputCreateRequest(
                template_id=1,
                data={"items": []},
                title="demo",
            ),
            current_user=User(id=123, username="tester", email=None, is_active=True),
            cdb=_CollectionsDB(),
            media_db=object(),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "render_failed"
    assert logger.errors == ["outputs render failed"]
    logged = "\n".join(logger.errors)
    assert "render backend exploded" not in logged
    assert "/private/output-template.md" not in logged


@pytest.mark.asyncio
async def test_outputs_create_directory_failure_log_is_sanitized(monkeypatch):
    class _Template:
        id = 1
        name = "Template"
        body = "hello"
        type = "newsletter_markdown"
        format = "md"

    class _CollectionsDB:
        def get_output_template(self, _template_id: int):
            return _Template()

    class _OutputDir:
        def mkdir(self, *args: Any, **kwargs: Any) -> None:
            raise OSError("mkdir exploded at /private/generated-outputs")

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "render_output_template", lambda *_args: "rendered")
    monkeypatch.setattr(outputs_endpoint, "_outputs_dir_for_user", lambda _user_id: _OutputDir())

    with pytest.raises(HTTPException) as exc_info:
        await outputs_endpoint.create_output(
            payload=outputs_endpoint.OutputCreateRequest(
                template_id=1,
                data={"items": []},
                title="demo",
            ),
            current_user=User(id=123, username="tester", email=None, is_active=True),
            cdb=_CollectionsDB(),
            media_db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "storage_unavailable"
    assert logger.errors == ["outputs directory creation failed"]
    logged = "\n".join(logger.errors)
    assert "mkdir exploded" not in logged
    assert "/private/generated-outputs" not in logged


@pytest.mark.asyncio
async def test_outputs_create_tts_generation_failure_log_is_sanitized(monkeypatch):
    class _Template:
        id = 1
        name = "Audio Template"
        body = "hello"
        type = "tts_audio"
        format = "mp3"

    class _CollectionsDB:
        def get_output_template(self, _template_id: int):
            return _Template()

    class _OutputDir:
        def mkdir(self, *args: Any, **kwargs: Any) -> None:
            return None

    async def _raise_tts_failure(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("tts backend exploded at /private/output-audio.mp3")

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "render_output_template", lambda *_args: "rendered")
    monkeypatch.setattr(outputs_endpoint, "_outputs_dir_for_user", lambda _user_id: _OutputDir())
    monkeypatch.setattr(outputs_endpoint, "_write_tts_audio_file", _raise_tts_failure)

    with pytest.raises(HTTPException) as exc_info:
        await outputs_endpoint.create_output(
            payload=outputs_endpoint.OutputCreateRequest(
                template_id=1,
                data={"items": []},
                title="demo",
            ),
            current_user=User(id=123, username="tester", email=None, is_active=True),
            cdb=_CollectionsDB(),
            media_db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "tts_generation_failed"
    assert logger.errors == ["outputs tts generation failed"]
    logged = "\n".join(logger.errors)
    assert "tts backend exploded" not in logged
    assert "/private/output-audio.mp3" not in logged


@pytest.mark.asyncio
async def test_outputs_create_write_failure_log_is_sanitized(monkeypatch):
    class _Template:
        id = 1
        name = "Markdown Template"
        body = "hello"
        type = "newsletter_markdown"
        format = "md"

    class _CollectionsDB:
        def get_output_template(self, _template_id: int):
            return _Template()

    class _OutputDir:
        def mkdir(self, *args: Any, **kwargs: Any) -> None:
            return None

    class _OutputPath:
        def write_text(self, *args: Any, **kwargs: Any) -> None:
            raise OSError("write exploded at /private/output.md")

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "render_output_template", lambda *_args: "rendered")
    monkeypatch.setattr(outputs_endpoint, "_outputs_dir_for_user", lambda _user_id: _OutputDir())
    monkeypatch.setattr(outputs_endpoint, "_resolve_output_path_for_user", lambda *_args: _OutputPath())

    with pytest.raises(HTTPException) as exc_info:
        await outputs_endpoint.create_output(
            payload=outputs_endpoint.OutputCreateRequest(
                template_id=1,
                data={"items": []},
                title="demo",
            ),
            current_user=User(id=123, username="tester", email=None, is_active=True),
            cdb=_CollectionsDB(),
            media_db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "write_failed"
    assert logger.errors == ["outputs file write failed"]
    logged = "\n".join(logger.errors)
    assert "write exploded" not in logged
    assert "/private/output.md" not in logged


@pytest.mark.asyncio
async def test_outputs_create_db_insert_failure_log_is_sanitized(monkeypatch):
    class _Template:
        id = 1
        name = "Markdown Template"
        body = "hello"
        type = "newsletter_markdown"
        format = "md"

    class _CollectionsDB:
        def get_output_template(self, _template_id: int):
            return _Template()

        def create_output_artifact(self, **_kwargs: Any):
            raise RuntimeError("output insert exploded at /private/outputs.db")

    class _OutputDir:
        def mkdir(self, *args: Any, **kwargs: Any) -> None:
            return None

    class _OutputPath:
        def write_text(self, *args: Any, **kwargs: Any) -> None:
            return None

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "render_output_template", lambda *_args: "rendered")
    monkeypatch.setattr(outputs_endpoint, "_outputs_dir_for_user", lambda _user_id: _OutputDir())
    monkeypatch.setattr(outputs_endpoint, "_resolve_output_path_for_user", lambda *_args: _OutputPath())
    monkeypatch.setattr(outputs_endpoint.os, "remove", lambda _path: None)

    with pytest.raises(HTTPException) as exc_info:
        await outputs_endpoint.create_output(
            payload=outputs_endpoint.OutputCreateRequest(
                template_id=1,
                data={"items": []},
                title="demo",
            ),
            current_user=User(id=123, username="tester", email=None, is_active=True),
            cdb=_CollectionsDB(),
            media_db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "db_insert_failed"
    assert logger.errors == ["outputs row insert failed"]
    logged = "\n".join(logger.errors)
    assert "output insert exploded" not in logged
    assert "/private/outputs.db" not in logged


@pytest.mark.asyncio
async def test_outputs_create_insert_cleanup_failure_log_is_sanitized(monkeypatch):
    class _Template:
        id = 1
        name = "Markdown Template"
        body = "hello"
        type = "newsletter_markdown"
        format = "md"

    class _CollectionsDB:
        def get_output_template(self, _template_id: int):
            return _Template()

        def create_output_artifact(self, **_kwargs: Any):
            raise RuntimeError("output insert exploded at /private/outputs.db")

    class _OutputDir:
        def mkdir(self, *args: Any, **kwargs: Any) -> None:
            return None

    class _OutputPath:
        def write_text(self, *args: Any, **kwargs: Any) -> None:
            return None

    def _raise_cleanup_failure(_path: Any) -> None:
        raise OSError("cleanup exploded at /private/output.md")

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "render_output_template", lambda *_args: "rendered")
    monkeypatch.setattr(outputs_endpoint, "_outputs_dir_for_user", lambda _user_id: _OutputDir())
    monkeypatch.setattr(outputs_endpoint, "_resolve_output_path_for_user", lambda *_args: _OutputPath())
    monkeypatch.setattr(outputs_endpoint.os, "remove", _raise_cleanup_failure)

    with pytest.raises(HTTPException) as exc_info:
        await outputs_endpoint.create_output(
            payload=outputs_endpoint.OutputCreateRequest(
                template_id=1,
                data={"items": []},
                title="demo",
            ),
            current_user=User(id=123, username="tester", email=None, is_active=True),
            cdb=_CollectionsDB(),
            media_db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "db_insert_failed"
    assert logger.errors == ["outputs row insert failed"]
    assert logger.warnings == ["outputs insert cleanup file removal failed"]
    logged = "\n".join(logger.errors + logger.warnings)
    assert "output insert exploded" not in logged
    assert "cleanup exploded" not in logged
    assert "/private/outputs.db" not in logged
    assert "/private/output.md" not in logged


@pytest.mark.asyncio
async def test_outputs_create_variant_cleanup_failure_logs_are_sanitized(monkeypatch):
    class _Template:
        id = 1
        name = "Markdown Template"
        body = "hello"
        type = "newsletter_markdown"
        format = "md"

    class _Row:
        id = 777
        title = "demo"
        storage_path = "demo.md"
        media_item_id = None
        created_at = "2024-01-01T00:00:00"

    class _CollectionsDB:
        def get_output_template(self, _template_id: int):
            return _Template()

        def get_default_output_template_by_type(self, _template_type: str):
            return None

        def create_output_artifact(self, **_kwargs: Any):
            return _Row()

        def delete_output_artifact(self, _output_id: int, *, hard: bool) -> bool:
            assert hard is True
            raise RuntimeError("cleanup row exploded at /private/outputs.db")

    class _OutputDir:
        def mkdir(self, *args: Any, **kwargs: Any) -> None:
            return None

    class _OutputPath:
        def write_text(self, *args: Any, **kwargs: Any) -> None:
            return None

        def exists(self) -> bool:
            return True

        def unlink(self) -> None:
            raise OSError("cleanup file exploded at /private/output.md")

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "render_output_template", lambda *_args: "rendered")
    monkeypatch.setattr(outputs_endpoint, "_outputs_dir_for_user", lambda _user_id: _OutputDir())
    monkeypatch.setattr(outputs_endpoint, "_resolve_output_path_for_user", lambda *_args: _OutputPath())

    with pytest.raises(HTTPException) as exc_info:
        await outputs_endpoint.create_output(
            payload=outputs_endpoint.OutputCreateRequest(
                template_id=1,
                data={"items": []},
                title="demo",
                generate_mece=True,
            ),
            current_user=User(id=123, username="tester", email=None, is_active=True),
            cdb=_CollectionsDB(),
            media_db=object(),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "mece_template_not_found"
    assert logger.warnings == [
        "failed to cleanup output file",
        "failed to cleanup output row",
    ]
    logged = "\n".join(logger.warnings)
    assert "777" not in logged
    assert "cleanup file exploded" not in logged
    assert "cleanup row exploded" not in logged
    assert "/private/output.md" not in logged
    assert "/private/outputs.db" not in logged


@pytest.mark.asyncio
async def test_outputs_list_invalid_path_fallback_log_is_sanitized(monkeypatch):
    row = SimpleNamespace(
        id=777,
        title="legacy",
        type="newsletter_markdown",
        format="md",
        storage_path="../private/output.md",
        media_item_id=None,
        created_at=datetime.utcnow().replace(microsecond=0).isoformat(),
        workspace_tag=None,
    )

    class _CollectionsDB:
        def list_output_artifacts(self, **_kwargs: Any):
            return [row], 1

    def _raise_invalid_path(**_kwargs: Any) -> str:
        raise HTTPException(status_code=400, detail="invalid path /private/output.md")

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "_normalize_output_storage_path_for_user", _raise_invalid_path)

    result = await outputs_endpoint.list_outputs(
        _current_user=User(id=123, username="tester", email=None, is_active=True),
        cdb=_CollectionsDB(),
    )

    assert result.items[0].storage_path == "../private/output.md"
    assert logger.warnings == ["outputs.list: invalid storage path skipped"]
    logged = "\n".join(logger.warnings)
    assert "777" not in logged
    assert "invalid path" not in logged
    assert "/private/output.md" not in logged


@pytest.mark.asyncio
async def test_outputs_list_deleted_invalid_path_fallback_log_is_sanitized(monkeypatch):
    row = SimpleNamespace(
        id=888,
        title="deleted",
        type="newsletter_markdown",
        format="md",
        storage_path="../private/deleted.md",
        media_item_id=None,
        created_at=datetime.utcnow().replace(microsecond=0).isoformat(),
        workspace_tag=None,
    )

    class _CollectionsDB:
        def list_output_artifacts(self, **_kwargs: Any):
            return [row], 1

    def _raise_invalid_path(**_kwargs: Any) -> str:
        raise HTTPException(status_code=400, detail="invalid path /private/deleted.md")

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "_normalize_output_storage_path_for_user", _raise_invalid_path)

    result = await outputs_endpoint.list_deleted_outputs(
        _current_user=User(id=123, username="tester", email=None, is_active=True),
        cdb=_CollectionsDB(),
    )

    assert result.items[0].storage_path == "../private/deleted.md"
    assert result.pagination.total == 1
    assert result.pagination.limit == 50
    assert result.pagination.offset == 0
    assert result.pagination.has_more is False
    assert logger.warnings == ["outputs.list_deleted: invalid storage path skipped"]
    logged = "\n".join(logger.warnings)
    assert "888" not in logged
    assert "invalid path" not in logged
    assert "/private/deleted.md" not in logged


@pytest.mark.asyncio
async def test_outputs_create_generic_failure_log_is_sanitized(monkeypatch):
    class _Template:
        id = 1
        name = "Markdown Template"
        body = "hello"
        type = "newsletter_markdown"
        format = "md"

    class _Row:
        id = 777
        title = "demo"
        storage_path = "demo.md"
        media_item_id = None
        created_at = "2024-01-01T00:00:00"

    class _CollectionsDB:
        def get_output_template(self, template_id: int):
            if template_id == 1:
                return _Template()
            raise RuntimeError("variant template backend exploded at /private/outputs.db")

        def create_output_artifact(self, **_kwargs: Any):
            return _Row()

        def delete_output_artifact(self, _output_id: int, *, hard: bool) -> bool:
            assert hard is True
            return True

    class _OutputDir:
        def mkdir(self, *args: Any, **kwargs: Any) -> None:
            return None

    class _OutputPath:
        def write_text(self, *args: Any, **kwargs: Any) -> None:
            return None

        def exists(self) -> bool:
            return False

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "render_output_template", lambda *_args: "rendered")
    monkeypatch.setattr(outputs_endpoint, "_outputs_dir_for_user", lambda _user_id: _OutputDir())
    monkeypatch.setattr(outputs_endpoint, "_resolve_output_path_for_user", lambda *_args: _OutputPath())

    with pytest.raises(HTTPException) as exc_info:
        await outputs_endpoint.create_output(
            payload=outputs_endpoint.OutputCreateRequest(
                template_id=1,
                data={"items": []},
                title="demo",
                generate_mece=True,
                mece_template_id=2,
            ),
            current_user=User(id=123, username="tester", email=None, is_active=True),
            cdb=_CollectionsDB(),
            media_db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "output_create_failed"
    assert logger.errors == ["outputs.create failed"]
    logged = "\n".join(logger.errors + logger.warnings)
    assert "variant template backend exploded" not in logged
    assert "/private/outputs.db" not in logged


@pytest.mark.asyncio
async def test_outputs_delete_tts_history_failure_log_is_sanitized(monkeypatch):
    class _CollectionsDB:
        def delete_output_artifact(self, _output_id: int, *, hard: bool) -> bool:
            assert hard is False
            return True

    class _MediaDB:
        def mark_tts_history_artifacts_deleted_for_output(self, **_kwargs: Any) -> None:
            raise RuntimeError("tts history backend exploded at /private/media.db")

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)

    result = await outputs_endpoint.delete_output(
        output_id=777,
        hard=False,
        delete_file=False,
        current_user=User(id=123, username="tester", email=None, is_active=True),
        cdb=_CollectionsDB(),
        media_db=_MediaDB(),
    )

    assert result == {"success": True, "file_deleted": False}
    assert logger.debugs == ["outputs.delete: failed to update tts_history"]
    logged = "\n".join(logger.debugs)
    assert "777" not in logged
    assert "tts history backend exploded" not in logged
    assert "/private/media.db" not in logged


@pytest.mark.asyncio
async def test_outputs_purge_enumerate_failure_log_is_sanitized(monkeypatch):
    class _CollectionsDB:
        user_id = 123

    def _raise_enumerate_failure(*args: Any, **kwargs: Any):
        raise RuntimeError("purge enumerate exploded at /private/outputs.db")

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "find_outputs_to_purge", _raise_enumerate_failure)

    result = await outputs_endpoint.purge_outputs(
        payload=outputs_endpoint.OutputsPurgeRequest(delete_files=False),
        current_user=User(id=123, username="tester", email=None, is_active=True),
        cdb=_CollectionsDB(),
    )

    assert result == {"removed": 0, "files_deleted": 0}
    assert logger.errors == ["outputs.purge: failed to enumerate purge candidates"]
    logged = "\n".join(logger.errors)
    assert "purge enumerate exploded" not in logged
    assert "/private/outputs.db" not in logged


@pytest.mark.asyncio
async def test_outputs_purge_db_delete_failure_log_is_sanitized(monkeypatch):
    class _CollectionsDB:
        user_id = 123

    def _raise_delete_failure(*args: Any, **kwargs: Any):
        raise RuntimeError("purge delete exploded at /private/outputs.db")

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "find_outputs_to_purge", lambda **_kwargs: {777: "output.md"})
    monkeypatch.setattr(outputs_endpoint, "delete_outputs_by_ids", _raise_delete_failure)

    result = await outputs_endpoint.purge_outputs(
        payload=outputs_endpoint.OutputsPurgeRequest(delete_files=False),
        current_user=User(id=123, username="tester", email=None, is_active=True),
        cdb=_CollectionsDB(),
    )

    assert result == {"removed": 0, "files_deleted": 0}
    assert logger.errors == ["outputs.purge: DB delete failed"]
    logged = "\n".join(logger.errors)
    assert "777" not in logged
    assert "purge delete exploded" not in logged
    assert "/private/outputs.db" not in logged


@pytest.mark.asyncio
async def test_outputs_purge_file_delete_failure_log_is_sanitized(monkeypatch):
    class _CollectionsDB:
        user_id = 123

    class _OutputPath:
        def exists(self) -> bool:
            return True

        def unlink(self) -> None:
            raise OSError("purge file delete exploded at /private/output.md")

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "find_outputs_to_purge", lambda **_kwargs: {777: "output.md"})
    monkeypatch.setattr(outputs_endpoint, "delete_outputs_by_ids", lambda **_kwargs: 1)
    monkeypatch.setattr(outputs_endpoint, "_normalize_output_storage_path_for_user", lambda **_kwargs: "output.md")
    monkeypatch.setattr(outputs_endpoint, "_resolve_output_path_for_user", lambda *_args: _OutputPath())

    result = await outputs_endpoint.purge_outputs(
        payload=outputs_endpoint.OutputsPurgeRequest(delete_files=True),
        current_user=User(id=123, username="tester", email=None, is_active=True),
        cdb=_CollectionsDB(),
    )

    assert result == {"removed": 1, "files_deleted": 0}
    assert logger.warnings == ["outputs.purge: failed to delete file"]
    logged = "\n".join(logger.warnings)
    assert "777" not in logged
    assert "output.md" not in logged
    assert "purge file delete exploded" not in logged
    assert "/private/output.md" not in logged


@pytest.mark.asyncio
async def test_outputs_update_old_file_cleanup_failure_log_is_sanitized(monkeypatch):
    class _Row:
        id = 777
        title = "Old Output"
        type = "newsletter_markdown"
        format = "md"
        storage_path = "old-output.md"
        media_item_id = None
        created_at = "2024-01-01T00:00:00"

    class _CollectionsDB:
        def get_output_artifact(self, _output_id: int):
            return _Row()

    class _OutputPath:
        def __init__(self, name: str) -> None:
            self.name = name

        @property
        def suffix(self) -> str:
            return "." + self.name.rsplit(".", 1)[-1]

        @property
        def stem(self) -> str:
            return self.name.rsplit(".", 1)[0]

        def read_text(self, **_kwargs: Any) -> str:
            return "# hello"

        def write_text(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def resolve(self) -> str:
            return f"/private/outputs/{self.name}"

        def exists(self) -> bool:
            return True

        def unlink(self) -> None:
            raise OSError("old output cleanup exploded at /private/outputs/old-output.md")

    def _resolve_path(_user_id: int, name: str) -> _OutputPath:
        return _OutputPath(name)

    def _update_output_artifact_db(**kwargs: Any):
        assert kwargs["new_format"] == "html"
        return _Row()

    logger = _LoggerStub()
    monkeypatch.setattr(outputs_endpoint, "logger", logger)
    monkeypatch.setattr(outputs_endpoint, "_resolve_output_path_for_user", _resolve_path)
    monkeypatch.setattr(outputs_endpoint, "update_output_artifact_db", _update_output_artifact_db)

    result = await outputs_endpoint.update_output(
        output_id=777,
        payload=outputs_endpoint.OutputUpdateRequest(format="html"),
        current_user=User(id=123, username="tester", email=None, is_active=True),
        cdb=_CollectionsDB(),
    )

    assert result.id == 777
    assert logger.warnings == ["failed to remove old output file"]
    logged = "\n".join(logger.warnings)
    assert "old-output.md" not in logged
    assert "old output cleanup exploded" not in logged
    assert "/private/outputs" not in logged


def test_outputs_preview_with_inline_data_and_generate(client_with_user, tmp_path):

    client = client_with_user

    # Create a template
    payload = {
        "name": "inline-demo",
        "type": "newsletter_markdown",
        "format": "md",
        "body": "# Daily Brief\nTop: {{ items[0].title if items else 'none' }}\n",
        "description": "Inline demo",
        "is_default": False,
    }
    r = client.post("/api/v1/outputs/templates", json=payload)
    assert r.status_code == 200, r.text
    tpl = r.json()
    tid = tpl["id"]

    # Preview with inline data
    inline_ctx = {
        "items": [
            {
                "title": "Example Story",
                "url": "https://example.com/x",
                "domain": "example.com",
                "summary": "S",
                "published_at": "2024-01-01",
                "tags": ["a"],
            }
        ]
    }
    r = client.post(f"/api/v1/outputs/templates/{tid}/preview", json={"template_id": tid, "data": inline_ctx})
    assert r.status_code == 200, r.text
    prev = r.json()
    assert "Example Story" in prev["rendered"]

    # Generate output with the same inline data
    r = client.post("/api/v1/outputs", json={"template_id": tid, "data": inline_ctx, "title": "demo"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["format"] == "md"
    path = _resolve_output_path_for_user(123, out["storage_path"])
    assert path.exists(), f"Output file missing at {path}"
    text = path.read_text(encoding="utf-8")
    assert "Example Story" in text

    # Get by id
    oid = out["id"]
    r = client.get(f"/api/v1/outputs/{oid}")
    assert r.status_code == 200
    meta = r.json()
    assert meta["id"] == oid

    # Download
    r = client.get(f"/api/v1/outputs/{oid}/download")
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/markdown")
    r = client.head(f"/api/v1/outputs/{oid}/download")
    assert r.status_code == 200
    assert int(r.headers.get("content-length", "0")) > 0
    r = client.get("/api/v1/outputs/download/by-name", params={"title": "demo", "format": "md"})
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/markdown")

    # List outputs
    r = client.get("/api/v1/outputs", params={"page": 1, "size": 10})
    assert r.status_code == 200
    lst = r.json()
    assert lst["total"] >= 1
    assert lst["pagination"]["total"] >= 1
    assert lst["pagination"]["limit"] == 10
    assert lst["pagination"]["offset"] == 0
    assert lst["has_more"] == lst["pagination"]["has_more"]
    assert lst["next_offset"] == lst["pagination"]["next_offset"]
    assert any(it["id"] == oid for it in lst.get("items", []))


def test_outputs_generate_variants_and_ingest(client_with_user, monkeypatch):
    client = client_with_user

    class DummyTTS:
        async def generate_speech(self, req):  # noqa: ARG002 - signature used by TTS service
            yield b"FAKEAUDIO"

    async def _fake_get_tts_service_v2(*args, **kwargs):  # noqa: ARG002
        return DummyTTS()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_service_v2.get_tts_service_v2",
        _fake_get_tts_service_v2,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.services.outputs_service.get_tts_service_v2",
        _fake_get_tts_service_v2,
    )

    base_payload = {
        "name": "base-briefing",
        "type": "briefing_markdown",
        "format": "md",
        "body": "# Briefing\n{{ items|length }} items",
        "description": "Base output",
        "is_default": False,
    }
    r = client.post("/api/v1/outputs/templates", json=base_payload)
    assert r.status_code == 200, r.text
    base_id = r.json()["id"]

    mece_payload = {
        "name": "mece-default",
        "type": "mece_markdown",
        "format": "md",
        "body": "# MECE\n{% for item in items %}- {{ item.title }}\n{% endfor %}",
        "description": "MECE output",
        "is_default": True,
    }
    r = client.post("/api/v1/outputs/templates", json=mece_payload)
    assert r.status_code == 200, r.text

    tts_payload = {
        "name": "tts-default",
        "type": "tts_audio",
        "format": "mp3",
        "body": "Audio briefing for {{ items|length }} items.",
        "description": "TTS output",
        "is_default": True,
    }
    r = client.post("/api/v1/outputs/templates", json=tts_payload)
    assert r.status_code == 200, r.text

    inline_ctx = {
        "items": [
            {
                "title": "Story A",
                "url": "https://example.com/a",
                "domain": "example.com",
                "summary": "A",
                "published_at": "2024-01-01",
                "tags": ["tag-a"],
            }
        ]
    }

    r = client.post(
        "/api/v1/outputs",
        json={
            "template_id": base_id,
            "data": inline_ctx,
            "title": "Daily",
            "generate_mece": True,
            "generate_tts": True,
            "ingest_to_media_db": True,
        },
    )
    assert r.status_code == 200, r.text
    base = r.json()
    assert base["media_item_id"] is not None

    r = client.get("/api/v1/outputs", params={"type": "mece_markdown"})
    assert r.status_code == 200
    mece_outputs = r.json()["items"]
    assert any(item.get("media_item_id") for item in mece_outputs)

    r = client.get("/api/v1/outputs", params={"type": "tts_audio"})
    assert r.status_code == 200
    tts_outputs = r.json()["items"]
    assert any(item.get("media_item_id") for item in tts_outputs)


def test_outputs_create_sanitizes_title_and_enforces_base_dir(client_with_user):

    client = client_with_user

    payload = {
        "name": "path-safety",
        "type": "newsletter_markdown",
        "format": "md",
        "body": "Hello {{ items|length }}",
        "description": "Path safety",
        "is_default": False,
    }
    r = client.post("/api/v1/outputs/templates", json=payload)
    assert r.status_code == 200, r.text
    tid = r.json()["id"]

    r = client.post("/api/v1/outputs", json={"template_id": tid, "data": {"items": []}, "title": "../outside"})
    assert r.status_code == 200, r.text
    out = r.json()
    out_path = _resolve_output_path_for_user(123, out["storage_path"])
    base_dir = DatabasePaths.get_user_base_directory(123) / "outputs"
    assert out_path.exists()
    assert out_path.resolve().is_relative_to(base_dir.resolve())


def test_outputs_download_rejects_storage_path_outside_base(client_with_user, tmp_path):

    client = client_with_user

    external = tmp_path / "outside.md"
    external.write_text("nope", encoding="utf-8")
    cdb = CollectionsDatabase.for_user(user_id=123)
    with pytest.raises(InvalidStoragePathError):
        cdb.create_output_artifact(
            type_="newsletter_markdown",
            title="outside",
            format_="md",
            storage_path=str(external),
            metadata_json=None,
        )
    row_id = _insert_output_row_raw(
        cdb,
        title="outside",
        format_="md",
        storage_path=str(external),
    )

    r = client.get(f"/api/v1/outputs/{row_id}/download")
    assert r.status_code == 400, r.text
    assert r.json().get("detail") == "invalid_path"


def test_outputs_download_normalizes_legacy_absolute_path(client_with_user):

    client = client_with_user

    base_dir = DatabasePaths.get_user_base_directory(123) / "outputs"
    base_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = base_dir / "legacy.md"
    legacy_path.write_text("legacy", encoding="utf-8")
    cdb = CollectionsDatabase.for_user(user_id=123)
    row_id = _insert_output_row_raw(
        cdb,
        title="legacy",
        format_="md",
        storage_path=str(legacy_path),
    )

    r = client.get(f"/api/v1/outputs/{row_id}/download")
    assert r.status_code == 200, r.text
    row = cdb.get_output_artifact(row_id)
    assert row.storage_path == "legacy.md"


def test_outputs_delete_skips_invalid_path_file_removal(client_with_user, tmp_path):

    client = client_with_user

    external = tmp_path / "external.txt"
    external.write_text("keep", encoding="utf-8")
    cdb = CollectionsDatabase.for_user(user_id=123)
    with pytest.raises(InvalidStoragePathError):
        cdb.create_output_artifact(
            type_="newsletter_markdown",
            title="outside-delete",
            format_="md",
            storage_path=str(external),
            metadata_json=None,
        )
    row_id = _insert_output_row_raw(
        cdb,
        title="outside-delete",
        format_="md",
        storage_path=str(external),
    )

    r = client.delete(
        f"/api/v1/outputs/{row_id}",
        params={"hard": "true", "delete_file": "true"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["success"] is True
    assert payload["file_deleted"] is False
    assert external.exists()
    with pytest.raises(KeyError):
        cdb.get_output_artifact(row_id, include_deleted=True)


def test_outputs_purge_skips_invalid_path_delete_files(client_with_user, tmp_path):

    client = client_with_user

    external = tmp_path / "purge.txt"
    external.write_text("keep", encoding="utf-8")
    cdb = CollectionsDatabase.for_user(user_id=123)
    past = (datetime.utcnow() - timedelta(days=1)).replace(microsecond=0).isoformat()
    with pytest.raises(InvalidStoragePathError):
        cdb.create_output_artifact(
            type_="newsletter_markdown",
            title="outside-purge",
            format_="md",
            storage_path=str(external),
            metadata_json=None,
            retention_until=past,
        )
    row_id = _insert_output_row_raw(
        cdb,
        title="outside-purge",
        format_="md",
        storage_path=str(external),
        retention_until=past,
    )

    r = client.post("/api/v1/outputs/purge", json={"delete_files": True})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["removed"] == 1
    assert external.exists()
    with pytest.raises(KeyError):
        cdb.get_output_artifact(row_id, include_deleted=True)


def test_outputs_resolve_path_rejects_traversal(client_with_user):  # noqa: ARG001 - fixture sets up USER_DB_BASE_DIR
    user_id = 123
    with pytest.raises(HTTPException) as excinfo:
        _resolve_output_path_for_user(user_id, "../outside.txt")
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "invalid_path"


def test_strip_html_for_tts_removes_tags():
    assert _strip_html_for_tts("Hello <b>World</b>") == "Hello World"


def test_strip_html_for_tts_keeps_unclosed_tag_literal():
    assert _strip_html_for_tts("Hello <b") == "Hello <b"


def test_strip_html_for_tts_keeps_empty_tag_literal():
    assert _strip_html_for_tts("a<>b") == "a<>b"


def test_strip_html_for_tts_keeps_trailing_unclosed_tag():
    assert _strip_html_for_tts("a<b>c<d") == "ac<d"
