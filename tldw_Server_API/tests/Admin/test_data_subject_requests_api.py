from __future__ import annotations

import os
import sqlite3
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.schemas.admin_schemas import DataSubjectRequestListResponse
from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta
from tldw_Server_API.app.main import app


def _setup_env(tmp_path) -> None:
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = "unit-test-api-key"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'users_test_dsr_api.db'}"
    os.environ["TLDW_DB_ALLOWED_BASE_DIRS"] = str(tmp_path)
    os.environ["TLDW_DB_BACKUP_PATH"] = str(tmp_path / "backups")
    os.environ["USER_DB_BASE_DIR"] = str(tmp_path / "user_dbs")


async def _reset_auth_state() -> None:
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()


async def _seed_user(*, username: str, email: str) -> int:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

    pool = await get_db_pool()
    # Reserve the login principal before allocating a separate subject. Startup
    # initializes the principal's real stores; these subject stores are scoped fixtures.
    await AuthnzUsersRepo(pool).ensure_single_user_admin_user(user_id=get_settings().SINGLE_USER_FIXED_ID)
    return await ensure_test_user(pool, username, email)


def _seed_subject_store_data(*, tmp_path, user_id: int, media_count: int, note_count: int, message_count: int, audit_count: int) -> None:
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    media_db_path = DatabasePaths.get_media_db_path(user_id)
    media_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(media_db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS Media (
                id INTEGER PRIMARY KEY,
                deleted INTEGER DEFAULT 0,
                is_trash INTEGER DEFAULT 0
            )
            """
        )
        for index in range(media_count):
            conn.execute(
                "INSERT INTO Media (id, deleted, is_trash) VALUES (?, 0, 0)",
                (index + 1,),
            )
        conn.execute("INSERT INTO Media (id, deleted, is_trash) VALUES (?, 1, 0)", (9001,))
        conn.execute("INSERT INTO Media (id, deleted, is_trash) VALUES (?, 0, 1)", (9002,))
        conn.commit()

    chacha_db_path = DatabasePaths.get_chacha_db_path(user_id)
    with sqlite3.connect(chacha_db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS notes (id TEXT PRIMARY KEY, deleted INTEGER DEFAULT 0)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                deleted INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                deleted INTEGER DEFAULT 0
            )
            """
        )
        for index in range(note_count):
            conn.execute("INSERT INTO notes (id, deleted) VALUES (?, 0)", (f"note-{index}",))
        conn.execute("INSERT INTO notes (id, deleted) VALUES (?, 1)", ("note-deleted",))
        conn.execute(
            "INSERT INTO conversations (id, client_id, deleted) VALUES (?, ?, 0)",
            ("conv-1", str(user_id)),
        )
        conn.execute(
            "INSERT INTO conversations (id, client_id, deleted) VALUES (?, ?, 1)",
            ("conv-deleted", str(user_id)),
        )
        for index in range(message_count):
            conn.execute(
                "INSERT INTO messages (id, conversation_id, deleted) VALUES (?, ?, 0)",
                (f"msg-{index}", "conv-1"),
            )
        conn.execute(
            "INSERT INTO messages (id, conversation_id, deleted) VALUES (?, ?, 1)",
            ("msg-deleted", "conv-1"),
        )
        conn.commit()

    audit_db_path = DatabasePaths.get_audit_db_path(user_id)
    with sqlite3.connect(audit_db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                context_user_id TEXT
            )
            """
        )
        for index in range(audit_count):
            conn.execute(
                """
                INSERT INTO audit_events (event_id, timestamp, category, event_type, severity, context_user_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"evt-{index}",
                    f"2026-03-10T12:00:0{index}Z",
                    "system",
                    "test.event",
                    "low",
                    str(user_id),
                ),
            )
        conn.commit()


@pytest.mark.asyncio
async def test_preview_returns_404_for_unknown_requester(tmp_path):
    _setup_env(tmp_path)
    await _reset_auth_state()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        response = client.post(
            "/api/v1/admin/data-subject-requests/preview",
            json={"requester_identifier": "missing@example.com"},
        )

    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "requester_not_found"


@pytest.mark.asyncio
async def test_preview_returns_500_when_subject_store_query_fails(tmp_path):
    _setup_env(tmp_path)
    await _reset_auth_state()
    user_id = await _seed_user(username="subject_user", email="subject@example.com")

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    media_db_path = DatabasePaths.get_media_db_path(user_id)
    media_db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(media_db_path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS unrelated_table (id INTEGER PRIMARY KEY)")
        conn.commit()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        response = client.post(
            "/api/v1/admin/data-subject-requests/preview",
            json={"requester_identifier": "subject@example.com"},
        )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "requester_data_unavailable"


@pytest.mark.asyncio
async def test_preview_returns_500_when_subject_store_is_missing(tmp_path):
    _setup_env(tmp_path)
    await _reset_auth_state()
    await _seed_user(username="subject_user", email="subject@example.com")

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        response = client.post(
            "/api/v1/admin/data-subject-requests/preview",
            json={"requester_identifier": "subject@example.com"},
        )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "requester_data_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["preview", "intake"])
@pytest.mark.parametrize("failure_at", ["manager", "list", "count"])
async def test_embedding_coverage_failure_is_sanitized_and_does_not_record_intake(
    monkeypatch, tmp_path, operation, failure_at
):
    from tldw_Server_API.app.core.config import settings
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    _setup_env(tmp_path)
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    await _reset_auth_state()
    user_id = await _seed_user(username="subject_user", email="subject@example.com")
    _seed_subject_store_data(
        tmp_path=tmp_path, user_id=user_id, media_count=1, note_count=1, message_count=1, audit_count=1
    )
    (tmp_path / "user_dbs" / str(user_id) / "chroma_storage").mkdir(parents=True)
    private_marker = "private-subject-embedding-content"
    manager = MagicMock()
    collection = MagicMock()
    collection.count.side_effect = RuntimeError(private_marker)
    manager.list_collections.return_value = [collection]
    get_manager = MagicMock(return_value=manager)
    if failure_at == "manager":
        get_manager.side_effect = ImportError(private_marker)
    elif failure_at == "list":
        manager.list_collections.side_effect = RuntimeError(private_marker)
    monkeypatch.setattr(service, "_get_chroma_manager_for_user", get_manager)
    payload = {
        "requester_identifier": "subject@example.com",
        "request_type": "erasure",
        "categories": ["embeddings"],
    }
    url = "/api/v1/admin/data-subject-requests"
    if operation == "preview":
        url += "/preview"
    else:
        payload["client_request_id"] = "unavailable-embedding-intake"

    with TestClient(app, headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}) as client:
        response = client.post(url, json=payload)
        history = client.get("/api/v1/admin/data-subject-requests")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "requester_data_unavailable"
    assert private_marker not in response.text
    assert history.status_code == 200, history.text
    assert history.json()["items"] == []


@pytest.mark.asyncio
async def test_notes_only_preview_succeeds_with_unrelated_stores_missing(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.services import admin_data_subject_requests_service as service

    _setup_env(tmp_path)
    await _reset_auth_state()
    user_id = await _seed_user(username="subject_user", email="subject@example.com")
    with sqlite3.connect(DatabasePaths.get_chacha_db_path(user_id)) as conn:
        conn.execute("CREATE TABLE notes (id TEXT PRIMARY KEY, deleted INTEGER DEFAULT 0)")
        conn.execute("INSERT INTO notes VALUES ('kept', 0), ('deleted', 1)")
    monkeypatch.setattr(service, "_get_chroma_manager_for_user", MagicMock(side_effect=RuntimeError("unavailable")))

    with TestClient(app, headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}) as client:
        response = client.post(
            "/api/v1/admin/data-subject-requests/preview",
            json={"requester_identifier": "subject@example.com", "categories": ["notes"]},
        )

    assert response.status_code == 200, response.text
    assert response.json()["counts"] == {"notes": 1}


@pytest.mark.asyncio
async def test_create_records_request_and_reuses_client_request_id(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    await _reset_auth_state()
    user_id = await _seed_user(username="subject_user", email="subject@example.com")
    _seed_subject_store_data(
        tmp_path=tmp_path,
        user_id=user_id,
        media_count=3,
        note_count=2,
        message_count=4,
        audit_count=5,
    )

    from tldw_Server_API.app.api.v1.endpoints.admin import admin_data_ops

    emitted: list[dict[str, object]] = []

    async def _fake_emit(*args, **kwargs):
        emitted.append(kwargs)

    monkeypatch.setattr(admin_data_ops, "_emit_admin_audit_event", _fake_emit)

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}
    payload = {
        "client_request_id": "dsr-client-1",
        "requester_identifier": "subject@example.com",
        "request_type": "export",
    }

    with TestClient(app, headers=headers) as client:
        first = client.post("/api/v1/admin/data-subject-requests", json=payload)
        second = client.post("/api/v1/admin/data-subject-requests", json=payload)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    first_item = first.json()["item"]
    second_item = second.json()["item"]
    assert first_item["id"] == second_item["id"]
    assert first_item["client_request_id"] == "dsr-client-1"
    assert first_item["status"] == "recorded"
    assert first_item["selected_categories"] == [
        "media_records",
        "chat_messages",
        "notes",
        "audit_events",
        "embeddings",
    ]

    summary_by_key = {
        entry["key"]: entry["count"]
        for entry in first_item["preview_summary"]
    }
    assert summary_by_key == {
        "media_records": 3,
        "chat_messages": 4,
        "notes": 2,
        "audit_events": 5,
        "embeddings": 0,
    }
    assert len(emitted) == 2
    assert emitted[0]["action"] == "data_subject_request.record"


@pytest.mark.asyncio
async def test_list_returns_newest_first_with_limit_offset(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    await _reset_auth_state()

    first_user_id = await _seed_user(username="subject_one", email="subject1@example.com")
    second_user_id = await _seed_user(username="subject_two", email="subject2@example.com")
    third_user_id = await _seed_user(username="subject_three", email="subject3@example.com")

    for user_id in (first_user_id, second_user_id, third_user_id):
        _seed_subject_store_data(
            tmp_path=tmp_path,
            user_id=user_id,
            media_count=1,
            note_count=1,
            message_count=1,
            audit_count=1,
        )

    from tldw_Server_API.app.api.v1.endpoints.admin import admin_data_ops

    async def _fake_emit(*args, **kwargs):
        return None

    monkeypatch.setattr(admin_data_ops, "_emit_admin_audit_event", _fake_emit)

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        for index, email in enumerate(
            ["subject1@example.com", "subject2@example.com", "subject3@example.com"],
            start=1,
        ):
            response = client.post(
                "/api/v1/admin/data-subject-requests",
                json={
                    "client_request_id": f"dsr-client-{index}",
                    "requester_identifier": email,
                    "request_type": "access",
                },
            )
            assert response.status_code == 200, response.text

        listed = client.get(
            "/api/v1/admin/data-subject-requests",
            params={"limit": 2, "offset": 1},
        )

    assert listed.status_code == 200, listed.text
    payload = listed.json()
    assert payload["total"] == 3
    assert payload["pagination"]["total"] == 3
    assert payload["pagination"]["limit"] == 2
    assert payload["pagination"]["offset"] == 1
    assert payload["pagination"]["has_more"] is False
    assert [item["client_request_id"] for item in payload["items"]] == [
        "dsr-client-2",
        "dsr-client-1",
    ]


def test_data_subject_request_list_response_defaults_pagination_aliases():
    response = DataSubjectRequestListResponse(
        items=[
            {
                "id": 1,
                "client_request_id": "dsr-client-1",
                "requester_identifier": "subject@example.com",
                "request_type": "access",
                "status": "pending",
                "selected_categories": ["authnz"],
                "preview_summary": [],
                "requested_at": "2026-03-10T12:00:00Z",
            }
        ],
        total=3,
        limit=2,
        offset=1,
        pagination=OffsetPaginationMeta(
            total=3,
            limit=2,
            offset=1,
            has_more=False,
            next_offset=None,
        ),
    )

    assert response.has_more is False
    assert response.next_offset is None


@pytest.mark.asyncio
async def test_preview_enforces_admin_scope(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    await _reset_auth_state()
    user_id = await _seed_user(username="subject_user", email="subject@example.com")
    _seed_subject_store_data(
        tmp_path=tmp_path,
        user_id=user_id,
        media_count=1,
        note_count=1,
        message_count=1,
        audit_count=1,
    )

    from tldw_Server_API.app.services import admin_data_subject_requests_service as dsr_service

    async def _deny_scope(*args, **kwargs):
        raise HTTPException(status_code=403, detail="scoped_out")

    monkeypatch.setattr(dsr_service.admin_scope_service, "is_platform_admin", lambda principal: False)
    monkeypatch.setattr(dsr_service.admin_scope_service, "enforce_admin_user_scope", _deny_scope)

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        response = client.post(
            "/api/v1/admin/data-subject-requests/preview",
            json={"requester_identifier": "subject@example.com"},
        )

    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "requester_not_found"


@pytest.mark.asyncio
async def test_preview_hides_out_of_scope_requesters_before_counting(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    await _reset_auth_state()
    user_id = await _seed_user(username="subject_user", email="subject@example.com")
    _seed_subject_store_data(
        tmp_path=tmp_path,
        user_id=user_id,
        media_count=1,
        note_count=1,
        message_count=1,
        audit_count=1,
    )

    from tldw_Server_API.app.services import admin_data_subject_requests_service as dsr_service

    monkeypatch.setattr(dsr_service.admin_scope_service, "is_platform_admin", lambda principal: False)

    async def _deny_scope(principal, target_user_id: int, *, require_hierarchy: bool) -> None:
        del principal, target_user_id, require_hierarchy
        raise HTTPException(status_code=403, detail="scoped_out")

    monkeypatch.setattr(dsr_service.admin_scope_service, "enforce_admin_user_scope", _deny_scope)

    def _unexpected_summary_build(*args, **kwargs):
        raise AssertionError("summary should not be built for out-of-scope requester")

    monkeypatch.setattr(dsr_service, "_build_summary_for_user", _unexpected_summary_build)

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        response = client.post(
            "/api/v1/admin/data-subject-requests/preview",
            json={"requester_identifier": "subject@example.com"},
        )

    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "requester_not_found"


@pytest.mark.asyncio
async def test_create_hides_out_of_scope_requesters_before_counting(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    await _reset_auth_state()
    user_id = await _seed_user(username="subject_user", email="subject@example.com")
    _seed_subject_store_data(
        tmp_path=tmp_path,
        user_id=user_id,
        media_count=1,
        note_count=1,
        message_count=1,
        audit_count=1,
    )

    from tldw_Server_API.app.services import admin_data_subject_requests_service as dsr_service

    monkeypatch.setattr(dsr_service.admin_scope_service, "is_platform_admin", lambda principal: False)

    async def _deny_scope(principal, target_user_id: int, *, require_hierarchy: bool) -> None:
        del principal, target_user_id, require_hierarchy
        raise HTTPException(status_code=403, detail="scoped_out")

    monkeypatch.setattr(dsr_service.admin_scope_service, "enforce_admin_user_scope", _deny_scope)

    def _unexpected_summary_build(*args, **kwargs):
        raise AssertionError("summary should not be built for out-of-scope requester")

    monkeypatch.setattr(dsr_service, "_build_summary_for_user", _unexpected_summary_build)

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        response = client.post(
            "/api/v1/admin/data-subject-requests",
            json={
                "client_request_id": "out-of-scope-dsr",
                "requester_identifier": "subject@example.com",
                "request_type": "access",
            },
        )

    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "requester_not_found"
