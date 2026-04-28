from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import chat_workflows_deps as deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


@contextmanager
def _capture_dependency_error_logs() -> Iterator[list[str]]:
    messages: list[str] = []
    sink_id = deps.logger.add(
        lambda message: messages.append(str(message)),
        filter=lambda record: record["name"] == deps.__name__,
        format="{message}",
        level="ERROR",
    )
    try:
        yield messages
    finally:
        deps.logger.remove(sink_id)


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/cw/me")
    async def me(ctx: dict[str, Any] = Depends(deps.get_chat_workflows_user)):
        return ctx

    return app


def test_chat_workflows_user_claims_permissions(monkeypatch):
    calls: dict[str, Any] = {}

    async def fake_get_request_user(request, api_key=None, token=None, legacy_token_header=None):
        calls["api_key"] = api_key
        calls["token"] = token
        return User(
            id=7,
            username="workflow-user",
            roles=["user"],
            permissions=["chat_workflows.run", "chat_workflows.write"],
            is_admin=False,
        )

    monkeypatch.setattr(deps, "get_request_user", fake_get_request_user, raising=True)

    app = _make_app()
    client = TestClient(app)
    response = client.get("/cw/me", headers={"Authorization": "Bearer token"})

    assert response.status_code == 200
    assert "chat_workflows.run" in response.json()["permissions"]
    assert response.json()["client_id"] == "web"
    assert calls["api_key"] is None
    assert calls["token"] == "token"


def test_chat_workflows_user_requires_auth_headers(monkeypatch):
    async def fake_get_request_user(*_args, **_kwargs):
        raise RuntimeError("should_not_be_called")

    monkeypatch.setattr(deps, "get_request_user", fake_get_request_user, raising=True)

    app = _make_app()
    client = TestClient(app)
    response = client.get("/cw/me")

    assert response.status_code == 401


def test_chat_workflows_db_cache_is_scoped_per_app(monkeypatch):
    created: list[tuple[str, str]] = []

    class FakeDB:
        def __init__(self, label: str):
            self.label = label
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def fake_create_chat_workflows_database(*, client_id, db_path, backend):
        created.append((client_id, str(db_path)))
        return FakeDB(f"{client_id}:{db_path}")

    monkeypatch.setattr(deps, "create_chat_workflows_database", fake_create_chat_workflows_database, raising=True)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: None, raising=True)
    monkeypatch.setattr(
        deps.DatabasePaths,
        "get_chat_workflows_db_path",
        staticmethod(lambda user_id: Path(f"/tmp/{user_id}.db")),
    )

    app_one = FastAPI()
    app_two = FastAPI()

    first = deps._get_or_create_chat_workflows_db(app_one, "user-1", "web")
    second = deps._get_or_create_chat_workflows_db(app_one, "user-1", "web")
    third = deps._get_or_create_chat_workflows_db(app_two, "user-1", "web")

    assert first is second
    assert third is not first
    assert len(created) == 2


def test_chat_workflows_db_create_failure_log_is_sanitized(monkeypatch):
    raw_user_id = "raw-user-token-42"
    raw_marker = "create raw marker"
    raw_path = "/private/chat-workflows-create.db"
    raw_token = "secret-token-create"

    def fake_create_chat_workflows_database(*, client_id, db_path, backend):
        raise RuntimeError(f"{raw_marker} {raw_path} {raw_token}")

    monkeypatch.setattr(deps, "create_chat_workflows_database", fake_create_chat_workflows_database, raising=True)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: None, raising=True)
    monkeypatch.setattr(
        deps.DatabasePaths,
        "get_chat_workflows_db_path",
        staticmethod(lambda user_id: Path(f"/tmp/{user_id}.db")),
    )

    app = FastAPI()

    with _capture_dependency_error_logs() as messages:
        with pytest.raises(HTTPException) as exc:
            deps._get_or_create_chat_workflows_db(app, raw_user_id, "web")

    rendered = "\n".join(messages)
    assert raw_marker not in rendered
    assert raw_path not in rendered
    assert raw_token not in rendered
    assert raw_user_id not in rendered
    assert "Failed to create ChatWorkflowsDatabase" in rendered
    assert "RuntimeError" in rendered
    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to initialize chat workflows database"


def test_shutdown_chat_workflows_deps_closes_only_target_app_instances(monkeypatch):
    class FakeDB:
        def __init__(self, label: str):
            self.label = label
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def fake_create_chat_workflows_database(*, client_id, db_path, backend):
        return FakeDB(f"{client_id}:{db_path}")

    monkeypatch.setattr(deps, "create_chat_workflows_database", fake_create_chat_workflows_database, raising=True)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: None, raising=True)
    monkeypatch.setattr(
        deps.DatabasePaths,
        "get_chat_workflows_db_path",
        staticmethod(lambda user_id: Path(f"/tmp/{user_id}.db")),
    )

    app_one = FastAPI()
    app_two = FastAPI()

    first = deps._get_or_create_chat_workflows_db(app_one, "user-1", "web")
    second = deps._get_or_create_chat_workflows_db(app_two, "user-1", "web")

    deps.shutdown_chat_workflows_deps(app_one)

    assert first.closed is True
    assert second.closed is False

    refreshed = deps._get_or_create_chat_workflows_db(app_one, "user-1", "web")

    assert refreshed is not first


def test_shutdown_chat_workflows_deps_close_failure_log_is_sanitized(monkeypatch):
    raw_marker = "shutdown raw marker"
    raw_path = "/private/chat-workflows-shutdown.db"
    raw_token = "secret-token-shutdown"

    class FailingCloseDB:
        def __init__(self) -> None:
            self.close_attempts = 0

        def close(self) -> None:
            self.close_attempts += 1
            raise RuntimeError(f"{raw_marker} {raw_path} {raw_token}")

    failing_db = FailingCloseDB()

    def fake_create_chat_workflows_database(*, client_id, db_path, backend):
        return failing_db

    monkeypatch.setattr(deps, "create_chat_workflows_database", fake_create_chat_workflows_database, raising=True)
    monkeypatch.setattr(deps, "get_content_backend_instance", lambda: None, raising=True)
    monkeypatch.setattr(
        deps.DatabasePaths,
        "get_chat_workflows_db_path",
        staticmethod(lambda user_id: Path(f"/tmp/{user_id}.db")),
    )

    app = FastAPI()
    deps._get_or_create_chat_workflows_db(app, "shutdown-user", "web")

    with _capture_dependency_error_logs() as messages:
        deps.shutdown_chat_workflows_deps(app)

    rendered = "\n".join(messages)
    assert raw_marker not in rendered
    assert raw_path not in rendered
    assert raw_token not in rendered
    assert "Failed to close ChatWorkflowsDatabase during shutdown" in rendered
    assert "RuntimeError" in rendered
    assert failing_db.close_attempts == 1
    assert not hasattr(app.state, deps._APP_STATE_KEY)
