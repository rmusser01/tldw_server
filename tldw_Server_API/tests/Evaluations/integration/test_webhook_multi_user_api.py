import json
import sqlite3
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI, Header, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.evaluations import (
    evaluations_webhooks as webhooks,
    evaluations_auth as eval_auth,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as service_module
from tldw_Server_API.app.core.Evaluations.webhook_manager import WebhookManager

pytestmark = [pytest.mark.integration]


def _seed_webhook(db_path: Path, *, user_id: str, url: str) -> None:
    EvaluationsDatabase(str(db_path))
    stored_user_id = user_id if str(user_id).startswith("user_") else f"user_{user_id}"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO webhook_registrations (user_id, url, secret, events, active, retry_count, timeout_seconds)
            VALUES (?, ?, ?, ?, 1, 3, 30)
            """,
            (stored_user_id, url, "s" * 32, json.dumps(["evaluation.completed"])),
        )
        conn.commit()


@pytest.fixture()
def multi_user_webhook_client(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "false")
    monkeypatch.delenv("EVALUATIONS_TEST_DB_PATH", raising=False)
    base_dir = tmp_path / "user_dbs"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    # Clear cached services to ensure per-user DBs are created under the temp base dir.
    service_module._service_instance = None
    try:
        service_module._service_instances_by_user.clear()
    except Exception:
        service_module._service_instances_by_user = {}  # type: ignore[assignment]

    app = FastAPI()
    app.include_router(webhooks.webhooks_router, prefix="/api/v1/evaluations")

    async def _verify_api_key(
        request: Request,
        x_api_key: str = Header(None, alias="X-API-KEY"),
    ) -> str:
        return request.headers.get("X-User-Id", "1")

    async def _get_eval_request_user(
        request: Request,
        _user_ctx: str = Depends(_verify_api_key),
    ) -> User:
        user_id = request.headers.get("X-User-Id", "1")
        return User(
            id=int(user_id),
            username=f"user_{user_id}",
            roles=["admin"],
            permissions=["evals.read", "evals.manage"],
            is_admin=True,
        )

    app.dependency_overrides[eval_auth.verify_api_key] = _verify_api_key
    app.dependency_overrides[eval_auth.get_eval_request_user] = _get_eval_request_user

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
    service_module._service_instance = None
    try:
        service_module._service_instances_by_user.clear()
    except Exception:
        service_module._service_instances_by_user = {}  # type: ignore[assignment]


def test_webhook_list_is_per_user(multi_user_webhook_client):
    client = multi_user_webhook_client

    db_path_user_1 = DatabasePaths.get_evaluations_db_path(1)
    db_path_user_2 = DatabasePaths.get_evaluations_db_path(2)
    _seed_webhook(db_path_user_1, user_id="1", url="https://example.com/u1")
    _seed_webhook(db_path_user_2, user_id="2", url="https://example.com/u2")

    resp_1 = client.get(
        "/api/v1/evaluations/webhooks",
        headers={"X-User-Id": "1", "X-API-KEY": "test"},
    )
    assert resp_1.status_code == 200
    payload_1 = resp_1.json()
    assert len(payload_1) == 1
    assert payload_1[0]["url"] == "https://example.com/u1"

    resp_2 = client.get(
        "/api/v1/evaluations/webhooks",
        headers={"X-User-Id": "2", "X-API-KEY": "test"},
    )
    assert resp_2.status_code == 200
    payload_2 = resp_2.json()
    assert len(payload_2) == 1
    assert payload_2[0]["url"] == "https://example.com/u2"


def test_webhook_test_uses_per_user_manager(multi_user_webhook_client, monkeypatch):
    client = multi_user_webhook_client

    db_path_user_1 = DatabasePaths.get_evaluations_db_path(1)
    db_path_user_2 = DatabasePaths.get_evaluations_db_path(2)
    _seed_webhook(db_path_user_1, user_id="1", url="https://example.com/u1")
    _seed_webhook(db_path_user_2, user_id="2", url="https://example.com/u2")

    called_paths = []

    async def _fake_test_webhook(self, user_id: str, url: str):
        cfg = getattr(self.db_adapter, "config", None)
        called_paths.append(getattr(cfg, "connection_string", None))
        return {"success": True}

    monkeypatch.setattr(WebhookManager, "test_webhook", _fake_test_webhook)

    resp_1 = client.post(
        "/api/v1/evaluations/webhooks/test",
        json={"url": "https://example.com/u1"},
        headers={"X-User-Id": "1", "X-API-KEY": "test"},
    )
    assert resp_1.status_code == 200

    resp_2 = client.post(
        "/api/v1/evaluations/webhooks/test",
        json={"url": "https://example.com/u2"},
        headers={"X-User-Id": "2", "X-API-KEY": "test"},
    )
    assert resp_2.status_code == 200

    assert called_paths == [str(db_path_user_1), str(db_path_user_2)]


def test_webhook_delete_is_url_based_and_scoped(multi_user_webhook_client):
    client = multi_user_webhook_client

    db_path_user_1 = DatabasePaths.get_evaluations_db_path(1)
    db_path_user_2 = DatabasePaths.get_evaluations_db_path(2)
    _seed_webhook(db_path_user_1, user_id="1", url="https://example.com/u1")
    _seed_webhook(db_path_user_2, user_id="2", url="https://example.com/u2")

    resp = client.delete(
        "/api/v1/evaluations/webhooks",
        params={"url": "https://example.com/u1"},
        headers={"X-User-Id": "1", "X-API-KEY": "test"},
    )
    assert resp.status_code == 200

    with sqlite3.connect(db_path_user_1) as conn:
        active = conn.execute(
            "SELECT active FROM webhook_registrations WHERE user_id = ? AND url = ?",
            ("user_1", "https://example.com/u1"),
        ).fetchone()[0]
        assert active == 0

    with sqlite3.connect(db_path_user_2) as conn:
        active = conn.execute(
            "SELECT active FROM webhook_registrations WHERE user_id = ? AND url = ?",
            ("user_2", "https://example.com/u2"),
        ).fetchone()[0]
        assert active == 1


def test_webhook_list_uses_canonical_string_scope_for_manager_binding(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MODE", "false")
    monkeypatch.delenv("EVALUATIONS_TEST_DB_PATH", raising=False)
    base_dir = tmp_path / "user_dbs"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    service_module._service_instance = None
    try:
        service_module._service_instances_by_user.clear()
    except Exception:
        service_module._service_instances_by_user = {}  # type: ignore[assignment]

    app = FastAPI()
    app.include_router(webhooks.webhooks_router, prefix="/api/v1/evaluations")

    async def _verify_api_key() -> str:
        return "tenant-user"

    class _EvalUser:
        id = 1
        id_str = "tenant-user"

    async def _get_eval_request_user(
        _user_ctx: str = Depends(_verify_api_key),
    ) -> _EvalUser:
        return _EvalUser()

    app.dependency_overrides[eval_auth.verify_api_key] = _verify_api_key
    app.dependency_overrides[eval_auth.get_eval_request_user] = _get_eval_request_user

    db_path_tenant = DatabasePaths.get_evaluations_db_path("tenant-user")
    db_path_numeric = DatabasePaths.get_evaluations_db_path(1)
    _seed_webhook(db_path_tenant, user_id="tenant-user", url="https://example.com/tenant")
    _seed_webhook(db_path_numeric, user_id="1", url="https://example.com/numeric")

    with TestClient(app) as client:
        resp = client.get("/api/v1/evaluations/webhooks", headers={"X-API-KEY": "test"})
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert len(payload) == 1
        assert payload[0]["url"] == "https://example.com/tenant"

    app.dependency_overrides.clear()
    service_module._service_instance = None
    try:
        service_module._service_instances_by_user.clear()
    except Exception:
        service_module._service_instances_by_user = {}  # type: ignore[assignment]
