"""Canonical recipient clone admission and status API tests."""

from __future__ import annotations

import inspect
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import sharing
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
    SharedWorkspaceAccessContext,
    SharedWorkspaceNotFound,
)
from tldw_Server_API.app.core.Sharing.shared_workspace_clone_operations import (
    build_clone_admission_command,
)

pytestmark = pytest.mark.integration

IDEMPOTENCY_KEY = "clone-request-0001"


def test_workspace_clone_queue_is_builtin_without_environment_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("JOBS_ALLOWED_QUEUES", raising=False)
    monkeypatch.delenv("JOBS_ALLOWED_QUEUES_SHARING", raising=False)
    manager = JobManager(tmp_path / "clone-default-queue.db")

    admission = manager.admit_idempotent_operation(
        build_clone_admission_command(
            share_id=42,
            recipient_user_id=9,
            requested_name=None,
            idempotency_key=IDEMPOTENCY_KEY,
        )
    )

    assert admission.job["domain"] == "sharing"
    assert admission.job["queue"] == "workspace-clone"


def _context(*, allow_clone: bool = True) -> SharedWorkspaceAccessContext:
    return SharedWorkspaceAccessContext(
        share_id=42,
        workspace_id="workspace-alpha",
        owner_user_id=7,
        recipient_user_id=9,
        share_scope_type="team",
        share_scope_id=11,
        access_level="view_chat",
        allow_clone=allow_clone,
        owner_display_name="Research owner",
        shared_at="2026-08-20T18:00:00+00:00",
        workspace={
            "id": "workspace-alpha",
            "name": "Evidence review",
            "description": "Review set",
        },
        policy_actions={
            "inspect_sources": {"allowed": True, "reason_code": None},
            "ask_grounded_questions": {"allowed": True, "reason_code": None},
            "add_sources": {"allowed": False, "reason_code": "shared_write_not_available"},
            "edit_workspace": {"allowed": False, "reason_code": "shared_write_not_available"},
            "clone_workspace": {
                "allowed": allow_clone,
                "reason_code": None if allow_clone else "owner_disabled",
            },
        },
    )


class _AccessService:
    def __init__(self, context: SharedWorkspaceAccessContext | None = None) -> None:
        self.context = context or _context()
        self.error: Exception | None = None
        self.calls: list[tuple[int, int]] = []

    async def resolve(self, *, share_id: int, recipient_user_id: int):
        self.calls.append((share_id, recipient_user_id))
        if self.error is not None:
            raise self.error
        return self.context


@pytest.fixture
def clone_api(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    monkeypatch.setenv("JOBS_METRICS_GAUGES_ENABLED", "false")
    manager = JobManager(tmp_path / "clone-api-jobs.db")
    service = _AccessService()
    audit_events: list[tuple[str, dict[str, Any]]] = []
    user = User(
        id=9,
        username="recipient",
        email="recipient@example.test",
        password_hash="hash",
    )
    principal = AuthPrincipal(
        kind="user",
        user_id=9,
        username="recipient",
        permissions=["sharing.read"],
    )

    async def _principal():
        return principal

    async def _user():
        return user

    async def _rate_limit(*_args, **_kwargs):
        return None

    async def _audit(_audit, event_type: str, **kwargs: Any):
        audit_events.append((event_type, kwargs))

    monkeypatch.setattr(auth_deps, "enforce_rbac_rate_limit", _rate_limit)
    monkeypatch.setattr(sharing, "_audit_log_best_effort", _audit)
    monkeypatch.setattr(sharing, "_get_audit_service", lambda: object())

    app = FastAPI()
    app.include_router(sharing.router, prefix="/api/v1")
    app.dependency_overrides[auth_deps.get_auth_principal] = _principal
    app.dependency_overrides[sharing.get_request_user] = _user
    app.dependency_overrides[sharing.get_shared_workspace_access_service] = lambda: service
    app.dependency_overrides[sharing.try_get_job_manager] = lambda: manager
    client = TestClient(app, raise_server_exceptions=False)
    return client, manager, service, audit_events, app


def _post(
    client: TestClient,
    *,
    key: str = IDEMPOTENCY_KEY,
    name: str | None = "Recipient copy",
):
    body = {} if name is None else {"name": name}
    return client.post(
        "/api/v1/sharing/shared-with-me/42/clone",
        headers={"Idempotency-Key": key},
        json=body,
    )


def _mark_completed(manager: JobManager, operation: dict[str, Any]) -> None:
    counts = {
        f"{kind}_{field}": 0
        for kind in ("sources", "notes", "artifacts", "media")
        for field in ("attempted", "copied", "failed")
    }
    counts["operation_owned_media_count"] = 0
    result = {
        "schema_version": 1,
        "outcome": "complete",
        "workspace_id": operation["workspace_id"],
        "name": "Recipient copy",
        "publication_confirmed": True,
        "counts": counts,
        "readiness": {
            "text_search": "ready",
            "citations": "ready",
            "vector_search": "needs_indexing",
        },
        "warnings": [],
    }
    with sqlite3.connect(manager.db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = 'completed', result = ?, completed_at = DATETIME('now'),
                updated_at = DATETIME('now')
            WHERE uuid = ?
            """,
            (json.dumps(result), operation["operation_id"]),
        )


@pytest.mark.parametrize("key", ["short", "has spaces 000000", "A" * 201])
def test_clone_post_requires_a_valid_exact_idempotency_key(clone_api, key: str) -> None:
    client, manager, service, _audit, _app = clone_api

    response = _post(client, key=key)

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_shared_workspace_request"
    assert manager.list_jobs(domain="sharing") == []
    assert service.calls == []


def test_clone_post_requires_idempotency_header(clone_api) -> None:
    client, manager, service, _audit, _app = clone_api

    response = client.post(
        "/api/v1/sharing/shared-with-me/42/clone",
        json={"name": "Recipient copy"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_shared_workspace_request"
    assert manager.list_jobs(domain="sharing") == []
    assert service.calls == []


def test_clone_post_returns_typed_unavailable_when_jobs_are_missing(clone_api) -> None:
    client, _manager, service, _audit, app = clone_api
    app.dependency_overrides[sharing.try_get_job_manager] = lambda: None

    response = _post(client)

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "clone_operation_unavailable"
    assert service.calls == []


def test_clone_post_requires_current_clone_permission(clone_api) -> None:
    client, manager, service, _audit, _app = clone_api
    service.context = _context(allow_clone=False)

    response = _post(client)

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "clone_not_allowed"
    assert manager.list_jobs(domain="sharing") == []


def test_clone_post_admits_one_durable_operation_and_audits_requested(clone_api) -> None:
    client, manager, service, audit_events, _app = clone_api

    response = _post(client, name="  Recipient   copy  ")

    assert response.status_code == 202
    operation = response.json()
    assert operation["status"] == "queued"
    assert operation["command"] == "shared_workspace_clone"
    assert operation["share_id"] == 42
    assert operation["poll_href"].endswith(f"/clone/{operation['operation_id']}")
    jobs = manager.list_jobs(domain="sharing", owner_user_id="9")
    assert len(jobs) == 1
    assert jobs[0]["idempotency_key"] is None
    assert IDEMPOTENCY_KEY not in json.dumps(jobs[0])
    assert service.calls == [(42, 9)]
    assert [event for event, _metadata in audit_events] == ["share.clone_requested"]
    assert audit_events[0][1]["metadata"] == {
        "operation_id": operation["operation_id"],
    }


def test_same_key_replays_after_revocation_without_resolving_share(clone_api) -> None:
    client, _manager, service, audit_events, _app = clone_api
    first = _post(client)
    service.error = SharedWorkspaceNotFound()

    replay = _post(client)

    assert replay.status_code == 202
    assert replay.json() == first.json()
    assert service.calls == [(42, 9)]
    assert len(audit_events) == 1


def test_same_key_with_different_request_returns_conflict_before_access(clone_api) -> None:
    client, _manager, service, _audit, _app = clone_api
    first = _post(client, name="First name")
    service.calls.clear()

    conflict = _post(client, name="Different name")

    assert first.status_code == 202
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "idempotency_key_reused"
    assert conflict.json()["detail"]["operation_id"] == first.json()["operation_id"]
    assert service.calls == []


def test_second_key_converges_same_request_or_rejects_different_request(clone_api) -> None:
    client, manager, _service, audit_events, _app = clone_api
    first = _post(client, key="clone-request-0001", name="Same name")
    converged = _post(client, key="clone-request-0002", name="Same name")
    conflict = _post(client, key="clone-request-0003", name="Different name")

    assert converged.status_code == 202
    assert converged.json()["operation_id"] == first.json()["operation_id"]
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "clone_already_in_progress"
    assert conflict.json()["detail"]["operation_id"] == first.json()["operation_id"]
    assert len(manager.list_jobs(domain="sharing")) == 1
    assert len(audit_events) == 1


def test_terminal_post_replay_returns_200_and_matches_get(clone_api) -> None:
    client, manager, _service, _audit, _app = clone_api
    queued = _post(client)
    _mark_completed(manager, queued.json())

    replay = _post(client)
    status_response = client.get(queued.json()["poll_href"])

    assert replay.status_code == 200
    assert status_response.status_code == 200
    assert replay.json() == status_response.json()
    assert replay.json()["status"] == "succeeded"
    assert replay.json()["result"]["readiness"]["vector_search"] == "needs_indexing"


def test_clone_get_is_owner_scoped_and_share_scoped(clone_api) -> None:
    client, manager, service, _audit, _app = clone_api
    foreign = manager.admit_idempotent_operation(
        build_clone_admission_command(
            share_id=42,
            recipient_user_id=10,
            requested_name=None,
            idempotency_key="foreign-request-001",
            now=datetime.now(timezone.utc),
        )
    ).job

    foreign_response = client.get(
        f"/api/v1/sharing/shared-with-me/42/clone/{foreign['uuid']}"
    )
    malformed_response = client.get(
        "/api/v1/sharing/shared-with-me/42/clone/not-a-uuid"
    )
    missing_response = client.get(
        f"/api/v1/sharing/shared-with-me/42/clone/{uuid4()}"
    )

    assert foreign_response.status_code == 404
    assert malformed_response.status_code == 404
    assert missing_response.status_code == 404
    assert service.calls == []


def test_clone_get_fails_closed_for_wrong_scope_and_malformed_terminal_result(
    clone_api,
) -> None:
    client, manager, service, _audit, _app = clone_api
    operation = _post(client).json()
    service.calls.clear()
    with sqlite3.connect(manager.db_path) as conn:
        conn.execute(
            "UPDATE jobs SET batch_group = 'share:99' WHERE uuid = ?",
            (operation["operation_id"],),
        )

    wrong_scope = client.get(operation["poll_href"])

    with sqlite3.connect(manager.db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET batch_group = 'share:42', status = 'completed', result = '{}'
            WHERE uuid = ?
            """,
            (operation["operation_id"],),
        )
    malformed = client.get(operation["poll_href"])

    assert wrong_scope.status_code == 404
    assert malformed.status_code == 503
    assert malformed.json()["detail"]["code"] == "clone_operation_unavailable"
    assert service.calls == []


def test_clone_routes_use_canonical_models_and_recipient_route_contract(clone_api) -> None:
    client, _manager, _service, _audit, _app = clone_api
    schema = client.get("/openapi.json").json()
    post = schema["paths"]["/api/v1/sharing/shared-with-me/{share_id}/clone"]["post"]
    get = schema["paths"][
        "/api/v1/sharing/shared-with-me/{share_id}/clone/{operation_id}"
    ]["get"]

    assert post["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/SharedWorkspaceCloneRequest"
    )
    for operation in (post, get):
        for success in ("200", "202"):
            if success in operation["responses"]:
                assert operation["responses"][success]["content"]["application/json"][
                    "schema"
                ]["$ref"].endswith("/SharedWorkspaceCloneOperationResponse")
        assert "HTTPValidationError" not in json.dumps(operation)
    assert "CloneWorkspaceRequest" not in schema["components"]["schemas"]
    assert "CloneWorkspaceResponse" not in schema["components"]["schemas"]

    routes = {
        route.path: route
        for route in sharing.router.routes
        if hasattr(route, "dependant")
    }
    assert isinstance(
        routes["/sharing/shared-with-me/{share_id}/clone"],
        sharing.SharedWorkspaceRecipientRoute,
    )
    assert isinstance(
        routes["/sharing/shared-with-me/{share_id}/clone/{operation_id}"],
        sharing.SharedWorkspaceRecipientRoute,
    )
    source = inspect.getsource(sharing)
    assert "BackgroundTasks" not in source
    assert "_run_clone_task" not in source
