from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import health
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import readiness_service
from tldw_Server_API.app.services.readiness_service import (
    ReadinessSnapshot,
    internal_readiness_payload,
    operator_readiness_payload,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("ready", [True, False])
async def test_compatibility_readiness_retains_typed_client_fields(
    monkeypatch: pytest.MonkeyPatch, ready: bool,
) -> None:
    """Legacy typed clients receive a readiness boolean and sanitized diagnostics."""
    import json
    from datetime import datetime

    from starlette.requests import Request

    snapshot = ReadinessSnapshot(
        ready, None if ready else "database_unavailable",
        {"database": {"status": "healthy" if ready else "unhealthy", "type": "sqlite"},
         "engine": {"queue_depth": 0}},
    )
    monkeypatch.setattr(readiness_service, "collect_readiness_snapshot", AsyncMock(return_value=snapshot))
    response = await health.api_readiness(Request({"type": "http", "app": FastAPI()}))
    body = json.loads(response.body)

    assert body["ready"] is ready
    assert body["db"] == {"ok": ready, "backend": "sqlite"}
    assert body["engine"] == {"queue_depth": 0}
    assert datetime.fromisoformat(body["time"]).tzinfo is not None
    assert body["time"].endswith("+00:00")
    assert response.status_code == (200 if ready else 503)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot", "field"),
    [
        (ReadinessSnapshot("false", None, {}), "ready"),
        (ReadinessSnapshot(True, None, {"engine": []}), "engine"),
        (ReadinessSnapshot(True, None, {"database": {"type": 42}}), "db"),
        (ReadinessSnapshot(True, None, {"workflows_db": []}), "workflows_db"),
        (ReadinessSnapshot(True, None, {"providers_initialized": "false"}), "providers_initialized"),
    ],
)
async def test_compatibility_readiness_rejects_malformed_collector_fields(
    monkeypatch: pytest.MonkeyPatch, snapshot: ReadinessSnapshot, field: str,
) -> None:
    """Validate the assembled body even though JSONResponse bypasses FastAPI validation."""
    monkeypatch.setattr(readiness_service, "collect_readiness_snapshot", AsyncMock(return_value=snapshot))
    with pytest.raises(ValidationError) as exc_info:
        await health.api_readiness(Request({"type": "http", "app": FastAPI()}))
    assert any(error["loc"][0] == field for error in exc_info.value.errors())


@pytest.mark.asyncio
@pytest.mark.parametrize("ready", [True, False])
async def test_compatibility_readiness_preserves_operator_details_and_dynamic_metrics(
    monkeypatch: pytest.MonkeyPatch, ready: bool,
) -> None:
    """Preserve sanitized optional sections, nullable metrics and forward-compatible detail."""
    details = {
        "database": {"status": "healthy" if ready else "unhealthy", "type": "postgresql", "pool_size": 3},
        "engine": {"queue_depth": None, "active_workflows": 2, "future_metric": {"values": [1, 2]}},
        "workflows_db": {"schema_version": None, "expected_version": None},
        "providers_initialized": True,
        "provider_health": {"local": {"ready": True, "latency": 0.25}},
        "otel_available": False,
        "rg_policy": {"version": 2, "store": None, "policies": None},
        "future_diagnostic": {"available": True},
    }
    snapshot = ReadinessSnapshot(ready, None if ready else "database_unavailable", details)
    monkeypatch.setattr(readiness_service, "collect_readiness_snapshot", AsyncMock(return_value=snapshot))
    response = await health.api_readiness(Request({"type": "http", "app": FastAPI()}))
    body = json.loads(response.body)
    assert {key: body[key] for key in details} == details
    assert body["ready"] is ready
    assert response.status_code == (200 if ready else 503)
    assert response.headers["Cache-Control"] == "no-store"
    if ready:
        assert "reason" not in body
    else:
        assert body["reason"] == "database_unavailable"


@pytest.mark.asyncio
async def test_compatibility_readiness_does_not_add_absent_operator_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A draining snapshot keeps its existing sparse response shape."""
    snapshot = ReadinessSnapshot(False, "shutdown_in_progress", {})
    monkeypatch.setattr(readiness_service, "collect_readiness_snapshot", AsyncMock(return_value=snapshot))
    response = await health.api_readiness(Request({"type": "http", "app": FastAPI()}))
    body = json.loads(response.body)
    assert set(body) == {"status", "reason", "ready", "engine", "db", "time"}
    assert body["db"] == {"ok": False, "backend": None}
    assert response.status_code == 503
    assert response.headers["Cache-Control"] == "no-store"


def test_compatibility_readiness_declares_success_and_unavailable_response_schema() -> None:
    """Typed clients receive the validated contract for both readiness outcomes."""
    app = FastAPI()
    app.include_router(health.router, prefix="/api/v1")
    schema: dict[str, Any] = app.openapi()
    responses = schema["paths"]["/api/v1/health/ready"]["get"]["responses"]
    for code in ("200", "503"):
        assert responses[code]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/ReadinessResponse",
        }


def test_internal_projection_discards_all_detail() -> None:
    snapshot = ReadinessSnapshot(
        ready=False,
        reason="database_unavailable",
        details={"database": {"type": "postgresql"}, "providers_initialized": False},
    )
    assert internal_readiness_payload(snapshot) == {"status": "not_ready"}


def test_operator_projection_keeps_only_sanitized_snapshot_detail() -> None:
    snapshot = ReadinessSnapshot(
        ready=True,
        reason=None,
        details={"database": {"status": "healthy", "type": "postgresql"}},
    )
    assert operator_readiness_payload(snapshot) == {
        "status": "ready",
        "database": {"status": "healthy", "type": "postgresql"},
    }


@pytest.mark.asyncio
async def test_resource_governor_policy_file_read_runs_off_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load fallback policy metadata away from the request event-loop thread."""

    policy_path = tmp_path / "policy.yml"
    policy_path.write_text("version: 3\npolicies:\n  default: {}\n", encoding="utf-8")
    monkeypatch.setenv("RG_POLICY_PATH", str(policy_path))
    app = FastAPI()
    event_loop_thread = threading.get_ident()
    observed_thread: list[int] = []
    original_loader = readiness_service._load_resource_governor_policy

    def recording_loader(path: Path) -> dict[str, object] | None:
        """Record the worker thread used by the synchronous file loader."""

        observed_thread.append(threading.get_ident())
        return original_loader(path)

    monkeypatch.setattr(readiness_service, "_load_resource_governor_policy", recording_loader)

    policy = await readiness_service._resource_governor_policy(app)

    assert policy == {"version": 3, "store": "file", "policies": 1}
    assert observed_thread and observed_thread[0] != event_loop_thread


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    ("/ready", "/health/ready", "/internal/ready", "/api/v1/readyz", "/api/v1/health/ready"),
)
async def test_each_readiness_route_uses_the_shared_snapshot_once(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    from tldw_Server_API.app import main

    app = FastAPI()
    app.add_api_route("/ready", main.readiness_check, methods=["GET"])
    app.add_api_route("/health/ready", main.readiness_alias, methods=["GET"])
    app.add_api_route("/internal/ready", main.internal_readiness_check, methods=["GET"])
    app.include_router(health.router, prefix="/api/v1")
    app.dependency_overrides[auth_deps.get_auth_principal] = lambda: AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="test",
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )
    snapshot = ReadinessSnapshot(True, None, {"database": {"status": "healthy", "type": "sqlite"}})
    collect = AsyncMock(return_value=snapshot)
    monkeypatch.setattr(readiness_service, "collect_readiness_snapshot", collect)

    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 43100))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path)

    assert response.status_code == 200
    assert collect.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot", "expected_status", "expected_body"),
    (
        (
            ReadinessSnapshot(True, None, {"database": {"status": "healthy", "type": "sqlite"}}),
            200,
            {"status": "ready"},
        ),
        (
            ReadinessSnapshot(False, "database_unavailable", {"database": {"type": "postgresql"}}),
            503,
            {"status": "not_ready"},
        ),
    ),
)
async def test_internal_readiness_projects_each_shared_snapshot_once(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: ReadinessSnapshot,
    expected_status: int,
    expected_body: dict[str, str],
) -> None:
    from tldw_Server_API.app import main

    app = FastAPI()
    app.add_api_route("/internal/ready", main.internal_readiness_check, methods=["GET"])
    collect = AsyncMock(return_value=snapshot)
    monkeypatch.setattr(readiness_service, "collect_readiness_snapshot", collect)

    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 43100))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/internal/ready")

    assert response.status_code == expected_status
    assert response.json() == expected_body
    collect.assert_awaited_once_with(app)


@pytest.mark.asyncio
async def test_public_liveness_never_collects_readiness_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app import main

    app = FastAPI()
    app.add_api_route("/health", main.health_check, methods=["GET"])
    collect = AsyncMock()
    monkeypatch.setattr(readiness_service, "collect_readiness_snapshot", collect)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert collect.await_count == 0
