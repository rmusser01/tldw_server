from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from starlette.requests import Request

from tldw_Server_API.app.services.readiness_service import is_loopback_peer


def _request(peer: str, headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/internal/ready",
            "headers": headers or [],
            "client": (peer, 43210),
            "server": ("app", 8000),
            "scheme": "http",
            "query_string": b"",
        }
    )


def test_remote_peer_cannot_spoof_loopback_with_forwarding_headers() -> None:
    request = _request(
        "172.30.0.2",
        [(b"x-forwarded-for", b"127.0.0.1"), (b"x-real-ip", b"127.0.0.1")],
    )
    assert is_loopback_peer(request) is False


def test_loopback_peer_is_allowed_for_internal_probe() -> None:
    assert is_loopback_peer(_request("127.0.0.1")) is True
    assert is_loopback_peer(_request("::1")) is True


@pytest.mark.asyncio
async def test_public_health_is_exact_minimal_liveness() -> None:
    from tldw_Server_API.app.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        head_response = await client.head("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["cache-control"] == "no-store"
    assert head_response.status_code == 200
    assert head_response.headers["content-type"].startswith("application/json")
    assert head_response.headers["cache-control"] == "no-store"
    assert head_response.content == b""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ready", "reason", "details", "expected_status", "expected_body"),
    (
        (True, None, {"database": {"type": "postgresql"}}, 200, {"status": "ready"}),
        (False, "database_unavailable", {"database": {"type": "postgresql"}}, 503, {"status": "not_ready"}),
    ),
)
async def test_internal_ready_is_loopback_only_and_detail_free(
    monkeypatch: pytest.MonkeyPatch,
    ready: bool,
    reason: str | None,
    details: dict[str, object],
    expected_status: int,
    expected_body: dict[str, str],
) -> None:
    from tldw_Server_API.app import main
    from tldw_Server_API.app.services import readiness_service

    test_app = FastAPI()
    test_app.add_api_route("/internal/ready", main.internal_readiness_check, methods=["GET", "HEAD"])

    async def _readiness(_: FastAPI) -> readiness_service.ReadinessSnapshot:
        return readiness_service.ReadinessSnapshot(ready, reason, details)

    monkeypatch.setattr(readiness_service, "collect_readiness_snapshot", _readiness)
    loopback = httpx.ASGITransport(app=test_app, client=("127.0.0.1", 43100))
    remote = httpx.ASGITransport(app=test_app, client=("172.30.0.2", 43100))
    async with httpx.AsyncClient(transport=loopback, base_url="http://test") as client:
        response = await client.get("/internal/ready")
        head_response = await client.head("/internal/ready")
    async with httpx.AsyncClient(transport=remote, base_url="http://test") as client:
        denied = await client.get("/internal/ready")

    assert response.status_code == expected_status
    assert response.json() == expected_body
    assert response.headers["cache-control"] == "no-store"
    assert head_response.status_code == expected_status
    assert head_response.headers["cache-control"] == "no-store"
    assert head_response.content == b""
    assert denied.status_code == 404
    assert denied.json() == {"detail": "Not Found"}
