from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Security.drain_gate_middleware import (
    CORS_EXPOSE_HEADERS,
)


@pytest.fixture()
def test_app(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MCP_DEBUG", "true")
    from tldw_Server_API.app.core.MCP_unified.config import get_config as get_mcp_config
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.services.app_lifecycle import reset_lifecycle_state

    if hasattr(get_mcp_config, "cache_clear"):
        get_mcp_config.cache_clear()
    reset_lifecycle_state(app)
    return app


@pytest.fixture()
def draining_client(test_app, auth_headers):
    from tldw_Server_API.app.services.app_lifecycle import get_or_create_lifecycle_state

    with TestClient(test_app, headers=auth_headers) as client:
        state = get_or_create_lifecycle_state(test_app)
        state.phase = "draining"
        state.ready = False
        state.draining = True
        yield client


def test_drain_gate_allows_health_and_liveness_but_rejects_mutation(test_app, draining_client):
    ok = draining_client.get("/health")
    head_ok = draining_client.head("/health")
    api_health_ok = draining_client.get("/api/v1/health")
    liveness_ok = draining_client.get("/api/v1/healthz")
    blocked = draining_client.post("/api/v1/chat/completions", json={"messages": []})
    if ok.status_code != 200:
        raise AssertionError(f"expected /health to stay open, got {ok.status_code}")
    if head_ok.status_code != 200:
        raise AssertionError(f"expected HEAD /health to stay open, got {head_ok.status_code}")
    if api_health_ok.status_code != 200:
        raise AssertionError(f"expected /api/v1/health to stay open, got {api_health_ok.status_code}")
    if liveness_ok.status_code != 200:
        raise AssertionError(f"expected /api/v1/healthz to stay open, got {liveness_ok.status_code}")
    if blocked.status_code != 503:
        raise AssertionError(f"expected drain gate to return 503, got {blocked.status_code}")
    if blocked.json()["reason"] != "shutdown_in_progress":
        raise AssertionError(f"unexpected drain reason: {blocked.json()['reason']!r}")


def test_drain_gate_rejects_non_control_plane_head_request(test_app, draining_client):
    blocked = draining_client.head("/api/v1/chat/completions")
    if blocked.status_code != 503:
        raise AssertionError(f"expected drain gate to reject HEAD /api/v1/chat/completions, got {blocked.status_code}")


def test_drain_gate_503_preserves_cors_headers_for_browser_clients(draining_client):
    blocked = draining_client.options(
        "/api/v1/chat/completions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": (
                "authorization,content-type,idempotency-key,if-match," "x-slides-accept-content-kinds"
            ),
        },
    )
    if blocked.status_code != 503:
        raise AssertionError(f"expected drain gate to return 503, got {blocked.status_code}")
    if blocked.headers.get("Access-Control-Allow-Origin") != "http://localhost:3000":
        raise AssertionError("drain gate should preserve CORS allow-origin on blocked responses")
    if blocked.headers.get("Access-Control-Allow-Methods") != "POST":
        raise AssertionError("drain gate should echo requested method for blocked preflight responses")
    allowed = {item.strip().lower() for item in blocked.headers.get("Access-Control-Allow-Headers", "").split(",")}
    if not {"idempotency-key", "if-match", "x-slides-accept-content-kinds"} <= allowed:
        raise AssertionError("drain preflight should allow standalone mutation and negotiation headers")
    exposed = {item.strip().lower() for item in blocked.headers.get("Access-Control-Expose-Headers", "").split(",")}
    required = {item.lower() for item in CORS_EXPOSE_HEADERS}
    if exposed != required:
        raise AssertionError(f"unexpected drain exposed-header policy: {sorted(exposed)!r}")
    if "origin" not in blocked.headers.get("Vary", "").lower():
        raise AssertionError("drain CORS response should vary by Origin")
    cors_config = draining_client.app.state._tldw_drain_gate_cors_config
    if cors_config["allow_credentials"]:
        if blocked.headers.get("Access-Control-Allow-Credentials") != "true":
            raise AssertionError("drain CORS response should preserve configured credential policy")
    elif "Access-Control-Allow-Credentials" in blocked.headers:
        raise AssertionError("drain CORS response should not widen the configured credential policy")


def test_drain_gate_actual_error_response_exposes_retry_and_download_headers(draining_client):
    blocked = draining_client.post(
        "/api/v1/slides/generations",
        headers={"Origin": "http://localhost:3000"},
        content=b"{}",
    )

    assert blocked.status_code == 503
    assert blocked.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "retry-after" in blocked.headers["access-control-expose-headers"].lower()
    assert "content-disposition" in blocked.headers["access-control-expose-headers"].lower()


def test_assert_may_start_work_raises_when_draining(test_app):
    from tldw_Server_API.app.services.app_lifecycle import (
        assert_may_start_work,
        get_or_create_lifecycle_state,
    )

    state = get_or_create_lifecycle_state(test_app)
    state.draining = True
    with pytest.raises(HTTPException) as excinfo:
        assert_may_start_work(test_app, kind="job_enqueue")
    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == {"message": "Shutdown in progress", "kind": "job_enqueue"}


def test_assert_may_start_work_noops_when_not_draining(test_app):
    from tldw_Server_API.app.services.app_lifecycle import assert_may_start_work

    assert_may_start_work(test_app, kind="job_enqueue")


def test_drain_gate_rejects_guarded_request_before_llm_budget_runs(test_app, draining_client, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.llm_budget_middleware import LLMBudgetMiddleware

    budget_called = {"value": False}

    async def _unexpected_budget_dispatch(self, request, call_next):
        budget_called["value"] = True
        raise AssertionError("LLMBudgetMiddleware should not run while draining")

    monkeypatch.setattr(LLMBudgetMiddleware, "dispatch", _unexpected_budget_dispatch)

    blocked = draining_client.post("/api/v1/chat/completions", json={"messages": []})
    if blocked.status_code != 503:
        raise AssertionError(f"expected drain gate to return 503, got {blocked.status_code}")
    if blocked.json()["reason"] != "shutdown_in_progress":
        raise AssertionError(f"unexpected drain reason: {blocked.json()['reason']!r}")
    if budget_called["value"]:
        raise AssertionError("LLMBudgetMiddleware was invoked for a drained request")


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/health"),
        ("HEAD", "/health"),
        ("GET", "/internal/ready"),
        ("HEAD", "/internal/ready"),
        ("GET", "/ready"),
        ("HEAD", "/ready"),
        ("GET", "/readyz"),
        ("HEAD", "/readyz"),
        ("HEAD", "/health/ready"),
        ("GET", "/healthz"),
        ("HEAD", "/healthz"),
        ("GET", "/api/v1/health"),
        ("HEAD", "/api/v1/health"),
        ("GET", "/api/v1/health/live"),
        ("HEAD", "/api/v1/health/live"),
        ("GET", "/api/v1/health/ready"),
        ("HEAD", "/api/v1/health/ready"),
        ("GET", "/api/v1/healthz"),
        ("HEAD", "/api/v1/healthz"),
        ("GET", "/api/v1/readyz"),
        ("HEAD", "/api/v1/readyz"),
    ],
)
def test_control_plane_probe_paths_are_allowlisted(method, path):
    from tldw_Server_API.app.core.Security.drain_gate_middleware import _is_allowlisted_control_plane_path

    request = SimpleNamespace(method=method, url=SimpleNamespace(path=path))
    if not _is_allowlisted_control_plane_path(request):
        raise AssertionError(f"Expected {method} {path} to be allowlisted during drain")
