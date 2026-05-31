"""
Integration tests for RAG health endpoints.
No mocks; assert JSON shape and expected status codes.
"""

import sys
import types

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints import rag_health as rag_health_ep
from tldw_Server_API.app.main import app
from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


pytestmark = pytest.mark.integration


class _LoggerStub:
    def __init__(self):
        self.errors = []

    def error(self, message, *args, **kwargs):
        self.errors.append(str(message))


def _install_logger_stub(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(rag_health_ep, "logger", logger_stub)
    return logger_stub


def _assert_sanitized_error_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.errors == [expected_message]
    assert "/private/" not in logger_stub.errors[0]
    assert "exploded" not in logger_stub.errors[0]


@pytest.fixture()
def client():
    async def _fake_get_auth_principal(_request: Request) -> AuthPrincipal:  # type: ignore[override]
        # Diagnostics-style principal with system.logs permission and admin flag for RAG health
        return AuthPrincipal(
            kind="service",
            user_id=None,
            api_key_id=None,
            subject="service:rag-health-test",
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=["system.logs"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(auth_deps.get_auth_principal, None)


def test_rag_liveness_and_readiness(client: TestClient):
    live = client.get("/api/v1/rag/health/live")
    assert live.status_code == 200
    assert live.json().get("status") == "alive"

    ready = client.get("/api/v1/rag/health/ready")
    assert ready.status_code in (200, 503)
    if ready.status_code == 200:
        assert ready.json().get("status") == "ready"


def test_rag_readiness_sanitizes_unexpected_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    def _raise_cache_error():
        raise RuntimeError("readiness backend unavailable")

    monkeypatch.setattr(rag_health_ep, "get_rag_cache", _raise_cache_error)

    response = client.get("/api/v1/rag/health/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Service not ready"}


def test_rag_full_health_and_cache_stats(client: TestClient):
    health = client.get("/api/v1/rag/health")
    assert health.status_code == 200
    h = health.json()
    assert isinstance(h, dict)
    assert "status" in h and "components" in h
    assert isinstance(h["components"], dict)

    cache = client.get("/api/v1/rag/cache/stats")
    assert cache.status_code in (200, 500)
    if cache.status_code == 200:
        stats = cache.json()
        assert isinstance(stats, dict)


def test_rag_health_components_when_present(client: TestClient):
    """If components are present in health report, validate basic shape and allowed statuses."""
    resp = client.get("/api/v1/rag/health")
    assert resp.status_code == 200
    h = resp.json()
    comps = h.get("components", {})
    assert isinstance(comps, dict)

    # Check known components if present
    for key in list(comps.keys()):
        comp = comps[key]
        assert isinstance(comp, dict)
        assert comp.get("status") in ("healthy", "degraded", "unhealthy")
        if key.startswith("circuit_breaker_"):
            # Circuit breaker component should include state and failure_rate
            if "state" not in comp or "failure_rate" not in comp:
                pytest.skip("Circuit breaker details not exposed in this environment")
        if key == "cache":
            # Cache component provides hit_rate/size where available
            if "hit_rate" not in comp or "size" not in comp:
                pytest.skip("Cache stats not exposed in this environment")
        if key == "metrics":
            # Metrics component should include recent_queries when available
            if "recent_queries" not in comp:
                pytest.skip("Metrics stats not exposed in this environment")
        if key == "batch_processor":
            # Batch processor should include active_jobs and success_rate
            if "active_jobs" not in comp or "success_rate" not in comp:
                pytest.skip("Batch processor stats not exposed in this environment")


def test_rag_cache_warm_sanitizes_unexpected_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    logger_stub = _install_logger_stub(monkeypatch)

    def _raise_cache_error():
        raise RuntimeError("warming backend exploded at /private/rag-cache.db")

    monkeypatch.setattr(rag_health_ep, "get_rag_cache", _raise_cache_error)

    response = client.get("/api/v1/rag/cache/warm")

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to get cache warming status"}
    _assert_sanitized_error_log(logger_stub, "Failed to get warming status")


def test_rag_cache_stats_sanitizes_unexpected_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    logger_stub = _install_logger_stub(monkeypatch)

    def _raise_cache_error():
        raise RuntimeError("cache stats backend exploded at /private/rag-cache.db")

    monkeypatch.setattr(rag_health_ep, "get_rag_cache", _raise_cache_error)

    response = client.get("/api/v1/rag/cache/stats")

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to retrieve cache statistics"}
    _assert_sanitized_error_log(logger_stub, "Failed to get cache statistics")


def test_rag_cache_clear_sanitizes_unexpected_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    logger_stub = _install_logger_stub(monkeypatch)

    def _raise_cache_error():
        raise RuntimeError("cache clear backend exploded at /private/rag-cache.db")

    monkeypatch.setattr(rag_health_ep, "get_rag_cache", _raise_cache_error)

    response = client.post("/api/v1/rag/cache/clear")

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to clear cache"}
    _assert_sanitized_error_log(logger_stub, "Failed to clear cache")


def test_rag_metrics_summary_sanitizes_unexpected_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    logger_stub = _install_logger_stub(monkeypatch)

    def _raise_metrics_error():
        raise RuntimeError("metrics backend exploded at /private/rag-metrics.db")

    monkeypatch.setattr(rag_health_ep, "get_metrics_collector", _raise_metrics_error)

    response = client.get("/api/v1/rag/metrics/summary")

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to get metrics summary"}
    _assert_sanitized_error_log(logger_stub, "Failed to get metrics summary")


def test_rag_cost_summary_sanitizes_unexpected_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    logger_stub = _install_logger_stub(monkeypatch)

    def _raise_cost_error():
        raise RuntimeError("cost tracker exploded at /private/rag-costs.db")

    fake_quick_wins = types.ModuleType("quick_wins")
    fake_quick_wins.get_cost_tracker = _raise_cost_error
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.quick_wins",
        fake_quick_wins,
    )

    response = client.get("/api/v1/rag/costs/summary")

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to get cost summary"}
    _assert_sanitized_error_log(logger_stub, "Failed to get cost summary")


def test_rag_batch_jobs_sanitizes_unexpected_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    logger_stub = _install_logger_stub(monkeypatch)

    def _raise_batch_error():
        raise RuntimeError("batch processor exploded at /private/rag-batch.db")

    monkeypatch.setattr(rag_health_ep, "get_batch_processor", _raise_batch_error)

    response = client.get("/api/v1/rag/batch/jobs")

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to get batch jobs"}
    _assert_sanitized_error_log(logger_stub, "Failed to get batch jobs")


def test_rag_quality_gate_sanitizes_unexpected_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    logger_stub = _install_logger_stub(monkeypatch)

    class _FailingEvaluator:
        def evaluate(self, metrics):
            _ = metrics
            raise RuntimeError("quality gate exploded at /private/rag-quality.db")

    fake_quality_gating = types.ModuleType("quality_gating")
    fake_quality_gating.GatingEvaluator = _FailingEvaluator
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.quality_gating",
        fake_quality_gating,
    )

    response = client.post("/api/v1/rag/quality-gate", json={"answer_relevance": 0.9})

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to run quality gate evaluation"}
    _assert_sanitized_error_log(logger_stub, "Quality gate evaluation failed")


def test_rag_baseline_save_sanitizes_unexpected_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    logger_stub = _install_logger_stub(monkeypatch)

    class _FailingDetector:
        def save_baseline(self, **kwargs):
            _ = kwargs
            raise RuntimeError("baseline backend exploded at /private/rag-regression.db")

    fake_regression = types.ModuleType("regression")
    fake_regression.RegressionDetector = _FailingDetector
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.regression",
        fake_regression,
    )

    response = client.post(
        "/api/v1/rag/baseline/save",
        json={"metrics": {"answer_relevance": 0.9}},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to save metric baseline"}
    _assert_sanitized_error_log(logger_stub, "Baseline save failed")


def test_rag_regression_check_get_sanitizes_unexpected_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    logger_stub = _install_logger_stub(monkeypatch)

    class _FailingDetector:
        def load_baseline(self, baseline_id):
            _ = baseline_id
            raise RuntimeError("regression backend exploded at /private/rag-regression.db")

    fake_regression = types.ModuleType("regression")
    fake_regression.RegressionDetector = _FailingDetector
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.regression",
        fake_regression,
    )

    response = client.get("/api/v1/rag/regression/check", params={"baseline_id": "latest"})

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to check regression"}
    _assert_sanitized_error_log(logger_stub, "Regression check failed")


def test_rag_regression_check_post_sanitizes_unexpected_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    logger_stub = _install_logger_stub(monkeypatch)

    class _FailingDetector:
        def check_regression(self, **kwargs):
            _ = kwargs
            raise RuntimeError("regression backend exploded at /private/rag-regression.db")

    fake_regression = types.ModuleType("regression")
    fake_regression.RegressionDetector = _FailingDetector
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.regression",
        fake_regression,
    )

    response = client.post(
        "/api/v1/rag/regression/check",
        params={"baseline_id": "latest"},
        json={"answer_relevance": 0.9},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to check regression"}
    _assert_sanitized_error_log(logger_stub, "Regression check failed")
